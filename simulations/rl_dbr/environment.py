from pathlib import Path
from typing import List, Literal, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from manusim.engine.orders import DemandOrder, ProductionOrder
from manusim.factory_sim import FactorySimulation
from stable_baselines3 import PPO


class DBRLEnv(FactorySimulation, gym.Env):
    def __init__(
        self,
        config: dict,
        resources: dict,
        products: dict,
        print_mode="all",
        seed: int = None,
    ):
        super().__init__(
            config,
            resources,
            products,
            print_mode,
            seed,
        )

    def _initiate_custom_env(self):
        self.scheduler_interval = self.config.get("scheduler_interval", 72)
        self.training = self.config.get("training", False)
        self.wip_multiplyer = self.config.get("wip_multiplyer", 0.2)
        self.fg_multiplyer = self.config.get("fg_multiplyer", 0.1)
        self.lost_sales_multiplyer = self.config.get("lost_sales_multiplyer", 0.3)

        self.schedule_max_size = self.config.get("schedule_max_size", 20)

        self.model_path = self.config.get("model_path", None)
        if self.model_path:
            self.model_path = Path(self.model_path)

        self._create_constraint_buffer()
        self._create_spaces()

    def _create_spaces(self):
        n_prod = len(self.products_config.keys())
        self.observation_space = spaces.Dict(
            {
                "constraint_buffer": spaces.Box(
                    low=0, high=np.inf, shape=(2,), dtype=np.float32
                ),
                "products": spaces.Box(
                    low=0, high=np.inf, shape=(n_prod, 2), dtype=np.float32
                ),
            }
        )

        self.action_space = spaces.MultiDiscrete([self.schedule_max_size + 1] * n_prod)

    def _start_custom_process(self):
        self.constraint_resource, self.utilization_df = self.define_constraint()
        self.env.process(self._process_demandOrders())

        for product in self.products_config.keys():
            self.env.process(self._update_finished_goods(product))

    def _create_constraint_buffer(self):
        # Constraint buffers
        self.constraint_buffer_level = 0
        self.constraint_buffer_wip = 0
        self.constraint_buffer_queue = 0

    def _update_finished_goods(self, product):
        qnt = self.products_config[product].get("shipping_buffer", 0)
        yield self.stores.finished_goods[product].put(qnt)

    def _create_custom_logs(self):
        self.logs.create_log("schedule_consumed", self.products_config.keys())
        self.logs.create_log("schedule_planned", self.products_config.keys())
        self.logs.create_log("constraint_buffer_wip", ["general"])
        self.logs.create_log("constraint_buffer_queue", ["general"])
        self.logs.create_log("constraint_buffer_level", ["general"])
        self.logs.create_log("constraint_buffer_target", ["general"])

    def _log_vars(
        self,
        variable: str,
        value,
        product: Optional[str] = None,
    ):
        if self.warmup_finished:
            if product:
                self.logs.log(variable, product, (self.env.now, value))
            else:
                self.logs.log(variable, "general", (self.env.now, round(value, 4)))

    def define_constraint(self) -> Tuple[str, pd.DataFrame]:
        df = pd.DataFrame(
            data=np.zeros(
                shape=(len(self.products_config), len(self.resources_config)),
                dtype=np.float32,
            ),
            index=list(self.products_config.keys()),
            columns=list(self.resources_config.keys()),
        )

        for product in self.products_config.keys():
            product_demand = self.products_config[product].get("demand")
            mean_arrival_rate = product_demand.get("freq").get("params")[0]
            quantity = product_demand.get("quantity").get("params")[0]

            for process in self.stores.processes_value_list[product]:
                mean_processing_time = process["processing_time"]["params"][0]
                resource = process["resource"]

                df.loc[product, resource] += mean_processing_time

            df.loc[product, :] = df.loc[product, :] * (1 / mean_arrival_rate) * quantity

        utilization_df = df.copy()
        constraint_resource = df.sum().sort_values(ascending=False).index[0]

        ccr_setup_time_params = self.stores.resources[constraint_resource].get(
            "setup", {"params": [0]}
        )
        self.ccr_setup_time = ccr_setup_time_params.get("params", [0])[0]

        return constraint_resource, utilization_df

    def _custom_order_in_resource_input(
        self, productionOrder: ProductionOrder, resource: str
    ):
        if resource == self.constraint_resource:
            product = productionOrder.product
            quantity = productionOrder.quantity
            actual_process = productionOrder.process_finished
            product_process = self.stores.processes_value_list[product][actual_process]
            product_processing_time = product_process["processing_time"]["params"][0]
            ccr_time = (product_processing_time * quantity) + self.ccr_setup_time

            self.constraint_buffer_wip -= ccr_time
            self.constraint_buffer_queue += ccr_time
            self.constraint_buffer_level = (
                self.constraint_buffer_wip + self.constraint_buffer_queue
            )

            self._log_vars("constraint_buffer_wip", self.constraint_buffer_wip)
            self._log_vars("constraint_buffer_queue", self.constraint_buffer_queue)
            self._log_vars("constraint_buffer_level", self.constraint_buffer_level)

    def _custom_order_out_resource_input(
        self, productionOrder: ProductionOrder, resource: str
    ):
        if resource == self.constraint_resource:
            product = productionOrder.product
            quantity = productionOrder.quantity
            actual_process = productionOrder.process_finished
            product_process = self.stores.processes_value_list[product][actual_process]
            product_processing_time = product_process["processing_time"]["params"][0]
            ccr_time = (product_processing_time * quantity) + self.ccr_setup_time

            self.constraint_buffer_queue -= ccr_time
            self.constraint_buffer_level = (
                self.constraint_buffer_wip + self.constraint_buffer_queue
            )

            self._log_vars("constraint_buffer_queue", self.constraint_buffer_queue)
            self._log_vars("constraint_buffer_level", self.constraint_buffer_level)

    def scheduler(self):
        if not self.training:
            if self.model_path is None:
                raise ValueError("Model not specified")
            if self.model_path.parent == "ppo":
                model = PPO.load(self.model_path)

            while True:
                obs = self._get_obs()
                actions, _ = model.predict(obs, deterministic=True)
                self.apply_actions(actions)
                yield self.env.timeout(self.scheduler_interval)
        else:
            while True:
                yield self.env.timeout(self.scheduler_interval)

    def apply_actions(self, actions):
        products = list(self.products_config.keys())
        orders: List[Tuple[ProductionOrder, float, float]] = []

        for aid, action in enumerate(actions):
            product = products[aid]

            ccr_processing_time = sum(
                [
                    process["processing_time"]["params"][0]
                    for process in self.stores.processes_value_list[product]
                    if process["resource"] == self.constraint_resource
                ]
            )

            self._log_vars("schedule_planned", product=product, value=action)
            if action <= 0:
                continue

            orders.append(
                (
                    ProductionOrder(
                        product=product,
                        quantity=action,
                        priority=0,
                    ),
                    ccr_processing_time,
                    self.stores.finished_goods[product].level,
                )
            )

        orders = sorted(orders, key=lambda x: x[-1], reverse=False)

        ccr_load_released = 0
        for productionOrder, ccr_time, _ in orders:
            product = productionOrder.product
            quantity = productionOrder.quantity

            if ccr_time == 0:
                productionOrder.schedule = self.env.now
            else:
                productionOrder.schedule = self.env.now + ccr_load_released
                ccr_time = (quantity * ccr_time) + self.ccr_setup_time
                ccr_load_released += ccr_time

            if quantity > 0:
                self.env.process(self.process_order(productionOrder, ccr_time))

    def process_order(self, productionOrder: ProductionOrder, ccr_add: float):
        if (
            productionOrder.schedule is not None
            and productionOrder.schedule > self.env.now
        ):
            delay = productionOrder.schedule - self.env.now
            yield self.env.timeout(delay)

        self.constraint_buffer_wip += ccr_add
        self.constraint_buffer_level = (
            self.constraint_buffer_wip + self.constraint_buffer_queue
        )

        self._log_vars("constraint_buffer_wip", value=self.constraint_buffer_wip)
        self._log_vars("constraint_buffer_level", value=self.constraint_buffer_level)

        self.env.process(self._release_order(productionOrder))

    def _process_demandOrders(self):
        while True:
            demandOrder: DemandOrder = yield self.stores.inbound_demand_orders.get()
            product = demandOrder.product
            yield self.stores.outbound_demand_orders[product].put(demandOrder)

    def _get_obs(self):
        constraint_buffer = np.array(
            [self.constraint_buffer_wip, self.constraint_buffer_queue],
            dtype=np.float32,
        )
        products_obs = []
        for product in self.products_config.keys():
            products_obs.append(
                [
                    self.stores.wip[product].level,
                    self.stores.finished_goods[product].level,
                ]
            )
        return {
            "constraint_buffer": constraint_buffer,
            "products": np.array(products_obs, dtype=np.float32),
        }

    def _get_info(self):
        return {}

    def calculate_reward(self, last_interval: float):
        products = list(self.products_config.keys())
        reward = 1 * len(products) if products else 0

        for product in products:
            wip = (self.wip_multiplyer * self.stores.wip[product].level) ** 2
            fg = (self.fg_multiplyer * self.stores.finished_goods[product].level) ** 2
            
            lost_sales = np.array(self.logs.get_log("lostSales", product))
            if lost_sales.size != 0:
                # Ensure we have at least 2 columns before slicing
                if lost_sales.ndim == 2 and lost_sales.shape[1] >= 2:
                    mask1 = lost_sales[:, 0] >= last_interval
                    mask2 = lost_sales[:, 1] > 0
                    lost_sales = lost_sales[mask1 & mask2]
                    ls = lost_sales.shape[0] * self.lost_sales_multiplyer
                else:
                    ls = 0
            else:
                ls = 0
            reward -= (wip + fg + ls)

        return reward / len(products) if products else 1

    def reset(self, seed=None, options=None):
        super().reset_simulation(seed=seed, log_save_path=self.log_save_path)
        self._initiate_custom_env()
        self._create_constraint_buffer()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, actions):
        # print("actions: ", actions)
        self.apply_actions(actions)

        last_interval = self.env.now
        self.env.run(until=last_interval + self.scheduler_interval)

        obs = self._get_obs()
        info = self._get_info()
        reward = self.calculate_reward(last_interval)
        # print("reward: ", reward)
        terminated = False
        if reward < 0:
            terminated = True
            reward = 0

        truncated = self.env.now >= self.run_until
        # print("obs: ", obs)
        
        return obs, reward, terminated, truncated, info
