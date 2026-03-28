from math import prod
from pathlib import Path
import pickle
from typing import List, Literal, Optional, Tuple


import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from manusim.engine.orders import DemandOrder, ProductionOrder
from manusim.factory_sim import FactorySimulation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize



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
        reward_cfg = self.config.get("reward") or {}
        self.wip_multiplier = reward_cfg.get("wip_multiplier")
        if self.wip_multiplier is None:
            self.wip_multiplier = self.config.get("wip_multiplyer", 0.2)
        self.fg_multiplier = reward_cfg.get("fg_multiplier")
        if self.fg_multiplier is None:
            self.fg_multiplier = self.config.get("fg_multiplyer", 0.1)
        self.lost_sales_multiplier = reward_cfg.get("lost_sales_multiplier")
        if self.lost_sales_multiplier is None:
            self.lost_sales_multiplier = self.config.get("lost_sales_multiplyer", 0.3)
        self.load_termination_threshold = self.config.get('load_termination_threshold', 0)
        self.space_scale_buffer = self.config.get('space_scale_buffer', 1)
        self.space_scale_products = self.config.get('space_scale_products', 1)

        self.schedule_max_size = self.config.get("schedule_max_size", 20)

        self.model_file = self.config.get("model_file", None)
        self.vec_norm_file = self.config.get("vec_norm_file", None)
        self.vec_normalize = None

        self._create_constraint_buffer()
        self._create_spaces()

        if not self.training and self.vec_norm_file:
            venv = DummyVecEnv([lambda: self])
            self.vec_normalize = VecNormalize.load(self.vec_norm_file, venv)
            print(f"Loaded normalization: {self.vec_norm_file}")
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = True


    def _create_spaces(self):
        self.observation_space = spaces.Dict(
            {
                "constraint_buffer": spaces.Box(
                    low=0, high=np.inf, shape=(2,), dtype=np.float32
                ),
                "products": spaces.Box(
                    low=0, high=np.inf, shape=(self.n_prod, 2), dtype=np.float32
                ),
            }
        )


        self.action_space = spaces.MultiDiscrete([self.schedule_max_size + 1] * self.n_prod)


    def _start_custom_process(self):
        self.constraint_resource, self.utilization_df = self.define_constraint()

        # Cache product list and count for performance
        self.products_list = list(self.products_config.keys())
        self.n_prod = len(self.products_list)

        # Track lost sales and deliveries per scheduler interval during training
        self.lost_sales_count = {product: 0 for product in self.products_list}
        self.interval_deliveries = {product: 0 for product in self.products_list}

        # CCR processing times will be cached after constraint_resource is defined
        self.ccr_processing_times = {}

        # Cache CCR processing times now that constraint_resource is defined
        for product in self.products_list:
            self.ccr_processing_times[product] = sum(
                process["processing_time"]["params"][0]
                for process in self.stores.processes_value_list[product]
                if process["resource"] == self.constraint_resource
            )
        
        self.env.process(self._process_demandOrders())


        for product in self.products_config.keys():
            self.env.process(self._update_finished_goods(product))


    def _create_constraint_buffer(self):
        # Constraint buffers
        # Note: constraint_buffer_level is computed as wip + queue, no need to store separately
        self.constraint_buffer_wip = 0
        self.constraint_buffer_queue = 0

    @property
    def constraint_buffer_level(self):
        """Compute constraint buffer level from wip and queue"""
        return self.constraint_buffer_wip + self.constraint_buffer_queue


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
        # Skip logging during training for performance optimization
        if self.training:
            return
        
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
            # constraint_buffer_level is now a property, no need to update


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
            # constraint_buffer_level is now a property, no need to update


            self._log_vars("constraint_buffer_queue", self.constraint_buffer_queue)
            self._log_vars("constraint_buffer_level", self.constraint_buffer_level)


    def scheduler(self):
        if not self.training:
            if self.model_file is None:
                raise ValueError("Model not specified")
            
            model = PPO.load(self.model_file)
            print(f"\nLoaded model: {self.model_file}")

            while True:
                obs = self._get_obs()
                print(f'{self.env.now} - OBS:\n{obs}\n')
                if self.vec_normalize:
                    obs = self.vec_normalize.normalize_obs(obs)
                    print(f'{self.env.now} - OBS normalized:\n{obs}\n')
                
                actions, _ = model.predict(obs, deterministic=True)
                print(f'{self.env.now} - Actions:\n{actions}\n')
                
                self.apply_actions(actions)
                yield self.env.timeout(self.scheduler_interval)
        else:
            while True:
                yield self.env.timeout(self.scheduler_interval)


    def apply_actions(self, actions):
        products = self.products_list  # Use cached list
        orders: List[Tuple[ProductionOrder, float, float]] = []


        for aid, action in enumerate(actions):
            product = products[aid]


            # Use cached ccr_processing_time
            ccr_processing_time = self.ccr_processing_times[product]


            # Only log if not training (logging disabled during training for performance)
            if not self.training:
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
        # constraint_buffer_level is now a property, no need to update


        self._log_vars("constraint_buffer_wip", value=self.constraint_buffer_wip)
        self._log_vars("constraint_buffer_level", value=self.constraint_buffer_level)


        self.env.process(self._release_order(productionOrder))


    def _process_demandOrders(self):
        while True:
            demandOrder: DemandOrder = yield self.stores.inbound_demand_orders.get()
            product = demandOrder.product
            yield self.stores.outbound_demand_orders[product].put(demandOrder)

    def _log_delivery_performance(self, demand_order: DemandOrder) -> None:
        """Override parent method to track lost sales and deliveries during training"""
        product = demand_order.product
        quantity = demand_order.quantity
        delivered = demand_order.delivered

        if self.training:
            if delivered:
                self.interval_deliveries[product] += quantity
            else:
                self.lost_sales_count[product] += quantity

        # Call parent method for logging (will be skipped if training due to _log_vars optimization)
        super()._log_delivery_performance(demand_order)


    def _get_obs(self):
        constraint_buffer = np.array(
            [self.constraint_buffer_wip, self.constraint_buffer_queue],
            dtype=np.float32,
        )
        
        # Pre-allocate array instead of list comprehension for better performance
        products_obs = np.zeros((self.n_prod, 2), dtype=np.float32)
        for idx, product in enumerate(self.products_list):
            products_obs[idx, 0] = self.stores.wip[product].level
            products_obs[idx, 1] = self.stores.finished_goods[product].level
        
        return {
            "constraint_buffer": constraint_buffer / self.space_scale_buffer,
            "products": products_obs / self.space_scale_products,
        }


    def _get_info(self):
        return {}

    def _sum_log_quantity_since(self, variable: str, product: str, since_time: float) -> float:
        arr = np.array(self.logs.get_log(variable, product))
        if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 2:
            return 0.0
        mask = arr[:, 0] >= since_time
        return float(np.sum(arr[mask, 1]))

    def calculate_reward(self, last_interval: float) -> Tuple[float, dict]:
        products = self.products_list
        if not products:
            return 0.0, {}

        if self.training:
            units_delivered = sum(self.interval_deliveries[p] for p in products)
            lost_sales_units = sum(self.lost_sales_count[p] for p in products)
            for p in products:
                self.interval_deliveries[p] = 0
                self.lost_sales_count[p] = 0
        else:
            units_delivered = sum(
                self._sum_log_quantity_since("deliveredOntime", p, last_interval)
                + self._sum_log_quantity_since("deliveredLate", p, last_interval)
                for p in products
            )
            lost_sales_units = sum(
                self._sum_log_quantity_since("lostSales", p, last_interval)
                for p in products
            )

        total_wip = sum(self.stores.wip[p].level for p in products)
        total_fg = sum(self.stores.finished_goods[p].level for p in products)
        wip_cost = self.wip_multiplier * total_wip
        fg_cost = self.fg_multiplier * total_fg

        lost_sales_penalty = self.lost_sales_multiplier * lost_sales_units
        reward_components = {
            "throughput": float(units_delivered),
            "lost_sales_penalty": float(lost_sales_penalty),
            "wip_penalty": float(wip_cost),
            "fg_penalty": float(fg_cost),
        }
        reward = (
            reward_components["throughput"]
            - reward_components["lost_sales_penalty"]
            - reward_components["wip_penalty"]
            - reward_components["fg_penalty"]
        )
        return reward, reward_components


    def reset(self, seed=None, options=None):
        super().reset_simulation(seed=seed, log_save_path=self.log_save_path)
        self._initiate_custom_env()
        self._create_constraint_buffer()
        self.lost_sales_count = {product: 0 for product in self.products_list}
        self.interval_deliveries = {product: 0 for product in self.products_list}
        obs = self._get_obs()
        info = self._get_info()
        return obs, info


    def step(self, actions):
        # print("actions: ", actions)
        self.apply_actions(actions)

        last_interval = self.env.now
        self.env.run(until=last_interval + self.scheduler_interval)

        obs = self._get_obs()
        reward, reward_components = self.calculate_reward(last_interval)
        info = self._get_info()
        info["reward_components"] = reward_components
        
        terminated = False
        total_load = sum(self.stores.wip[p].level + self.stores.finished_goods[p].level for p in self.products_list)
        if self.load_termination_threshold > 0 and total_load>=self.load_termination_threshold:
            terminated = True

        truncated = self.env.now >= self.run_until
        # print("obs: ", obs)
        
        return obs, reward, terminated, truncated, info
