from enum import Enum
from typing import List, Literal, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
from manusim.engine.orders import DemandOrder, ProductionOrder
from manusim.experiment import ExperimentRunner
from manusim.factory_sim import FactorySimulation
from manusim.metrics import AggMethod, ExperimentMetrics, MetricParams
from omegaconf import DictConfig


class DBRproducts(Enum):
    shippingBufferLevel = MetricParams(AggMethod.mean, False)
    shippingBufferTarget = MetricParams(AggMethod.mean, False)
    scheduleConsumed = MetricParams(AggMethod.mean, False)
    schedulePlanned = MetricParams(AggMethod.mean, False)


class DBRgeneral(Enum):
    constraintBufferWip = MetricParams(AggMethod.mean, False)
    constraintBufferQueue = MetricParams(AggMethod.mean, False)
    constraintBufferLevel = MetricParams(AggMethod.mean, False)
    constraintBufferTarget = MetricParams(AggMethod.mean, False)


class DBRSimulation(FactorySimulation):
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
        self.order_release_limit = self.config.get("order_release_limit", float("inf"))
        self.ccr_release_limit = self.config.get("ccr_release_limit", False)
        self.scheduler_interval = self.config.get("scheduler_interval", 72)
        self.cb_target_level = self.config.get("cb_target_level", float("inf"))
        self.min_lotsize = self.config.get("min_lotsize", 0)
        self.max_lotsize_multiplyer = self.config.get("max_lotsize_multiplyer", 1)

        self._create_constraint_buffer()
        self._create_shipping_buffers()

    def _start_custom_process(self):
        self.constraint_resource, self.utilization_df = self.define_constraint()
        # self.env.process(self._update_constraint_buffer(self.constraint_resource))
        self.env.process(self._process_demandOrders())

        # Update buffers
        self.buffers_update_multiplyer = self.config.get("buffers_update_multiplyer", 1)
        self.buffers_update_warmup = self.config.get("buffers_update_warmup", 0)
        self.sb_update: bool = self.config.get("sb_update", False)
        self.cb_update: bool = self.config.get("cb_update", False)
        self.env.process(self.adjust_buffers())

        for product in self.products_config.keys():
            self.env.process(self._update_finished_goods(product))

    def _create_constraint_buffer(self):
        # Constraint buffers
        self.constraint_buffer_target = self.cb_target_level
        self.constraint_buffer_level = 0
        self.constraint_buffer_wip = 0
        self.constraint_buffer_queue = 0
        self.constraint_buffer_penetration = []

    def _create_shipping_buffers(self):
        # Shipping_buffer

        self.shipping_buffer = {
            p: self.products_config[p].get("shipping_buffer", 0)
            for p in self.products_config.keys()
        }
        self.shipping_buffer_level = {
            p: self.products_config[p].get("shipping_buffer", 0)
            for p in self.products_config.keys()
        }
        self.shipping_buffer_penetration = {p: [] for p in self.products_config.keys()}

        # for product in self.products_config.keys():
        #     qnt = self.products_config[product].get("shipping_buffer", 0)
        #     self.stores.finished_goods[product].put(qnt)

    def _update_finished_goods(self, product):
        qnt = self.products_config[product].get("shipping_buffer", 0)
        yield self.stores.finished_goods[product].put(qnt)

    def _create_custom_logs(self):

        for dbr_product in DBRproducts:
            self.logs.create_log(dbr_product.name, self.products_config.keys())

        for general in DBRgeneral:
            self.logs.create_log(general.name, ["general"])

    def _log_vars(
        self,
        variable,
        value,
        product: Optional[float] = None,
    ):
        if self.warmup_finished:
            if variable == "constraint_buffer_level":
                pass
                # print(self.log_general.constraint_buffer_level)
            match variable:
                case "constraint_buffer_wip":
                    self.log_general.constraint_buffer_wip.append(
                        (self.env.now, round(value, 4))
                    )
                case "constraint_buffer_queue":
                    self.log_general.constraint_buffer_queue.append(
                        (self.env.now, round(value, 4))
                    )
                case "constraint_buffer_level":
                    self.log_general.constraint_buffer_level.append(
                        (self.env.now, round(value, 4))
                    )
                case "constraint_buffer_target":
                    self.log_general.constraint_buffer_target.append(
                        (self.env.now, value)
                    )
                case "shipping_buffer_level":
                    self.log_product.shipping_buffer_level[product].append(
                        (self.env.now, value)
                    )
                case "shipping_buffer_target":
                    self.log_product.shipping_buffer_target[product].append(
                        (self.env.now, value)
                    )
                case "schedule_consumed":
                    self.log_product.schedule_consumed[product].append(
                        (self.env.now, value)
                    )
                case "schedule_planned":
                    self.log_product.schedule_planned[product].append(
                        (self.env.now, value)
                    )

    def define_constraint(self) -> Tuple[str, pd.DataFrame]:
        df = pd.DataFrame(
            data=np.zeros(
                shape=(len(self.products_config), len(self.resources_config)),
                dtype=np.float32,
            ),
            index=self.products_config.keys(),
            columns=self.resources_config.keys(),
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
            "setup", {"params": None}
        )
        self.ccr_setup_time = ccr_setup_time_params.get("params", [0])[0]

        return constraint_resource, utilization_df

    def _custom_order_in_resource_input(
        self, productionOrder: ProductionOrder, resource: str
    ):
        # Process if constraint ressource
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

            self.logs.log(
                DBRgeneral.constraintBufferWip.name,
                key="general",
                value=(self.env.now, self.constraint_buffer_wip),
            )
            self.logs.log(
                DBRgeneral.constraintBufferQueue.name,
                key="general",
                value=(self.env.now, self.constraint_buffer_queue),
            )
            self.logs.log(
                DBRgeneral.constraintBufferLevel.name,
                key="general",
                value=(self.env.now, self.constraint_buffer_level),
            )

    def _custom_order_out_resource_input(
        self, productionOrder: ProductionOrder, resource: str
    ):
        # Process if constraint ressource
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

            self.logs.log(
                DBRgeneral.constraintBufferQueue.name,
                key="general",
                value=(self.env.now, self.constraint_buffer_queue),
            )
            self.logs.log(
                DBRgeneral.constraintBufferLevel.name,
                key="general",
                value=(self.env.now, self.constraint_buffer_level),
            )

    def calculate_shipping_buffer(self, product):
        self.shipping_buffer_level[product] = (
            self.stores.wip[product].level + self.stores.finished_goods[product].level
        )

        return self.shipping_buffer_level[product]

    # Start refactor
    def scheduler(self):
        ccr_setup_time_params = self.stores.resources[self.constraint_resource].get(
            "setup", {"params": None}
        )
        ccr_setup_time = ccr_setup_time_params.get("params", [0])[0]
        while True:
            orders: List[Tuple[ProductionOrder, float, float]] = []
            for product in self.stores.products.keys():
                ccr_processing_time = sum(
                    [
                        process["processing_time"]["params"][0]
                        for process in self.stores.processes_value_list[product]
                        if process["resource"] == self.constraint_resource
                    ]
                )

                replenishment, penetration = self.calculate_replenishment(product)

                self.logs.log(
                    variable=DBRproducts.scheduleConsumed.name,
                    key=product,
                    value=(self.env.now, replenishment),
                )

                # Order quantity
                max_size = round(self.max_lotsize_multiplyer * self.shipping_buffer[product], 0)
                quantity = max(replenishment, self.min_lotsize)
                quantity = min(quantity, max_size)
                self.logs.log(
                    variable=DBRproducts.schedulePlanned.name,
                    key=product,
                    value=(self.env.now, quantity),
                )

                # Check of need replenishment
                if replenishment <= 0:
                    continue

                orders.append(
                    (
                        # Production order
                        ProductionOrder(
                            product=product,
                            quantity=quantity,
                            priority=round(
                                penetration / self.shipping_buffer[product], 3
                            ),
                        ),
                        # ccr processin time
                        ccr_processing_time,
                        # Release priority
                        round(replenishment / self.shipping_buffer[product], 3),
                    )
                )

            # Ordenate by priority
            orders = list(sorted(orders, key=lambda x: x[-1], reverse=True))

            # Set ccr safe load
            if self.ccr_release_limit:
                ccr_safe_load = self.scheduler_interval * self.ccr_release_limit
            else:
                ccr_safe_load = self.constraint_buffer_target - round(
                    self.constraint_buffer_level, 4
                )

            # Release orders based on priority
            orders_released = 0
            ccr_load_released = 0
            ccr_accept_orders = True
            for productionOrder, ccr_time, _ in orders:
                product = productionOrder.product
                quantity = productionOrder.quantity
                release_order = False

                if ccr_time == 0:
                    ccr_time = 0
                    productionOrder.schedule = self.env.now
                    release_order = True
                else:
                    ccr_time = (quantity * ccr_time) + ccr_setup_time

                    if (
                        ccr_safe_load > 0
                        and ccr_safe_load > ccr_load_released
                        and orders_released < self.order_release_limit
                        and ccr_accept_orders
                    ):
                        productionOrder.schedule = self.env.now + ccr_load_released
                        ccr_load_released += ccr_time
                        release_order = True
                        orders_released += 1
                    else:
                        ccr_accept_orders = False

                if quantity > 0 and release_order:
                    self.env.process(self.process_order(productionOrder, ccr_time))

            yield self.env.timeout(self.scheduler_interval)

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

        self.logs.log(
            variable=DBRgeneral.constraintBufferWip.name,
            key="general",
            value=(self.env.now, self.constraint_buffer_wip),
        )
        self.logs.log(
            variable=DBRgeneral.constraintBufferLevel.name,
            key="general",
            value=(self.env.now, self.constraint_buffer_level),
        )

        self.env.process(self._release_order(productionOrder))

    def order_selection(self, resource):
        orders: List[ProductionOrder] = self.stores.resource_input[resource].items

        for id, production_order in enumerate(orders):
            # Get all orders ahead in the system
            ahead_orders: List[ProductionOrder] = []
            for resource_ in self.stores.resources.keys():
                ahead_orders.extend(self.stores.resource_input[resource_].items)
                ahead_orders.extend(self.stores.resource_output[resource_].items)
                ahead_orders.extend(self.stores.resource_transport[resource_].items)
                ahead_orders.extend(self.stores.resource_processing[resource_].items)

            product = production_order.product
            released = production_order.released

            # Calculate quantity of orders ahead for same product
            ahead_quantity = [
                order.quantity
                for order in ahead_orders
                if order.released < released and order.product == product
            ]

            # Calculate priority
            orders[id].priority = (
                sum(ahead_quantity) + self.stores.finished_goods[product].level
            ) / self.shipping_buffer[product]

        # Return order with lowest priority
        if len(orders) > 0:
            selected_order = min(orders, key=lambda x: x.priority)
            # Get order from queue
            productionOrder = yield self.stores.resource_input[resource].get(
                lambda x: x.id == selected_order.id
            )
        else:
            productionOrder = yield self.stores.resource_input[resource].get()

        return productionOrder

    def calculate_replenishment(self, product):
        finished_goods = self.stores.finished_goods[product].level
        target_level = self.shipping_buffer[product]

        penetration = target_level - finished_goods
        replenishment = target_level - self.calculate_shipping_buffer(product)

        return replenishment, penetration

    def _custom_fg_reduced(self, product):
        # Update penetration
        if self.warmup_finished:

            penetration = (
                self.shipping_buffer[product]
                - self.stores.finished_goods[product].level
            )
            self.shipping_buffer_penetration[product].append(
                [self.env.now, penetration]
            )

            self.logs.log(
                variable=DBRproducts.shippingBufferLevel.name,
                key=product,
                value=(self.env.now, self.calculate_shipping_buffer(product)),
            )

        return

    def adjust_constraint_buffer(self, interval) -> bool:

        red_limit = round(self.constraint_buffer_target * (1 / 3), 2)
        green_limit = round(self.constraint_buffer_target * (2 / 3), 2)

        cb_levels = np.array(
            self.logs.get_log(DBRgeneral.constraintBufferQueue.name, "general")
        )

        cb_updated = False

        if cb_levels.shape[0] > 0:

            cb_levels = cb_levels[cb_levels[:, 0] >= interval]

            red_counter = cb_levels[cb_levels[:, 1] < red_limit].shape[0]
            green_counter = cb_levels[cb_levels[:, 1] > green_limit].shape[0]

            cb_updates = cb_levels.shape[0]

            if red_counter >= cb_updates * 0.5:
                self.constraint_buffer_target += round(
                    self.constraint_buffer_target * 0.05, 0
                )
                cb_updated = True

            elif green_counter >= cb_updates * 0.8:
                self.constraint_buffer_target -= 1
                cb_updated = True

            # print("==== Constraint Buffer =====")
            # print(
            #     f"target/level: {self.constraint_buffer_target}/{self.constraint_buffer_level:.0f}"
            # )
            # print(
            #     f"WIP/Queue: {self.logs.get_last_log_value(DBRgeneral.constraintBufferWip.name,"general")[1]}/{self.logs.get_last_log_value(DBRgeneral.constraintBufferQueue.name,"general")[1]}"
            # )
            # print(f"g/r/t: {green_counter}/{red_counter}/{cb_updates}")

        return cb_updated

    def adjust_shipping_buffer(self, product, interval) -> bool:

        red_penetration = round(self.shipping_buffer[product] * (2 / 3), 2)
        green_penetration = round(self.shipping_buffer[product] * (1 / 3), 2)

        sb_penetrations = np.array(self.shipping_buffer_penetration[product])

        sb_updated = False
        if sb_penetrations.shape[0] > 0:

            sb_penetrations = sb_penetrations[sb_penetrations[:, 0] >= interval]

            red_counter = sb_penetrations[
                sb_penetrations[:, 1] >= red_penetration
            ].shape[0]

            green_counter = sb_penetrations[
                sb_penetrations[:, 1] < green_penetration
            ].shape[0]

            sb_updates = sb_penetrations.shape[0]
            if red_counter > sb_updates * 0.5:
                self.shipping_buffer[product] += round(
                    self.shipping_buffer[product] * 0.05, 0
                )
                sb_updated = True

            elif green_counter == sb_updates:
                self.shipping_buffer[product] -= 1
                sb_updated = True

        return sb_updated

    def adjust_buffers(self):

        self.logs.log(
            variable=DBRgeneral.constraintBufferTarget.name,
            key="general",
            value=(self.env.now, self.constraint_buffer_target),
        )

        window_analysis = self.scheduler_interval * self.buffers_update_multiplyer

        yield self.env.timeout(self.buffers_update_warmup)

        while True:

            # print(f"==== {self.env.now:.2f} ====")

            cb_updated = False
            interval = self.env.now - window_analysis
            if self.cb_update:
                cb_updated = self.adjust_constraint_buffer(interval)

            if not cb_updated and self.sb_update:
                for product in self.products_config.keys():
                    self.adjust_shipping_buffer(product, interval)

                # print(
                #     " | ".join(
                #         [
                #             str(self.shipping_buffer[p])
                #             for p in self.products_config.keys()
                #         ]
                #     )
                # )

            # Log buffers
            self.logs.log(
                variable=DBRgeneral.constraintBufferTarget.name,
                key="general",
                value=(self.env.now, self.constraint_buffer_target),
            )
            for product in self.products_config.keys():
                self.logs.log(
                    variable=DBRproducts.shippingBufferTarget.name,
                    key=product,
                    value=(self.env.now, self.shipping_buffer[product]),
                )

            yield self.env.timeout(window_analysis)

    def _process_demandOrders(self):
        while True:
            demandOrder: DemandOrder = yield self.stores.inbound_demand_orders.get()
            product = demandOrder.product
            yield self.stores.outbound_demand_orders[product].put(demandOrder)

    def dbr_general_metrics(self, saved_logs: bool = False):
        df_list = []
        for metric in DBRgeneral:
            metric_df = self.logs.get_variable_logs(
                variable=metric.name, saved_logs=saved_logs
            )
            metric_df = metric_df.pivot_table(
                values="value", index="key", columns="variable", aggfunc="mean"
            )

            df_list.append(metric_df)
        return pd.concat(df_list, axis=1)

    def dbr_product_metrics(self, saved_logs: bool = False):
        df_list = []
        for metric in DBRproducts:
            metric_df = self.logs.get_variable_logs(
                variable=metric.name, saved_logs=saved_logs
            )
            metric_df = metric_df.pivot_table(
                values="value", index="key", columns="variable", aggfunc="mean"
            )

            df_list.append(metric_df)
        return pd.concat(df_list, axis=1)

    def save_custom_metrics(self, save_path, saved_logs=False):
        products_df = self.dbr_product_metrics(saved_logs=saved_logs)
        general_df = self.dbr_general_metrics(saved_logs=saved_logs)

        save_path.mkdir(exist_ok=True, parents=True)
        products_df.to_csv(save_path / "metrics_dbr_product.csv")
        general_df.to_csv(save_path / "metrics_dbr_general.csv")

    def print_custom_metrics(self):
        """Print DBR metrics"""

        # Shipping buffer print
        print("DBR - SHIPPING BUFFER:")
        dbr_product_df = self.dbr_product_metrics()
        if not dbr_product_df.empty:
            print(dbr_product_df)
            print("\n")
        else:
            print("Empty metrics")
            print("\n")

        # Constraint buffer print
        print("DBR - CONSTRAINT BUFFER:")
        dbr_general_df = self.dbr_general_metrics()
        if not dbr_general_df.empty:
            print(dbr_general_df)
            print("\n")
        else:
            print("Empty metrics")
            print("\n")


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config",
)
def main(cfg: DictConfig):
    """Main execution function."""
    sim = DBRSimulation(
        config=cfg.simulation,
        resources=cfg.resources,
        products=cfg.products,
        print_mode=cfg.simulation.print_mode,
    )

    experiment = ExperimentRunner(
        simulation=sim,
        number_of_runs=cfg.experiment.number_of_runs,
        save_logs=cfg.experiment.save_logs,
        run_name=cfg.experiment.name,
        seed=cfg.experiment.seed,
    )
    experiment.run_experiment()

    metrics = ExperimentMetrics(experiment.save_folder_path, config=cfg)

    metrics.read_runs_metrics()
    stats_df = metrics.save_stats(0.95, 0.05)
    print("=" * 50)
    print("Experiment Stats")
    print("=" * 50)
    print(stats_df)
    print("=" * 50)


if __name__ == "__main__":
    exit(main())
