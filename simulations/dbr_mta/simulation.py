from typing import Tuple, List, Literal, Optional

import pandas as pd
import numpy as np

from manusim.factory_sim import FactorySimulation
from manusim.experiment import ExperimentRunner
from manusim.engine.orders import ProductionOrder, DemandOrder
from manusim.engine.cli_config import create_experiment_parser
from manusim.engine.utils import load_yaml


class DBRSimulation(FactorySimulation):
    def __init__(
        self,
        config,
        resources,
        products,
        save_logs=True,
        print_mode="metrics",
        seed=None,
        queue_order_selection=None,
    ):
        super().__init__(
            config,
            resources,
            products,
            save_logs,
            print_mode,
            seed,
            queue_order_selection,
        )

        self.order_release_limit = self.config.get("order_release_limit", float("inf"))
        self.ccr_release_limit = self.config.get("ccr_release_limit", False)
        self.scheduler_interval = self.config.get("scheduler_interval", 72)
        self.cb_target_level = self.config.get("cb_target_level", float("inf"))

        self._create_constraint_buffer()
        self._create_shipping_buffers()

    def _start_custom_process(self):
        self.contraint_resource, self.utilization_df = self.define_constraint()
        self.env.process(self._update_constraint_buffer(self.contraint_resource))
        self.env.process(self._process_demandOrders())
        # Update finihed goods
        self.sb_update: bool = self.config.get("sb_update", False)
        self.cb_update: bool = self.config.get("cb_update", False)
        for product in self.products_config.keys():
            self.env.process(self._update_finished_goods(product))
            if self.sb_update:
                self.env.process(self.adjust_shipping_buffer(product))

    def _create_constraint_buffer(self):
        # Constraint buffers
        self.constraint_buffer = self.config.get("cb_target_level", float("inf"))
        self.constraint_buffer_level = 0

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

        for product in self.products_config.keys():
            qnt = self.products_config[product].get("shipping_buffer", 0)
            self.stores.finished_goods[product].put(qnt)

    def _update_finished_goods(self, product):
        qnt = self.products_config[product].get("shipping_buffer", 0)
        yield self.stores.finished_goods[product].put(qnt)

    def _create_custom_logs(self):
        custom_logs = {
            "products": {
                "shipping_buffer_level": {p: [] for p in self.products_config.keys()},
                "shipping_buffer_target": {p: [] for p in self.products_config.keys()},
                "schedule_consumed": {p: [] for p in self.products_config.keys()},
            },
            "general": {"constraint_buffer_level": [], "constraint_buffer_target": []},
        }
        return custom_logs

    def _log_vars(
        self,
        variable: Literal[
            "constraint_buffer_level",
            "constraint_buffer_target",
            "shipping_buffer_level",
            "shipping_buffer_target",
            "schedule_consumed",
        ],
        value,
        product: Optional[float] = None,
    ):
        if self.warmup <= self.env.now and self.save_logs:
            match variable:
                case "constraint_buffer_level":
                    self.log_general.constraint_buffer_level.append(
                        (self.env.now, value)
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

    def _register_custom_logs(self):
        def register_logs():
            yield self.env.timeout(self.warmup)
            while True:
                self._log_vars(
                    "constraint_buffer_level", value=self.constraint_buffer_level
                )
                self._log_vars("constraint_buffer_target", value=self.constraint_buffer)

                for product in self.products_config:
                    self._log_vars(
                        "shipping_buffer_level",
                        product=product,
                        value=self.calculate_shipping_buffer(product),
                    )
                    self._log_vars(
                        "shipping_buffer_target",
                        product=product,
                        value=self.shipping_buffer[product],
                    )
                yield self.env.timeout(self.log_interval)

        # self.env.process(register_logs())
        pass

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
        return constraint_resource, utilization_df

    def _update_constraint_buffer(self, constraint):
        ccr_setup_time_params = self.stores.resources[self.contraint_resource].get(
            "setup", {"params": None}
        )
        ccr_setup_time = ccr_setup_time_params.get("params", [0])[0]
        while True:
            productionOrder: ProductionOrder = yield self.stores.resource_finished[
                constraint
            ].get()
            product = productionOrder.product
            quantity = productionOrder.quantity
            actual_process = productionOrder.process_finished - 1
            product_process = self.stores.processes_value_list[product][actual_process]
            product_processing_time = product_process["processing_time"]["params"][0]
            ccr_time = product_processing_time * quantity + ccr_setup_time
            self.constraint_buffer_level -= ccr_time

            self._log_vars("constraint_buffer_level", self.constraint_buffer_level)

    def calculate_shipping_buffer(self, product):
        self.shipping_buffer_level[product] = (
            self.stores.wip[product].level + self.stores.finished_goods[product].level
        )

        return self.shipping_buffer_level[product]

    # Start refactor
    def scheduler(self):
        ccr_setup_time_params = self.stores.resources[self.contraint_resource].get(
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
                        if process["resource"] == self.contraint_resource
                    ]
                )

                replenishment, penetration = self.calculate_replenishment(product)

                self._log_vars(
                    "schedule_consumed", product=product, value=replenishment
                )

                min_lotsize = self.config.get("min_lotsize", 0)

                orders.append(
                    (
                        # Production order
                        ProductionOrder(
                            product=product,
                            quantity=max(replenishment, min_lotsize),
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
                ccr_safe_load = self.constraint_buffer - self.constraint_buffer_level

            # Release orders based on priority
            if ccr_safe_load > 0:
                orders_released = 0
                for productionOrder, ccr_time, _ in orders:
                    product = productionOrder.product
                    quantity = productionOrder.quantity
                    release_order = False
                    if ccr_time > 0:
                        if (
                            ccr_safe_load > 0
                            and orders_released < self.order_release_limit
                        ):
                            ccr_time = (quantity * ccr_time) + ccr_setup_time
                            productionOrder.schedule = self.env.now + ccr_time
                            release_order = True
                            orders_released += 1
                    else:
                        ccr_time = 0
                        productionOrder.schedule = self.env.now
                        release_order = True

                    if quantity > 0 and release_order:
                        self.env.process(self.process_order(productionOrder, ccr_time))
                        ccr_safe_load -= ccr_time

            yield self.env.timeout(self.scheduler_interval)

    def process_order(self, productionOrder: ProductionOrder, ccr_add: float):
        if (
            productionOrder.schedule is not None
            and productionOrder.schedule > self.env.now
        ):
            delay = productionOrder.schedule - self.env.now
            yield self.env.timeout(delay)
        product = productionOrder.product
        self.constraint_buffer_level += ccr_add

        self._log_vars("constraint_buffer_level", value=self.constraint_buffer_level)

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
        replenishment = max(target_level - self.calculate_shipping_buffer(product), 0)

        return replenishment, penetration

    def process_fg_reduce(self, product):
        # Update penetration
        if self.env.now >= self.warmup:
            penetration = (
                self.shipping_buffer[product]
                - self.stores.finished_goods[product].level
            )
            self.shipping_buffer_penetration[product].append(
                [self.env.now, penetration]
            )
            self._log_vars(
                "shipping_buffer_level",
                product=product,
                value=self.calculate_shipping_buffer(product),
            )

        return

    def adjust_shipping_buffer(self, product):
        adjust_multiply = 20
        adjust_interval = self.scheduler_interval * adjust_multiply
        yield self.env.timeout(self.warmup)
        while True:
            red_penetration = round(self.shipping_buffer[product] * (2 / 3), 2)
            green_penetration = round(self.shipping_buffer[product] * (1 / 3), 2)

            interval = self.env.now - adjust_interval
            total_penetrations = np.array(self.shipping_buffer_penetration[product])

            if total_penetrations.shape[0] > 0:
                self.shipping_buffer_penetration[product] = list(
                    filter(
                        lambda x: x[0] > interval,
                        self.shipping_buffer_penetration[product],
                    )
                )
                total_penetrations = total_penetrations[
                    total_penetrations[:, 0] >= interval
                ]

                red_counter = total_penetrations[
                    total_penetrations[:, 1] >= red_penetration
                ].shape[0]
                green_counter = total_penetrations[
                    total_penetrations[:, 1] < green_penetration
                ].shape[0]

                if product == "produto10":
                    print(
                        f"{self.env.now} - {interval} - {self.shipping_buffer[product]}:{red_penetration}:{green_penetration} - {red_counter}/{len(total_penetrations)} - {green_counter}/{len(total_penetrations)}"
                    )

                if red_counter > (total_penetrations.shape[0] * 0.8):
                    self.shipping_buffer[product] += 1
                    yield self.env.timeout(adjust_interval)

                elif green_counter == total_penetrations.shape[0]:
                    self.shipping_buffer[product] -= 1
                    yield self.env.timeout(adjust_interval)

                # Log buffer
                self._log_vars(
                    "shipping_buffer_target",
                    product=product,
                    value=self.shipping_buffer[product],
                )

            yield self.env.timeout(adjust_interval)

    def _process_demandOrders(self):
        while True:
            demandOrder: DemandOrder = yield self.stores.inbound_demand_orders.get()
            product = demandOrder.product
            yield self.stores.outbound_demand_orders[product].put(demandOrder)

    def print_custom_metrics(self):
        """Print DBR metrics"""

        # Shipping buffer print
        print("DBR - SHIPPING BUFFER:")
        logs_df = self.log_product.to_dataframe()
        logs_sb = logs_df.loc[
            logs_df["variable"].isin(
                ["shipping_buffer_target", "shipping_buffer_level"]
            )
        ]
        if not logs_sb.empty:
            logs_sb = logs_sb.pivot_table(
                values="value", index="product", columns="variable"
            )
            print(logs_sb)
            print("\n")
        else:
            print("Empty metrics")
            print("\n")

        # Constraint buffer print
        print("DBR - CONSTRAINT BUFFER:")
        logs_df = self.log_general.to_dataframe()
        logs_cb = logs_df.loc[
            logs_df["variable"].isin(
                ["constraint_buffer_target", "constraint_buffer_level"]
            )
        ]
        if not logs_cb.empty:
            print(logs_cb[["variable", "value"]].groupby("variable").mean())
            print("\n")
        else:
            print("Empty metrics")
            print("\n")


def main():
    """Main execution function."""
    parser = create_experiment_parser()
    args = parser.parse_args()

    # Determine paths
    if args.save_folder is None:
        raise ValueError("Experiment folder not specified")

    save_folder = args.save_folder
    config_path = args.config
    products_path = args.products
    resources_path = args.resources
    # Load configurations
    try:
        config = load_yaml(config_path)
        resources_cfg = load_yaml(resources_path)
        products_cfg = load_yaml(products_path)
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        return 1
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    sim = DBRSimulation(
        config=config,
        resources=resources_cfg,
        products=products_cfg,
        save_logs=True,
        print_mode="metrics",
        seed=args.exp_seed,
    )

    # Create and run experiment
    # try:
    experiment = ExperimentRunner(
        simulation=sim,
        number_of_runs=args.number_of_runs,
        save_folder_path=save_folder,
        run_name=args.name,
        seed=args.exp_seed,
    )
    experiment.run_experiment()


if __name__ == "__main__":
    exit(main())
