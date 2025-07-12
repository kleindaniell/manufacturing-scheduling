from typing import Tuple

import pandas as pd
import numpy as np

from manusim.factory_sim import FactorySimulation
from manusim.experiment import ExperimentRunner
from manusim.engine.orders import ProductionOrder, DemandOrder
from manusim.engine.cli_config import create_experiment_parser
from manusim.engine.utils import load_yaml


class ArticleSimulation(FactorySimulation):
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

        self._create_constraint_buffer()
        self._create_shipping_buffers()

    def _start_custom_process(self):
        self.contraint_resource, self.utilization_df = self.define_constraint()
        self.env.process(self._update_buffer(self.contraint_resource))

    def _create_constraint_buffer(self):

        # Constraint buffers
        self.cb_level = self.config.get("cb_level", 0)
        self.constraint_buffer = self.cb_level
        self.constraint_buffer_level = 0

    def _create_shipping_buffers(self):
        # Shipping_buffer
        self.shipping_buffer = self.config.get("sb_level", 0)
        self.shipping_buffer_level = {}

        for product in self.products_config:
            # self.shipping_buffer[product] = self.products_config[product].get(
            #     "shipping_buffer", 0
            # )
            self.shipping_buffer_level[product] = 0

    def _create_custom_logs(self):
        custom_logs = {
            "products": {
                "shipping_buffer": {p: [] for p in self.products_config.keys()},
            },
            "general": {"constraint_buffer": []},
        }
        return custom_logs

    def _register_custom_logs(self):
        def register_logs():
            yield self.env.timeout(self.warmup)
            while True:
                self.log_general.constraint_buffer.append(
                    (self.env.now, self.constraint_buffer_level)
                )
                for product in self.products_config:
                    self.log_product.shipping_buffer[product].append(
                        (self.env.now, self.calculate_shipping_buffer(product))
                    )
                yield self.env.timeout(self.log_interval)

        self.env.process(register_logs())

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

    def _update_buffer(self, constraint):
        while True:
            productionOrder: ProductionOrder = yield self.stores.resource_finished[
                constraint
            ].get()
            product = productionOrder.product
            actual_process = productionOrder.process_finished - 1
            product_process = self.stores.processes_value_list[product][actual_process]
            product_processing_time = product_process["processing_time"]["params"][0]
            self.constraint_buffer_level -= product_processing_time

    def calculate_shipping_buffer(self, product):
        self.shipping_buffer_level[product] = (
            self.stores.wip[product].level + self.stores.finished_goods[product].level
        )

        return self.shipping_buffer_level

    def scheduler(self):
        while True:
            demandOrder: DemandOrder = yield self.stores.inbound_demand_orders.get()
            product = demandOrder.product
            quantity = demandOrder.quantity
            duedate = demandOrder.duedate

            ccr_processing_time = sum(
                [
                    process["processing_time"]["params"][0]
                    for process in self.stores.processes_value_list[product]
                    if process["resource"] == self.contraint_resource
                ]
            )

            if ccr_processing_time > 0:
                # schedule = (
                #     duedate
                #     - (self.shipping_buffer + ccr_processing_time)
                #     - self.constraint_buffer
                # )

                buffer_diff = self.constraint_buffer_level - self.constraint_buffer
                schedule = (
                    self.env.now + buffer_diff if buffer_diff > 0 else self.env.now
                )

            else:
                schedule = self.env.now

            productionOrder = ProductionOrder(product=product, quantity=quantity)
            productionOrder.schedule = schedule
            productionOrder.duedate = demandOrder.duedate
            productionOrder.priority = 0
            self.env.process(
                self.process_order(productionOrder, ccr_processing_time, demandOrder)
            )
            yield self.stores.outbound_demand_orders[product].put(demandOrder)

    def process_order(
        self, productionOrder: ProductionOrder, ccr_processing_time: float, do
    ):

        if (
            productionOrder.schedule is not None
            and productionOrder.schedule > self.env.now
        ):
            delay = productionOrder.schedule - self.env.now
            yield self.env.timeout(delay)

        self.constraint_buffer_level += ccr_processing_time
        # print(f"=== Release: {self.env.now} -> \n{productionOrder}\n{do}")
        self.env.process(self._release_order(productionOrder))


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

    sim = ArticleSimulation(
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
