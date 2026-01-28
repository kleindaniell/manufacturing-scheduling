import hydra
from environment import DBRLEnv
from manusim.experiment import ExperimentRunner
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="simulation_config",
)
def main(cfg: DictConfig):
    """Main execution function."""
    env = DBRLEnv(
        config=cfg.simulation,
        resources=cfg.resources,
        products=cfg.products,
        print_mode=cfg.simulation.print_mode,
    )

    experiment = ExperimentRunner(
        simulation=env,
        number_of_runs=cfg.experiment.number_of_runs,
        save_logs=cfg.experiment.save_logs,
        run_name=cfg.experiment.name,
        seed=cfg.experiment.seed,
    )
    experiment.run_experiment()


if __name__ == "__main__":
    main()
