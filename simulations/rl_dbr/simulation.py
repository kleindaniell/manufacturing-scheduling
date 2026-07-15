import hydra
from pathlib import Path
from environment import DBRLEnv
from manusim.experiment import ExperimentRunner, make_simulation_factory
from manusim.metrics import ExperimentMetrics
from omegaconf import DictConfig, OmegaConf


def _resolve_path(path: str, base_dir: Path) -> str:
    """Resolve path relative to base_dir if not absolute."""
    p = Path(path)
    return str(base_dir / p) if not p.is_absolute() else path


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="simulation_config",
)
def main(cfg: DictConfig):
    """Main execution function."""
    if not cfg.simulation.training:
        from hydra.utils import get_original_cwd
        base = Path(get_original_cwd())
        if cfg.simulation.get("model_file"):
            cfg.simulation.model_file = _resolve_path(cfg.simulation.model_file, base)
        if cfg.simulation.get("vec_norm_file"):
            cfg.simulation.vec_norm_file = _resolve_path(cfg.simulation.vec_norm_file, base)

    sim_kwargs = {
        "config": OmegaConf.to_container(cfg.simulation, resolve=True),
        "resources": OmegaConf.to_container(cfg.resources, resolve=True),
        "products": OmegaConf.to_container(cfg.products, resolve=True),
        "print_mode": cfg.simulation.print_mode,
    }
    simulation_factory = make_simulation_factory(DBRLEnv, **sim_kwargs)

    experiment = ExperimentRunner(
        simulation=simulation_factory(),
        simulation_factory=simulation_factory,
        number_of_runs=cfg.experiment.number_of_runs,
        n_jobs=cfg.experiment.get("n_jobs", 1),
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
