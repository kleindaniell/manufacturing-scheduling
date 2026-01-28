import numpy as np
import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from environment import DBRLEnv


def make_env(cfg: DictConfig, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.
    """

    def __init__():
        env = DBRLEnv(
            config=cfg.environment,
            resources=cfg.resources,
            products=cfg.products,
            print_mode=cfg.environment.print_mode,
        )

        return env

    return __init__


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="train_config",
)
def main(cfg: DictConfig):
    """Main execution function."""
    env = make_vec_env(
        make_env(cfg, 0),
        n_envs=cfg.training.n_envs,
        seed=123,
        vec_env_cls=SubprocVecEnv,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=cfg.training.learning_rate,
        verbose=cfg.training.verbose,
        tensorboard_log=cfg.training.tensorboard_log,
        batch_size=cfg.training.batch_size,
        n_epochs=cfg.training.n_epochs,
        n_steps=cfg.training.n_steps,
    )

    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        tb_log_name=cfg.training.tb_log_name,
    )

    model.save(cfg.training.model_save_path)


if __name__ == "__main__":
    main()
