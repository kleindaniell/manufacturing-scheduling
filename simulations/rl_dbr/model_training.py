from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from diagnostics_callback import DiagnosticsCallback
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
    # Resolve model save path relative to original cwd so checkpoints go to
    # simulations/rl_dbr/models/ppo/ even when Hydra changes cwd to the run dir.
    orig_cwd = Path(get_original_cwd())
    model_save_path = orig_cwd / cfg.training.model_save_path
    model_save_path.mkdir(parents=True, exist_ok=True)

    vec_step_freq = max(cfg.training.callback_save_freq // cfg.training.n_envs, 1)

    env = make_vec_env(
        make_env(cfg, 0),
        n_envs=cfg.training.n_envs,
        seed=123,
        vec_env_cls=SubprocVecEnv,
    )

    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=200.0)

    checkpoint_callback = CheckpointCallback(
        save_freq=vec_step_freq,
        save_path=str(model_save_path),
        name_prefix=cfg.training.callback_model_prefix,
        save_vecnormalize=True,
        verbose=2,
    )

    eval_env = make_env(cfg, 0)()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_save_path / "best_model"),
        log_path=str(model_save_path / "eval_logs"),
        eval_freq=vec_step_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    diagnostics_cb = DiagnosticsCallback(
        entropy_n_samples=getattr(
            cfg.training, "diagnostics_entropy_samples", 512
        ),
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
        callback=[checkpoint_callback, eval_callback, diagnostics_cb],
    )

    model.save(str(model_save_path))
    # env.save(str(model_save_path) + "_vecnormalize.pkl")


if __name__ == "__main__":
    main()
