from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from diagnostics_callback import DiagnosticsCallback
from environment import DBRLEnv


def _resolve_training_path(path_value: str | None, orig_cwd: Path) -> Path | None:
    if path_value is None:
        return None
    p = Path(path_value)
    return p if p.is_absolute() else (orig_cwd / p)


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
    eval_step_freq = max(cfg.training.callback_eval_freq // cfg.training.n_envs, 1)

    env = make_vec_env(
        make_env(cfg, 0),
        n_envs=cfg.training.n_envs,
        seed=123,
        vec_env_cls=SubprocVecEnv,
    )

    resume_cfg = getattr(cfg.training, "resume", None)
    resume_enabled = bool(getattr(resume_cfg, "enabled", False))
    if resume_enabled:
        resume_model_path = _resolve_training_path(
            getattr(resume_cfg, "model_file", None),
            orig_cwd,
        )
        resume_vecnorm_path = _resolve_training_path(
            getattr(resume_cfg, "vecnormalize_file", None),
            orig_cwd,
        )
        if resume_model_path is None:
            raise ValueError(
                "Resume enabled, but training.resume.model_file is not set."
            )
        if resume_vecnorm_path is None:
            raise ValueError(
                "Resume enabled, but training.resume.vecnormalize_file is not set."
            )
        if not resume_model_path.exists():
            raise FileNotFoundError(
                f"Resume model file does not exist: {resume_model_path}"
            )
        if not resume_vecnorm_path.exists():
            raise FileNotFoundError(
                f"Resume VecNormalize file does not exist: {resume_vecnorm_path}"
            )
        env = VecNormalize.load(str(resume_vecnorm_path), env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10000.0,
            clip_reward=10000.0,
            training=True,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=vec_step_freq,
        save_path=str(model_save_path),
        name_prefix=cfg.training.callback_model_prefix,
        save_vecnormalize=True,
        verbose=2,
    )

    eval_env = make_vec_env(
        make_env(cfg, 0),
        n_envs=1,
        seed=456,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10000.0,
        clip_reward=10000.0,
        training=False,
    )
    # Use the same running obs/reward stats as training so evaluation matches policy inputs.
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    eval_callback = EvalCallback(
        eval_env,
        # best_model_save_path=str(model_save_path / "best_model"),
        log_path=str(model_save_path / "eval_logs"),
        eval_freq=eval_step_freq,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )

    diagnostics_cb = DiagnosticsCallback(
        entropy_n_samples=getattr(
            cfg.training, "diagnostics_entropy_samples", 512
        ),
    )

    schedule = getattr(cfg.training, "learning_rate_schedule", "constant")
    learning_rate_start = float(cfg.training.learning_rate)
    if schedule == "linear":
        learning_rate = get_linear_fn(
            learning_rate_start,
            float(
                getattr(cfg.training, "learning_rate_end", 1e-5)
            ),
            1.0,
        )
    else:
        learning_rate = lambda _: learning_rate_start
    initial_learning_rate = float(learning_rate(1.0))
    print(
        f"Using learning rate schedule={schedule} "
        f"initial_lr={initial_learning_rate:.10f}"
    )

    policy_net_arch = getattr(cfg.training, "policy_net_arch", None)
    policy_kwargs = (
        {"net_arch": [int(x) for x in policy_net_arch]}
        if policy_net_arch is not None and len(policy_net_arch) > 0
        else None
    )

    ppo_kwargs = dict(
        learning_rate=learning_rate,
        verbose=cfg.training.verbose,
        tensorboard_log=cfg.training.tensorboard_log,
        batch_size=cfg.training.batch_size,
        n_epochs=cfg.training.n_epochs,
        n_steps=cfg.training.n_steps,
    )
    if policy_kwargs is not None:
        ppo_kwargs["policy_kwargs"] = policy_kwargs

    if resume_enabled:
        model = PPO.load(
            str(resume_model_path),
            env=env,
            tensorboard_log=cfg.training.tensorboard_log,
        )
        # PPO checkpoints restore optimizer and LR schedule. Re-apply the
        # configured schedule so resumed runs honor the current config.
        model.learning_rate = learning_rate
        model.lr_schedule = learning_rate
        for param_group in model.policy.optimizer.param_groups:
            param_group["lr"] = initial_learning_rate
    else:
        model = PPO("MultiInputPolicy", env, **ppo_kwargs)

    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        tb_log_name=cfg.training.tb_log_name,
        callback=[checkpoint_callback, eval_callback, diagnostics_cb],
        reset_num_timesteps=bool(
            getattr(resume_cfg, "reset_num_timesteps", False)
        ),
    )

    model.save(str(model_save_path))
    env.save(str(model_save_path) + "_vecnormalize.pkl")


if __name__ == "__main__":
    main()
