"""TensorBoard diagnostics: reward decomposition from env info and per-dimension policy entropy."""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

REWARD_COMPONENT_KEYS = (
    "throughput",
    "lost_sales_penalty",
    "wip_penalty",
    "fg_penalty",
)


class DiagnosticsCallback(BaseCallback):
    """Accumulate reward_components from infos during rollout; log means and entropy at rollout end."""

    def __init__(self, entropy_n_samples: int = 512, verbose: int = 0):
        super().__init__(verbose)
        self.entropy_n_samples = entropy_n_samples
        self._reward_sum: dict[str, float] = {k: 0.0 for k in REWARD_COMPONENT_KEYS}
        self._diag_steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True
        for info in infos:
            if not isinstance(info, dict):
                continue
            rc = info.get("reward_components")
            if not rc:
                continue
            for k in REWARD_COMPONENT_KEYS:
                self._reward_sum[k] += float(rc.get(k, 0.0))
            self._diag_steps += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._diag_steps > 0:
            for k in REWARD_COMPONENT_KEYS:
                self.logger.record(
                    f"reward/decomp/{k}",
                    self._reward_sum[k] / self._diag_steps,
                )
        self._reward_sum = {k: 0.0 for k in REWARD_COMPONENT_KEYS}
        self._diag_steps = 0

        buf = self.model.rollout_buffer
        n = min(self.entropy_n_samples, buf.buffer_size)
        if n <= 0:
            return
        idx = np.random.randint(0, buf.buffer_size, size=n)
        policy = self.model.policy
        try:
            obs_batch_np = {k: buf.observations[k][idx, 0] for k in buf.observations}
        except (KeyError, TypeError, IndexError):
            return
        obs_tensor, _ = policy.obs_to_tensor(obs_batch_np)
        with th.no_grad():
            dist = policy.get_distribution(obs_tensor)
            dists = getattr(dist, "distribution", None)
            if dists is None:
                return
            for i, categorical in enumerate(dists):
                self.logger.record(
                    f"policy/entropy/product_{i}",
                    float(categorical.entropy().mean().item()),
                )
