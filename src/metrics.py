"""Training diagnostics for GRPO/V-GRPO mechanism validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor


@dataclass
class StepMetrics:
    mean_sigma_g: float
    frac_sigma_g_zero: float
    mean_entropy: float
    mean_abs_advantage: float
    frac_advantage_clipped: float
    sigma_global: float
    reward_mean: float
    reward_std: float

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


def compute_group_sigmas(rewards: Tensor) -> Tensor:
    return rewards.std(dim=-1, unbiased=False)


def compute_step_metrics(
    rewards: Tensor,
    advantages: Tensor,
    entropies: Tensor,
    sigma_global: float,
    clip_c: float,
) -> StepMetrics:
    # Module goal: compute mechanism-aligned diagnostics each training step.
    # Inputs: batch rewards/advantages, policy entropy, sigma_global and clip C.
    # Outputs: scalar metrics for logging dashboards and reproducibility checks.
    # Paper mapping: sigma_G behavior, entropy shift, clipping activity.
    sigmas = compute_group_sigmas(rewards)
    clipped = advantages.abs() >= (clip_c - 1e-12)

    return StepMetrics(
        mean_sigma_g=float(sigmas.mean().item()),
        frac_sigma_g_zero=float((sigmas == 0).float().mean().item()),
        mean_entropy=float(entropies.mean().item()),
        mean_abs_advantage=float(advantages.abs().mean().item()),
        frac_advantage_clipped=float(clipped.float().mean().item()),
        sigma_global=float(sigma_global),
        reward_mean=float(rewards.mean().item()),
        reward_std=float(rewards.std(unbiased=False).item()),
    )
