"""Advantage computation utilities for GRPO and V-GRPO."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_grpo_advantage(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    # Module goal: compute baseline GRPO normalized group advantages.
    # Inputs: one response group reward tensor (G,).
    # Outputs: GRPO advantage tensor (G,) with population-variance normalization.
    # Paper mapping: Eq.(1) per-group sigma_G denominator.
    mu = rewards.mean()
    sigma = rewards.std(unbiased=False)
    return (rewards - mu) / (sigma + eps)


def compute_vgrpo_advantage(
    rewards: Tensor,
    sigma_global: float,
    clip_C: float = 10.0,
    eps: float = 1e-8,
) -> Tensor:
    # Module goal: compute V-GRPO advantages with task-global denominator.
    # Inputs: reward tensor (G,) and external sigma_hat_global(T) scalar.
    # Outputs: clipped advantage tensor (G,).
    # Paper mapping: Eq.(2) global normalization + Eq.(3) hard clip.
    mu = rewards.mean()
    adv = (rewards - mu) / (float(sigma_global) + eps)
    return adv.clamp(-clip_C, clip_C)
