"""GRPO/V-GRPO trainer utilities with switchable advantage denominator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import Tensor

from config import TrainConfig
from src.advantage import compute_grpo_advantage, compute_vgrpo_advantage
from src.metrics import StepMetrics, compute_step_metrics
from src.variance_tracker import GlobalVarianceTracker


@dataclass
class TrainerOutput:
    advantages: Tensor
    metrics: StepMetrics


class VGRPOTrainerCore:
    # Module goal: provide GRPO/V-GRPO advantage path with tracker integration.
    # Inputs: reward tensor (B,G), task labels (B), entropy tensor.
    # Outputs: computed advantages and diagnostic metrics for logging.
    # Paper mapping: Eq.(1)->Eq.(2) denominator swap + Eq.(3) clipping.
    def __init__(
        self,
        config: TrainConfig,
        variance_tracker: GlobalVarianceTracker | None = None,
        update_before_compute: bool = True,
    ) -> None:
        self.config = config
        self.variance_tracker = variance_tracker or GlobalVarianceTracker(alpha=config.ema_alpha)
        self.update_before_compute = update_before_compute

    def _sync_sigma(self, sigma: float) -> float:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return sigma

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sigma_t = torch.tensor([sigma], device=dev)
        torch.distributed.broadcast(sigma_t, src=0)
        return float(sigma_t.item())

    def compute_advantages(self, rewards: Tensor, task_types: Sequence[str]) -> Tensor:
        if rewards.ndim != 2:
            raise ValueError("rewards must be shape (B, G)")
        if len(task_types) != rewards.shape[0]:
            raise ValueError("task_types length must match batch size")

        if not self.config.use_vgrpo:
            return torch.stack(
                [compute_grpo_advantage(rewards[b], eps=self.config.eps) for b in range(rewards.shape[0])]
            )

        advs: List[Tensor] = []
        for b in range(rewards.shape[0]):
            task = task_types[b]
            group_rewards = rewards[b]

            if self.update_before_compute:
                self.variance_tracker.update(task, group_rewards.detach().cpu().tolist())

            sigma_global = self._sync_sigma(self.variance_tracker.get(task))
            adv = compute_vgrpo_advantage(
                group_rewards,
                sigma_global=sigma_global,
                clip_C=self.config.vgrpo_clip_c,
                eps=self.config.eps,
            )
            advs.append(adv)

            if not self.update_before_compute:
                self.variance_tracker.update(task, group_rewards.detach().cpu().tolist())

        return torch.stack(advs)

    def process_step(self, rewards: Tensor, task_types: Sequence[str], entropies: Tensor) -> TrainerOutput:
        advantages = self.compute_advantages(rewards, task_types)
        if task_types:
            # Report task-global sigma as an unweighted average across task types.
            # This keeps the metric comparable across mixed-task batches without
            # over-emphasizing whichever task appears most frequently in one step.
            # Trade-off: if tasks have very different scales, this summary metric
            # can hide per-task variance differences; inspect per-task logs as needed.
            unique_tasks = sorted(set(task_types))
            sigmas = [self.variance_tracker.get(task) for task in unique_tasks]
            sigma_global = float(sum(sigmas) / len(sigmas))
        else:
            sigma_global = 1.0
        metrics = compute_step_metrics(
            rewards=rewards,
            advantages=advantages,
            entropies=entropies,
            sigma_global=sigma_global,
            clip_c=self.config.vgrpo_clip_c,
        )
        return TrainerOutput(advantages=advantages, metrics=metrics)
