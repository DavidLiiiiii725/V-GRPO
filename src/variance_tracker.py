"""EMA tracker for task-conditioned global reward variance used by V-GRPO."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable


@dataclass
class GlobalVarianceTracker:
    # Module goal: maintain per-task EMA variance estimates.
    # Inputs: task labels and per-group reward lists from each batch.
    # Outputs: task-level global standard deviation sigma_hat_global(T).
    # Paper mapping: Eq.(2) EMA update of task-global variance.
    alpha: float = 0.1
    init_var: float = 1.0
    _ema_var: Dict[str, float] = field(default_factory=dict)

    def update(self, task_type: str, batch_rewards: Iterable[float]) -> None:
        rewards = list(float(x) for x in batch_rewards)
        if not rewards:
            return
        mean_r = sum(rewards) / len(rewards)
        batch_var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)

        if task_type not in self._ema_var:
            self._ema_var[task_type] = batch_var
            return

        prev = self._ema_var[task_type]
        self._ema_var[task_type] = self.alpha * batch_var + (1.0 - self.alpha) * prev

    def get(self, task_type: str) -> float:
        var = self._ema_var.get(task_type, self.init_var)
        return math.sqrt(max(var, 0.0))

    def state_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "init_var": self.init_var,
            "ema_var": dict(self._ema_var),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.alpha = float(sd.get("alpha", self.alpha))
        self.init_var = float(sd.get("init_var", self.init_var))
        ema_var = sd.get("ema_var", {})
        self._ema_var = {str(k): float(v) for k, v in ema_var.items()}
