"""V-GRPO source package."""

from src.variance_tracker import GlobalVarianceTracker

try:
    from src.advantage import compute_grpo_advantage, compute_vgrpo_advantage
except ModuleNotFoundError as e:
    if e.name != "torch":
        raise
    compute_grpo_advantage = None
    compute_vgrpo_advantage = None

__all__ = ["compute_grpo_advantage", "compute_vgrpo_advantage", "GlobalVarianceTracker"]
