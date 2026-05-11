"""V-GRPO source package."""

from src.variance_tracker import GlobalVarianceTracker

try:
    from src.advantage import compute_grpo_advantage, compute_vgrpo_advantage
except ModuleNotFoundError as e:
    if e.name != "torch":
        raise

    def _raise_torch_import_error(*args, **kwargs):
        raise ImportError(
            "torch is required to use advantage functions; install torch to use "
            "compute_grpo_advantage/compute_vgrpo_advantage."
        ) from e

    compute_grpo_advantage = _raise_torch_import_error
    compute_vgrpo_advantage = _raise_torch_import_error

__all__ = ["compute_grpo_advantage", "compute_vgrpo_advantage", "GlobalVarianceTracker"]
