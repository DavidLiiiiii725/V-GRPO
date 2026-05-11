"""Centralized experiment configuration for GRPO/V-GRPO reproduction."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrainConfig:
    # Module goal: collect all training hyperparameters in one place.
    # Inputs: optional runtime overrides from CLI scripts.
    # Outputs: serializable config object consumed by trainers/scripts.
    # Paper mapping: Eq.(2) EMA alpha and Eq.(3) advantage clip live here.
    learning_rate: float = 1e-6
    group_size: int = 8
    batch_size: int = 16
    max_response_length: int = 1024
    kl_coef: float = 0.04
    clip_epsilon: float = 0.2
    total_steps: int = 1000
    eval_every: int = 100
    ema_alpha: float = 0.1
    vgrpo_clip_c: float = 10.0
    eps: float = 1e-8
    use_vgrpo: bool = False

    reward_correct: float = 1.0
    reward_incorrect: float = 0.0

    model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    dataset_name: str = "hendrycks/competition_math"
    train_split: str = "train"
    eval_split: str = "test"

    output_dir: str = "results"
    run_name: str = "default"
    seed: int = 42


def as_dict(cfg: TrainConfig) -> Dict[str, Any]:
    return asdict(cfg)


def ensure_output_dirs(cfg: TrainConfig) -> None:
    base = Path(cfg.output_dir)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (base / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base / "tables").mkdir(parents=True, exist_ok=True)
