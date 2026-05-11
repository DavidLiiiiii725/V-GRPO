"""Evaluation helpers for per-level MATH accuracy reporting."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterable

from src.level_utils import parse_level
from src.reward import math_reward


def evaluate_by_level(
    generate_fn: Callable[[str], str],
    dataset: Iterable[dict],
    levels: Iterable[int] = (1, 2, 3, 4, 5),
) -> Dict[int, float]:
    # Module goal: compute MATH accuracy bucketed by difficulty level.
    # Inputs: generation function and iterable examples with level/answer.
    # Outputs: level->accuracy dict for result tables.
    # Paper mapping: Table-style per-level comparison (mechanism signature).
    total = defaultdict(int)
    correct = defaultdict(int)
    target_levels = set(int(l) for l in levels)

    for ex in dataset:
        level = parse_level(ex["level"])
        if level not in target_levels:
            continue

        pred = generate_fn(ex["prompt"])
        rew = math_reward(pred, ex["answer"])
        total[level] += 1
        if rew > 0:
            correct[level] += 1

    return {
        level: (correct[level] / total[level] if total[level] else 0.0)
        for level in sorted(target_levels)
    }
