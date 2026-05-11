"""Dataset loading utilities for MATH with level/task annotations."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from datasets import Dataset, load_dataset


def load_math_dataset(
    dataset_name: str = "hendrycks/competition_math",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    # Module goal: load official MATH split and preserve difficulty levels.
    # Inputs: HF dataset id, split name, optional sample cap.
    # Outputs: dataset rows containing problem/solution/level/task_type.
    # Paper mapping: per-level analysis and single-task tagging for MATH.
    ds = load_dataset(dataset_name, split=split)

    if "level" not in ds.column_names:
        raise ValueError("MATH dataset split does not contain 'level' field")

    if "task_type" not in ds.column_names:
        ds = ds.add_column("task_type", ["math"] * len(ds))

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds


def group_by_level(dataset: Dataset, levels: Iterable[int] = (1, 2, 3, 4, 5)) -> Dict[int, Dataset]:
    grouped: Dict[int, Dataset] = {}
    for level in levels:
        level_int = int(level)

        def _is_level(row: dict) -> bool:
            return int(row["level"]) == level_int

        grouped[level_int] = dataset.filter(_is_level)
    return grouped


def to_prompt_answer_pairs(dataset: Dataset) -> List[dict]:
    pairs = []
    for row in dataset:
        prompt = row.get("problem") or row.get("question")
        answer = row.get("solution") or row.get("answer")
        if not prompt or not answer:
            raise ValueError("Dataset row missing required prompt/answer fields")

        pairs.append(
            {
                "prompt": prompt,
                "answer": answer,
                "level": int(row["level"]),
                "task_type": row.get("task_type", "math"),
            }
        )
    return pairs
