#!/usr/bin/env python
"""Run per-level MATH evaluation from stored predictions or a stub generator."""

from __future__ import annotations

import argparse
import json

from src.data import load_math_dataset, to_prompt_answer_pairs
from src.eval import evaluate_by_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output", default="results/tables/eval_math.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_math_dataset(split="test", max_samples=args.max_samples)
    examples = to_prompt_answer_pairs(ds)

    def greedy_stub(_: str) -> str:
        return ""

    result = evaluate_by_level(greedy_stub, examples)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(result)


if __name__ == "__main__":
    main()
