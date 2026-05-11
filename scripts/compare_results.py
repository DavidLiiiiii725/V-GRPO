#!/usr/bin/env python
"""Compare GRPO and V-GRPO evaluation outputs into a markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grpo", required=True)
    parser.add_argument("--vgrpo", required=True)
    parser.add_argument("--output", default="results/tables/comparison.md")
    args = parser.parse_args()

    for name in ("grpo", "vgrpo"):
        path = Path(getattr(args, name))
        if not path.is_file():
            parser.error(
                f"--{name} file not found: {path}. "
                f"Generate it first with scripts/eval_math.py --output {path}"
            )
    return args


def main() -> None:
    args = parse_args()
    with open(args.grpo, "r", encoding="utf-8") as f:
        grpo = {int(k): float(v) for k, v in json.load(f).items()}
    with open(args.vgrpo, "r", encoding="utf-8") as f:
        vgrpo = {int(k): float(v) for k, v in json.load(f).items()}

    levels = sorted(set(grpo) | set(vgrpo))
    lines = [
        "| Level | GRPO | V-GRPO | Delta |",
        "|---|---:|---:|---:|",
    ]
    for lv in levels:
        g = grpo.get(lv, 0.0)
        v = vgrpo.get(lv, 0.0)
        lines.append(f"| {lv} | {g:.4f} | {v:.4f} | {v-g:+.4f} |")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
