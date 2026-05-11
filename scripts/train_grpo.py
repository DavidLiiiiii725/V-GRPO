#!/usr/bin/env python
"""Entry point for baseline GRPO training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TrainConfig, ensure_output_dirs, as_dict
from src.grpo_trainer import VGRPOTrainerCore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="grpo_baseline")
    parser.add_argument("--total_steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(use_vgrpo=False, run_name=args.run_name, total_steps=args.total_steps)
    ensure_output_dirs(cfg)

    trainer = VGRPOTrainerCore(cfg)
    for step in range(cfg.total_steps):
        rewards = torch.rand(cfg.batch_size, cfg.group_size)
        entropies = torch.rand(cfg.batch_size)
        task_types = ["math"] * cfg.batch_size
        out = trainer.process_step(rewards, task_types, entropies)
        if step % max(1, cfg.eval_every) == 0:
            print(f"step={step} metrics={out.metrics.to_dict()}")

    out_path = Path(cfg.output_dir) / "logs" / f"{cfg.run_name}_config.json"
    out_path.write_text(json.dumps(as_dict(cfg), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
