"""Rule-based reward for MATH benchmark answers."""

from __future__ import annotations

import re
from fractions import Fraction


_BOXED_PATTERN = re.compile(r"\\\\boxed\s*\{([^{}]+)\}")
_FRACTION_PATTERN = re.compile(r"^[-+]?\\d+\\s*/\\s*[-+]?\\d+$")


def _extract_final_answer(text: str) -> str:
    boxed = _BOXED_PATTERN.findall(text)
    if boxed:
        return boxed[-1].strip()

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _normalize_answer(answer: str) -> str:
    a = answer.strip()
    a = a.replace("$", "")
    a = a.replace("\\left", "").replace("\\right", "")
    a = a.replace(" ", "")
    a = a.replace("\\frac", "frac")

    if a.startswith("frac{") and a.endswith("}"):
        inner = a[len("frac{") : -1]
        parts = inner.split("}{")
        if len(parts) == 2:
            a = f"{parts[0]}/{parts[1]}"

    if _FRACTION_PATTERN.match(a):
        try:
            frac = Fraction(a)
            return f"{frac.numerator}/{frac.denominator}"
        except ZeroDivisionError:
            return a

    return a.lower()


def math_reward(
    response: str,
    ground_truth_answer: str,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
) -> float:
    # Module goal: provide deterministic rule-based scalar reward for MATH.
    # Inputs: model response and reference answer string.
    # Outputs: configurable binary reward (default 1/0).
    # Paper mapping: reward signal used by GRPO/V-GRPO training loop.
    pred = _normalize_answer(_extract_final_answer(response))
    gold = _normalize_answer(_extract_final_answer(ground_truth_answer))
    return float(correct_reward if pred == gold else incorrect_reward)
