"""Rule-based reward for MATH benchmark answers."""

from __future__ import annotations

import re
from fractions import Fraction


_FRACTION_PATTERN = re.compile(r"^[-+]?\d+\s*/\s*[-+]?\d+$")


def _extract_final_answer(text: str) -> str:
    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx != -1:
        i = idx + len(marker)
        depth = 1
        chunk = []
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
                chunk.append(ch)
            elif ch == "}":
                depth -= 1
                if depth > 0:
                    chunk.append(ch)
            else:
                chunk.append(ch)
            i += 1
        if depth == 0:
            return "".join(chunk).strip()

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
        except (ValueError, ZeroDivisionError):
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
