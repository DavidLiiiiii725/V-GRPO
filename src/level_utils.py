"""Utilities for parsing MATH difficulty levels."""

from __future__ import annotations

import re


_LEVEL_INT_PATTERN = re.compile(r"\d+")


def parse_level(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = _LEVEL_INT_PATTERN.search(value)
        if match:
            return int(match.group(0))
    raise ValueError(f"Invalid level value: {value!r}")
