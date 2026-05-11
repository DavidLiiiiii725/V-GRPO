"""Utilities for parsing MATH difficulty levels."""

from __future__ import annotations

import re


_LEVEL_PATTERN = re.compile(r"level\s*(\d+)", flags=re.IGNORECASE)


def parse_level(value: object) -> int:
    parsed: int
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
        else:
            match = _LEVEL_PATTERN.fullmatch(stripped)
            if not match:
                raise ValueError(f"Invalid level value: {value!r}")
            parsed = int(match.group(1))
    else:
        raise ValueError(f"Invalid level value: {value!r}")

    if parsed <= 0:
        raise ValueError(f"Invalid level value: {value!r}")
    return parsed
