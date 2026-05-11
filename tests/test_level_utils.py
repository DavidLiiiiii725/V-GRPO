import pytest

from src.level_utils import parse_level


def test_parse_level_accepts_numeric_and_string_values():
    assert parse_level(3) == 3
    assert parse_level(4.0) == 4
    assert parse_level("5") == 5
    assert parse_level("Level 2") == 2


def test_parse_level_rejects_invalid_values():
    with pytest.raises(ValueError):
        parse_level("unknown")
