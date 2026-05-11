from src.reward import math_reward


def test_boxed_integer():
    assert math_reward("最终答案是 \\boxed{42}", "\\boxed{42}") == 1.0


def test_boxed_fraction_equivalent():
    assert math_reward("\\boxed{\\frac{1}{2}}", "\\boxed{1/2}") == 1.0


def test_symbolic_text_not_equal_when_mismatch():
    assert math_reward("\\boxed{x^2+1}", "\\boxed{x^2+2}") == 0.0


def test_plain_text_last_line_extraction():
    response = "分析过程...\n7"
    gt = "7"
    assert math_reward(response, gt) == 1.0
