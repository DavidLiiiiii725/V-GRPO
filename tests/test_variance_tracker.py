import math
import random

from src.variance_tracker import GlobalVarianceTracker


def test_constant_rewards_variance_goes_to_zero():
    t = GlobalVarianceTracker(alpha=0.1)
    for _ in range(20):
        t.update("math", [1.0] * 8)
    assert t.get("math") < 1e-6


def test_ema_tracks_normal_variance_region():
    random.seed(0)
    t = GlobalVarianceTracker(alpha=0.2)
    for _ in range(200):
        batch = [random.gauss(0.0, 2.0) for _ in range(64)]
        t.update("math", batch)
    sigma = t.get("math")
    assert 1.5 <= sigma <= 2.5


def test_task_isolation():
    t = GlobalVarianceTracker(alpha=0.5)
    t.update("A", [0.0, 0.0, 10.0, 10.0])
    before_b = t.get("B")
    t.update("A", [0.0, 1.0, 2.0, 3.0])
    after_b = t.get("B")
    assert math.isclose(before_b, after_b)
