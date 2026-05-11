import torch

from src.advantage import compute_grpo_advantage, compute_vgrpo_advantage


def test_case_a_values_close_to_paper_numbers():
    rewards = torch.tensor([7.2, 7.0, 7.1, 6.9, 7.3, 7.0, 7.1, 7.2], dtype=torch.float32)
    grpo = compute_grpo_advantage(rewards)
    best_idx = int(torch.argmax(rewards).item())
    assert grpo[best_idx].item() == pytest_approx(1.63, abs_=0.05)

    vgrpo = compute_vgrpo_advantage(rewards, sigma_global=1.5)
    assert vgrpo[best_idx].item() == pytest_approx(0.13, abs_=0.03)


def test_case_b_values_close_to_paper_numbers():
    rewards = torch.tensor([100, 100, 100, 100, 0, 0, 0, 0], dtype=torch.float32)
    vgrpo = compute_vgrpo_advantage(rewards, sigma_global=30.0)
    assert float(vgrpo.max().item()) == pytest_approx(1.67, abs_=0.05)
    assert float(vgrpo.min().item()) == pytest_approx(-1.67, abs_=0.05)


def test_clipping_works():
    rewards = torch.tensor([5000.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    vgrpo = compute_vgrpo_advantage(rewards, sigma_global=1.0, clip_C=10.0)
    assert torch.all(vgrpo <= 10.0)
    assert torch.all(vgrpo >= -10.0)


def pytest_approx(value, abs_):
    import pytest

    return pytest.approx(value, abs=abs_)
