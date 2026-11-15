import pytest
import torch

from msc.tta.adapt import window
from src.models.kf_unet2d import Unet2DRollout
from src.models.pdearena import Unet
from src.pde.ns import NSVorticity

S, T = 16, 9


@pytest.fixture
def model() -> Unet2DRollout:
    torch.manual_seed(0)
    net = Unet(n_input_scalar_components=1, n_input_vector_components=0,
               n_output_scalar_components=1, n_output_vector_components=0,
               time_history=1, time_future=1, hidden_channels=8, activation="gelu",
               norm=True, padding_mode="circular")
    return Unet2DRollout(net, residual=True, output_factor=0.1)


@pytest.mark.parametrize("k", [2, 4, 8])
def test_window_t_interval_reproduces_the_full_trajectory_residual(k):
    """The dt trap: a k+1-frame window must see the full trajectory's frame spacing, not 1/k."""
    torch.manual_seed(0)
    traj = torch.randn(2, S, S, T)
    full, _ = NSVorticity(re=500, t_interval=1.0).residual(traj)
    ns = NSVorticity(re=500, t_interval=window.window_t_interval(1.0, T, k))
    for t0 in (0, T - 1 - k):
        win, _ = ns.residual(traj[..., t0:t0 + k + 1])
        assert torch.allclose(win, full[..., t0:t0 + k - 1], atol=1e-4)


def test_window_frames_match_the_full_rollout(model):
    ic = torch.randn(2, S, S)
    with torch.no_grad():
        full = model.rollout(ic[:, None], T)
    for seed in range(5):
        pred, t0 = window.sample_window(model, ic, T, 3, torch.Generator().manual_seed(seed))
        assert pred.shape == (2, 1, S, S, 4)
        assert torch.allclose(pred[:, 0], full[..., t0:t0 + 4], atol=1e-5)


def test_only_the_predicted_frames_carry_grad(model):
    pred, _ = window.sample_window(model, torch.randn(1, S, S), T, 2, torch.Generator().manual_seed(1))
    params = list(model.parameters())
    first = torch.autograd.grad(pred[..., 0].sum(), params, allow_unused=True, retain_graph=True)
    last = torch.autograd.grad(pred[..., -1].sum(), params, allow_unused=True)
    assert all(g is None or not g.any() for g in first)
    assert any(g is not None and g.any() for g in last)


def test_start_index_covers_both_ends_and_full_window_starts_at_the_ic(model):
    g, ic = torch.Generator().manual_seed(0), torch.randn(1, S, S)
    starts = {window.sample_window(model, ic, T, 6, g)[1] for _ in range(40)}
    assert starts == {0, 1, 2}
    pred, t0 = window.sample_window(model, ic, T, T - 1, g)
    assert t0 == 0 and torch.equal(pred[:, 0, ..., 0], ic)


def test_window_rejects_a_window_longer_than_the_trajectory(model):
    with pytest.raises(AssertionError):
        window.sample_window(model, torch.randn(1, S, S), T, T)
