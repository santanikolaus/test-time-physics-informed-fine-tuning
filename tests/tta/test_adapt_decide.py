import numpy as np
import pytest
import torch

from msc.tta.adapt import decide


def _shear_windows(nu: float, S: int = 32, T: int = 9, t_interval: float = 0.125) -> torch.Tensor:
    """Exact Kolmogorov solution: laminar shear -cos(4y)/(4 nu) plus a decaying sin(2y) mode, no advection."""
    y = torch.linspace(0, 2 * np.pi, S + 1, dtype=torch.float64)[:-1].reshape(1, S)
    t = torch.linspace(0, t_interval, T, dtype=torch.float64)
    field = (-torch.cos(4 * y) / (4 * nu))[..., None] + torch.sin(2 * y)[..., None] * torch.exp(-4 * nu * t)
    return field.expand(S, S, T).unsqueeze(0).clone()


@pytest.mark.parametrize("nu", [1 / 100, 1 / 500])
def test_nu_hat_recovers_viscosity_of_exact_solution(nu):
    assert decide.nu_hat(_shear_windows(nu), t_interval=0.125) == pytest.approx(nu, rel=1e-3)


def test_gate_adapts_only_beyond_tolerance():
    assert not decide.gate(0.0102, 0.01, tol=np.log(1.5))
    assert decide.gate(0.002, 0.01, tol=np.log(1.5))
    assert decide.gate(0.05, 0.01, tol=np.log(1.5))


def test_window_error_pools_frames_one_to_m_over_leading_axes():
    gt = np.ones((3, 2, 4, 10))
    err = np.zeros_like(gt)
    err[0, ..., 1:3] = 0.5
    err[1, ..., 9] = 7.0
    out = decide.window_error(err, gt, m=2)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [0.5, 0.0, 0.0])


def test_stop_index_aborts_when_probe_shows_no_drop():
    steps = np.arange(0, 301, 50)
    assert decide.stop_index(steps, np.ones(len(steps)), probe=100, w=100, tau=0.01) == (0, "abort")


def test_stop_index_returns_argmin_at_plateau():
    steps = np.arange(0, 501, 50)
    signal = np.array([1.0, 0.8, 0.6, 0.5, 0.45, 0.449, 0.449, 0.46, 0.47, 0.48, 0.49])
    i, outcome = decide.stop_index(steps, signal, probe=100, w=100, tau=0.01)
    assert outcome == "stop"
    assert i == 5


def test_stop_index_short_run_never_adapts():
    assert decide.stop_index(np.array([0, 10, 20]), np.array([1.0, 0.5, 0.2]), probe=100, w=50, tau=0.01) == (0, "short")
