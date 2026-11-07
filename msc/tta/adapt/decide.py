"""Deployment decisions from a few observed frames: viscosity estimate, start gate, stop rule."""
import numpy as np
import torch

from src.pde.ns import NSVorticity


def nu_hat(w: torch.Tensor, t_interval: float) -> float:
    """Estimates the viscosity by least squares on the vorticity equation, which is linear in nu.

    Args:
      w: (B, S, S, T) observed vorticity windows, T >= 3 consecutive frames.
      t_interval: physical time spanned by the T frames.

    Returns:
      nu minimising ||(w_t + u.grad w - f) - nu lap w||^2 pooled over all windows and interior frames.
    """
    ns = NSVorticity(re=1.0, t_interval=t_interval)
    _, (wt, adv, diff) = ns.residual(w)
    a = wt + adv - ns.get_forcing(w.shape[1], w.device)
    lap = -diff
    return float((a * lap).sum() / (lap * lap).sum())


def gate(nu_est: float, nu_src: float, tol: float) -> bool:
    """Returns True (adapt) iff the estimated viscosity differs from the source one by more than tol in log space."""
    return abs(np.log(nu_est / nu_src)) > tol


def window_error(err_pt: np.ndarray, gt_pt: np.ndarray, m: int) -> np.ndarray:
    """Relative error power over predicted frames 1..m, pooled over samples and shells.

    Args:
      err_pt: (..., N, n_bands, T) error power, as stored by forward_bands (leading axes e.g. steps).
      gt_pt: same shape, GT power.
      m: number of observed frames after the IC.

    Returns:
      Array of the leading shape (scalar for a single snapshot).
    """
    window = slice(1, m + 1)
    return err_pt[..., window].sum((-3, -2, -1)) / gt_pt[..., window].sum((-3, -2, -1))


def stop_index(steps: np.ndarray, signal: np.ndarray, probe: int, w: int, tau: float,
               delta: float = 0.02) -> tuple:
    """Causal probe-abort + plateau stop rule on a lower-is-better signal, using only past snapshots.

    Args:
      steps: (S,) ascending optimizer steps of the snapshots, steps[0] == 0.
      signal: (S,) monitoring error per snapshot.
      probe: probe length in optimizer steps; abort unless signal fell by delta by then.
      w: plateau look-back window in optimizer steps.
      tau: minimum improvement of the running minimum over w, relative to signal[0].
      delta: relative drop required at the probe.

    Returns:
      (index of the chosen snapshot, outcome in {"short", "abort", "stop", "censored"}).
    """
    if steps[-1] < probe:
        return 0, "short"
    ip = int(np.searchsorted(steps, probe))
    if signal[ip] >= (1 - delta) * signal[0]:
        return 0, "abort"
    run_min = np.minimum.accumulate(signal)
    for i in range(ip + 1, len(steps)):
        j = int(np.searchsorted(steps, steps[i] - w, side="right")) - 1
        if j >= 0 and run_min[j] - run_min[i] < tau * signal[0]:
            return int(np.argmin(signal[: i + 1])), "stop"
    return int(np.argmin(signal)), "censored"
