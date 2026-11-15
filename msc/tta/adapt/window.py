"""Short-window rollouts for windowed physics adaptation of a recurrent one-step model."""
import torch

from src.models.kf_unet2d import Unet2DRollout


def window_t_interval(t_interval: float, T: int, k: int) -> float:
    """Returns the physical span of a k-step window, keeping the full trajectory's frame spacing.

    Args:
      t_interval: physical span of the full T-frame trajectory, as NSVorticity reads it.
      T: frames of the full trajectory, IC included.
      k: steps in the window, 1 <= k <= T - 1.

    Returns:
      t_interval * k / (T - 1), so NSVorticity's dt on k + 1 frames equals the full one.
    """
    assert 1 <= k <= T - 1, f"window of {k} steps does not fit a {T}-frame trajectory"
    return t_interval * k / (T - 1)


def gather(traj: torch.Tensor, t0: torch.Tensor, k: int) -> torch.Tensor:
    """Cuts k + 1 frames from each start in t0 and stacks them window-major along the batch.

    Args:
      traj: (B, S, S, T) trajectory.
      t0: (n,) start indices, each in [0, T - 1 - k].
      k: steps per window.

    Returns:
      (n * B, S, S, k + 1); rows [i * B:(i + 1) * B] hold the window starting at t0[i].
    """
    return torch.cat([traj[..., t:t + k + 1] for t in t0.tolist()])


def sample_windows(model: Unet2DRollout, ic: torch.Tensor, T: int, k: int, n: int,
                   generator: torch.Generator = None) -> tuple:
    """Rolls out from ic without grad, then re-rolls k steps with grad from n detached own states.

    Args:
      model: recurrent one-step model; only time_history 1 is supported.
      ic: (B, S, S) initial vorticity.
      T: frames of the full no-grad rollout, IC included.
      k: grad-carrying steps per window, 1 <= k <= T - 1.
      n: windows per call, drawn independently.
      generator: CPU generator for the start indices; None uses the global RNG.

    Returns:
      A tuple (pred, t0): pred (n * B, 1, S, S, k + 1), window-major as in gather(), holds
      the detached state at each start followed by k predictions that carry grad; t0 (n,)
      holds the starts, uniform in [0, T - 1 - k].
    """
    assert model.net.time_history == 1, "window sampling supports time_history 1 only"
    assert 1 <= k <= T - 1, f"window of {k} steps does not fit a {T}-frame trajectory"
    t0 = torch.randint(0, T - k, (n,), generator=generator)
    with torch.no_grad():
        starts = gather(model.rollout(ic[:, None], T), t0, 0)
    return model.rollout(starts.permute(0, 3, 1, 2), k + 1).unsqueeze(1), t0
