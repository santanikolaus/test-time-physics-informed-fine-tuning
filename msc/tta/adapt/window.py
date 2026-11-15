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


def sample_window(model: Unet2DRollout, ic: torch.Tensor, T: int, k: int,
                  generator: torch.Generator = None) -> tuple:
    """Rolls out from ic without grad, then re-rolls k steps with grad from a detached own state.

    Args:
      model: recurrent one-step model; only time_history 1 is supported.
      ic: (B, S, S) initial vorticity.
      T: frames of the full no-grad rollout, IC included.
      k: grad-carrying steps, 1 <= k <= T - 1.
      generator: CPU generator for the start index; None uses the global RNG.

    Returns:
      A tuple (pred, t0): pred (B, 1, S, S, k + 1) holds the detached state at frame
      t0 followed by k predictions that carry grad; t0 is uniform in [0, T - 1 - k].
    """
    assert model.net.time_history == 1, "window sampling supports time_history 1 only"
    assert 1 <= k <= T - 1, f"window of {k} steps does not fit a {T}-frame trajectory"
    t0 = int(torch.randint(0, T - k, (1,), generator=generator))
    with torch.no_grad():
        start = model.rollout(ic[:, None], t0 + 1)[..., t0] if t0 else ic
    return model.rollout(start.detach()[:, None], k + 1).unsqueeze(1), t0
