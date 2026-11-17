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


def sample_starts(model: Unet2DRollout, ic: torch.Tensor, T: int, k: int, n: int,
                  generator: torch.Generator = None) -> tuple:
    """Rolls out from ic without grad and picks n own states as window starts.

    Args:
      model: recurrent one-step model; only time_history 1 is supported.
      ic: (B, S, S) initial vorticity.
      T: frames of the full no-grad rollout, IC included.
      k: steps per window, 1 <= k <= T - 1.
      n: windows, drawn independently.
      generator: CPU generator for the start indices; None uses the global RNG.

    Returns:
      A tuple (starts, t0): starts (n * B, S, S) detached, window-major as in gather();
      t0 (n,) start indices, uniform in [0, T - 1 - k].
    """
    assert model.net.time_history == 1, "window sampling supports time_history 1 only"
    assert 1 <= k <= T - 1, f"window of {k} steps does not fit a {T}-frame trajectory"
    t0 = torch.randint(0, T - k, (n,), generator=generator)
    with torch.no_grad():
        return gather(model.rollout(ic[:, None], T), t0, 0)[..., 0], t0


def observed_rollout(model: Unet2DRollout, ic: torch.Tensor, m: int) -> torch.Tensor:
    """Rolls m steps with grad from the IC, covering only the observed frames.

    Args:
      model: recurrent one-step model; only time_history 1 is supported.
      ic: (B, S, S) initial vorticity.
      m: observed frames after the IC, m >= 1.

    Returns:
      (B, 1, S, S, m + 1): the IC followed by m predictions that carry grad.
    """
    assert model.net.time_history == 1, "observed rollout supports time_history 1 only"
    assert m >= 1, f"need at least one observed frame, got {m}"
    return model.rollout(ic[:, None], m + 1).unsqueeze(1)


def sample_windows(model: Unet2DRollout, ic: torch.Tensor, T: int, k: int, n: int,
                   generator: torch.Generator = None) -> tuple:
    """Re-rolls k steps with grad from n detached own states, as drawn by sample_starts.

    Returns:
      A tuple (pred, t0): pred (n * B, 1, S, S, k + 1), window-major as in gather(), holds
      the detached state at each start followed by k predictions that carry grad; t0 as
      in sample_starts.
    """
    starts, t0 = sample_starts(model, ic, T, k, n, generator)
    return model.rollout(starts[:, None], k + 1).unsqueeze(1), t0


def backward_windows(model: Unet2DRollout, loss_fn, starts: torch.Tensor, target: torch.Tensor,
                     k: int, chunk: "int | None" = None) -> dict:
    """Backpropagates the window-mean loss chunk by chunk, so peak memory scales with chunk, not n.

    Exact for losses that average per-sample terms over the batch, as KFLoss's LpLoss does.

    Args:
      model: recurrent one-step model; gradients accumulate into its parameters.
      loss_fn: KFLoss built for k + 1-frame windows.
      starts: (N, S, S) detached window starts.
      target: (N, S, S, k + 1) matching GT windows.
      k: grad-carrying steps per window.
      chunk: windows per backward; None backpropagates all N at once.

    Returns:
      The loss parts averaged over all N windows, detached.
    """
    n = len(starts)
    total = {}
    for i in range(0, n, chunk or n):
        pred = model.rollout(starts[i:i + (chunk or n), None], k + 1).unsqueeze(1)
        parts = loss_fn(pred, target[i:i + len(pred)])
        share = len(pred) / n
        (share * parts["loss"]).backward()
        for key, value in parts.items():
            total[key] = total.get(key, 0.0) + share * value.detach()
    return total
