import pytest
import torch
from omegaconf import OmegaConf

from src.models.kf_module_onestep import KFLitModuleOneStep


def _make_cfg(pde_weight):
    return OmegaConf.create({
        "model": {
            "model_arch": "unet2d",
            "hidden_channels": 8,
            "time_history": 4,
            "activation": "gelu",
            "norm": True,
            "ch_mults": [1, 2, 2, 4],
            "is_attn": [False, False, False, False],
            "mid_attn": False,
            "n_blocks": 2,
            "use1x1": False,
        },
        "loss": {
            "re": 100.0,
            "t_interval": 1.0,
            "data_weight": 1.0,
            "pde_weight": pde_weight,
            "ic_weight": 0.0,
            "pde_horizon": 3,
        },
        "opt": {"learning_rate": 1.0e-3},
        "data": {
            "T": 16,
            "time_scale": 1.0,
            "temporal_pad": 0,
            "pad_mode": "zero",
            "n_context": 1,
        },
    })


def _make_batch(B=2, S=16, T=16):
    return {"y": torch.randn(B, S, S, T)}


@pytest.mark.parametrize("pde_weight", [0.0, 1.0], ids=["data_only", "pde_rollout"])
def test_training_step_trains_inner_net(pde_weight):
    """training_step must return a finite scalar loss and backprop into model.net.

    S=16 != time_history=4 so a wrong permute/window-slide would break shape
    or feed garbage rather than silently pass; the grad-flow check on
    model.net rules out a loss that computes but is detached from the net.
    """
    torch.manual_seed(0)
    module = KFLitModuleOneStep(_make_cfg(pde_weight))
    batch = _make_batch()

    loss = module.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in module.model.net.parameters() if p.grad is not None]
    assert len(grads) > 0
    total_abs_grad = sum(g.abs().sum().item() for g in grads)
    assert total_abs_grad > 0.0


class _Oracle(torch.nn.Module):
    """Returns the exact scaled increment from its (possibly noisy) input to gt; records the input."""

    def __init__(self, gt, output_factor):
        super().__init__()
        self.gt, self.of, self.seen = gt, output_factor, None

    def forward(self, x):
        self.seen = x[:, -1, 0].clone()
        return ((self.gt - self.seen) / self.of)[:, None, None]


@pytest.mark.parametrize("noise", [0.0, 0.5])
def test_input_noise_keeps_clean_target(noise):
    """With opt.input_noise the net sees a noisy window but must still map it to the clean next frame."""
    cfg = _make_cfg(0.0)
    cfg.model.residual, cfg.model.output_factor, cfg.opt.input_noise = True, 0.25, noise
    module = KFLitModuleOneStep(cfg)
    batch = _make_batch(T=5)
    gt, last = batch["y"][..., 4], batch["y"][..., 3]
    module.model.net = _Oracle(gt, 0.25)
    rng = torch.get_rng_state()

    loss = module.training_step(batch, 0)

    assert loss.item() < 1e-10
    noisy = not torch.allclose(module.model.net.seen, last)
    assert noisy == bool(noise)
    if not noise:
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        torch.randint(0, 1, (1,))
        assert torch.equal(after, torch.get_rng_state()), "noise 0 must draw no extra random numbers"
