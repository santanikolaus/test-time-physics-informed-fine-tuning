"""True training-step peak memory for high-capacity FNO at 128² with the PINO loss.

Unlike fno_capacity_probe (trivial y² loss), this drives the real KFLitModule
training_step — data + PDE-residual + IC terms — so the autograd graph through the
physics FFTs is included. That is the memory that actually decides OOM in training.

    CUDA_VISIBLE_DEVICES=4 python scripts/fno_trainmem_probe.py
"""
import torch
from omegaconf import OmegaConf
from src.models.kf_module import KFLitModule

S, T, TPAD = 128, 65, 5            # Re100 native 128², sub_t=2 → 65 frames, temporal_pad 5
# (mx,my,mt,hidden,n_layers) at 128² — full spatial spectrum (Nyquist 64), descending capacity.
CANDIDATES = [
    (32, 32, 8, 128, 6),   # Run D as designed
    (24, 24, 8, 128, 8),   # Run E as designed
    (32, 32, 8, 128, 4),   # modes alone (Run A), sanity anchor
    (16, 16, 8, 128, 6),   # fallback: half the modes of D
    (16, 16, 8, 128, 8),   # fallback: half the modes of E
    (8, 8, 8, 128, 8),     # layers alone (Run C), sanity anchor
]


def _config(mx, my, mt, hidden, n_layers=4):
    return {
        "model": {
            "model_arch": "fno", "data_channels": 4, "out_channels": 1,
            "n_modes": [mx, my, mt], "hidden_channels": hidden, "n_layers": n_layers,
            "lifting_channel_ratio": 0, "projection_channel_ratio": 2,
            "domain_padding": 0.0, "positional_embedding": None, "norm": None,
            "fno_skip": "linear", "implementation": "factorized",
            "use_channel_mlp": False, "channel_mlp_expansion": 0.5,
            "channel_mlp_dropout": 0.0, "separable": False, "factorization": None,
            "rank": 1.0, "fixed_rank_modes": False, "stabilizer": "None",
        },
        "loss": {"re": 100, "t_interval": 1.0, "data_weight": 5.0,
                 "pde_weight": 1.0, "ic_weight": 1.0},
        "opt": {"learning_rate": 1e-3, "weight_decay": 0.0,
                "milestones": [25, 50, 75, 100], "gamma": 0.5},
        "data": {"T": T, "time_scale": 1.0, "temporal_pad": TPAD},
    }


def probe(mx, my, mt, hidden, dev, n_layers=4):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    mod = KFLitModule(OmegaConf.create(_config(mx, my, mt, hidden, n_layers))).to(dev)
    mod._device_override = dev
    n = sum(p.numel() for p in mod.model.parameters())
    opt = torch.optim.Adam(mod.model.parameters(), lr=1e-3)
    batch = {"x": torch.randn(1, S, S, device=dev),
             "y": torch.randn(1, S, S, T, device=dev)}
    # KFLitModule.training_step uses self.device (Lightning); patch for standalone run.
    mod.to(dev)
    loss = mod.training_step(batch, 0)
    loss.backward()
    opt.step()
    torch.cuda.synchronize(dev)
    peak = torch.cuda.max_memory_allocated(dev) / 1024**3
    del mod, opt, batch, loss
    return n, peak


def main():
    dev = torch.device("cuda:0")
    tot = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    print(f"{torch.cuda.get_device_name(dev)}  total {tot:.1f} GB | 128² PINO train step\n")
    print(f"{'modes':>14} {'h':>4} {'layers':>6} {'params':>9} {'peak_GB':>8}  status")
    for mx, my, mt, hidden, n_layers in CANDIDATES:
        tag = f"{f'[{mx},{my},{mt}]':>14} {hidden:>4} {n_layers:>6}"
        try:
            n, peak = probe(mx, my, mt, hidden, dev, n_layers)
            print(f"{tag} {n/1e6:>8.1f}M {peak:>8.2f}  OK")
        except RuntimeError as e:
            msg = "OOM" if "out of memory" in str(e).lower() else f"ERR {str(e)[:50]}"
            print(f"{tag} {'-':>9} {'-':>8}  {msg}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
