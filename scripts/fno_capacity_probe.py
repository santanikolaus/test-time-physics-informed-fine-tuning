"""Probe peak GPU memory + param count for FNO3d at high spectral-mode capacity.

Builds the SAME operator as train_kf (build_fno_kf / neuralop), runs one
forward+backward+Adam step on a (1,4,S,S,T) batch, reports params and peak mem.
OOM on one config is caught and reported; the sweep continues.

Run on the reserved card:
    CUDA_VISIBLE_DEVICES=4 python scripts/fno_capacity_probe.py
"""
import torch
from src.models.kf_fno import build_fno_kf

# (S, mx, my, mt, hidden) — temporal modes fixed at op100's 8; push spatial.
CANDIDATES = [
    # 128² full spatial spectrum (Nyquist 64); push width/temporal into the headroom
    (128,  64,  64,  8,  96),
    (128,  64,  64,  8, 112),
    (128,  64,  64, 16,  64),
    (128,  64,  64, 16,  90),
    # 256² full spatial spectrum (Nyquist 128)
    (256, 128, 128,  8,  48),
    (256, 128, 128, 16,  32),
]
T_PAD = 70  # 65 effective frames + temporal_pad 5


def _cfg(mx, my, mt, hidden):
    return {
        "model_arch": "fno", "data_channels": 4, "out_channels": 1,
        "n_modes": [mx, my, mt], "hidden_channels": hidden, "n_layers": 4,
        "lifting_channel_ratio": 0, "projection_channel_ratio": 2,
        "domain_padding": 0.0, "positional_embedding": None, "norm": None,
        "fno_skip": "linear", "implementation": "factorized",
        "use_channel_mlp": False, "channel_mlp_expansion": 0.5,
        "channel_mlp_dropout": 0.0, "separable": False, "factorization": None,
        "rank": 1.0, "fixed_rank_modes": False, "stabilizer": "None",
    }


def probe(S, mx, my, mt, hidden, dev):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    model = build_fno_kf(_cfg(mx, my, mt, hidden)).to(dev)
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(1, 4, S, S, T_PAD, device=dev)
    y = model(x)
    loss = y.float().pow(2).mean()
    loss.backward()
    opt.step()
    torch.cuda.synchronize(dev)
    peak = torch.cuda.max_memory_allocated(dev) / 1024**3
    del model, opt, x, y, loss
    return n, peak


def main():
    dev = torch.device("cuda:0")
    print(f"device {torch.cuda.get_device_name(dev)} "
          f"total {torch.cuda.get_device_properties(dev).total_memory/1024**3:.1f} GB\n")
    print(f"{'S':>4} {'modes':>14} {'h':>4} {'params':>10} {'peak_GB':>8}  status")
    for S, mx, my, mt, hidden in CANDIDATES:
        tag = f"{S:>4} {f'[{mx},{my},{mt}]':>14} {hidden:>4}"
        try:
            n, peak = probe(S, mx, my, mt, hidden, dev)
            print(f"{tag} {n/1e6:>9.1f}M {peak:>8.2f}  OK")
        except RuntimeError as e:
            msg = "OOM" if "out of memory" in str(e).lower() else f"ERR {str(e)[:40]}"
            print(f"{tag} {'-':>10} {'-':>8}  {msg}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
