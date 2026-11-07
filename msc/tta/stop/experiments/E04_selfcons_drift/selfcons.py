"""Computes self-consistency restart and checkpoint-drift band powers over a banked adaptation run's weight snapshots."""
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from msc.tta import setup
from msc.tta.adapt import adapt
from msc.tta.eval import eval as ev
from src.models.kf_fno import kf_forward

STEPS = range(0, 501, 20)
T0 = (16, 32)
SHELLS = 33


def power(field, kinf):
    """Returns (SHELLS, T) float32 band power of one (1, S, S, T) field."""
    return ev.band_power_t(field, kinf, SHELLS).astype(np.float32)


def main(weights_dir, out_path):
    """Forwards every scheduled checkpoint on held-out + pool ICs and saves GT, restart and drift powers."""
    device = torch.device("cuda")
    files = {int(p.stem[4:]): p for p in Path(weights_dir).glob("step*.pt")}
    saved = torch.load(files[max(files)], weights_only=False, map_location="cpu")
    cfg = OmegaConf.create(saved["cfg"])
    model, train_cfg = setup.load_model(saved["base_ckpt"], device)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    pool, heldout, target_cfg = adapt.build_splits(cfg, train_cfg)
    items = [heldout[i] for i in range(len(heldout))] + [pool[i] for i in range(len(pool))]
    d = target_cfg.data
    fwd = lambda ic, T: kf_forward(model, ic, T, time_scale=d.time_scale, temporal_pad=d.temporal_pad,
                                   pad_mode=d.pad_mode).squeeze(1)
    S = items[0]["y"].shape[0]
    kinf = ev.cheb_bins(S, device)
    out = {k: [] for k in ["pred", "gt", "err", "drift"] + [f"sc{t}_{x}" for t in T0 for x in ("a", "b", "d")]}
    prev = [None] * len(items)
    steps = [s for s in STEPS if s == 0 or s in files]
    for s in steps:
        model.load_state_dict(base_state if s == 0 else
                              torch.load(files[s], weights_only=False, map_location=device)["state_dict"])
        model.eval()
        rows = {k: [] for k in out}
        for i, item in enumerate(items):
            ic, gt = item["x"].unsqueeze(0).to(device), item["y"].unsqueeze(0).to(device)
            T = gt.shape[-1]
            with torch.no_grad():
                u = fwd(ic, T)
                rows["pred"].append(power(u, kinf))
                rows["gt"].append(power(gt, kinf))
                rows["err"].append(power(u - gt, kinf))
                rows["drift"].append(power(u - prev[i].to(device), kinf) if prev[i] is not None
                                     else np.zeros((SHELLS, T), np.float32))
                for t0 in T0:
                    a = u[..., t0:]
                    b = fwd(u[..., t0], T)[..., :T - t0]
                    rows[f"sc{t0}_a"].append(power(a, kinf))
                    rows[f"sc{t0}_b"].append(power(b, kinf))
                    rows[f"sc{t0}_d"].append(power(a - b, kinf))
            prev[i] = u.cpu()
        for k in out:
            out[k].append(np.stack(rows[k]))
        print(f"step {s} done", flush=True)
    np.savez_compressed(out_path, step=np.array(steps), n_heldout=len(heldout),
                        **{k: np.stack(v) for k, v in out.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
