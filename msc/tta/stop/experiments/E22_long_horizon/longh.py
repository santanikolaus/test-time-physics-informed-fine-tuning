"""E22: 640-step held-out readout on the long val chains (Re LONGH_RE, default 500); per model band H (0.9, 0.8), first-65 sanity H, band amplitude over time.

Usage (repo root): [LONGH_RE=1000] [LONGH_RES=256] python longh.py OUT.npz label=ckpt[:weights.pt] ...
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

from msc.tta import setup
from msc.tta.eval import eval as ev
from src.datasets.kf_dataset import KFDataset

RE = int(os.environ.get("LONGH_RE", 500))
RES = int(os.environ.get("LONGH_RES", 128))
LONG = Path(f"/system/user/studentwork/wehofer/data/ns/NS_fine_Re{RE}_T128_U10_res{RES}_val.npy")
CHAINS = range(240, 270)
G = {"all": slice(1, None), "k1-4": slice(1, 5), "k5-7": slice(5, 8), "k8-16": slice(8, 17), "k17+": slice(17, None)}
AMP_FRAMES = (64, 160, 320, 640)


def readout(label: str, spec: str, ds, device) -> dict:
    """Rolls one model over ds and returns its per-sample band-pooled correlation curves and band power.

    Args:
      label: row name.
      spec: "ckpt" or "ckpt:weights.pt".
      ds: long KFDataset.
      device: torch device.

    Returns:
      {"<label>_corr_<band>": (N, T), "<label>_pred_pt": (N, n_bands, T), "<label>_gt_pt": ...}.
    """
    ckpt, _, weights = spec.partition(":")
    model, cfg = setup.load_model(ckpt, device, weights or None)
    raw = ev.forward_bands(model, ds, device, regime=setup.Regime(op_re=int(cfg.data.get("re", 500)), test_re=RE),
                           time_scale=cfg.data.time_scale, temporal_pad=cfg.data.temporal_pad,
                           pad_mode=cfg.data.pad_mode, t_interval=10.0, residuals=False)
    p, g, e = raw["pred_pt"], raw["gt_pt"], raw["err_pt"]
    out = {f"{label}_corr_{b}": ev.corr_curve(p, g, e, bands=s) for b, s in G.items()}
    out.update({f"{label}_pred_pt": p, f"{label}_gt_pt": g})
    row = []
    for b in G:
        c = out[f"{label}_corr_{b}"]
        row.append(f"{ev.time_to_threshold(c, 0.9).mean():5.1f}/{ev.time_to_threshold(c, 0.8).mean():5.1f}")
    cens = (ev.time_to_threshold(out[f"{label}_corr_all"], 0.9) == c.shape[-1]).mean()
    first = ev.time_to_threshold(out[f"{label}_corr_all"][:, :65], 0.9).mean()
    amp = " ".join(f"{np.sqrt(p[:, G['k17+'], t].sum() / g[:, G['k17+'], t].sum()):.2f}" for t in AMP_FRAMES)
    amp_lo = " ".join(f"{np.sqrt(p[:, G['k5-7'], t].sum() / g[:, G['k5-7'], t].sum()):.2f}" for t in AMP_FRAMES)
    print(f"{label:14s} H0.9/H0.8 " + " ".join(row) + f" | cens {cens:.2f} | H65 {first:4.1f} | "
          f"amp k5-7 {amp_lo} | amp k17+ {amp}", flush=True)
    return out


def main(out_path: str, specs: list) -> None:
    """Reads out every label=spec on the long file, saves all curves to out_path."""
    device = torch.device("cuda")
    ds = KFDataset(str(LONG), len(CHAINS), offset=0, sub_t=2)
    print(f"long val: {len(ds)} chains x {ds[0]['y'].shape[-1]} frames; bands " + " ".join(G)
          + f"; amp frames {AMP_FRAMES}", flush=True)
    res = {}
    for s in specs:
        label, _, spec = s.partition("=")
        res.update(readout(label, spec, ds, device))
    np.savez_compressed(out_path, **res)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
