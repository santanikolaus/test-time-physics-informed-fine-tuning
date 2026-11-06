"""Extracts per-step GT readouts and label-free feature series from banked TTA runs into one compact npz each."""
import glob
import sys
from pathlib import Path

import numpy as np

from msc.tta.eval import eval as ev

RHO = 0.9
SHELLS = np.arange(1, 33)
KMAX = 33
WINDOWS = {"early": slice(1, 9), "mid": slice(28, 37), "late": slice(55, 63), "all": slice(1, 63)}


def windows(x):
    """Averages (S, B, T) series over the WINDOWS frame ranges.

    Args:
      x: (S, B, T) per-shell per-frame series, T >= 63.

    Returns:
      (S, KMAX, len(WINDOWS)) float32, shells 0..KMAX-1.
    """
    return np.stack([x[:, :KMAX, w].mean(-1) for w in WINDOWS.values()], -1).astype(np.float32)



def horizons(pred, gt, err):
    """Returns mean correlation horizons per snapshot.

    Args:
      pred: (S, N, B, T) predicted power.
      gt: (S, N, B, T) GT power.
      err: (S, N, B, T) error power.

    Returns:
      (H_all (S,), H_k (S, len(SHELLS))) in frames, mean over N.
    """
    S = pred.shape[0]
    h_all = np.array([ev.time_to_threshold(ev.corr_curve(pred[s], gt[s], err[s], bands=slice(1, None)), RHO).mean()
                      for s in range(S)])
    h_k = np.array([[ev.time_to_threshold(ev.corr_curve(pred[s], gt[s], err[s], bands=slice(k, k + 1)), RHO).mean()
                     for k in SHELLS] for s in range(S)])
    return h_all, h_k


def features(path):
    """Builds the feature dict of one run; keys prefixed gt_ use labels, all others are label-free."""
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files if k.startswith("meta_") or k.startswith("losses_")}
    out["step"] = d["step"]
    for side in ("heldout", "pool"):
        pred, gt, err = d[f"{side}_pred_pt"], d[f"{side}_gt_pt"], d[f"{side}_err_pt"]
        h_all, h_k = horizons(pred, gt, err)
        out[f"gt_{side}_H_all"], out[f"gt_{side}_H_k"] = h_all, h_k
        out[f"gt_{side}_rel_all"] = np.array([ev.rel_l2(err[s], gt[s], bands=slice(1, None)) for s in range(len(pred))])
        out[f"gt_{side}_rel_k7"] = np.array([ev.rel_l2(err[s], gt[s], bands=slice(1, 8)) for s in range(len(pred))])
        out[f"gt_{side}_E_w"] = windows(gt[:1].mean(1))[0]
        out[f"{side}_E_ic"] = gt[0, :, :KMAX, 0].mean(0).astype(np.float32)
        out[f"{side}_E_ic_n"] = gt[0, :, :KMAX, 0].astype(np.float32)
        out[f"{side}_E_pred_w"] = windows(pred.mean(1))
        out[f"{side}_E_pred_tot_t"] = pred[:, :, 1:].sum(2).mean(1).astype(np.float32)
        out[f"{side}_res_w"] = windows(d[f"{side}_pde_res_pred_pt"].mean(1))
        out[f"{side}_res_op_w"] = windows(d[f"{side}_pde_res_pred_op_pt"].mean(1))
        out[f"{side}_w1wc_t"] = np.nanmean(d[f"{side}_w1wc_t"], 1).astype(np.float32)
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], Path(sys.argv[2])
    shard, n_shards = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(glob.glob(src + "/*/*.npz"))[shard::n_shards]:
        name = p.split("/")[-2] + "__" + Path(p).stem
        if (dst / f"{name}.npz").exists():
            continue
        try:
            f = features(p)
        except Exception as e:
            print("FAIL", p, e, file=sys.stderr, flush=True)
            continue
        if len(f["step"]) < 4:
            continue
        np.savez_compressed(dst / f"{name}.npz", **f)
        print(name, str(f["meta_heldout_split"]), f["gt_heldout_H_all"][[0, -1]].round(2), flush=True)
