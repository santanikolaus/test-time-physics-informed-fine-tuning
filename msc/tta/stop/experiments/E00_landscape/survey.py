"""Surveys banked TTA runs: GT error vs step and label-free signal tracking, one row per run."""
import glob
import sys

import numpy as np
from scipy.stats import spearmanr

K7 = slice(1, 8)
ALL = slice(1, None)


def rel(err, gt, bands):
    """Returns pooled rel_l2 per snapshot step, (S,)."""
    return np.sqrt(err[:, :, bands].sum((1, 2, 3)) / gt[:, :, bands].sum((1, 2, 3)))


def rms(res, bands=slice(None)):
    """Returns pooled residual RMS per snapshot step, (S,)."""
    return np.sqrt(res[:, :, bands].sum((1, 2, 3)))


def row(path):
    """Summarises one run: oracle-best step, gains, and Spearman of label-free signals vs heldout GT."""
    d = np.load(path, allow_pickle=True)
    step = d["step"]
    if len(step) < 4:
        return None
    gt_h = rel(d["heldout_err_pt"], d["heldout_gt_pt"], K7)
    gt_all = rel(d["heldout_err_pt"], d["heldout_gt_pt"], ALL)
    sig = {
        "res_pool": rms(d["pool_pde_res_pred_pt"]),
        "res_held": rms(d["heldout_pde_res_pred_pt"]),
        "res_held_k7": rms(d["heldout_pde_res_pred_pt"], K7),
    }
    loss = d["losses_loss"]
    idx = np.clip(step - 1, 0, len(loss) - 1)
    sig["loss"] = np.where(step == 0, np.nan, loss[idx])
    b = int(np.argmin(gt_h))
    out = dict(run=path.split("/")[-2] + "/" + path.split("/")[-1][:-4], S=int(step[-1]), dstep=int(step[1] - step[0]),
               gt0=gt_h[0], gtbest=gt_h[b], best_step=int(step[b]), gtfin=gt_h[-1], all0=gt_all[0], allfin=gt_all[-1])
    for k, v in sig.items():
        m = ~np.isnan(v)
        out[f"rho_{k}"] = spearmanr(v[m], gt_h[m])[0]
        out[f"argmin_{k}"] = int(step[m][np.argmin(v[m])])
    return out


if __name__ == "__main__":
    rows = []
    for p in sorted(glob.glob(sys.argv[1] + "/*/*.npz")):
        try:
            r = row(p)
        except Exception as e:
            print("FAIL", p, e, file=sys.stderr)
            continue
        if r:
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    np.save(sys.argv[2], rows, allow_pickle=True)
