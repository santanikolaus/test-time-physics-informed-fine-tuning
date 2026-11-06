"""Scores label-free stop/start rules on E01 feature files by GT horizon regret."""
import glob
import sys

import numpy as np

K = slice(1, 17)
W = {"early": 0, "mid": 1, "late": 2, "all": 3}


def mismatch(f, side, window, ic_side=None):
    """Returns invariant-measure mismatch M(s) = mean_k |log E_pred(k,window) / E_ic(k)|, shells 1..16.

    Args:
      f: E01 feature dict.
      side: "heldout" or "pool" — whose predictions are read.
      window: key of W.
      ic_side: whose IC ensemble is the reference; defaults to side.

    Returns:
      (S,) mismatch per snapshot.
    """
    e_ic = f[f"{ic_side or side}_E_ic"][K]
    return np.abs(np.log(f[f"{side}_E_pred_w"][:, K, W[window]] / e_ic)).mean(1)


def shape_mismatch(f, side, window, ic_side=None):
    """Returns amplitude-free spectral-shape mismatch: std over shells 1..16 of log E_pred / E_ic, per snapshot."""
    e_ic = f[f"{ic_side or side}_E_ic"][K]
    return np.log(f[f"{side}_E_pred_w"][:, K, W[window]] / e_ic).std(1)


def residual(f, side, window="all", op=False):
    """Returns pooled residual power per snapshot, shells 1..32."""
    return f[f"{side}_res{'_op' if op else ''}_w"][:, 1:, W[window]].sum(1)


def rule_argmin(series):
    """Stops at the snapshot index minimising a label-free series."""
    return int(np.nanargmin(series))


def rule_gate(series, delta):
    """Argmin, but returns 0 (do not adapt) unless the minimum improves on step 0 by more than delta (relative)."""
    i = int(np.nanargmin(series))
    return i if series[i] < series[0] * (1 - delta) else 0


RULES = {
    "declared_S": lambda f: len(f["step"]) - 1,
    "start_S0": lambda f: 0,
    "min_loss": lambda f: rule_argmin(np.r_[np.nan, f["losses_loss"][np.clip(f["step"][1:] - 1, 0, None)]]),
    "min_res_pool": lambda f: rule_argmin(residual(f, "pool")),
    "min_res_held": lambda f: rule_argmin(residual(f, "heldout")),
    "min_M_late_held": lambda f: rule_argmin(mismatch(f, "heldout", "late")),
    "min_M_all_held": lambda f: rule_argmin(mismatch(f, "heldout", "all")),
    "min_M_late_pool": lambda f: rule_argmin(mismatch(f, "pool", "late")),
    "gate05_M_late_held": lambda f: rule_gate(mismatch(f, "heldout", "late"), 0.05),
    "gate05_M_late_pool": lambda f: rule_gate(mismatch(f, "pool", "late"), 0.05),
    "min_shape_late_pool": lambda f: rule_argmin(shape_mismatch(f, "pool", "late")),
    "min_shape_late_held": lambda f: rule_argmin(shape_mismatch(f, "heldout", "late")),
    "min_shape_all_pool": lambda f: rule_argmin(shape_mismatch(f, "pool", "all")),
    "gate05_shape_late_pool": lambda f: rule_gate(shape_mismatch(f, "pool", "late"), 0.05),
}


def score(path):
    """Returns one row: run tags, GT trajectory summary, and per-rule (step, ΔH vs S=0, regret vs oracle, H_k2 dip)."""
    f = dict(np.load(path, allow_pickle=True))
    h, hk2 = f["gt_heldout_H_all"], f["gt_heldout_H_k"][:, 1]
    best = int(np.argmax(h))
    row = {"run": path.split("/")[-1][:-4], "split": str(f["meta_heldout_split"]).split()[0],
           "H0": h[0], "Hbest": h[best], "s_best": int(f["step"][best]), "HS": h[-1]}
    for name, rule in RULES.items():
        i = rule(f)
        row[name] = (int(f["step"][i]), h[i] - h[0], h[best] - h[i], hk2[i] - hk2[0])
    return row


if __name__ == "__main__":
    rows = [score(p) for p in sorted(glob.glob(sys.argv[1] + "/*.npz"))]
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    rows = [r for r in rows if r["split"] == split]
    pos = [r for r in rows if r["Hbest"] - r["H0"] > 1.0]
    neg = [r for r in rows if r["Hbest"] - r["H0"] <= 1.0]
    print(f"split={split} runs={len(rows)} positive(oracle gain>1 frame)={len(pos)} negative={len(neg)}\n")
    print("| rule | pos: mean ΔH vs S0 | pos: mean regret | neg: mean ΔH vs S0 | neg: worst ΔH | all: k2 ΔH<-0.5 count |")
    print("|---|---|---|---|---|---|")
    for name in RULES:
        p = np.array([r[name] for r in pos]) if pos else np.zeros((0, 4))
        n = np.array([r[name] for r in neg]) if neg else np.zeros((0, 4))
        a = np.array([r[name] for r in rows])
        print(f"| {name} | {p[:, 1].mean():.2f} | {p[:, 2].mean():.2f} | "
              f"{n[:, 1].mean() if len(n) else np.nan:.2f} | {n[:, 1].min() if len(n) else np.nan:.2f} | "
              f"{int((a[:, 3] < -0.5).sum())} |")
    print("\n| run | H0 | Hbest@s | HS | " + " | ".join(RULES) + " |")
    for r in rows:
        print(f"| {r['run'][:60]} | {r['H0']:.1f} | {r['Hbest']:.1f}@{r['s_best']} | {r['HS']:.1f} | "
              + " | ".join(f"{r[k][0]}:{r[k][1]:+.1f}" for k in RULES) + " |")
