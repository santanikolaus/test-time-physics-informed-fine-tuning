"""Scores start-gate statistics on 5-IC pools: AUROC and detection rate, in-distribution vs shifted, both directions."""
import sys

import numpy as np

K = slice(1, 17)
B3 = slice(9, 17)
LATE = slice(55, 63)
POOL, DRAWS = 5, 2000
DIRECTIONS = {"forward Re100->Re500": ("3z5bxjzp_re100", "3z5bxjzp_re500"),
              "reverse Re500->Re100": ("qr6rs0jb_re500", "qr6rs0jb_re100")}


def stats(z, key, idx):
    """Returns (shape sigma, energy-level M, B3 residual raw, B3 residual / IC B3 power) for one IC pool."""
    pred = z[f"{key}_pred_pt"][idx][:, K, LATE].mean((0, 2))
    ic = z[f"{key}_gt_pt"][idx][:, K, 0].mean(0)
    lr = np.log(pred / ic)
    res = z[f"{key}_pde_res_pred_op_pt"][idx][:, B3].sum((1, 2)).mean()
    ic_b3 = z[f"{key}_gt_pt"][idx][:, B3, 0].sum(1).mean()
    return lr.std(), np.abs(lr).mean(), res, res / ic_b3


def auroc(neg, pos):
    """Returns P(score_pos > score_neg) via ranks."""
    s = np.r_[neg, pos]
    r = s.argsort().argsort() + 1
    return (r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


if __name__ == "__main__":
    z = np.load(sys.argv[1])
    rng = np.random.default_rng(0)
    names = ("shape σ", "energy M", "B3 resid", "B3 resid / IC")
    print("| direction | statistic | ID median | OOD median | AUROC | OOD detected @ ID p95 |\n|---|---|---|---|---|---|")
    for direction, (id_key, ood_key) in DIRECTIONS.items():
        n = z[f"{id_key}_pred_pt"].shape[0]
        draw = lambda key: np.array([stats(z, key, rng.choice(n, POOL, replace=False)) for _ in range(DRAWS)])
        a, b = draw(id_key), draw(ood_key)
        for j, name in enumerate(names):
            thr = np.quantile(a[:, j], 0.95)
            print(f"| {direction} | {name} | {np.median(a[:, j]):.4g} | {np.median(b[:, j]):.4g} | "
                  f"{auroc(a[:, j], b[:, j]):.3f} | {(b[:, j] > thr).mean():.2f} |")
