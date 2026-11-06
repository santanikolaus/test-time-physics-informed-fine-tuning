"""Scores the E06 stop rule when the signal comes from only k validation ICs with m observed frames each."""
import sys

import numpy as np

from causal import causal_stop
from stop_short import load, summary

KS, MS, DRAWS = (2, 3, 5), (2, 3, 4, 8), 20
RULE = (200, 100, 0.01)


if __name__ == "__main__":
    runs = load(sys.argv[1])
    raw = {f["name"]: (lambda z: (z["heldout_err_pt"], z["heldout_gt_pt"]))(np.load(f["raw_path"])) for f in runs}
    rng = np.random.default_rng(0)
    print("| k val ICs | m frames | pos ΔH | pos regret | neg mean ΔH | neg worst (mean over draws) | neg worst (any draw) |")
    print("|---|---|---|---|---|---|---|")
    for k in KS:
        for m in MS:
            res = []
            for _ in range(DRAWS):
                rows = []
                for f in runs:
                    err, gt = raw[f["name"]]
                    idx = rng.choice(err.shape[1], k, replace=False)
                    sig = err[:, idx, :, 1:m + 1].sum((1, 2, 3)) / gt[:, idx, :, 1:m + 1].sum((1, 2, 3))
                    h = f["gt_heldout_H_all"]
                    i, _ = causal_stop(f["step"], sig, *RULE)
                    rows.append((f["name"], f["family"], h.max() - h[0] > 1.0, h[i] - h[0], h.max() - h[i]))
                res.append(summary(rows)[:4])
            r = np.array(res)
            print(f"| {k} | {m} | {r[:, 0].mean():.2f} | {r[:, 1].mean():.2f} | {r[:, 2].mean():.2f} | "
                  f"{r[:, 3].mean():.2f} | {r[:, 3].min():.2f} |")
