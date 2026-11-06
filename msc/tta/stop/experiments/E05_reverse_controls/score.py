"""Applies the frozen E02b stop rule and the E03 two-sided gate statistic to E05 runs, without retuning."""
import glob
import sys

import numpy as np

from causal import causal_stop, shape

FROZEN = ("1-16", "late", 200, 100, 0.01)
SIGMA_SRC = {"3z5bxjzp": 0.243, "qr6rs0jb": 0.742}


if __name__ == "__main__":
    print("| run | ΔH s500 | ΔH max (s) | σ(0) | log σ(0)/σ_src | σ(200)/σ(0) | rule s* | outcome | ΔH at s* | regret |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(glob.glob(sys.argv[1] + "/e05_*.npz")):
        f = dict(np.load(p, allow_pickle=True))
        k, win, probe, w, tau = FROZEN
        h, steps, sig = f["gt_heldout_H_all"], f["step"], shape(f, "pool", k, win)
        i, outcome = causal_stop(steps, sig, probe, w, tau)
        ip = int(np.searchsorted(steps, probe))
        print(f"| {p.split('__')[-1][:-4]} | {h[-1] - h[0]:+.2f} | {h.max() - h[0]:+.2f} ({steps[h.argmax()]}) | "
              f"{sig[0]:.3f} | {np.log(sig[0] / SIGMA_SRC[str(f['meta_ckpt'])]):+.2f} | {sig[ip] / sig[0]:.3f} | "
              f"{steps[i]} | {outcome} | {h[i] - h[0]:+.2f} | {h.max() - h[i]:.2f} |")
