"""Tabulates GT horizon against self-consistency and drift signals per checkpoint for E04 outputs."""
import glob
import sys

import numpy as np

from msc.tta.eval import eval as ev

RHO = 0.9
ALL = slice(1, None)


def horizon(a, b, d, bands=ALL):
    """Returns mean first-cross horizon of the pooled correlation between two power-matched fields."""
    return float(ev.time_to_threshold(ev.corr_curve(a, b, d, bands=bands), RHO).mean())


def mean_corr(a, b, d, bands=ALL):
    """Returns the correlation averaged over frames and samples (continuous, unquantised)."""
    return float(ev.corr_curve(a, b, d, bands=bands).mean())


def table(path):
    """Prints one row per checkpoint: GT H, GT H_k2, self-consistency (held-out, pool), drift."""
    z = np.load(path)
    n = int(z["n_heldout"])
    h, p = slice(0, n), slice(n, None)
    print(f"\n## {path.split('/')[-1][:-4]}\n")
    print("| step | GT H | GT H_k2 | SC16 H held | SC32 H held | SC16 ρ̄ held | SC16 ρ̄ pool | drift late held |")
    print("|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(z["step"]):
        g = [z[k][i] for k in ("pred", "gt", "err")]
        sc = {t: [z[f"sc{t}_{x}"][i] for x in "abd"] for t in (16, 32)}
        drift = z["drift"][i][h, 1:, 55:63].sum() / z["pred"][i][h, 1:, 55:63].sum()
        print(f"| {s} | {horizon(*(x[h] for x in g)):.2f} | {horizon(*(x[h] for x in g), bands=slice(2, 3)):.2f} | "
              f"{horizon(*(x[h] for x in sc[16])):.2f} | {horizon(*(x[h] for x in sc[32])):.2f} | "
              f"{mean_corr(*(x[h] for x in sc[16])):.4f} | {mean_corr(*(x[p] for x in sc[16])):.4f} | {drift:.4f} |")


if __name__ == "__main__":
    for path in sorted(glob.glob(sys.argv[1] + "/*.npz")):
        table(path)
