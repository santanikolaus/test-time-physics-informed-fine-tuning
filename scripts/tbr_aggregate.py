"""Aggregate time_band_resolved npz outputs into the decision view.

Reads the per-run npz files, prints (a) the summary table, (b) the k<=7 error
time curve e(t) sampled at a few lead times, and (c) the spectral split of the
error: relative error and error-power fraction in cumulative wavenumber bands.
The k<=7 early/late and the low- vs high-k split together locate the 0.44.
"""
import argparse
import json

import numpy as np

K_REP = 7
BAND_GROUPS = [(0, 7), (8, 16), (17, 32), (33, 64), (65, 128)]


def load(npzs):
    runs = {}
    for p in npzs:
        d = np.load(p, allow_pickle=True)
        if "summary_json" in d:
            for s in json.loads(str(d["summary_json"])):
                runs.setdefault(s["name"], {})["summary"] = s
        for k in d.files:
            if "__" in k:
                name, field = k.split("__")
                runs.setdefault(name, {})[field] = d[k]
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    args = ap.parse_args()
    runs = load(args.npz)

    order = ["op500@8-anchor", "FNO@16", "FNO@32", "UNO@32"]
    names = [n for n in order if n in runs] + [n for n in runs if n not in order]

    print("=" * 96)
    print(f"{'run':<16}{'val_l2':>9}{'err_full':>10}{'err_k7':>9}{'early_k7':>10}{'late_k7':>9}{'late/early':>11}")
    print("-" * 96)
    for n in names:
        s = runs[n].get("summary")
        if not s:
            continue
        print(f"{n:<16}{s['val_l2_val']:>9.4f}{s['err_full']:>10.4f}{s['err_k7']:>9.4f}"
              f"{s['early']:>10.4f}{s['late']:>9.4f}{s['ratio']:>11.2f}")

    print("\nk<=7 error e(t) at lead times (fraction of horizon):")
    print(f"{'run':<16}" + "".join(f"{f't={q}':>9}" for q in ["0", "1/8", "1/4", "1/2", "3/4", "end"]))
    print("-" * 70)
    for n in names:
        et = runs[n].get("err_t")
        if et is None:
            continue
        T = len(et)
        idx = [0, T // 8, T // 4, T // 2, 3 * T // 4, T - 1]
        print(f"{n:<16}" + "".join(f"{et[i]:>9.4f}" for i in idx))

    print("\nspectral split — relative error sqrt(Σerr/Σgt) in cumulative-disjoint k bands:")
    print(f"{'run':<16}" + "".join(f"{f'k{lo}-{hi}':>11}" for lo, hi in BAND_GROUPS) + f"{'errfrac_k<=7':>14}")
    print("-" * 90)
    for n in names:
        bp_err = runs[n].get("bp_err")
        bp_gt = runs[n].get("bp_gt")
        if bp_err is None:
            continue
        cells = []
        for lo, hi in BAND_GROUPS:
            e = bp_err[lo:hi + 1].sum()
            g = bp_gt[lo:hi + 1].sum()
            cells.append(np.sqrt(e / g) if g > 0 else np.nan)
        frac_k7 = bp_err[:K_REP + 1].sum() / bp_err.sum()
        print(f"{n:<16}" + "".join(f"{c:>11.4f}" for c in cells) + f"{frac_k7:>14.3f}")


if __name__ == "__main__":
    main()
