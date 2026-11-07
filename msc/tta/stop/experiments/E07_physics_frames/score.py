"""Scores E07 runs: GT horizon per step, frozen monitoring stop rule, per-cell table physics x observed frames."""
import glob
import sys
from collections import defaultdict

import numpy as np

from msc.tta.adapt import decide
from msc.tta.eval import eval as ev

RHO, RULE, M = 0.9, (200, 100, 0.01), 8


def horizon_series(z):
    """Returns whole-field correlation horizon on held-out ICs per snapshot, (S,)."""
    return np.array([ev.time_to_threshold(ev.corr_curve(p, g, e, bands=slice(1, None)), RHO).mean()
                     for p, g, e in zip(z["heldout_pred_pt"], z["heldout_gt_pt"], z["heldout_err_pt"])])


def cell(z):
    """Returns the Table-2 cell label (physics on/off, observed frames)."""
    return f"{'physics' if str(z['meta_objective']) == 'physics' else 'no-physics'} f{int(z['meta_obs_frames'])}"


if __name__ == "__main__":
    rows = defaultdict(list)
    print("| run | cell | H0 | ΔH s1000 | ΔH oracle (s) | s* | outcome | ΔH ours | regret |\n|---|---|---|---|---|---|---|---|---|")
    for p in sorted(glob.glob(sys.argv[1] + "/*.npz")):
        z = np.load(p)
        h, steps = horizon_series(z), z["step"]
        sig = decide.window_error(z["monitor_err_pt"], z["monitor_gt_pt"], M)
        i, outcome = decide.stop_index(steps, sig, *RULE)
        r = (h[-1] - h[0], h.max() - h[0], h[i] - h[0])
        rows[cell(z)].append(r)
        np.savez(f"{sys.argv[2]}/{p.split('/')[-1]}", step=steps, H=h, monitor_e8=sig)
        print(f"| {p.split('/')[-1][:-4]} | {cell(z)} | {h[0]:.2f} | {r[0]:+.2f} | {r[1]:+.2f} ({steps[h.argmax()]}) | "
              f"{steps[i]} | {outcome} | {r[2]:+.2f} | {r[1] - r[2]:.2f} |")
    print("\n| cell | n | ΔH fixed S=1000 | ΔH oracle | ΔH ours | captured |\n|---|---|---|---|---|---|")
    for c in sorted(rows):
        a = np.array(rows[c])
        m, s = a.mean(0), a.std(0)
        cap = m[2] / m[1] if m[1] > 0 else np.nan
        print(f"| {c} | {len(a)} | {m[0]:+.2f} ± {s[0]:.2f} | {m[1]:+.2f} ± {s[1]:.2f} | {m[2]:+.2f} ± {s[2]:.2f} | {cap:.0%} |")
