"""Scores the causal probe-abort + plateau stop rule on E01 features over a settings grid."""
import glob
import itertools
import sys

import numpy as np

KS = {"1-8": slice(1, 9), "1-16": slice(1, 17), "4-16": slice(4, 17), "1-32": slice(1, 33)}
WIN = {"mid": 1, "late": 2, "all": 3}
DELTA = 0.02
BUDGET_TESTS = ("ie46pt7g", "pm1135ol")


def shape(f, side, k, win):
    """Returns sigma(s) = std over shells of log(E_pred / E_IC), (S,)."""
    return np.log(f[f"{side}_E_pred_w"][:, KS[k], WIN[win]] / f[f"{side}_E_ic"][KS[k]]).std(1)


def causal_stop(steps, sigma, probe, w, tau):
    """Applies probe-abort + plateau using only past snapshots.

    Args:
      steps: (S,) optimizer step of each snapshot, ascending, steps[0] == 0.
      sigma: (S,) label-free series, lower = better.
      probe: probe length P in optimizer steps.
      w: plateau look-back window in optimizer steps.
      tau: minimum relative improvement of the running minimum over w.

    Returns:
      (snapshot index chosen, outcome in {"abort", "short", "stop", "censored"}).
    """
    if steps[-1] < probe:
        return 0, "short"
    ip = int(np.searchsorted(steps, probe))
    if sigma[ip] >= (1 - DELTA) * sigma[0]:
        return 0, "abort"
    run_min = np.minimum.accumulate(sigma)
    for i in range(ip + 1, len(steps)):
        j = int(np.searchsorted(steps, steps[i] - w, side="right")) - 1
        if j >= 0 and run_min[j] - run_min[i] < tau * sigma[0]:
            return int(np.argmin(sigma[: i + 1])), "stop"
    return int(np.argmin(sigma)), "censored"


def load(feat_dir, split):
    """Loads E01 features of one split, collapsing duplicate configs, tagging protocol family."""
    runs, seen = [], set()
    for p in sorted(glob.glob(feat_dir + "/*.npz")):
        f = dict(np.load(p, allow_pickle=True))
        if not str(f["meta_heldout_split"]).startswith(split):
            continue
        name = p.split("/")[-1][:-4]
        cfg = name.rsplit("_", 1)[0]
        if cfg in seen:
            continue
        seen.add(cfg)
        f["name"] = name
        f["protocol"] = (str(f["meta_ckpt"]) == "3z5bxjzp" and float(f["meta_lr"]) <= 3e-4
                         and int(f["meta_steps"]) >= 200)
        runs.append(f)
    return runs


def evaluate(runs, side, k, win, probe, w, tau):
    """Returns per-run (name, protocol, positive, ΔH at s*, regret, step*, outcome, ΔH_k2)."""
    out = []
    for f in runs:
        h, hk2 = f["gt_heldout_H_all"], f["gt_heldout_H_k"][:, 1]
        i, outcome = causal_stop(f["step"], shape(f, side, k, win), probe, w, tau)
        out.append((f["name"], f["protocol"], h.max() - h[0] > 1.0, h[i] - h[0], h.max() - h[i],
                    int(f["step"][i]), outcome, hk2[i] - hk2[0]))
    return out


def summary(rows, family):
    """Returns (pos mean ΔH, pos mean regret, neg mean ΔH, neg worst ΔH, n_pos, n_neg) for one family."""
    r = [x for x in rows if x[1] == family]
    pos = np.array([x[3:5] for x in r if x[2]]).reshape(-1, 2)
    neg = np.array([x[3] for x in r if not x[2]])
    m = lambda a: float(a.mean()) if len(a) else np.nan
    return m(pos[:, 0]), m(pos[:, 1]), m(neg), float(neg.min()) if len(neg) else np.nan, len(pos), len(neg)


if __name__ == "__main__":
    runs = load(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "val")
    grid = []
    for k, win, probe, w, tau in itertools.product(KS, WIN, (60, 100, 150, 200, 250), (50, 100), (0.01, 0.02)):
        rows = evaluate(runs, "pool", k, win, probe, w, tau)
        grid.append(((k, win, probe, w, tau), summary(rows, True), summary(rows, False)))
    print(f"runs={len(runs)} protocol={sum(f['protocol'] for f in runs)}\n")
    print("| K | win | P | w | tau | proto pos ΔH | proto pos regret | proto neg ΔH | proto neg worst | stress pos ΔH | stress neg ΔH | stress neg worst |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s, pr, st in sorted(grid, key=lambda g: -g[1][0]):
        print("| " + " | ".join(map(str, s)) + " | " + " | ".join(f"{v:.2f}" for v in pr[:4] + st[:1] + st[2:4]) + " |")
    p, n = grid[0][1][4], grid[0][1][5]
    print(f"\nprotocol n_pos={p} n_neg={n}; baselines: declared S, never adapt")
    for name, pick in (("declared_S", lambda f: len(f["step"]) - 1), ("never", lambda f: 0)):
        rows = [(f["name"], f["protocol"], f["gt_heldout_H_all"].max() - f["gt_heldout_H_all"][0] > 1,
                 f["gt_heldout_H_all"][pick(f)] - f["gt_heldout_H_all"][0],
                 f["gt_heldout_H_all"].max() - f["gt_heldout_H_all"][pick(f)]) for f in runs]
        print(name, "protocol", np.round(summary(rows, True)[:4], 2), "stress", np.round(summary(rows, False)[:4], 2))
    ok = [g for g in grid if g[1][2] >= -0.5]
    best = max(ok, key=lambda g: g[1][0])[0] if ok else grid[0][0]
    print(f"\nfrozen setting {best}; per-run (protocol first):")
    for side in ("pool", "heldout"):
        print(f"\n[{side} ICs]")
        rows = evaluate(runs, side, *best)
        print("| run | proto | pos | ΔH | regret | s* | outcome | ΔH_k2 |\n|---|---|---|---|---|---|---|---|")
        for x in sorted(rows, key=lambda x: (not x[1], x[0])):
            tag = " BUDGET" if any(b in x[0] for b in BUDGET_TESTS) else ""
            print(f"| {x[0][:70]}{tag} | {int(x[1])} | {int(x[2])} | {x[3]:+.2f} | {x[4]:.2f} | {x[5]} | {x[6]} | {x[7]:+.2f} |")
