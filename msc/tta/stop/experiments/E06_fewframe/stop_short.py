"""Scores short-window observed-frame error (pool ICs, frames 1..m) as a causal stop signal against GT horizon."""
import glob
import itertools
import sys

import numpy as np

from causal import causal_stop

MS = (1, 2, 4, 8)
ADAPT = "/system/user/studentwork/wehofer/pino-lightning/msc/tta/outputs/adapt"


def short_err(raw, side, m):
    """Returns relative error power over frames 1..m, all shells, per step (S,)."""
    err, gt = raw[f"{side}_err_pt"][..., 1:m + 1], raw[f"{side}_gt_pt"][..., 1:m + 1]
    return err.sum((1, 2, 3)) / gt.sum((1, 2, 3))


def tag(f):
    """Returns the run family used for reporting."""
    ckpt, tgt, obj = (str(f[k]) for k in ("meta_ckpt", "meta_target_re", "meta_objective"))
    pde = str(f.get("meta_pde_re", tgt))
    pool = str(f.get("meta_pool_path", ""))
    if "tau" in pool:
        return "A5 " + pool.split("pool_")[-1].split("_res")[0]
    direction = "fwd" if ckpt == "3z5bxjzp" else "rev"
    if obj != "physics":
        return f"{direction} {obj}"
    if (ckpt, tgt) in (("3z5bxjzp", "100"), ("qr6rs0jb", "500")):
        return f"{direction} noshift"
    return f"{direction} oracle-nu" if pde == tgt else f"{direction} nu{pde}"


def load(feat_dir):
    """Loads val E01 features joined with short-window errors from the raw npz, deduplicated by config."""
    runs, seen = [], set()
    for p in sorted(glob.glob(feat_dir + "/*.npz")):
        f = dict(np.load(p, allow_pickle=True))
        name = p.split("/")[-1][:-4]
        if not str(f["meta_heldout_split"]).startswith("val") or name.rsplit("_", 1)[0] in seen:
            continue
        seen.add(name.rsplit("_", 1)[0])
        d, stem = name.split("__")
        raw = np.load(f"{ADAPT}/{d}/{stem}.npz")
        for side in ("pool", "heldout"):
            for m in MS:
                f[f"{side}_e{m}"] = short_err(raw, side, m)
        f["name"], f["family"], f["raw_path"] = stem, tag(f), f"{ADAPT}/{d}/{stem}.npz"
        runs.append(f)
    return runs


def evaluate(runs, side, m, probe, w, tau):
    """Returns per-run (name, family, positive, ΔH at s*, regret, s*, outcome, ΔH_k2)."""
    out = []
    for f in runs:
        h, hk2 = f["gt_heldout_H_all"], f["gt_heldout_H_k"][:, 1]
        i, outcome = causal_stop(f["step"], f[f"{side}_e{m}"], probe, w, tau)
        out.append((f["name"], f["family"], h.max() - h[0] > 1.0, h[i] - h[0], h.max() - h[i],
                    int(f["step"][i]), outcome, hk2[i] - hk2[0]))
    return out


def summary(rows):
    """Returns (pos mean ΔH, pos mean regret, neg mean ΔH, neg worst ΔH, n_pos, n_neg)."""
    pos = np.array([x[3:5] for x in rows if x[2]]).reshape(-1, 2)
    neg = np.array([x[3] for x in rows if not x[2]])
    m = lambda a: float(a.mean()) if len(a) else np.nan
    return m(pos[:, 0]), m(pos[:, 1]), m(neg), float(neg.min()) if len(neg) else np.nan, len(pos), len(neg)


if __name__ == "__main__":
    runs = load(sys.argv[1])
    print(f"runs={len(runs)} families: " + ", ".join(f"{k}={sum(f['family'] == k for f in runs)}"
                                                       for k in sorted({f['family'] for f in runs})))
    print("\n| side | m | P | w | tau | pos ΔH | pos regret | neg ΔH | neg worst | n_pos | n_neg |\n|---|---|---|---|---|---|---|---|---|---|---|")
    grid = []
    for side, m, probe, w, tau in itertools.product(("pool", "heldout"), MS, (25, 50, 100, 200), (50, 100), (0.01, 0.02)):
        grid.append(((side, m, probe, w, tau), summary(evaluate(runs, side, m, probe, w, tau))))
    for s, r in [g for side in ("pool", "heldout") for g in sorted((g for g in grid if g[0][0] == side), key=lambda g: -g[1][0])[:15]]:
        print("| " + " | ".join(map(str, s)) + " | " + " | ".join(f"{v:.2f}" for v in r[:4]) + f" | {r[4]} | {r[5]} |")
    print("\nbaselines (declared S, oracle):")
    for name, pick in (("declared_S", lambda f: len(f["step"]) - 1), ("oracle", lambda f: int(f["gt_heldout_H_all"].argmax()))):
        rows = [(f["name"], f["family"], f["gt_heldout_H_all"].max() - f["gt_heldout_H_all"][0] > 1,
                 f["gt_heldout_H_all"][pick(f)] - f["gt_heldout_H_all"][0],
                 f["gt_heldout_H_all"].max() - f["gt_heldout_H_all"][pick(f)]) for f in runs]
        print(name, np.round(summary(rows)[:4], 2))
    pool = [g for g in grid if g[0][0] == "pool" and g[1][2] >= -0.5]
    best = max(pool, key=lambda g: g[1][0])[0]
    print(f"\nselected (pool side, neg ΔH ≥ -0.5, max pos ΔH): {best}\n")
    print("| run | family | pos | ΔH | regret | s* | outcome | ΔH_k2 |\n|---|---|---|---|---|---|---|---|")
    for x in sorted(evaluate(runs, *best), key=lambda x: (x[1], x[0])):
        print(f"| {x[0][:60]} | {x[1]} | {int(x[2])} | {x[3]:+.2f} | {x[4]:.2f} | {x[5]} | {x[6]} | {x[7]:+.2f} |")
