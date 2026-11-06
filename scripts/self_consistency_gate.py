"""Step 1 — self-consistency TRUTH gate (frozen operator, forward-only, no training).

Question (roadmap step 2a, sharpened): does per-instance LATE self-inconsistency
rank-track per-instance LATE one-shot error vs GT, both in the k<=7 band?
  pass  -> self-inconsistency is a label-free signal that tracks truth -> proceed.
  fail  (Spearman ~0 / negative) -> chaos correlate -> self-consistency direction DEAD.

Per instance i, restart m (u = F(ic) = one-shot field):
  self_incons(i,m) = relL2_{k<=7, late frames}( F(u[m]) , u[m:] )
                     -- model vs its OWN one-shot tail after restarting from own frame m.
  gt_late(i)       = relL2_{k<=7, late frames}( u , GT )
                     -- the one-shot model error we ultimately want to lower.
Both use eval.py band machinery + the canonical late window (last T//8 frames),
so gt_late reproduces eval.per_instance_k7['late'].

Reported per op, per m: Spearman rho (rank-tracking, the necessary test) + Pearson
+ p-value + n. Also rho vs aggr error for context. Label-free: no GT in the signal.

Run: CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD python scripts/self_consistency_gate.py --ops op100 op500
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from msc.tta import setup
from msc.tta.eval import cheb_bins, band_power_t, K_REP
from scripts.chain_gate import CKPTS

DATA_RE = 500
HELDOUT = (200, 300)
EPS = 1e-30
OUT = setup.ROOT / "msc" / "tta" / "outputs" / "selfcons"


def _kf(model, ic, T):
    return kf_forward(model, ic, T, time_scale=setup.TIME_SCALE,
                      temporal_pad=setup.TEMPORAL_PAD).squeeze(1)        # (1,S,S,T)


def _late_relL2(num_field, den_field, kinf, n_bands, nE):
    """relL2 over the k<=7 band, last nE frames. num/den_field: (1,S,S,Tt)."""
    num_t = band_power_t(num_field, kinf, n_bands)[:K_REP + 1].sum(0)    # (Tt,)
    den_t = band_power_t(den_field, kinf, n_bands)[:K_REP + 1].sum(0)
    return float(np.sqrt(num_t[-nE:].sum() / (den_t[-nE:].sum() + EPS)))


def run_op(model, dataset, device, strides) -> dict:
    S = dataset[0]["y"].shape[0]
    T = dataset[0]["y"].shape[-1]
    n_bands = S // 2 + 1
    kinf = cheb_bins(S, device)
    nE = max(1, T // 8)

    N = len(dataset)
    gt_late = np.zeros(N)
    gt_aggr = np.zeros(N)
    incons = {m: np.zeros(N) for m in strides}
    for i in range(N):
        ic = dataset[i]["x"].unsqueeze(0).to(device)                    # (1,S,S)
        gt = dataset[i]["y"].unsqueeze(0).to(device)                    # (1,S,S,T)
        with torch.no_grad():
            u = _kf(model, ic, T)                                       # one-shot (1,S,S,T)
            gt_late[i] = _late_relL2(u - gt, gt, kinf, n_bands, nE)
            ep = band_power_t(u - gt, kinf, n_bands)[:K_REP + 1]
            gp = band_power_t(gt, kinf, n_bands)[:K_REP + 1]
            gt_aggr[i] = float(np.sqrt(ep.sum() / (gp.sum() + EPS)))
            for m in strides:
                v_m = _kf(model, u[:, :, :, m], T)                      # restart from own frame m
                seg = v_m[..., 0:T - m]                                 # abs frames m..T-1
                u_tail = u[..., m:T]
                incons[m][i] = _late_relL2(seg - u_tail, u_tail, kinf, n_bands, nE)

    stats = {}
    for m in strides:
        rho_l, p_l = spearmanr(incons[m], gt_late)
        r_l, _ = pearsonr(incons[m], gt_late)
        rho_a, p_a = spearmanr(incons[m], gt_aggr)
        stats[m] = {"spearman_late": float(rho_l), "p_late": float(p_l),
                    "pearson_late": float(r_l),
                    "spearman_aggr": float(rho_a), "p_aggr": float(p_a),
                    "incons_mean": float(incons[m].mean())}
    return {"strides": strides, "nE": nE, "n": N, "stats": stats,
            "gt_late": gt_late.tolist(), "gt_aggr": gt_aggr.tolist(),
            "incons": {m: incons[m].tolist() for m in strides}}


def _plot(op, res):
    strides = res["strides"]
    fig, ax = plt.subplots(1, len(strides), figsize=(5 * len(strides), 4.5), squeeze=False)
    gt = np.array(res["gt_late"])
    for j, m in enumerate(strides):
        a = ax[0][j]
        a.scatter(res["incons"][m], gt, s=14, alpha=0.6)
        st = res["stats"][m]
        a.set_title(f"{op}  m={m}\nSpearman {st['spearman_late']:.3f} (p={st['p_late']:.1e})")
        a.set_xlabel("late self-inconsistency (k<=7)")
        a.set_ylabel("GT late error (k<=7)")
        a.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / f"selfcons_gate_{op}.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Self-consistency truth gate (Spearman)")
    ap.add_argument("--ops", nargs="+", default=["op100", "op500"])
    ap.add_argument("--strides", type=int, nargs="+", default=[16, 32])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heldout = KFDataset(str(setup.data_path(DATA_RE)), n_samples=HELDOUT[1] - HELDOUT[0],
                        offset=HELDOUT[0], sub_t=setup.SUB_T)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"self-consistency GATE  strides={args.strides}  heldout={HELDOUT} "
          f"n={len(heldout)}  device={device}\n", flush=True)
    summary = {}
    for op in args.ops:
        model = setup.load_model(CKPTS[op], device)
        res = run_op(model, heldout, device, args.strides)
        _plot(op, res)
        summary[op] = res["stats"]
        print(f"== {op} ==  (n={res['n']}, late=last {res['nE']} frames)")
        for m in args.strides:
            st = res["stats"][m]
            print(f"  m={m:>2}: Spearman(late) {st['spearman_late']:+.3f} p={st['p_late']:.1e} | "
                  f"Pearson(late) {st['pearson_late']:+.3f} | Spearman(aggr) {st['spearman_aggr']:+.3f} "
                  f"p={st['p_aggr']:.1e} | <incons>={st['incons_mean']:.3f}", flush=True)
        (OUT / f"selfcons_gate_{op}.json").write_text(json.dumps(res, indent=2))
    print(f"\nVerdict guide: Spearman(late) >= ~0.4 (p<0.05) on a working m -> signal tracks "
          f"truth, proceed to dose-response. ~0 / negative -> chaos correlate, direction dead.")
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
