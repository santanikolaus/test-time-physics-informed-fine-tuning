"""Step 2 — temporal-payoff gate (frozen operator, forward-only, no training).

Ceiling test for any d/dt-term regularizer (flags: residual ~wt is wiggly). Time-axis
mirror of the spatial band-limit (E_bandK). For each held-out instance, low-pass the
MODEL one-shot trajectory to a sweep of temporal-mode budgets K_t and measure late
k<=7 error vs GT. GT low-passed to the same K_t = the floor cost of the truncation itself.

  err_model(K_t) = late-k7( trunc_t(F(ic), K_t) , GT )     -- does smoothing time help?
  err_gt(K_t)    = late-k7( trunc_t(GT, K_t)    , GT )     -- truncation's own cost
  full           = late-k7( F(ic) , GT )                    -- baseline (reproduces refs)

If min_Kt err_model(K_t) >= full -> late error sits in the temporally-smooth band, no
wt-targeting regularizer can move it. If err_model(K_t) < full toward op500's 0.473 ->
recoverable error lives in high time-freq -> temporal regularization has payoff.

Run: CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD python scripts/temporal_payoff_gate.py --ops op100 op500
"""
import argparse
import json

import numpy as np
import torch

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from msc.tta import setup
from msc.tta.eval import cheb_bins, band_power_t, K_REP
from scripts.chain_gate import CKPTS
from scripts.temporal_budget import trunc_time

DATA_RE = 500
HELDOUT = (200, 300)
EPS = 1e-30
OUT = setup.ROOT / "msc" / "tta" / "outputs" / "selfcons"


def _kf(model, ic, T):
    return kf_forward(model, ic, T, time_scale=setup.TIME_SCALE,
                      temporal_pad=setup.TEMPORAL_PAD).squeeze(1)        # (1,S,S,T)


def split_k7(pred, gt, kinf, n_bands, nE):
    """(1,S,S,T) pred,gt -> (early, late, aggr) k<=7 rel-L2 (canonical per_instance_k7 reduction)."""
    ep = band_power_t(pred - gt, kinf, n_bands)[:K_REP + 1]              # (K_REP+1, T)
    gp = band_power_t(gt, kinf, n_bands)[:K_REP + 1]
    err_t = np.sqrt(ep.sum(0) / (gp.sum(0) + EPS))                      # (T,)
    return (float(err_t[1:1 + nE].mean()), float(err_t[-nE:].mean()),
            float(np.sqrt(ep.sum() / (gp.sum() + EPS))))


def run_op(model, dataset, device, budgets) -> dict:
    S = dataset[0]["y"].shape[0]
    T = dataset[0]["y"].shape[-1]
    n_bands = S // 2 + 1
    kinf = cheb_bins(S, device)
    nE = max(1, T // 8)
    N = len(dataset)

    # per-(budget) accumulators of per-instance (early,late,aggr); 'full' = untruncated model
    keys = ["full"] + [f"Kt{k}" for k in budgets]
    model_late = {k: np.zeros(N) for k in keys}
    model_early = {k: np.zeros(N) for k in keys}
    gt_late = {f"Kt{k}": np.zeros(N) for k in budgets}                  # truncation floor
    for i in range(N):
        ic = dataset[i]["x"].unsqueeze(0).to(device)
        gt = dataset[i]["y"].unsqueeze(0).to(device)
        with torch.no_grad():
            u = _kf(model, ic, T)
            e, l, _ = split_k7(u, gt, kinf, n_bands, nE)
            model_early["full"][i], model_late["full"][i] = e, l
            for k in budgets:
                u_lp = trunc_time(u, n_tmodes=k)
                e, l, _ = split_k7(u_lp, gt, kinf, n_bands, nE)
                model_early[f"Kt{k}"][i], model_late[f"Kt{k}"][i] = e, l
                _, gl, _ = split_k7(trunc_time(gt, n_tmodes=k), gt, kinf, n_bands, nE)
                gt_late[f"Kt{k}"][i] = gl

    rows = {k: {"model_late": float(model_late[k].mean()),
                "model_early": float(model_early[k].mean())} for k in keys}
    for k in budgets:
        rows[f"Kt{k}"]["gt_late"] = float(gt_late[f"Kt{k}"].mean())
    return {"budgets": budgets, "nE": nE, "n": N, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="Temporal-payoff gate (time-axis band sweep)")
    ap.add_argument("--ops", nargs="+", default=["op100", "op500"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[3, 4, 5, 6, 8, 10, 12, 16, 24])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heldout = KFDataset(str(setup.data_path(DATA_RE)), n_samples=HELDOUT[1] - HELDOUT[0],
                        offset=HELDOUT[0], sub_t=setup.SUB_T)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"temporal-payoff GATE  budgets={args.budgets} (FNO time n_modes=8)  "
          f"heldout={HELDOUT} n={len(heldout)}  device={device}\n", flush=True)
    summary = {}
    for op in args.ops:
        model = setup.load_model(CKPTS[op], device)
        res = run_op(model, heldout, device, args.budgets)
        summary[op] = res["rows"]
        full = res["rows"]["full"]["model_late"]
        best_k = min(args.budgets, key=lambda k: res["rows"][f"Kt{k}"]["model_late"])
        best = res["rows"][f"Kt{best_k}"]["model_late"]
        print(f"== {op} ==  (n={res['n']}, late=last {res['nE']} frames)  full late={full:.3f}")
        print(f"  {'K_t':>4} {'model_late':>11} {'Δ vs full':>10} {'gt_late(floor)':>15} {'model_early':>12}")
        for k in args.budgets:
            r = res["rows"][f"Kt{k}"]
            print(f"  {k:>4} {r['model_late']:>11.3f} {r['model_late'] - full:>+10.3f} "
                  f"{r['gt_late']:>15.3f} {r['model_early']:>12.3f}", flush=True)
        print(f"  best model_late = {best:.3f} @ K_t={best_k}  (Δ vs full {best - full:+.3f})\n", flush=True)
        (OUT / f"temporal_payoff_{op}.json").write_text(json.dumps(res, indent=2))
    print("read: best model_late < full by a real margin -> recoverable error in high time-freq "
          "(temporal reg has payoff). best >= full -> late error is temporally smooth, lever dead.")
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
