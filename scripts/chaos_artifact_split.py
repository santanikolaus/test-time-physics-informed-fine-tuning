"""Chaos-vs-artifact split — is the late wall unfixable chaos or fixable physics error?

For each held-out instance, take the operator's one-shot field at a reprojection
frame t_r, hand it to the TRUE NS solver (src.solver.periodic.NavierStokes2d, the
generator of the data), integrate to t=1, and split the late k<=7 error:

  A = relL2(pred_late , GT_late)   total operator late error      (= our 0.678/0.473)
  B = relL2(pred_late , sol_late)  pred vs solver-continuation of its OWN field
                                   = FIXABLE artifact (operator violates physics)
  C = relL2(sol_late  , GT_late)   solver-continuation vs truth
                                   = UNFIXABLE chaos (mid-field error amplified)
  all normalised by ||GT_late|| so A ~ budget of B (fixable) + C (chaos).

CONTROL: reproject from the TRUE field GT[t_r] -> C_ctrl should be small (only
float32-roundoff chaos); validates that solver forcing/Re/dt/resolution match the
data. If C_ctrl ~ A, the setup is wrong, not the physics.

Read: C ~ A (B~0) -> wall is CHAOS, unfixable from frame 0. B ~ A (C small) -> wall
is fixable operator artifact. Sweep t_r to see chaos grow with horizon.

Run: CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD python scripts/chaos_artifact_split.py
"""
import argparse
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from src.solver.periodic import NavierStokes2d
from msc.tta import setup, eval as ev
from scripts.chain_gate import CKPTS

DATA_RE = 500
HELDOUT = (200, 300)
OUT = setup.ROOT / "msc" / "tta" / "outputs" / "chaos_split"
LO = slice(0, ev.K_REP + 1)        # k<=7


def kf_forcing(S, device, dtype):
    """f = -4cos(4Y), Y varying along axis 1 — identical to src/solver/generate_kf.py."""
    t = torch.linspace(0, 2 * math.pi, S + 1, dtype=dtype, device=device)[:-1]
    _, Y = torch.meshgrid(t, t, indexing="ij")
    return -4.0 * torch.cos(4.0 * Y)


def solve_forward(solver, seed, f, t_r: int, T: int, dt: float, device) -> torch.Tensor:
    """Integrate the TRUE solver from `seed` (S,S float64 vorticity at frame t_r) one
    dt per frame to t=1. Returns (S,S,T) float32, frames t_r..T-1 filled, NaN before."""
    S = seed.shape[-1]
    out = torch.full((S, S, T), float("nan"), device=device)
    out[:, :, t_r] = seed.float()
    w = seed.unsqueeze(0)                                   # (1,S,S)
    for fr in range(t_r + 1, T):
        w = solver.advance(w, f, T=dt, Re=DATA_RE, adaptive=True)
        out[:, :, fr] = w[0].float()
    return out


def k7_late_power(field, kinf, n_bands, nlate) -> float:
    """k<=7 power summed over the last `nlate` frames + space, of (1,S,S,T) field."""
    p = ev.band_power_t(field, kinf, n_bands)[LO]          # (K+1, T)
    return float(p[:, -nlate:].sum())


def run(model, dataset, solver, f, strides, T, dt, kinf, n_bands, nlate, device, want_ctrl):
    keys = ("A", "B", "C", "den") + (("Cctrl",) if want_ctrl else ())
    acc = {tr: {k: 0.0 for k in keys} for tr in strides}
    for i in range(len(dataset)):
        gt = dataset[i]["y"].unsqueeze(0).to(device)        # (1,S,S,T)
        with torch.no_grad():
            pred = kf_forward(model, gt[:, :, :, 0], T, time_scale=setup.TIME_SCALE,
                              temporal_pad=setup.TEMPORAL_PAD).squeeze(1) if model is not None else None
        for tr in strides:
            late = slice(T - nlate, T)
            d = acc[tr]
            d["den"] += k7_late_power(gt[..., late], kinf, n_bands, nlate)
            if model is not None:
                sol = solve_forward(solver, pred[0, :, :, tr].double(), f, tr, T, dt, device)[None]
                d["A"] += k7_late_power(pred[..., late] - gt[..., late], kinf, n_bands, nlate)
                d["B"] += k7_late_power(pred[..., late] - sol[..., late], kinf, n_bands, nlate)
                d["C"] += k7_late_power(sol[..., late] - gt[..., late], kinf, n_bands, nlate)
            if want_ctrl:
                solg = solve_forward(solver, gt[0, :, :, tr].double(), f, tr, T, dt, device)[None]
                d["Cctrl"] += k7_late_power(solg[..., late] - gt[..., late], kinf, n_bands, nlate)
    out = {}
    for tr in strides:
        d = acc[tr]
        rel = lambda k: float(np.sqrt(d[k] / (d["den"] + 1e-30)))
        row = {k: rel(k) for k in keys if k != "den"}
        out[tr] = row
    return out


def main():
    ap = argparse.ArgumentParser(description="Chaos-vs-artifact split via solver reprojection")
    ap.add_argument("--ops", nargs="+", default=["op100", "op500"])
    ap.add_argument("--strides", type=int, nargs="+", default=[16, 32, 48])
    ap.add_argument("--n", type=int, default=None, help="cap instances (smoke)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = KFDataset(str(setup.data_path(DATA_RE)), n_samples=HELDOUT[1] - HELDOUT[0],
                     offset=HELDOUT[0], sub_t=setup.SUB_T)
    dataset = full if args.n is None else torch.utils.data.Subset(full, range(min(args.n, len(full))))
    S, T = dataset[0]["y"].shape[0], dataset[0]["y"].shape[-1]
    dt = setup.T_INTERVAL / (T - 1)
    nlate = max(1, T // 8)
    kinf, n_bands = ev.cheb_bins(S, device), S // 2 + 1
    solver = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=device, dtype=torch.float64)
    f = kf_forcing(S, device, torch.float64)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"chaos-split  strides={args.strides}  heldout={HELDOUT} n={len(dataset)} "
          f"T={T} dt={dt:.5f} device={device}\n", flush=True)

    # control once (operator-independent): solver continuation of the TRUE field
    ctrl = run(None, dataset, solver, f, args.strides, T, dt, kinf, n_bands, nlate, device, want_ctrl=True)
    print("CONTROL  C_ctrl (solver from GT@t_r vs GT, should be small):")
    for tr in args.strides:
        print(f"  t_r={tr}: C_ctrl={ctrl[tr]['Cctrl']:.4f}", flush=True)

    summary = {"control": {tr: ctrl[tr]["Cctrl"] for tr in args.strides}}
    for op in args.ops:
        model = setup.load_model(CKPTS[op], device)
        res = run(model, dataset, solver, f, args.strides, T, dt, kinf, n_bands, nlate, device, want_ctrl=False)
        print(f"\n== {op} ==  (late k<=7, normalised by ||GT||)")
        print(f"  {'t_r':>4} {'A total':>9} {'B fixable':>10} {'C chaos':>9}")
        for tr in args.strides:
            r = res[tr]
            print(f"  {tr:>4} {r['A']:>9.4f} {r['B']:>10.4f} {r['C']:>9.4f}", flush=True)
        summary[op] = res
        _plot(op, res, ctrl, args.strides)
    (OUT / "chaos_split.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {OUT}", flush=True)


def _plot(op, res, ctrl, strides):
    x = np.arange(len(strides))
    A = [res[t]["A"] for t in strides]; B = [res[t]["B"] for t in strides]
    C = [res[t]["C"] for t in strides]; Cc = [ctrl[t]["Cctrl"] for t in strides]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, A, "o-", label="A total (pred vs GT)")
    ax.plot(x, B, "s-", label="B fixable (pred vs solver)")
    ax.plot(x, C, "^-", label="C chaos (solver vs GT)")
    ax.plot(x, Cc, "x--", color="gray", label="C_ctrl (GT-restart)")
    ax.set_xticks(x); ax.set_xticklabels([f"t_r={t}" for t in strides])
    ax.set_ylabel("late k≤7 rel-L2"); ax.set_title(f"{op} — chaos vs artifact split")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"chaos_split_{op}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
