"""Step 2b — dose-response gate (per-instance TTIA, the first lever that touches weights).

Steerability test for the self-consistency signal validated by self_consistency_gate.py.
Per held-out instance: clone the frozen op, take K Adam steps that LOWER self-inconsistency
at the m restart (restart F(u[m]) detached = teacher, optimise the one-shot tail toward it;
pure lever, NO physics / NO ic-anchor), and track GT error.

  loss          = relL2_{k<=7}( F(ic)[m:] , stopgrad F(F(ic)[m]) )   over restart frames m..T-1
  si_late       = relL2_{k<=7,late}( F(F(ic)[m]) , F(ic)[m:] )       (the validated signal)
  gt_late/early = relL2_{k<=7, late/early}( F(ic) , GT )             (truth + forgetting guard)
  restart_gt_l  = relL2_{k<=7,late}( F(F(ic)[m]) , GT[m:] )          (teacher quality)

Read (pre-registered): si_late DOWN & gt_late DOWN, correlated across instances (Spearman
Δsi vs Δgt_late > 0) -> STEERABLE lever -> green-light training. si_late DOWN but gt_late
flat/UP -> chaos co-variate / trivial collapse -> self-consistency direction dies here.

Run: CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD python scripts/self_consistency_dose.py --ops op100 op500
"""
import argparse
import copy
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from src.pde.ns import cheb_lowpass
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
    num_t = band_power_t(num_field, kinf, n_bands)[:K_REP + 1].sum(0)
    den_t = band_power_t(den_field, kinf, n_bands)[:K_REP + 1].sum(0)
    return float(np.sqrt(num_t[-nE:].sum() / (den_t[-nE:].sum() + EPS)))


def split_k7(pred, gt, kinf, n_bands, nE):
    ep = band_power_t(pred - gt, kinf, n_bands)[:K_REP + 1]
    gp = band_power_t(gt, kinf, n_bands)[:K_REP + 1]
    err_t = np.sqrt(ep.sum(0) / (gp.sum(0) + EPS))
    return float(err_t[1:1 + nE].mean()), float(err_t[-nE:].mean())


def self_incons_loss(model, ic, m, T, ic_weight):
    """consist + ic_weight*ic_anchor. consist = differentiable k<=7 relL2 between one-shot
    tail and the DETACHED restart teacher; ic_anchor pins frame-0 to the (label-free) IC to
    forbid trivial collapse. NO physics term -> any GT-late move is attributable to consist.
    Teacher forwards (no_grad) run + free BEFORE the grad forward -> peak = one fwd/bwd."""
    with torch.no_grad():                                           # teacher: no graph, freed before grad forward
        u_const = _kf(model, ic, T)
        tl = cheb_lowpass(_kf(model, u_const[:, :, :, m], T)[..., 0:T - m], K_REP)
    u = _kf(model, ic, T)                                           # grad branch (the only retained graph)
    pl = cheb_lowpass(u[..., m:T], K_REP)
    consist = ((pl - tl) ** 2).sum() / (tl ** 2).sum().clamp_min(EPS)
    ic_anchor = ((u[..., 0] - ic) ** 2).sum() / (ic ** 2).sum().clamp_min(EPS)
    return consist + ic_weight * ic_anchor                          # squared relL2 terms (monotone)


@torch.no_grad()
def measure(model, ic, gt, m, T, kinf, n_bands, nE):
    u = _kf(model, ic, T)
    ge, gl = split_k7(u, gt, kinf, n_bands, nE)
    v = _kf(model, u[:, :, :, m], T)
    seg, tail = v[..., 0:T - m], u[..., m:T]
    si = _late_relL2(seg - tail, tail, kinf, n_bands, nE)
    rgl = _late_relL2(seg - gt[..., m:T], gt[..., m:T], kinf, n_bands, nE)
    return {"gt_early": ge, "gt_late": gl, "si_late": si, "restart_gt_late": rgl}


def run_op(base, dataset, device, m, steps, lr, ic_weight, measure_every) -> dict:
    S, T = dataset[0]["y"].shape[0], dataset[0]["y"].shape[-1]
    n_bands, kinf, nE = S // 2 + 1, cheb_bins(S, device), max(1, T // 8)
    N = len(dataset)
    keys = ("gt_early", "gt_late", "si_late", "restart_gt_late")
    rec0 = {k: np.zeros(N) for k in keys}
    recF = {k: np.zeros(N) for k in keys}
    sched = sorted(set([0] + list(range(measure_every, steps, measure_every)) + [steps]))
    csum = {s: np.zeros(3) for s in sched}                          # mean (si_late, gt_late, gt_early)

    for i in range(N):
        ic = dataset[i]["x"].unsqueeze(0).to(device)
        gt = dataset[i]["y"].unsqueeze(0).to(device)
        model = setup.enable_gradient_checkpointing(copy.deepcopy(base))
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        ms = measure(model, ic, gt, m, T, kinf, n_bands, nE)
        for k in keys:
            rec0[k][i] = ms[k]
        csum[0] += [ms["si_late"], ms["gt_late"], ms["gt_early"]]
        for s in range(steps):
            opt.zero_grad()
            self_incons_loss(model, ic, m, T, ic_weight).backward()
            opt.step()
            if (s + 1) in sched:
                ms = measure(model, ic, gt, m, T, kinf, n_bands, nE)
                csum[s + 1] += [ms["si_late"], ms["gt_late"], ms["gt_early"]]
        for k in keys:                                              # ms holds the final scheduled measure
            recF[k][i] = ms[k]
    curve = [[float(s)] + [float(x) for x in csum[s] / N] for s in sched]

    d_si = recF["si_late"] - rec0["si_late"]
    d_gl = recF["gt_late"] - rec0["gt_late"]
    moved = d_si < 0                                                # instances where the lever pushed
    rho, p = spearmanr(d_si, d_gl)
    try:
        w_p = float(wilcoxon(recF["gt_late"], rec0["gt_late"]).pvalue)
    except ValueError:
        w_p = float("nan")
    return {
        "m": m, "steps": steps, "lr": lr, "ic_weight": ic_weight, "n": N, "nE": nE,
        "si_late": [float(rec0["si_late"].mean()), float(recF["si_late"].mean())],
        "gt_late": [float(rec0["gt_late"].mean()), float(recF["gt_late"].mean())],
        "gt_early": [float(rec0["gt_early"].mean()), float(recF["gt_early"].mean())],
        "restart_gt_late0": float(rec0["restart_gt_late"].mean()),
        "dose_spearman": float(rho), "dose_p": float(p),
        "gt_late_wilcoxon_p": w_p,
        "frac_si_down": float(moved.mean()),
        "frac_gtlate_down_given_si_down": float((d_gl[moved] < 0).mean()) if moved.any() else float("nan"),
        "mean_d_gtlate": float(d_gl.mean()),
        "mean_d_gtlate_given_si_down": float(d_gl[moved].mean()) if moved.any() else float("nan"),
        "curve": curve,
        "d_si": d_si.tolist(), "d_gl": d_gl.tolist(),
    }


def _plot(op, r):
    c = np.array(r["curve"])                                        # cols: step, si_late, gt_late, gt_early
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(c[:, 0], c[:, 1], "o-", label="si_late (objective)")
    ax[0].plot(c[:, 0], c[:, 2], "o-", label="gt_late (truth)")
    ax[0].plot(c[:, 0], c[:, 3], "o-", label="gt_early (guard)")
    ax[0].set_title(f"{op}  m={r['m']} lr={r['lr']}  dose curve (mean n={r['n']})")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("rel-L2 k<=7"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].scatter(r["d_si"], r["d_gl"], s=16, alpha=0.6)
    ax[1].axhline(0, color="k", lw=0.5); ax[1].axvline(0, color="k", lw=0.5)
    ax[1].set_title(f"dose-response  Spearman {r['dose_spearman']:.3f} (p={r['dose_p']:.1e})")
    ax[1].set_xlabel("Δ si_late"); ax[1].set_ylabel("Δ gt_late"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / f"dose_{op}.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Self-consistency dose-response gate")
    ap.add_argument("--ops", nargs="+", default=["op100", "op500"])
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ic_weight", type=float, default=5.0)
    ap.add_argument("--measure_every", type=int, default=10)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = KFDataset(str(setup.data_path(DATA_RE)), n_samples=HELDOUT[1] - HELDOUT[0],
                     offset=HELDOUT[0], sub_t=setup.SUB_T)
    dataset = torch.utils.data.Subset(full, range(min(args.n, len(full))))
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"dose-response GATE (transductive, consist+{args.ic_weight}*ic, NO physics)  m={args.m} "
          f"steps={args.steps} lr={args.lr} n={len(dataset)} heldout={HELDOUT}  device={device}\n", flush=True)
    for op in args.ops:
        base = setup.load_model(CKPTS[op], device)
        r = run_op(base, dataset, device, args.m, args.steps, args.lr, args.ic_weight, args.measure_every)
        _plot(op, r)
        (OUT / f"dose_{op}.json").write_text(json.dumps(r, indent=2))
        print(f"== {op} ==  (n={r['n']}, m={r['m']}, {r['steps']} steps, lr={r['lr']})")
        print(f"  si_late  {r['si_late'][0]:.3f} -> {r['si_late'][1]:.3f}   (objective; frac down {r['frac_si_down']:.2f})")
        print(f"  gt_late  {r['gt_late'][0]:.3f} -> {r['gt_late'][1]:.3f}   (Δ̄ {r['mean_d_gtlate']:+.3f}, wilcoxon p={r['gt_late_wilcoxon_p']:.1e})")
        print(f"  gt_early {r['gt_early'][0]:.3f} -> {r['gt_early'][1]:.3f}  (collapse/forgetting guard)")
        print(f"  restart_gt_late@0 {r['restart_gt_late0']:.3f} vs gt_late@0 {r['gt_late'][0]:.3f}  (teacher better if <)")
        print(f"  DOSE  Spearman(Δsi, Δgt_late) {r['dose_spearman']:+.3f} (p={r['dose_p']:.1e}) | "
              f"of si-down: {r['frac_gtlate_down_given_si_down']:.2f} also gt_late-down, "
              f"Δ̄gt_late {r['mean_d_gtlate_given_si_down']:+.3f}\n", flush=True)
    print("read: gt_late DOWN with si_late DOWN, positive dose Spearman, gt_early stable -> "
          "STEERABLE. gt_late flat/UP or only via gt_early blowup -> chaos co-variate / collapse.")
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
