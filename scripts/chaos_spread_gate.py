"""Gate 1 — is the late wall irreducible chaos spread, and does the operator under-disperse?

No training. Perturb the true IC by band-limited (k≤7) noise of relative size δ, then roll
TWO ensembles from the SAME perturbed ICs and compare their late k≤7 spread:

  σ_phys : TRUE solver (src.solver.periodic.NavierStokes2d) from each perturbed IC
           = irreducible physical chaos spread for IC uncertainty δ.
  σ_op   : the EXISTING operator from each perturbed IC
           = how much the operator's output varies with the same IC noise.

Readings (pooled, ‖GT‖-normalised; spread reported for BOTH early (post-IC) and late k≤7):
  - growth = σ_phys_late / σ_phys_early = DIRECT chaos amplification over the horizon. ~1 ⇒ weak/no
    sensitive-dependence (the wall is REPRESENTATIONAL); ≫1 and saturating in δ ⇒ chaotic wall.
    (σ_phys roughly linear in δ with growth≈1 ⇒ NOT chaos; this is the decisive number.)
  - σ_op ≪ σ_phys ⇒ the operator UNDER-disperses (blurs); cov_op = relL2(GT,ens-mean)/σ_op ≫1 ⇒
    GT outside the operator spread (overconfident) — the deterministic-emulator failure.
  - phys_mean ≪ op_single (wall) and σ_phys < wall ⇒ the wall is NOT chaos amplified from a small
    IC uncertainty (operator gets the exact IC; solver control = 0).

CONTROL: δ=0 solver from the true IC vs GT ⇒ ~0 (validates solver Re/forcing/dt/resolution).
OOD: run --data_re 300 (in-dist) vs 500 (OOD), each with its operator; σ_phys at fixed δ should
grow with Re (faster chaos) — the physics-level OOD signal. Operator/data Re must match.

Run: CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD python scripts/chaos_spread_gate.py \
        --data_re 500 --ckpt adapt_op100=<rundir>/adapted.ckpt --deltas 0.05 0.1 0.2
Self-test (toy, no solver/model): python scripts/chaos_spread_gate.py --selftest
"""
import argparse
import math

import numpy as np
import torch

from torch.utils.data import Subset

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from src.pde.ns import cheb_lowpass
from src.solver.periodic import NavierStokes2d
from msc.tta import setup, eval as ev

HELDOUT = (200, 300)
CKPTS = {"op100": "pretrain-kol/pvqq97sq/checkpoints/best.ckpt",
         "op300": "pretrain-kol/1iix0n42/checkpoints/best.ckpt",
         "op500": "pretrain-kol/38o0kj3y/checkpoints/best.ckpt"}


def kf_forcing(S, device, dtype):
    """f = -4cos(4Y), Y along axis 1 — identical to chaos_artifact_split / generate_kf."""
    t = torch.linspace(0, 2 * math.pi, S + 1, dtype=dtype, device=device)[:-1]
    _, Y = torch.meshgrid(t, t, indexing="ij")
    return -4.0 * torch.cos(4.0 * Y)


def band_noise(S, kmax, device, gen):
    n = torch.randn(S, S, generator=gen, device=device)
    return cheb_lowpass(n[None, :, :, None], kmax)[0, :, :, 0]


def perturb_ic(ic, delta, kmax, gen):
    """ic (S,S) -> ic + band-limited noise scaled to ‖noise‖/‖ic‖ = delta."""
    n = band_noise(ic.shape[-1], kmax, ic.device, gen)
    return ic + n / (n.norm() + 1e-30) * delta * ic.norm()


def solve_from_ic(solver, ic, f, T, dt, re, device):
    """ic (S,S) float64 -> (S,S,T) float32 true-solver trajectory, frames 0..T-1."""
    S = ic.shape[-1]
    out = torch.empty((S, S, T), device=device)
    out[:, :, 0] = ic.float()
    w = ic.unsqueeze(0)
    for fr in range(1, T):
        w = solver.advance(w, f, T=dt, Re=re, adaptive=True)
        out[:, :, fr] = w[0].float()
    return out


def pooled_rel(num: float, den: float) -> float:
    """sqrt(pooled error power / pooled GT power) — ‖GT‖-normalised relL2."""
    return float(np.sqrt(num / (den + 1e-30)))


def k7_win(field, kinf, n_bands, win) -> float:
    """k≤7 power over frame-window `win` (+ space) of (S,S,T) field."""
    p = ev.band_power_t(field[None], kinf, n_bands)[:ev.K_REP + 1]
    return float(p[:, win].sum())


def operator_traj(model, ic, T):
    """ic (S,S) -> operator one-shot (S,S,T)."""
    return kf_forward(model, ic[None].float(), T, time_scale=setup.TIME_SCALE,
                      temporal_pad=setup.TEMPORAL_PAD).squeeze(1).squeeze(0)


def run(model, dataset, solver, f, deltas, M, re, T, dt, kinf, n_bands, nlate, device):
    we, wl = slice(1, 1 + nlate), slice(T - nlate, T)           # early (post-IC) vs late window
    den = {"e": 0.0, "l": 0.0}
    sing = {"op": 0.0, "ctrl": 0.0}
    acc = {d: {"sph_e": 0.0, "sph_l": 0.0, "mph_e": 0.0, "mph_l": 0.0,
               "sop": 0.0, "mop": 0.0, "pm": 0.0} for d in deltas}
    for i in range(len(dataset)):
        gt = dataset[i]["y"].to(device)                         # (S,S,T)
        ic = gt[:, :, 0]
        den["e"] += k7_win(gt, kinf, n_bands, we)
        den["l"] += k7_win(gt, kinf, n_bands, wl)
        with torch.no_grad():
            pred = operator_traj(model, ic, T)
        sing["op"] += k7_win(pred - gt, kinf, n_bands, wl)
        sing["ctrl"] += k7_win(solve_from_ic(solver, ic.double(), f, T, dt, re, device) - gt,
                               kinf, n_bands, wl)
        gen = torch.Generator(device=device)
        for d in deltas:
            mph, mop = [], []
            for m in range(M):
                gen.manual_seed(i * 100003 + m * 97 + int(d * 1e4))
                pic = perturb_ic(ic, d, ev.K_REP, gen)
                mph.append(solve_from_ic(solver, pic.double(), f, T, dt, re, device))
                with torch.no_grad():
                    mop.append(operator_traj(model, pic, T))
            Sp, So = torch.stack(mph), torch.stack(mop)         # (M,S,S,T)
            bph, bop = Sp.mean(0), So.mean(0)
            acc[d]["sph_e"] += float(np.mean([k7_win(Sp[m] - bph, kinf, n_bands, we) for m in range(M)]))
            acc[d]["sph_l"] += float(np.mean([k7_win(Sp[m] - bph, kinf, n_bands, wl) for m in range(M)]))
            acc[d]["mph_e"] += k7_win(bph - gt, kinf, n_bands, we)
            acc[d]["mph_l"] += k7_win(bph - gt, kinf, n_bands, wl)
            acc[d]["sop"] += float(np.mean([k7_win(So[m] - bop, kinf, n_bands, wl) for m in range(M)]))
            acc[d]["mop"] += k7_win(bop - gt, kinf, n_bands, wl)
            acc[d]["pm"] += k7_win(pred - bph, kinf, n_bands, wl)
    out = {"op_single": pooled_rel(sing["op"], den["l"]), "ctrl": pooled_rel(sing["ctrl"], den["l"]),
           "deltas": {}}
    for d in deltas:
        a = acc[d]
        out["deltas"][d] = {"sph_e": pooled_rel(a["sph_e"], den["e"]),
                            "sph_l": pooled_rel(a["sph_l"], den["l"]),
                            "mph_e": pooled_rel(a["mph_e"], den["e"]),
                            "mph_l": pooled_rel(a["mph_l"], den["l"]),
                            "sop": pooled_rel(a["sop"], den["l"]),
                            "mop": pooled_rel(a["mop"], den["l"]),
                            "pm": pooled_rel(a["pm"], den["l"])}
    return out


def report(label, res):
    print(f"\n=== {label} ===  (late k≤7, ‖GT‖-normalised)")
    print(f"  control δ=0 (solver from true IC vs GT): {res['ctrl']:.4f}  [should be ~0]")
    print(f"  op_single (deterministic operator wall):  {res['op_single']:.4f}")
    print(f"  {'δ':>6} {'σph_e':>7} {'σph_l':>7} {'growth':>7} {'σ_op':>7} {'op_mean':>8} "
          f"{'pred-Em':>8} {'cov_op':>7}")
    for d, m in res["deltas"].items():
        growth = m["sph_l"] / (m["sph_e"] + 1e-30)
        cop = m["mop"] / (m["sop"] + 1e-30)
        print(f"  {d:>6.3f} {m['sph_e']:>7.4f} {m['sph_l']:>7.4f} {growth:>7.2f} {m['sop']:>7.4f} "
              f"{m['mop']:>8.4f} {m['pm']:>8.4f} {cop:>7.2f}")
    print("  growth=σph_l/σph_e = direct chaos amplification over the horizon: ~1 ⇒ weak/no sensitive-")
    print("    dependence (REPRESENTATIONAL wall); ≫1 (and saturating in δ) ⇒ chaotic wall.")
    print("  σ_op≪σ_phys ⇒ IC-perturbation ensemble under-disperses; cov_op≫1 ⇒ GT outside spread.")
    print("  pred-Em=relL2(pred,phys_mean); op_single (wall) above. phys_mean≪wall + σ_phys<wall ⇒ not chaos-from-IC.")


def main():
    ap = argparse.ArgumentParser(description="Gate 1 — chaos spread vs operator dispersion")
    ap.add_argument("--data_re", type=int, default=500)
    ap.add_argument("--ops", nargs="+", default=None, help="keys in CKPTS (op100/op300/op500)")
    ap.add_argument("--ckpt", nargs="+", default=None, help="label=path (overrides --ops)")
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--M", type=int, default=8, help="ensemble members")
    ap.add_argument("--n", type=int, default=20, help="held-out instances")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return

    ckpts: dict[str, str] = {}
    if args.ckpt:
        for it in args.ckpt:
            k, v = it.split("=", 1)
            ckpts[k] = v
    else:
        ckpts = {op: CKPTS[op] for op in (args.ops or ["op500"])}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = KFDataset(str(setup.data_path(args.data_re)), n_samples=HELDOUT[1] - HELDOUT[0],
                     offset=HELDOUT[0], sub_t=setup.SUB_T)
    ds = Subset(full, range(min(args.n, len(full))))
    S, T = ds[0]["y"].shape[0], ds[0]["y"].shape[-1]
    dt = setup.T_INTERVAL / (T - 1)
    nlate = max(1, T // 8)
    kinf, n_bands = ev.cheb_bins(S, device), S // 2 + 1
    solver = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=device, dtype=torch.float64)
    f = kf_forcing(S, device, torch.float64)
    print(f"chaos-spread gate  data_re={args.data_re} n={len(ds)} M={args.M} deltas={args.deltas} "
          f"T={T} dt={dt:.5f} device={device}", flush=True)
    for label, ck in ckpts.items():
        res = run(setup.load_model(ck, device), ds, solver, f, args.deltas, args.M, args.data_re,
                  T, dt, kinf, n_bands, nlate, device)
        report(f"{label} @ Re{args.data_re}", res)


# ---------------------------------------------------------------------------- tests
def selftest():
    S = 16
    dev = torch.device("cpu")
    kinf, n_bands = ev.cheb_bins(S, dev), S // 2 + 1
    w = slice(-2, None)                                          # window for the toy checks
    gen = torch.Generator(device=dev)

    gen.manual_seed(0)
    ic = torch.randn(S, S)
    pic = perturb_ic(ic, 0.1, ev.K_REP, gen)
    assert abs((pic - ic).norm() / ic.norm() - 0.1) < 1e-5, "perturb relative magnitude"

    hi = (pic - ic) - cheb_lowpass((pic - ic)[None, :, :, None], ev.K_REP)[0, :, :, 0]
    assert hi.norm() / (pic - ic).norm() < 1e-5, "perturbation is band-limited k≤7"

    zero = torch.zeros(S, S, 6)
    assert k7_win(zero, kinf, n_bands, w) == 0.0, "k7_win of zero is 0"
    g = cheb_lowpass(torch.randn(1, S, S, 6), ev.K_REP)[0]
    assert k7_win(g, kinf, n_bands, w) > 0, "k7_win of signal positive"
    assert k7_win(g, kinf, n_bands, slice(0, 3)) != k7_win(g, kinf, n_bands, slice(3, 6)), \
        "k7_win is window-selective (early ≠ late)"

    a = cheb_lowpass(torch.randn(1, S, S, 6), ev.K_REP)[0]
    members = torch.stack([g + a, g - a])                       # mean = g, dev = ±a
    bmean = members.mean(0)
    assert (bmean - g).norm() < 1e-5, "ensemble mean recovers center"
    sp = np.mean([k7_win(members[m] - bmean, kinf, n_bands, w) for m in range(2)])
    ka = k7_win(a, kinf, n_bands, w)
    assert abs(sp - ka) < 1e-5 * ka, "spread = ±a deviation power"

    asy = torch.stack([g + a, g + 2 * a])                       # mean = g+1.5a, dev = ±0.5a
    bmean_asy = asy.mean(0)
    sp_asy = np.mean([k7_win(asy[m] - bmean_asy, kinf, n_bands, w) for m in range(2)])
    assert abs(sp_asy - k7_win(0.5 * a, kinf, n_bands, w)) < 1e-5 * sp_asy, "asymmetric spread"

    assert abs(k7_win(2 * a, kinf, n_bands, w) - 4 * ka) < 1e-4 * ka, "k7_win is power (∝ amp²)"

    same = torch.stack([g, g, g])                               # δ=0 analogue
    bm = same.mean(0)
    zero_sp = np.mean([k7_win(same[m] - bm, kinf, n_bands, w) for m in range(3)])
    assert zero_sp < 1e-6 * k7_win(g, kinf, n_bands, w), "identical members -> zero spread"

    cov = pooled_rel(4.0, 1.0) / pooled_rel(1.0, 1.0)           # = sqrt(4/1)/sqrt(1/1) = 2
    assert abs(cov - 2.0) < 1e-9, "coverage = relL2(GT,mean)/σ (ratio of sqrt-power, not power)"

    print("selftest OK (perturb, k7_win power-scaling+windowing, sym+asym spread, zero-spread, coverage ratio)")


if __name__ == "__main__":
    main()
