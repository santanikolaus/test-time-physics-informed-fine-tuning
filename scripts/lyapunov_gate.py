"""Lyapunov gate — maximal Lyapunov exponent λ(Re) for forced 2D KF (Nastac Alg. 1, PRF 2.094606).

Perturb a held-out IC by ε‖ic‖ in a unit direction, roll BOTH the reference and the perturbed IC
through the TRUE solver with a FIXED frozen substep (adaptive=False ⇒ identical, field-independent
schedule — no numerical drift masquerading as chaos), and fit λ = slope of ln‖δ(t)‖ over the linear
region. The region is the max-R² window subject to a minimum-e-fold dynamic-range gate; if no window
clears the gate the separation never left the transient ⇒ report "no resolvable linear region ⇒
t_p ≥ window" rather than a manufactured slope.

  t_p = 1/λ   (predictability time, one e-fold)      N_cap = t_p / dt_record   (in 128-step units)

Two clocks: full-field ‖δ‖ → canonical maximal λ; k≤7 ‖δ‖ → band error-growth rate (NOT a Lyapunov
exponent — nonlinear coupling feeds high-k and the restricted norm converges to global λ_max). The
ε-sweep verifies the tangent regime: λ is ε-invariant; smaller ε buys a longer linear region.

Resolutions believed energy-resolved: Re100@128², Re500@256². Pass --res to compare.

Determinism: ε=0 ⇒ identical trajectories, zero separation, fit → None.
Integration check (run BEFORE trusting λ): the frozen substep is set from the IC CFL, capped at
dt_record. Rerun with --dt_factor 0.5; if λ shifts, the base trajectory is under-integrated — halve
until λ stops moving. The ε-sweep guards the perturbation regime, NOT the integration regime. Do not
revert to adaptive: that reintroduces schedule-dependent numerical drift between the two trajectories.

Sanity gate (Nastac): t_p brackets between the smallest-scale time and the eddy-turnover time.

Run: PYTHONPATH=$PWD python scripts/lyapunov_gate.py --data_re 100 --res 128 --eps 1e-3 1e-5 1e-7
     PYTHONPATH=$PWD python scripts/lyapunov_gate.py --data_re 500 --res 256 --eps 1e-3 1e-5 1e-7
Self-test (no solver/data): python scripts/lyapunov_gate.py --selftest
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from torch.utils.data import Subset

from src.datasets.kf_dataset import KFDataset
from src.pde.ns import cheb_lowpass
from src.solver.periodic import NavierStokes2d
from msc.tta import setup, eval as ev


def kf_forcing(S, device, dtype):
    """f = -4cos(4Y), Y along axis 1 — identical to chaos_spread_gate / generate_kf."""
    t = torch.linspace(0, 2 * math.pi, S + 1, dtype=dtype, device=device)[:-1]
    _, Y = torch.meshgrid(t, t, indexing="ij")
    return -4.0 * torch.cos(4.0 * Y)


def eddy_time(ic):
    """Enstrophy-based eddy-turnover time τ ≈ 1/ω_rms for a vorticity field (S,S) — the clock that
    anchors T and N_cap in turnover units (Nastac sanity gate: t_p brackets τ_η..τ_eddy)."""
    return 1.0 / (float((ic ** 2).mean().sqrt()) + 1e-30)


def unit_dir(S, kmax, banded, device, gen, dtype):
    """Unit-L2 perturbation direction; banded ⇒ restricted to k≤kmax (else white/full-spectrum)."""
    d = torch.randn(S, S, generator=gen, device=device, dtype=dtype)
    if banded:
        d = cheb_lowpass(d[None, :, :, None], kmax)[0, :, :, 0]
    return d / (d.norm() + 1e-30)


def perturb(ic, eps, d):
    """ic (S,S) + ε‖ic‖·d̂  ⇒  ‖perturb-ic‖/‖ic‖ = ε exactly (d unit-norm)."""
    return ic + eps * ic.norm() * d


def band_norm(field, kmax):
    """L2 norm of the k≤kmax content of a (S,S) field."""
    return float(cheb_lowpass(field[None, :, :, None], kmax)[0, :, :, 0].norm())


def frozen_dt(solver, ic, f, Re, dt_record):
    """CFL step from the reference IC, capped at dt_record, then FROZEN — used for both trajectories."""
    w_h = torch.fft.rfft2(ic.unsqueeze(0))
    q, v = solver.velocity_field(solver.stream_function(w_h, real_space=False), real_space=True)
    return min(solver.time_step(q, v, f, Re), dt_record)


def separation(solver, ic_ref, ic_pert, f, n_frames, dt_record, Re, dt_sub, kmax, roll_mult=1):
    """Roll ref+pert in lockstep (one fixed delta_t) ⇒ per-frame full-field and k≤7 ‖δ(t)‖.

    roll_mult>1 rolls past the data window (Nastac rolls to SATURATION to expose the linear
    region between initial-response and plateau; forced KF is stationary so this is valid).
    Returns (full, band): arrays length (n_frames-1)*roll_mult+1, full[0]=‖ic_pert-ic_ref‖.
    """
    n_eff = (n_frames - 1) * roll_mult + 1
    w = torch.stack([ic_ref, ic_pert])
    full = np.empty(n_eff)
    band = np.empty(n_eff)
    d0 = ic_pert - ic_ref
    full[0], band[0] = float(d0.norm()), band_norm(d0, kmax)
    for fr in range(1, n_eff):
        w = solver.advance(w, f, T=dt_record, Re=Re, adaptive=False, delta_t=dt_sub)
        d = w[1] - w[0]
        full[fr], band[fr] = float(d.norm()), band_norm(d, kmax)
    return full, band


def fit_lyapunov(t, ln_sep, min_efolds, min_pts):
    """λ = slope of ln_sep vs t over the max-R² window with ≥min_efolds rise and ≥min_pts points.

    ln_sep is the NATURAL log of the separation ⇒ slope is λ directly (no log10→ln factor).
    Returns dict(lmbda, r2, i, j, efolds) or None if no window clears the dynamic-range gate.
    """
    n = len(t)
    best = None
    for i in range(n):
        for j in range(i + min_pts - 1, n):
            rise = ln_sep[j] - ln_sep[i]
            if rise < min_efolds:
                continue
            tt, yy = t[i:j + 1], ln_sep[i:j + 1]
            slope, intercept = np.polyfit(tt, yy, 1)
            if slope <= 0:
                continue
            pred = slope * tt + intercept
            ss_res = float(np.sum((yy - pred) ** 2))
            ss_tot = float(np.sum((yy - yy.mean()) ** 2)) + 1e-30
            r2 = 1.0 - ss_res / ss_tot
            if best is None or r2 > best["r2"]:
                best = {"lmbda": float(slope), "r2": r2, "i": i, "j": j, "efolds": float(rise)}
    return best


def measure(solver, ds, f, Re, eps_list, n_dirs, n_frames, dt_record, kmax,
            banded, min_efolds, min_pts, dt_factor, roll_mult, device, dtype):
    """Per-ε aggregate λ over instances×directions for one clock (full or k≤7)."""
    n_eff = (n_frames - 1) * roll_mult + 1
    t = np.arange(n_eff) * dt_record
    stride = max(1, n_eff // 200)
    tf = t[::stride]
    out = {}
    for eps in eps_list:
        lam_full, lam_band, n_fit, n_total = [], [], 0, 0
        amp_f, amp_b = [], []
        for i in range(len(ds)):
            ic = ds[i]["y"][:, :, 0].to(device).to(dtype)
            dt_sub = frozen_dt(solver, ic, f, Re, dt_record) * dt_factor
            gen = torch.Generator(device=device)
            for m in range(n_dirs):
                gen.manual_seed(i * 100003 + m * 97 + int(eps * 1e9))
                d = unit_dir(ic.shape[-1], kmax, banded, device, gen, dtype)
                full, band = separation(solver, ic, perturb(ic, eps, d), f,
                                         n_frames, dt_record, Re, dt_sub, kmax, roll_mult)
                n_total += 1
                amp_f.append(amplification(full[:n_frames]))
                amp_b.append(amplification(band[:n_frames]))
                ff = fit_lyapunov(tf, np.log(full[::stride] + 1e-300), min_efolds, min_pts)
                bf = fit_lyapunov(tf, np.log(band[::stride] + 1e-300), min_efolds, min_pts)
                if ff is not None:
                    lam_full.append(ff["lmbda"]); n_fit += 1
                if bf is not None:
                    lam_band.append(bf["lmbda"])
        out[eps] = {"full": _stat(lam_full), "band": _stat(lam_band),
                    "amp_full": _amp_stat(amp_f), "amp_band": _amp_stat(amp_b),
                    "n_fit": n_fit, "n_total": n_total}
    return out, (n_frames - 1) * dt_record


def amplification(curve):
    """(end, peak) separation amplification vs t=0: ‖δ(T)‖/‖δ(0)‖ and max‖δ‖/‖δ(0)‖.

    Gate-free — quantifies the null when no linear region clears the e-fold gate. end<e ⇒ the
    perturbation has not grown one e-fold over the window ⇒ t_p > T empirically (Nastac Eq. 6).
    """
    d0 = float(curve[0]) + 1e-300
    return float(curve[-1]) / d0, float(curve.max()) / d0


def _stat(xs):
    if not xs:
        return None
    a = np.asarray(xs)
    return {"mean": float(a.mean()), "std": float(a.std()), "n": len(a)}


def _amp_stat(pairs):
    a = np.asarray(pairs)
    return {"end": float(np.median(a[:, 0])), "peak": float(np.median(a[:, 1])), "n": len(a)}


def report(label, res, horizon):
    print(f"\n=== {label} ===  horizon T={horizon:.4f}  (λ in 1/time; t_p=1/λ; N_cap=t_p/dt)")
    dt = horizon / 128.0
    for eps, r in res.items():
        print(f"  ε={eps:.0e}  fits {r['n_fit']}/{r['n_total']}")
        for clock in ("full", "band"):
            s = r[clock]
            tag = "λ_full(max) " if clock == "full" else "g_k≤7(band)"
            if s is None:
                print(f"    {tag}: no resolvable linear region ⇒ t_p ≥ horizon")
                continue
            lam = s["mean"]
            tp = 1.0 / lam if lam > 0 else float("inf")
            print(f"    {tag}: {lam:.4f} ± {s['std']:.4f} (n={s['n']})  "
                  f"t_p={tp:.3f}  N_cap={tp / dt:.1f}  λ·T={lam * horizon:.3f}")
        for clock, key in (("full", "amp_full"), ("k≤7", "amp_band")):
            a = r[key]
            ef = math.log(a["end"]) if a["end"] > 0 else float("-inf")
            ncap = 128.0 / ef if ef > 0 else float("inf")
            bound = "t_p ≥ T (chaos beyond window)" if ef < 1.0 else "t_p < T (chaos bites in-window)"
            print(f"    amp {clock:>4}: A_end={a['end']:.3f} (e-folds={ef:.3f}) A_peak={a['peak']:.3f}"
                  f"  ⇒ {bound}; extrap N_cap≈{ncap:.0f}")
    print("  e-folds(A_end)<1 ⇒ horizon beyond window EMPIRICALLY (Nastac Eq.6, no extrapolation).")
    print("  extrap N_cap=128/e-folds assumes exponential continuation at the window-mean rate —")
    print("  likely OPTIMISTIC if the window is still pre-exponential (true λ from a later linear")
    print("  region would be larger ⇒ shorter horizon). A_peak≫A_end ⇒ growth-then-saturation.")


def resolve_data(data_re, res):
    paths = yaml.safe_load((setup.ROOT / "documentation" / "paths.yaml").read_text())["data"]
    if res == 128:
        return setup.data_path(data_re)
    return Path(paths[f"ns_re{data_re}_res{res}"])


def main():
    ap = argparse.ArgumentParser(description="Lyapunov gate — λ(Re) for forced 2D KF (Nastac Alg.1)")
    ap.add_argument("--data_re", type=int, default=100)
    ap.add_argument("--res", type=int, default=128, choices=[128, 256, 512])
    ap.add_argument("--eps", nargs="+", type=float, default=[1e-3, 1e-5, 1e-7])
    ap.add_argument("--n", type=int, default=8, help="held-out instances")
    ap.add_argument("--dirs", type=int, default=2, help="perturbation directions per instance")
    ap.add_argument("--spectrum", choices=["white", "band"], default="white",
                    help="white = full-spectrum (maximal λ); band = k≤kmax direction")
    ap.add_argument("--kmax", type=int, default=ev.K_REP)
    ap.add_argument("--min_efolds", type=float, default=1.0, help="dynamic-range gate on the fit")
    ap.add_argument("--min_pts", type=int, default=5)
    ap.add_argument("--dt_factor", type=float, default=1.0,
                    help="scale the frozen substep; rerun at 0.5 to check timestep convergence of λ")
    ap.add_argument("--roll_mult", type=int, default=1,
                    help="roll past the data window by this factor to expose the linear region (Nastac)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    path = resolve_data(args.data_re, args.res)
    full = KFDataset(str(path), n_samples=setup.N_TEST, offset=setup.OFFSET_TEST, sub_t=1)
    ds = Subset(full, range(min(args.n, len(full))))
    S, T = ds[0]["y"].shape[0], ds[0]["y"].shape[-1]
    dt_record = setup.T_INTERVAL / (T - 1)
    tau = float(np.median([eddy_time(ds[i]["y"][:, :, 0]) for i in range(len(ds))]))
    solver = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=device, dtype=dtype)
    f = kf_forcing(S, device, dtype)
    print(f"lyapunov gate  data_re={args.data_re} res={S} n={len(ds)} dirs={args.dirs} "
          f"spectrum={args.spectrum} kmax={args.kmax} frames={T} roll_mult={args.roll_mult} "
          f"dt_record={dt_record:.5f} eps={args.eps} device={device}\n"
          f"  units: τ_eddy≈1/ω_rms={tau:.4f}  ⇒  data window T={setup.T_INTERVAL} = "
          f"{setup.T_INTERVAL / tau:.2f} eddy-turnovers; roll spans {args.roll_mult * setup.T_INTERVAL / tau:.2f}",
          flush=True)
    res, horizon = measure(solver, ds, f, args.data_re, args.eps, args.dirs, T, dt_record,
                           args.kmax, banded=(args.spectrum == "band"),
                           min_efolds=args.min_efolds, min_pts=args.min_pts,
                           dt_factor=args.dt_factor, roll_mult=args.roll_mult, device=device, dtype=dtype)
    report(f"Re{args.data_re}@{S}² ({args.spectrum})", res, horizon)


# ---------------------------------------------------------------------------- tests
def selftest():
    """Minimal pins; the comprehensive toy battery lives in tests/tta/test_lyapunov_gate.py."""
    S = 16

    ic = torch.randn(S, S, dtype=torch.float64)
    d = torch.randn(S, S, dtype=torch.float64)
    d = d / d.norm()
    p = perturb(ic, 0.1, d)
    assert abs((p - ic).norm() / ic.norm() - 0.1) < 1e-9, "perturb relative magnitude = ε"

    gen = torch.Generator()
    db = unit_dir(S, ev.K_REP, True, torch.device("cpu"), gen, torch.float64)
    assert abs(float(db.norm()) - 1.0) < 1e-9, "unit_dir is unit-norm"
    hi = db - cheb_lowpass(db[None, :, :, None], ev.K_REP)[0, :, :, 0]
    assert hi.norm() / db.norm() < 1e-6, "banded direction is k≤7"

    pure_hi = torch.randn(S, S, dtype=torch.float64)
    pure_hi = pure_hi - cheb_lowpass(pure_hi[None, :, :, None], ev.K_REP)[0, :, :, 0]
    assert band_norm(pure_hi, ev.K_REP) / pure_hi.norm() < 1e-6, "band_norm ≈ 0 for pure high-k"

    t = np.arange(40) * 0.02
    lam_true = 1.7
    sep = np.exp(lam_true * t)
    fit = fit_lyapunov(t, np.log(sep), min_efolds=1.0, min_pts=5)
    assert fit is not None and abs(fit["lmbda"] - lam_true) < 1e-6, "recover λ from ln-linear curve"
    bad = fit_lyapunov(t, np.log10(sep), min_efolds=0.4, min_pts=5)
    assert bad is not None and abs(bad["lmbda"] - lam_true / math.log(10)) < 1e-6, \
        "log10 input ⇒ slope is λ/ln10 (pins the natural-log convention at the call site)"

    flat = np.full(40, 3.0) + 1e-6 * np.random.randn(40)
    assert fit_lyapunov(t, flat, min_efolds=1.0, min_pts=5) is None, \
        "flat/transient-only curve ⇒ no region clears the dynamic-range gate"

    trans = np.full(15, -8.0)
    lin = -8.0 + lam_true * (np.arange(30) * 0.02)
    sat = np.full(10, lin[-1])
    three = np.concatenate([trans, lin, sat])
    tt = np.arange(len(three)) * 0.02
    f3 = fit_lyapunov(tt, three, min_efolds=0.5, min_pts=5)
    assert f3 is not None and abs(f3["lmbda"] - lam_true) < 0.2, \
        "three-region curve ⇒ recovers the linear-region slope, not transient/saturation"

    assert abs(eddy_time(torch.full((8, 8), 2.0, dtype=torch.float64)) - 0.5) < 1e-9, \
        "eddy_time = 1/ω_rms (ω_rms=2 ⇒ τ=0.5)"

    end, peak = amplification(np.array([1.0, 8.0, 4.0]))
    assert end == 4.0 and peak == 8.0, "amplification = (last/first, max/first)"
    st = _amp_stat([(4.0, 8.0), (2.0, 4.0)])
    assert st["end"] == 3.0 and st["peak"] == 6.0 and st["n"] == 2, "amp stat = medians"
    assert abs(128.0 / math.log(amplification(np.array([1.0, math.e]))[0]) - 128.0) < 1e-9, \
        "A_end=e ⇒ one e-fold ⇒ extrap N_cap=128 (boundary t_p=T)"

    print("selftest OK (perturb ε, unit/banded dir, band_norm, λ-recovery, ln-vs-log10, "
          "gate-rejects-flat, three-region, amplification+N_cap)")


if __name__ == "__main__":
    main()
