"""Gate 2 — phase-spread test. Does the late k≤7 PHASE intrinsically decorrelate?

Phase analog of Gate 1 (chaos_spread_gate, which is POWER-based ⇒ phase-blind).
Phase is 54–66% of the late k≤7 wall (amp/phase decomp), and whether THAT phase is
intrinsically IC-sensitive is the migrate-fork decider — untested until here.

Per held-out IC: M perturbed copies w0 + δ·(k≤7 band-limited noise), δ bracketing the
operator's OWN early error (~0.1–0.3 rel). Roll the TRUE solver for all. Per spatial
Fourier mode per frame, ensemble circular dispersion D(k,t) = 1 − |⟨e^{iφ_m}⟩_M|,
φ = angle(ŵ(k)); energy-weighted (by the unperturbed solve's |ŵ_ref|²) over k≤7 modes,
windowed early (frames 1..8) / late (last T/8). Range 0 (phase-locked) → 1 (decorrelated).

  READ WITH chain_gate (oracle re-anchor beats one-shot 100% @256², 0.528→0.313). Gate 2
  bounds ONLY the one-shot-from-IC map (full-horizon IC→late phase), NOT re-anchoring methods:
    D_late LOW  ⇒ one-shot SHOULD predict late-from-IC ⇒ the 0.55 wall is an optimization /
      objective failure ⇒ fix the loss IN PLACE, NO migration. (Clean, decisive.)
    D_late → 1 (fast, small δ) ⇒ one-shot-from-IC irreducibly walled ⇒ MUST re-anchor ⇒ points
      TOWARD self-consistency / autoregressive (NOT consolidate); D_late then CALIBRATES how good
      the re-anchor fields must be (the self-consistency-training bar); oracle−model gap ~0.22 = prize.
  (Full-horizon D bounds one-shot only; the re-anchor ceiling needs a stride-window D — follow-up.)
  VALID ONLY IF control ≪ wall (solver chaos == data chaos).

256² by default: the true solver is faithful there (128² under-resolves — the reason
Path B moved to 256²), and 256² is the wall we actually care about. control = k≤7
relL2(unperturbed solve − GT) flags solver faithfulness; the phase metric is
self-referential (ensemble spread + solver's own |ŵ_ref|²), independent of solver-GT match.

Reuses chaos_spread_gate.{kf_forcing,perturb_ic,solve_from_ic}. NEW: phase_dispersion().
Run:  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD python scripts/gate2_phase_spread.py --n 20
Self-test: PYTHONPATH=$PWD python scripts/gate2_phase_spread.py --selftest
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from src.datasets.kf_dataset import KFDataset
from src.pde.ns import cheb_lowpass
from src.solver.periodic import NavierStokes2d
from msc.tta import setup, eval as ev
from scripts.chaos_spread_gate import kf_forcing, perturb_ic, solve_from_ic

RES256 = "/system/user/studentwork/wehofer/data/ns/NS_fine_Re500_T128_res256_part0.npy"
EPS = 1e-12


def phase_dispersion(ensemble, kinf, kmax, win, ref) -> float:
    """Energy-weighted ensemble circular phase dispersion over k≤kmax modes in window `win`.

    ensemble (M,S,S,T) real; ref (S,S,T) real = energy-weight source (unperturbed solve).
    Per mode per frame D = 1 − |mean_M e^{iφ}|; weight |F(ref)|²; return scalar in [0,1].
    """
    Fh = torch.fft.fft2(ensemble, dim=(1, 2))                  # (M,S,S,T) complex
    unit = Fh / (Fh.abs() + EPS)                               # e^{iφ}
    D = 1.0 - unit.mean(0).abs()                               # (S,S,T) in [0,1]
    W = torch.fft.fft2(ref, dim=(1, 2)).abs() ** 2            # (S,S,T) energy weights
    mask = (kinf <= kmax)[:, :, None].to(W.dtype)             # (S,S,1) k≤kmax modes
    Wm = (W * mask)[:, :, win]
    return float((Wm * D[:, :, win]).sum() / (Wm.sum() + 1e-30))


def run(dataset, solver, f, deltas, M, re, T, dt, kinf, kmax, we, wl, device) -> dict:
    n_bands = kinf.shape[0] // 2 + 1
    ctrl_num = ctrl_den = 0.0
    acc = {d: {"De": [], "Dl": []} for d in deltas}
    for i in range(len(dataset)):
        gt = dataset[i]["y"].to(device)                        # (S,S,T)
        ic = gt[:, :, 0]
        ref = solve_from_ic(solver, ic.double(), f, T, dt, re, device)   # unperturbed solve
        ctrl_num += ev.band_power_t((ref - gt)[None], kinf, n_bands)[:ev.K_REP + 1][:, wl].sum()
        ctrl_den += ev.band_power_t(gt[None], kinf, n_bands)[:ev.K_REP + 1][:, wl].sum()
        gen = torch.Generator(device=device)
        for d in deltas:
            members = []
            for m in range(M):
                gen.manual_seed(i * 100003 + m * 97 + int(d * 1e4))
                pic = perturb_ic(ic, d, kmax, gen)
                members.append(solve_from_ic(solver, pic.double(), f, T, dt, re, device))
            ens = torch.stack(members)                         # (M,S,S,T)
            acc[d]["De"].append(phase_dispersion(ens, kinf, kmax, we, ref))
            acc[d]["Dl"].append(phase_dispersion(ens, kinf, kmax, wl, ref))
    ctrl = float(np.sqrt(ctrl_num / (ctrl_den + 1e-30)))
    out = {"control": ctrl, "deltas": {}}
    for d in deltas:
        De, Dl = float(np.mean(acc[d]["De"])), float(np.mean(acc[d]["Dl"]))
        out["deltas"][d] = {"D_early": De, "D_late": Dl, "ratio": Dl / (De + 1e-30)}
    return out


def report(res, kmax):
    print(f"\n=== Gate 2 phase-spread (k≤{kmax}, energy-weighted circular dispersion) ===")
    print(f"  control = k≤7 relL2(unperturbed solve − GT) = {res['control']:.4f}  [~0 ⇒ solver faithful]")
    print(f"  {'δ':>6} {'D_early':>8} {'D_late':>8} {'late/early':>11}")
    for d, m in res["deltas"].items():
        print(f"  {d:>6.3f} {m['D_early']:>8.4f} {m['D_late']:>8.4f} {m['ratio']:>11.2f}")
    print("  READ WITH chain_gate (oracle re-anchor beats one-shot 100%): D_late bounds ONLY one-shot-")
    print("    from-IC. D_late LOW ⇒ one-shot should work ⇒ wall = optimization/objective failure ⇒ fix")
    print("    loss IN PLACE (no migration). D_late→1 ⇒ must re-anchor ⇒ self-consistency/AR (NOT")
    print("    consolidate); D_late calibrates the re-anchor-field bar. VALID only if control ≪ wall.")


def main():
    ap = argparse.ArgumentParser(description="Gate 2 — phase-spread test")
    ap.add_argument("--data", default=RES256)
    ap.add_argument("--offset", type=int, default=260)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    ap.add_argument("--kmax", type=int, default=ev.K_REP)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="scripts/outputs/gate2_phase_spread.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    full = KFDataset(args.data, n_samples=args.n, offset=args.offset, sub_t=setup.SUB_T)
    ds = Subset(full, range(min(args.n, len(full))))
    S, T = ds[0]["y"].shape[0], ds[0]["y"].shape[-1]
    dt = setup.T_INTERVAL / (T - 1)
    nlate = max(1, T // 8)
    we, wl = slice(1, 1 + nlate), slice(T - nlate, T)
    kinf = ev.cheb_bins(S, device)
    solver = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=device, dtype=torch.float64)
    f = kf_forcing(S, device, torch.float64)
    print(f"Gate 2  data offset={args.offset} n={len(ds)} M={args.M} deltas={args.deltas} "
          f"S={S} T={T} dt={dt:.5f} kmax={args.kmax} device={device}", flush=True)
    res = run(ds, solver, f, args.deltas, args.M, 500, T, dt, kinf, args.kmax, we, wl, device)
    report(res, args.kmax)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nsaved -> {out}", flush=True)


def selftest():
    S, M, T = 16, 8, 4
    dev = torch.device("cpu")
    kinf = ev.cheb_bins(S, dev)
    win = slice(0, T)
    ref = cheb_lowpass(torch.randn(1, S, S, T), ev.K_REP)[0].abs() + 0.5   # positive energy ref

    base = cheb_lowpass(torch.randn(1, S, S, T), ev.K_REP)[0]
    ident = base[None].repeat(M, 1, 1, 1)
    d0 = phase_dispersion(ident, kinf, ev.K_REP, win, ref)
    assert d0 < 1e-6, f"identical ensemble -> 0, got {d0}"

    gen = torch.Generator(device=dev)
    rand = torch.stack([cheb_lowpass(torch.randn(1, S, S, T, generator=gen.manual_seed(m), device=dev),
                                     ev.K_REP)[0] for m in range(64)])
    dr = phase_dispersion(rand, kinf, ev.K_REP, win, ref)
    assert dr > 0.8, f"independent random phases -> ~1, got {dr}"

    half = torch.stack([base if m < M // 2 else -base for m in range(M)])   # phase flip π on half
    dh = phase_dispersion(half, kinf, ev.K_REP, win, ref)
    assert 0.9 < dh <= 1.0 + 1e-6, f"antipodal split -> ~1 (|mean|=0), got {dh}"

    assert phase_dispersion(ident, kinf, ev.K_REP, win, ref * 3.0) < 1e-6, "ref scale invariance @ D=0"
    print("selftest OK (identical->0, random->~1, antipodal->~1, ref-scale-invariant)")


if __name__ == "__main__":
    main()
