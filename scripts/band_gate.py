"""
Decision gate (roadmap F3) — does the Re500 OOD/adaptation signal live in the
FNO-representable band k<=7 (n_modes=8 -> Chebyshev modes 0..7)?

For op300@test500 (pre-TTA), band-resolves by k_inf = max(|kx|,|ky|):
  err(k)    = power of (û - GT)            -> where the accuracy gap lives
  res_u(k)  = power of (residual(û) - f)   -> where the TTA objective pulls
  res_gt(k) = power of (residual(GT)- f)   -> is GT low-k self-consistent?

Pre-registered decision:
  Option A LIVE  iff  bandlim relL2 err(k<=7) >= ERR_THRESH (0.10)
                 AND  residual(û) energy fraction in k<=7 >= RESW_THRESH (0.20)
  GT low-k CLEAN iff  residual(GT) energy fraction in k<=7 <= GTRES_THRESH (0.15)
Else Option A is hollow (nothing to fix / objective-metric mismatch) -> lean B.

Run:
  PYTHONPATH=$PWD python scripts/band_gate.py \\
      --ckpt pretrain-kol/1iix0n42/checkpoints/best.ckpt --op_re 300 --test_re 500
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf, kf_forward
from src.pde.ns import NSVorticity
import alias_check as ac   # reuse MODEL_CFG, constants, data_path, load_model

K_REP        = 7      # FNO-representable Chebyshev band (n_modes=8 -> modes 0..7)
ERR_THRESH   = 0.10   # min k<=7 rel-L2 error for "a real gap to close"
RESW_THRESH  = 0.20   # min residual(û) energy fraction in k<=7 for "objective pulls here"
GTRES_THRESH = 0.15   # max residual(GT) fraction in k<=7 for "GT low-k clean"


def cheb_bins(S: int, device) -> torch.Tensor:
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KX, KY = np.meshgrid(k, k)
    return torch.from_numpy(np.maximum(np.abs(KX), np.abs(KY))).to(device)  # (S,S)


def band_power(field: torch.Tensor, kinf: torch.Tensor, n_bands: int) -> np.ndarray:
    """(B,S,S,T) real -> (n_bands,) batch+time-summed power per Chebyshev band."""
    fh = torch.fft.fft2(field, dim=(1, 2))
    p = (fh.real ** 2 + fh.imag ** 2).sum(dim=(0, 3))   # (S,S)
    return np.array([float(p[kinf == ki].sum()) for ki in range(n_bands)])


def band_power_t(field: torch.Tensor, kinf: torch.Tensor, n_bands: int) -> np.ndarray:
    """(B,S,S,T) real -> (n_bands, T) batch-summed power per Chebyshev band, per frame."""
    fh = torch.fft.fft2(field, dim=(1, 2))
    p = (fh.real ** 2 + fh.imag ** 2).sum(dim=0)        # (S,S,T)
    out = np.zeros((n_bands, p.shape[-1]))
    for ki in range(n_bands):
        out[ki] = p[kinf == ki].sum(dim=0).cpu().numpy()  # (T,)
    return out


def resid_minus_forcing(w: torch.Tensor, nu: float) -> torch.Tensor:
    ns = NSVorticity(re=1.0 / nu, t_interval=ac.T_INTERVAL)
    S, T = w.shape[1], w.shape[3]
    forcing = ns.get_forcing(S, w.device).expand(w.shape[0], S, S, T - 2)
    Du, _ = ns.residual(w)
    return Du - forcing                                  # (B,S,S,T-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--op_re", type=int, required=True)
    ap.add_argument("--test_re", type=int, default=500)
    ap.add_argument("--out", default="scripts/outputs/band_gate.npz")
    ap.add_argument("--plot", default="scripts/outputs/band_gate.png")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = args.ckpt if Path(args.ckpt).is_absolute() else str(ac._ROOT / args.ckpt)
    print(f"Device={device}  op{args.op_re}@test{args.test_re}\n  ckpt={ckpt}\n")

    model = ac.load_model(ckpt, device)
    ds = KFDataset(str(ac.data_path(args.test_re)), n_samples=ac.N_TEST,
                   offset=ac.OFFSET_TEST, sub_t=ac.SUB_T)

    S = ds[0]["y"].shape[0]
    T_eff = ds[0]["y"].shape[-1]
    n_bands = S // 2 + 1
    kinf = cheb_bins(S, device)
    nu_u, nu_gt = 1.0 / args.op_re, 1.0 / args.test_re

    # u/gt/err kept time-resolved (n_bands,T); residuals time-summed (gate only)
    u_pt   = np.zeros((n_bands, T_eff))
    gt_pt  = np.zeros((n_bands, T_eff))
    err_pt = np.zeros((n_bands, T_eff))
    bp_resu, bp_resgt = np.zeros(n_bands), np.zeros(n_bands)
    for i in range(len(ds)):
        ic = ds[i]["x"].unsqueeze(0).to(device)
        gt = ds[i]["y"].unsqueeze(0).to(device)
        T = gt.shape[-1]
        with torch.no_grad():
            uhat = kf_forward(model, ic, T, time_scale=ac.TIME_SCALE,
                              temporal_pad=ac.TEMPORAL_PAD).squeeze(1)   # (1,S,S,T)
        u_pt    += band_power_t(uhat,      kinf, n_bands)
        gt_pt   += band_power_t(gt,        kinf, n_bands)
        err_pt  += band_power_t(uhat - gt, kinf, n_bands)
        bp_resu  += band_power(resid_minus_forcing(uhat, nu_u),  kinf, n_bands)
        bp_resgt += band_power(resid_minus_forcing(gt,   nu_gt), kinf, n_bands)
    bp = {"u": u_pt.sum(1), "gt": gt_pt.sum(1), "err": err_pt.sum(1),
          "res_u": bp_resu, "res_gt": bp_resgt}

    lo = slice(0, K_REP + 1)   # k<=7
    err_k7   = float(np.sqrt(bp["err"][lo].sum()   / bp["gt"][lo].sum()))
    err_full = float(np.sqrt(bp["err"].sum()       / bp["gt"].sum()))
    resu_f7  = float(bp["res_u"][lo].sum()  / bp["res_u"].sum())
    resgt_f7 = float(bp["res_gt"][lo].sum() / bp["res_gt"].sum())

    print(f"{'band-limited rel-L2 error  k<=7':<38}{err_k7:.4f}   (full-field {err_full:.4f})")
    print(f"{'residual(û)  energy fraction k<=7':<38}{resu_f7:.3f}")
    print(f"{'residual(GT) energy fraction k<=7':<38}{resgt_f7:.3f}")

    gap_ok   = err_k7   >= ERR_THRESH
    pull_ok  = resu_f7  >= RESW_THRESH
    gt_clean = resgt_f7 <= GTRES_THRESH
    a_live   = gap_ok and pull_ok
    print("\n--- pre-registered gate ---")
    print(f"  gap to close (err k<=7 >= {ERR_THRESH})          : {'PASS' if gap_ok else 'FAIL'}")
    print(f"  objective pulls in band (res_u frac >= {RESW_THRESH}) : {'PASS' if pull_ok else 'FAIL'}")
    print(f"  GT low-k clean (res_gt frac <= {GTRES_THRESH})        : {'PASS' if gt_clean else 'FAIL'}")
    print(f"\n  OPTION A: {'LIVE' if a_live else 'HOLLOW -> lean B'}"
          f"{'' if gt_clean else '   [WARN: GT low-k may be contaminated]'}")

    # --- early-vs-late split of the k<=7 error (static physics vs chaotic drift) ---
    err_t = np.sqrt(err_pt[lo].sum(0) / (gt_pt[lo].sum(0) + 1e-30))   # (T,)
    nE = max(1, T_eff // 8)
    early = float(err_t[1:1 + nE].mean())          # skip t=0 (shared IC)
    late  = float(err_t[-nE:].mean())
    ratio = late / (early + 1e-12)
    static_ok = early >= 0.15
    drift_dom = (early < 0.10) and (ratio >= 2.0)
    print("\n--- early-vs-late k<=7 error (rollout time) ---")
    print(f"  early (t=1..{nE})        : {early:.4f}")
    print(f"  late  (last {nE} frames) : {late:.4f}")
    print(f"  late/early ratio         : {ratio:.2f}")
    verdict = ("STATIC component material -> TTA-addressable" if static_ok
               else "DRIFT-dominated -> check #2 may null orthogonally to resolution"
               if drift_dom else "MIXED -> inspect curve")
    print(f"  reading: {verdict}")

    np.savez(Path(args.out), kinf_max=n_bands - 1, K_REP=K_REP,
             **{f"bp_{k}": v for k, v in bp.items()},
             err_k7=err_k7, err_full=err_full, resu_f7=resu_f7, resgt_f7=resgt_f7,
             err_t=err_t, early=early, late=late)
    print(f"\nSaved → {args.out}")

    k = np.arange(n_bands)
    cum_err = np.sqrt(np.cumsum(bp["err"]) / np.cumsum(bp["gt"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(k[1:], np.sqrt(bp["err"][1:] / (bp["gt"][1:] + 1e-30)), "o-", ms=3, label="per-band rel error")
    ax1.plot(k[1:], cum_err[1:], "s--", ms=3, label="cumulative band-limited rel-L2")
    ax1.axvline(K_REP, color="red", ls=":", label=f"FNO band k={K_REP}")
    ax1.axhline(ERR_THRESH, color="grey", ls=":", lw=0.8)
    ax1.set_xlabel("Chebyshev band k"); ax1.set_ylabel("rel error û vs GT")
    ax1.set_title("Where the accuracy gap lives"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(k, np.cumsum(bp["res_u"]) / bp["res_u"].sum(), "o-", ms=3, label="residual(û) cum. frac")
    ax2.plot(k, np.cumsum(bp["res_gt"]) / bp["res_gt"].sum(), "s--", ms=3, label="residual(GT) cum. frac")
    ax2.axvline(K_REP, color="red", ls=":", label=f"FNO band k={K_REP}")
    ax2.set_xlabel("Chebyshev band k"); ax2.set_ylabel("cumulative energy fraction")
    ax2.set_title("Where the residual objective pulls"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"Saved → {args.plot}")

    figt, axt = plt.subplots(figsize=(7, 5))
    tt = np.arange(T_eff)
    axt.plot(tt, err_t, "o-", ms=3)
    axt.axvspan(1, nE, color="green", alpha=0.12, label=f"early (t=1..{nE})")
    axt.axvspan(T_eff - nE, T_eff - 1, color="red", alpha=0.12, label=f"late (last {nE})")
    axt.axhline(0.15, color="grey", ls=":", lw=0.8, label="static threshold 0.15")
    axt.set_xlabel("rollout frame t"); axt.set_ylabel("band-limited rel error k<=7")
    axt.set_title("k<=7 error vs rollout time\nlarge early = static (TTA-addressable);  ~0 early then rising = chaotic drift")
    axt.legend(fontsize=8); axt.grid(alpha=0.3)
    figt.tight_layout()
    tplot = str(Path(args.plot).with_name("band_gate_time.png"))
    figt.savefig(tplot, dpi=150)
    print(f"Saved → {tplot}")


if __name__ == "__main__":
    main()
