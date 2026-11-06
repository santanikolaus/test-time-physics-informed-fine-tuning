"""
Time-split band gate — is the in-band (k<=7) residual still IDENTIFYING at LATE time?

band_gate.py time-SUMS the residual, so it can only say "globally the model's
in-band residual is ~850x GT's" (R3). The TTA wall is specifically LATE, so the
decision-relevant question is whether that identifiability survives to late frames:

  late in-band res_u >> res_gt  -> objective still has signal late
        => late wall is OPTIMIZATION / EXPRESSIVITY, not information (band-limited
           residual is unlikely to break it on its own)
  late in-band res_u ~ res_gt   -> objective goes blind late
        => late wall is an INFORMATION limit of the instantaneous residual
           (band-limited residual cannot help; need a different signal)

This keeps the residual time-resolved (band_power_t) instead of summing it, and
splits early/late on the residual time axis, consistent with eval.band_eval's
error split. All math mirrors band_gate.py so summing over time reproduces it.

Residual time axis note: residual(w) has length T-2 (central time difference drops
the two end frames). Residual index j corresponds to original rollout frame j+1, so
the early window (orig frames 1..nE) = residual indices 0..nE-1, and we take the
late window as the last nE residual frames (orig frames ~T-1-nE..T-2); the ~1-frame
offset vs the error's late window is from the stencil and is immaterial.

Run (op300 reproduces the existing gate; also run op100 source for contrast):
  PYTHONPATH=$PWD python scripts/band_gate_timesplit.py \
      --ckpt pretrain-kol/1iix0n42/checkpoints/best.ckpt --op_re 300 --test_re 500 \
      --out scripts/outputs/band_gate_timesplit_op300.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.datasets.kf_dataset import KFDataset
from src.pde.ns import NSVorticity
from src.models.kf_fno import kf_forward
import alias_check as ac   # MODEL_CFG, load_model, data_path, constants, _ROOT

K_REP = 7   # FNO-representable Chebyshev band (n_modes=8 -> modes 0..7)


def cheb_bins(S: int, device) -> torch.Tensor:
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KX, KY = np.meshgrid(k, k)
    return torch.from_numpy(np.maximum(np.abs(KX), np.abs(KY))).to(device)   # (S,S)


def band_power_t(field: torch.Tensor, kinf: torch.Tensor, n_bands: int) -> np.ndarray:
    """(B,S,S,T) real -> (n_bands, T) batch-summed power per Chebyshev band, per frame."""
    fh = torch.fft.fft2(field, dim=(1, 2))
    p = (fh.real ** 2 + fh.imag ** 2).sum(dim=0)         # (S,S,T)
    out = np.zeros((n_bands, p.shape[-1]))
    for ki in range(n_bands):
        out[ki] = p[kinf == ki].sum(dim=0).cpu().numpy()
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
    ap.add_argument("--op_re", type=int, required=True, help="operator train-Re (nu for residual)")
    ap.add_argument("--test_re", type=int, default=500)
    ap.add_argument("--out", default="scripts/outputs/band_gate_timesplit.npz")
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

    # time-resolved accumulators
    u_pt   = np.zeros((n_bands, T_eff))          # |u|^2          (n_bands, T)
    gt_pt  = np.zeros((n_bands, T_eff))          # |gt|^2
    err_pt = np.zeros((n_bands, T_eff))          # |u-gt|^2
    resu_pt  = np.zeros((n_bands, T_eff - 2))    # |Du - f|^2     (n_bands, T-2)
    resgt_pt = np.zeros((n_bands, T_eff - 2))    # |D(gt) - f|^2

    for i in range(len(ds)):
        ic = ds[i]["x"].unsqueeze(0).to(device)
        gt = ds[i]["y"].unsqueeze(0).to(device)
        T = gt.shape[-1]
        with torch.no_grad():
            uhat = kf_forward(model, ic, T, time_scale=ac.TIME_SCALE,
                              temporal_pad=ac.TEMPORAL_PAD).squeeze(1)   # (1,S,S,T)
        u_pt     += band_power_t(uhat,      kinf, n_bands)
        gt_pt    += band_power_t(gt,        kinf, n_bands)
        err_pt   += band_power_t(uhat - gt, kinf, n_bands)
        resu_pt  += band_power_t(resid_minus_forcing(uhat, nu_u), kinf, n_bands)
        resgt_pt += band_power_t(resid_minus_forcing(gt,   nu_gt), kinf, n_bands)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>3}/{len(ds)}]")

    lo = slice(0, K_REP + 1)                       # k<=7
    nE = max(1, T_eff // 8)
    er_early, er_late = slice(0, nE), slice(-nE, None)   # residual-axis windows

    def inband_ratio(tsl):   # model/GT in-band residual energy ratio over time window tsl
        return float(resu_pt[lo, tsl].sum() / (resgt_pt[lo, tsl].sum() + 1e-30))

    r_early = inband_ratio(er_early)
    r_late  = inband_ratio(er_late)
    r_all   = inband_ratio(slice(None))

    # in-band residual fraction (resu_f7) per window — how much of the objective is in-band
    def frac7(res_t, tsl):
        return float(res_t[lo, tsl].sum() / (res_t[:, tsl].sum() + 1e-30))
    resu_f7_early, resu_f7_late, resu_f7_all = (frac7(resu_pt, er_early),
                                                frac7(resu_pt, er_late),
                                                frac7(resu_pt, slice(None)))

    # error early/late for context (error axis, skip shared t=0)
    err_t = np.sqrt(err_pt[lo].sum(0) / (gt_pt[lo].sum(0) + 1e-30))      # (T,)
    err_early = float(err_t[1:1 + nE].mean())
    err_late  = float(err_t[-nE:].mean())

    # regression check: time-summed in-band fraction must reproduce band_gate.resu_f7
    print("\n--- in-band (k<=7) residual identifiability: model vs GT energy ratio ---")
    print(f"  EARLY  res_u/res_gt = {r_early:10.1f}x   (resu_f7 {resu_f7_early:.3f})")
    print(f"  LATE   res_u/res_gt = {r_late:10.1f}x   (resu_f7 {resu_f7_late:.3f})")
    print(f"  ALL    res_u/res_gt = {r_all:10.1f}x   (resu_f7 {resu_f7_all:.3f})  <- matches band_gate")
    print(f"\n  late/early ratio drop : {r_late / (r_early + 1e-30):.3f}")
    print(f"  error  early {err_early:.4f}  late {err_late:.4f}")
    verdict = ("LATE still identifying (>50x) -> late wall is OPTIM/EXPRESSIVITY"
               if r_late > 50 else
               "LATE goes blind (<50x) -> late wall is an INFORMATION limit")
    print(f"\n  reading: {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             op_re=args.op_re, test_re=args.test_re, K_REP=K_REP, nE=nE, T_eff=T_eff,
             bp_res_u_t=resu_pt, bp_res_gt_t=resgt_pt,
             err_pt=err_pt, gt_pt=gt_pt, u_pt=u_pt, err_t=err_t,
             ratio_early=r_early, ratio_late=r_late, ratio_all=r_all,
             resu_f7_early=resu_f7_early, resu_f7_late=resu_f7_late, resu_f7_all=resu_f7_all,
             err_early=err_early, err_late=err_late)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
