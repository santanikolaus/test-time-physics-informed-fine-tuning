"""
Open-check #1 — is the Re=500 OOD residual signal aliasing-contaminated?

Computes the PDE residual of the OOD-TTA target — a lower-Re operator's
prediction û on Re=500 data — on progressively finer grids, by spectrally
zero-padding the SAME fixed 128² field (exact band-limited interpolation, no new
information, operator NOT re-run). Quadratic aliasing in the non-dealiased
advection term is removed once the grid resolves the product (≈2× the field's
top wavenumber), so:

  residual(û) flat across grids   ⇒ signal is NOT aliasing-dominated → TTA-clean
  residual(û) drops at finer grid ⇒ 128² residual is partly numerical aliasing

GT Re=500 is carried as a cross-check: its 128² residual must reproduce the
gt_residual_check diagonal (≈0.666), validating the pipeline + upsampler.
E(k) of û and GT (at 128²) is reported to show where the tail sits vs Nyquist.

Run:
  PYTHONPATH=$PWD python scripts/alias_check.py \\
      --ckpt pretrain-kol/1iix0n42/checkpoints/best.ckpt --op_re 300 --test_re 500
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf, kf_forward
from src.pde.ns import NSVorticity

# identical architecture + inference settings to infer_re500_id.py
MODEL_CFG = {
    "model_arch": "fno", "data_channels": 4, "out_channels": 1,
    "n_modes": [8, 8, 8], "hidden_channels": 128, "n_layers": 4,
    "lifting_channel_ratio": 0, "projection_channel_ratio": 2,
    "domain_padding": 0.0, "positional_embedding": None, "norm": None,
    "fno_skip": "linear", "implementation": "factorized",
    "use_channel_mlp": False, "channel_mlp_expansion": 0.5,
    "channel_mlp_dropout": 0.0, "separable": False, "factorization": None,
    "rank": 1.0, "fixed_rank_modes": False, "stabilizer": "None",
}
N_TEST, OFFSET_TEST, SUB_T = 40, 260, 2
TIME_SCALE, TEMPORAL_PAD, T_INTERVAL = 1.0, 5, 1.0
GRIDS = [128, 192, 256]

_ROOT       = Path(__file__).parent.parent
_PATHS      = yaml.safe_load((_ROOT / "documentation" / "paths.yaml").read_text())
DATA_ROOT   = Path(_PATHS["data"]["ns"])


def data_path(re: int) -> Path:
    if re == 1000:
        return DATA_ROOT / "NS_fine_Re1000_T128_indep.npy"
    return DATA_ROOT / f"NS_fine_Re{re}_T128_part0.npy"


def lp_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a - b).reshape(a.shape[0], -1)
    ref  = b.reshape(b.shape[0], -1)
    return float((diff.norm(dim=1) / (ref.norm(dim=1) + 1e-8)).mean().item())


def spectral_upsample(w: torch.Tensor, S_new: int) -> torch.Tensor:
    """(B,S,S,T) real → (B,S_new,S_new,T) real, exact band-limited interpolation.
    Zero-pads the centered spectrum; amplitude scaled by (S_new/S)^2 for the
    unnormalized FFT convention. No-op when S_new == S."""
    B, S, _, T = w.shape
    if S_new == S:
        return w
    wh = torch.fft.fftshift(torch.fft.fft2(w, dim=(1, 2)), dim=(1, 2))
    out = torch.zeros(B, S_new, S_new, T, dtype=wh.dtype, device=w.device)
    lo = (S_new - S) // 2
    out[:, lo:lo + S, lo:lo + S, :] = wh
    out = torch.fft.ifft2(torch.fft.ifftshift(out, dim=(1, 2)), dim=(1, 2)).real
    return out * (S_new / S) ** 2


def residual_relL2(w: torch.Tensor, nu: float) -> float:
    """rel-L2(NS residual, forcing) for a vorticity trajectory batch (B,S,S,T)."""
    ns = NSVorticity(re=1.0 / nu, t_interval=T_INTERVAL)
    S, T = w.shape[1], w.shape[3]
    forcing = ns.get_forcing(S, w.device).expand(w.shape[0], S, S, T - 2)
    Du, _ = ns.residual(w)
    return lp_rel(Du, forcing)


def radial_Ek(w: torch.Tensor) -> np.ndarray:
    """Time+batch-averaged radial energy spectrum of (B,S,S,T) vorticity."""
    S = w.shape[1]
    fh = torch.fft.fft2(w, dim=(1, 2))
    power = (fh.real ** 2 + fh.imag ** 2).mean(dim=(0, 3)).cpu().numpy() / (S * S)
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KX, KY = np.meshgrid(k, k)
    K = np.round(np.sqrt(KX ** 2 + KY ** 2)).astype(int)
    return np.array([power[K == ki].sum() for ki in range(S // 2 + 1)])


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = build_fno_kf(MODEL_CFG)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
             if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="operator checkpoint (rel to repo root or abs)")
    ap.add_argument("--op_re", type=int, required=True, help="operator training Re (sets ν for residual)")
    ap.add_argument("--test_re", type=int, default=500, help="OOD test Re")
    ap.add_argument("--out", default="scripts/outputs/alias_check.npz")
    ap.add_argument("--plot", default="scripts/outputs/alias_check.png")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = args.ckpt if Path(args.ckpt).is_absolute() else str(_ROOT / args.ckpt)
    print(f"Device={device}  op_re={args.op_re}  test_re={args.test_re}\n  ckpt={ckpt}\n")

    model = load_model(ckpt, device)

    # build û on test_re and the matching GT, both (N,S,S,T)
    ds = KFDataset(str(data_path(args.test_re)), n_samples=N_TEST, offset=OFFSET_TEST, sub_t=SUB_T)
    preds, gts = [], []
    for i in range(len(ds)):
        ic = ds[i]["x"].unsqueeze(0).to(device)
        gt = ds[i]["y"].unsqueeze(0).to(device)
        T = gt.shape[-1]
        with torch.no_grad():
            p = kf_forward(model, ic, T, time_scale=TIME_SCALE, temporal_pad=TEMPORAL_PAD)
        preds.append(p.squeeze(1))      # (1,S,S,T)
        gts.append(gt)
    uhat = torch.cat(preds, 0)          # (N,S,S,T)
    gt   = torch.cat(gts, 0)
    print(f"û={tuple(uhat.shape)}  gt={tuple(gt.shape)}")

    # upsampler self-check: 256 even-subgrid must equal the original field
    up = spectral_upsample(uhat[:1], 256)
    err = (up[:, ::2, ::2, :] - uhat[:1]).abs().max().item()
    assert err < 1e-3, f"upsampler subgrid mismatch: {err:.2e}"
    print(f"upsampler self-check max_err={err:.2e}  (ok)\n")

    nu_u  = 1.0 / args.op_re      # detector convention: "satisfies op_re physics?"
    nu_gt = 1.0 / args.test_re    # GT vs its own physics (matches gt_residual_check)

    print(f"{'grid':<8}{'residual(û) ν=1/'+str(args.op_re):<26}{'residual(GT) ν=1/'+str(args.test_re):<26}")
    print("-" * 60)
    res_u  = {S: [] for S in GRIDS}
    res_gt = {S: [] for S in GRIDS}
    for n in range(uhat.shape[0]):                      # per-traj → bounded GPU memory
        for S_new in GRIDS:
            res_u[S_new].append(residual_relL2(spectral_upsample(uhat[n:n + 1], S_new), nu_u))
            res_gt[S_new].append(residual_relL2(spectral_upsample(gt[n:n + 1], S_new), nu_gt))
    res_u  = {S: float(np.mean(v)) for S, v in res_u.items()}
    res_gt = {S: float(np.mean(v)) for S, v in res_gt.items()}
    for S_new in GRIDS:
        print(f"{S_new:<8}{res_u[S_new]:<26.4f}{res_gt[S_new]:<26.4f}")

    drop_u  = 1 - res_u[GRIDS[-1]]  / res_u[GRIDS[0]]
    print(f"\nû residual drop {GRIDS[0]}→{GRIDS[-1]}: {drop_u*100:.1f}%   "
          f"(large ⇒ TTA signal aliasing-contaminated; small ⇒ clean)")

    ek_u  = radial_Ek(uhat)
    ek_gt = radial_Ek(gt)

    np.savez(Path(args.out),
             grids=np.array(GRIDS),
             res_u=np.array([res_u[s] for s in GRIDS]),
             res_gt=np.array([res_gt[s] for s in GRIDS]),
             ek_u=ek_u, ek_gt=ek_gt, op_re=args.op_re, test_re=args.test_re)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaved → {args.out}")

    fig, (axr, axe) = plt.subplots(1, 2, figsize=(13, 5))
    axr.plot(GRIDS, [res_u[s] for s in GRIDS], "o-", label=f"û  (op{args.op_re}@test{args.test_re})")
    axr.plot(GRIDS, [res_gt[s] for s in GRIDS], "s--", label=f"GT Re={args.test_re}")
    axr.set_xlabel("residual eval grid (spectral upsample of fixed field)")
    axr.set_ylabel("residual rel-L2"); axr.set_xticks(GRIDS)
    axr.set_title("Grid-refinement of residual\nflat = alias-free   drop = 128² aliased")
    axr.legend(); axr.grid(True, alpha=0.3)

    k = np.arange(len(ek_u))[1:]
    axe.semilogy(k, ek_u[1:], label="û")
    axe.semilogy(k, ek_gt[1:], "--", label=f"GT Re={args.test_re}")
    axe.axvline(8, color="red", ls=":", lw=1, label="n_modes=8")
    axe.axvline(64, color="k", ls=":", lw=1, label="Nyquist (128²)")
    axe.set_xlabel("wavenumber k"); axe.set_ylabel("E(k)")
    axe.set_title("Spectrum of prediction vs GT (128²)")
    axe.legend(fontsize=8); axe.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"Saved → {args.plot}")


if __name__ == "__main__":
    main()
