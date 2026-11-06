"""Temporal-mode point at 256 — is the time-mode axis still the dead lever?

Inference-only version of the temporal question on the converged 256 FNO@16
(adb4tfh0, n_modes=[16,16,8]). For each held-out instance, low-pass the model
one-shot trajectory to a sweep of temporal-mode budgets K_t and measure late
k<=7 error vs GT; gt_late(K_t) = the cost of the truncation itself (how much GT
late content lives beyond K_t time-modes).

  CLINCHER: model_late(K_t) >= full for EVERY K_t -> truncating the model recovers
            nothing; its late error already lives in the smooth (<=8-mode) band.
            Time-mode lever DEAD @256.
  SUPPORT : gt_late(8) read vs the model full-late wall (~0.57), NOT vs zero. At 128
            gt_late(8)~=0.19 -- not small absolutely, but << the 0.5 wall: 8 t-modes
            can represent the late field far better than the model achieves, so the
            wall is not a time-representability limit.

This is the cheap, student09-runnable equivalent of training a 16-time-mode model
(256 training needs >=24 GB; out of scope here). Mirrors the 128 temporal_budget /
temporal_payoff result ("8 t-modes hold late 0.19 << wall", lever dead) on 256.
Reuses run_op / trunc_time from scripts.temporal_payoff_gate.

Run: PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python scripts/temporal_payoff_256.py --device cuda
"""
import argparse
import json
from pathlib import Path

import torch

from src.datasets.kf_dataset import KFDataset
from scripts.temporal_payoff_gate import run_op
from scripts.time_band_resolved import load_model, fno_cfg, RES256, SUB_T, TEST_OFFSET, TEST_N

FNO16_CKPT = "pathB-256/adb4tfh0/checkpoints/best.ckpt"
OUT = Path("scripts/outputs/temporal_payoff_256.json")


def main():
    ap = argparse.ArgumentParser(description="Temporal-payoff gate on 256 FNO@16")
    ap.add_argument("--ckpt", default=FNO16_CKPT)
    ap.add_argument("--budgets", type=int, nargs="+", default=[3, 4, 5, 6, 8, 10, 12, 16, 24])
    ap.add_argument("--offset", type=int, default=TEST_OFFSET)
    ap.add_argument("--n", type=int, default=TEST_N)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    ck = args.ckpt
    if not Path(ck).is_absolute():
        from msc.tta import setup
        ck = str(setup.ROOT / ck)
    model = load_model(fno_cfg([16, 16, 8]), ck, device)

    ds = KFDataset(RES256, n_samples=args.n, offset=args.offset, sub_t=SUB_T)
    print(f"temporal-payoff @256  FNO@16 (model time n_modes=8)  budgets={args.budgets}\n"
          f"  split offset={args.offset} n={args.n}  S={ds[0]['y'].shape[0]} T={ds[0]['y'].shape[-1]}  "
          f"device={device}\n", flush=True)

    res = run_op(model, ds, device, args.budgets)
    full = res["rows"]["full"]["model_late"]
    best_k = min(args.budgets, key=lambda k: res["rows"][f"Kt{k}"]["model_late"])
    best = res["rows"][f"Kt{best_k}"]["model_late"]
    print(f"full late={full:.3f}  (n={res['n']}, late=last {res['nE']} frames)")
    print(f"  {'K_t':>4} {'model_late':>11} {'Δ vs full':>10} {'gt_late(floor)':>15} {'model_early':>12}")
    for k in args.budgets:
        r = res["rows"][f"Kt{k}"]
        print(f"  {k:>4} {r['model_late']:>11.3f} {r['model_late'] - full:>+10.3f} "
              f"{r['gt_late']:>15.3f} {r['model_early']:>12.3f}", flush=True)
    print(f"  best model_late = {best:.3f} @ K_t={best_k}  (Δ vs full {best - full:+.3f})")
    print("\nread (verdict=#1): model_late(K_t) >= full at EVERY K_t -> time-mode lever DEAD "
          "(truncation recovers nothing). gt_late(8) read vs the ~0.57 wall, NOT vs 0: "
          "gt_late(8) << wall -> 8 t-modes already over-represent the late field.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
