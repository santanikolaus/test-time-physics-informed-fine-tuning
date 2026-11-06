"""Paired op-quality test at 256 — chaos vs systematic discriminator.

Forward two operators of KNOWN different quality (default op300, op500) over the
SAME 256 Re500 ICs and compare per-trajectory late/aggr k<=7 error. Paired,
because chaos -> large instance variance; the question is whether the BETTER
operator beats the worse one on the SAME IC.

  frac wins ~ 0.5, p > 0.05  -> chaos: model quality does not move late error.
  frac wins high, p tiny      -> SYSTEMATIC: late error is learnable model deficiency.

Mirrors the 128 finding (op500 > op300 35/40, p=1e-6) on the 256 target. A
SYSTEMATIC result licenses "the 256 late k<=7 field contains learnable structure
(not pure chaos)" -- NOT "the 256-native models' 0.57 wall is the reducible part".
Pure inference; the pretrained ops are n8@128, never trained on this 256 file.
Even under any trajectory-index overlap the paired contrast still isolates quality
(both ops share it) -- no quality-confounding leakage. Reuses eval.per_instance_k7
(canonical k<=7 reduction) and chain_gate.CKPTS.

Run: PYTHONPATH=$PWD python scripts/paired_op_quality_256.py --ops op300 op500 --device cpu
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon, binomtest

from src.datasets.kf_dataset import KFDataset
from msc.tta import setup
from msc.tta.eval import per_instance_k7
from scripts.chain_gate import CKPTS

RES256 = "/system/user/studentwork/wehofer/data/ns/NS_fine_Re500_T128_res256_part0.npy"


def paired(better: np.ndarray, worse: np.ndarray) -> dict:
    """worse - better per IC; positive = better op has the lower error (wins)."""
    d = worse - better
    wins = int((d > 0).sum())
    n = int(d.size)
    p_w = float(wilcoxon(worse, better).pvalue) if np.any(d != 0) else 1.0
    p_sign = float(binomtest(wins, n, 0.5).pvalue)
    return {"better_mean": float(better.mean()), "worse_mean": float(worse.mean()),
            "delta_mean": float(d.mean()), "frac_better_wins": wins / n,
            "wins": wins, "n": n, "wilcoxon_p": p_w, "sign_p": p_sign}


def main():
    ap = argparse.ArgumentParser(description="Paired op-quality test (chaos vs systematic) @256")
    ap.add_argument("--ops", nargs=2, default=["op300", "op500"],
                    help="worse better (the second is the expected-better op)")
    ap.add_argument("--offset", type=int, default=200)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="scripts/outputs/paired_op_quality_256.json")
    args = ap.parse_args()
    device = torch.device(args.device)
    worse_op, better_op = args.ops

    ds = KFDataset(RES256, n_samples=args.n, offset=args.offset, sub_t=setup.SUB_T)
    print(f"paired op-quality @256  worse={worse_op} better={better_op}  "
          f"split offset={args.offset} n={args.n}  S={ds[0]['y'].shape[0]} T={ds[0]['y'].shape[-1]}  "
          f"device={device}\n", flush=True)

    metrics = {}
    for op in (worse_op, better_op):
        model = setup.load_model(CKPTS[op], device)
        m = per_instance_k7(model, ds, device)
        metrics[op] = m
        print(f"== {op} ==  early={m['early'].mean():.4f}  late={m['late'].mean():.4f}  "
              f"aggr={m['aggr'].mean():.4f}", flush=True)

    out = {"worse": worse_op, "better": better_op, "offset": args.offset, "n": args.n}
    print()
    for metric in ("late", "aggr"):
        rep = paired(metrics[better_op][metric], metrics[worse_op][metric])
        out[metric] = rep
        verdict = "SYSTEMATIC" if (rep["frac_better_wins"] >= 0.65 and rep["sign_p"] < 0.05) else \
                  ("CHAOS-LIKE" if rep["sign_p"] > 0.05 else "mixed")
        print(f"PAIRED {metric:>4}: {better_op}={rep['better_mean']:.4f} {worse_op}={rep['worse_mean']:.4f}  "
              f"Δ̄={rep['delta_mean']:+.4f}  {better_op} wins {rep['wins']}/{rep['n']} "
              f"({rep['frac_better_wins']:.0%})  wilcoxon_p={rep['wilcoxon_p']:.2e}  "
              f"sign_p={rep['sign_p']:.2e}  -> {verdict}")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
