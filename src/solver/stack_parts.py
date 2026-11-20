"""Stack per-chain GT part files into one KFDataset-layout file.

generate_trajectories.py writes resumable per-chain parts (T+1,S,S); KFDataset needs a
single stacked file (N,T+1,S,S). Parts are copied in chain order via a memmap (no full set
in RAM), at native resolution; for a 512^2 -> 256^2 downsample use downsample.py.
Val-split chains can come from a second pattern (e.g. long _U10 chains), truncated to the
first part's frame count. Prints the one-step increment std (= model output_factor).

Run (server):
    python -m src.solver.stack_parts --re 100 --res 256
    python -m src.solver.stack_parts --re 1000 --res 128 --parts-dir .../E24/parts \
        --pattern 'res{res}_part{j}.npy' --val-pattern 'res{res}_U10_part{j}.npy'
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).parents[2]
DATA_ROOT = Path(yaml.safe_load((_ROOT / "msc" / "configs" / "paths.yaml").read_text())["data"]["ns"])
SPLIT = yaml.safe_load((_ROOT / "msc" / "configs" / "configs.yaml").read_text())["split"]
PATTERN = "NS_fine_Re{re}_T128_res{res}_part{j}.npy"


def stack(paths: list, out: Path) -> np.memmap:
    """Copies per-chain parts into one (N, T+1, S, S) float32 memmap, T+1 from the first part.

    Args:
      paths: one part path per chain, in output order; longer parts are cut to the first's frames.
      out: output .npy path; must not exist.

    Returns:
      The flushed memmap.
    """
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"missing {len(missing)} parts, e.g. {missing[:3]}")
    T1, S, _ = np.load(paths[0], mmap_mode="r").shape
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(len(paths), T1, S, S))
    for i, p in enumerate(paths):
        mm[i] = np.load(p, mmap_mode="r")[:T1]
    mm.flush()
    return mm


def increment_std(chains, sub_t: int) -> float:
    """Std of one-step increments w[t+sub_t] - w[t] over all chains, frames and pixels.

    Args:
      chains: iterable of (T+1, S, S) arrays.
      sub_t: temporal stride of the model step (2 = dt 1/64 on T128 data).

    Returns:
      The pooled increment std.
    """
    s = q = c = 0.0
    for w in chains:
        d = np.diff(np.asarray(w[::sub_t], dtype=np.float64), axis=0)
        s, q, c = s + d.sum(), q + (d * d).sum(), c + d.size
    return float(np.sqrt(q / c - (s / c) ** 2))


def chain_paths(parts: Path, pattern: str, val_pattern: "str | None", re: int, res: int, chains) -> list:
    """Returns one part path per chain; val-split chains use val_pattern when given."""
    val = range(SPLIT["val"]["offset"], SPLIT["val"]["offset"] + SPLIT["val"]["n"])
    return [parts / (val_pattern if val_pattern and j in val else pattern).format(re=re, res=res, j=j)
            for j in chains]


def main():
    ap = argparse.ArgumentParser(description="Stack per-chain parts into one KFDataset file")
    ap.add_argument("--re", type=int, required=True)
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=SPLIT["total"])
    ap.add_argument("--parts-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pattern", default=PATTERN)
    ap.add_argument("--val-pattern", default=None)
    ap.add_argument("--sub-t", type=int, default=2)
    ap.add_argument("--std-only", action="store_true", help="print the increment std of existing train parts, no stacking")
    args = ap.parse_args()
    parts = Path(args.parts_dir or DATA_ROOT / f"re{args.re}_res{args.res}")
    train = range(SPLIT["train"]["offset"], SPLIT["train"]["offset"] + SPLIT["train"]["n"])

    if args.std_only:
        ps = [p for p in chain_paths(parts, args.pattern, None, args.re, args.res, train) if p.exists()]
        std = increment_std((np.load(p, mmap_mode="r") for p in ps), args.sub_t)
        print(f"increment std (sub_t {args.sub_t}) over {len(ps)}/{len(train)} train parts: {std:.4f}")
        return

    out = Path(args.out or DATA_ROOT / f"NS_fine_Re{args.re}_T128_res{args.res}_part0.npy")
    chains = range(args.start, args.start + args.n)
    mm = stack(chain_paths(parts, args.pattern, args.val_pattern, args.re, args.res, chains), out)
    print(f"done -> {out} {mm.shape}")
    if args.start == 0 and args.n >= len(train):
        print(f"increment std (sub_t {args.sub_t}) over train chains: {increment_std((mm[j] for j in train), args.sub_t):.4f}")


if __name__ == "__main__":
    main()
