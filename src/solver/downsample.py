"""Materialize a downsampled NS GT dataset from a high-res source.

Two source formats supported:
  --source-file  stacked (N,T+1,S,S) npy  [Re100 256^2, Re500 256^2]
  --parts-dir    per-chain part files      [Re500 512^2, generation layout]

Run (server, where the data lives):
    # Re500 512->256 spectral (per-chain parts)
    python -m src.solver.downsample --re 500 --s-out 256

    # Re500 / Re100 256->128 spectral (stacked file)
    python -m src.solver.downsample --re 500 --s-out 128 \
        --source-file /path/to/NS_fine_Re500_T128_res256_part0.npy
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml


def spectral_resample(field: torch.Tensor, s_out: int) -> torch.Tensor:
    """Spectral (FFT-crop) downsample (n,S,S,T) real -> (n,s_out,s_out,T), s_out<S.

    Ideal sharp low-pass at the new Nyquist: keeps modes |k|<=s_out/2 EXACTLY (so
    every retained shell, incl. k<=7, is byte-identical to the source) and aliases
    nothing. Amplitudes scale by (s_out/S)^2 to stay physical across the grid change.
    This is the discretization-consistent restriction for pseudo-spectral periodic
    data; strided/box downsampling would alias or attenuate resolved modes."""
    S = field.shape[1]
    fh = torch.fft.fftshift(torch.fft.fft2(field, dim=(1, 2)), dim=(1, 2))
    c, h = S // 2, s_out // 2
    crop = torch.fft.ifftshift(fh[:, c - h:c + h, c - h:c + h, :], dim=(1, 2))
    return (torch.fft.ifft2(crop, dim=(1, 2)).real * (s_out ** 2) / (S ** 2))


def spatial_resample_strided(field: torch.Tensor, s_out: int) -> torch.Tensor:
    """Strided downsample (n,S,S,T) -> (n,s_out,s_out,T).

    Aliases high-k modes into low-k bands (common ML-paper shortcut, not
    physically correct for pseudo-spectral data). s_out must exactly divide S."""
    S = field.shape[1]
    assert S % s_out == 0, f"s_out={s_out} does not divide S={S}"
    step = S // s_out
    return field[:, ::step, ::step, :]


_METHODS = {
    "spectral": spectral_resample,
    "strided":  spatial_resample_strided,
}

_ROOT = Path(__file__).parents[2]
DATA_ROOT = Path(yaml.safe_load((_ROOT / "msc" / "configs" / "paths.yaml").read_text())["data"]["ns"])


def resample_part(arr: np.ndarray, s_out: int, device, method: str = "spectral") -> np.ndarray:
    """(T+1,S,S) float32 -> (T+1,s_out,s_out) float32 downsample via method."""
    fn = _METHODS[method]
    f = torch.from_numpy(np.ascontiguousarray(arr.transpose(1, 2, 0)))[None].to(device)
    ds = fn(f, s_out)[0]                                      # (s_out,s_out,T+1)
    return ds.permute(2, 0, 1).contiguous().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Materialize downsampled NS GT")
    ap.add_argument("--re", type=int, required=True, help="Reynolds number (for output filename)")
    ap.add_argument("--s-out", type=int, default=256)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--method", choices=list(_METHODS), default="spectral")
    ap.add_argument("--source-file", default=None,
                    help="stacked (N,T+1,S,S) npy source; alternative to --parts-dir")
    ap.add_argument("--parts-dir", default=str(DATA_ROOT / "re500_res512"),
                    help="directory of per-chain part files (used when --source-file is omitted)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    suffix = "" if args.method == "spectral" else f"_{args.method}"
    out = Path(args.out or DATA_ROOT / f"NS_fine_Re{args.re}_T128_res{args.s_out}{suffix}_part0.npy")

    src = None
    if args.source_file:
        src = np.load(args.source_file, mmap_mode="r")        # (N, T+1, S, S)
        Tp1 = src.shape[1]
    else:
        parts = Path(args.parts_dir)
        sample = np.load(parts / f"NS_fine_Re{args.re}_T128_res512_part0.npy", mmap_mode="r")
        Tp1 = sample.shape[0]

    mm = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.float32, shape=(args.n, Tp1, args.s_out, args.s_out))
    print(f"re={args.re} method={args.method} | out {out}\nshape {(args.n, Tp1, args.s_out, args.s_out)} device {device}")

    written = 0
    for j in range(args.n):
        if src is not None:
            arr = np.array(src[j])                            # (T+1, S, S)
        else:
            p = Path(args.parts_dir) / f"NS_fine_Re{args.re}_T128_res512_part{j}.npy"
            if not p.exists():
                print(f"  skip missing part{j}")
                continue
            arr = np.load(p)
        mm[j] = resample_part(arr, args.s_out, device, args.method)
        written += 1
        if written % 50 == 0:
            print(f"  {written}/{args.n}")
    mm.flush()
    print(f"done; wrote {written}/{args.n} trajectories -> {out}")


if __name__ == "__main__":
    main()
