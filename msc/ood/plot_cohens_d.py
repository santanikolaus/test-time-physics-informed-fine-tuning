"""
Cohen's d heatmap — 4 terms × 5x5 (op_re × test_re) grid.

Usage:
    python msc/ood/plot_cohens_d.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

TERMS   = ["Du", "wt", "adv", "diff"]
RE_LIST = [100, 200, 300, 500, 1000]
OUTPUTS = Path(__file__).parent / "outputs"


def score(arr: np.ndarray) -> np.ndarray:
    return arr.mean(axis=-1)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return (b.mean() - a.mean()) / pooled


def build_matrix(term: str) -> np.ndarray:
    mat = np.full((len(RE_LIST), len(RE_LIST)), np.nan)
    for i, op_re in enumerate(RE_LIST):
        id_scores = score(np.load(OUTPUTS / f"residuals_op{op_re}_test{op_re}.npz")[term])
        for j, test_re in enumerate(RE_LIST):
            if test_re == op_re:
                continue
            ood_scores = score(np.load(OUTPUTS / f"residuals_op{op_re}_test{test_re}.npz")[term])
            mat[i, j] = cohens_d(id_scores, ood_scores)
    return mat


def main() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    labels = [str(r) for r in RE_LIST]
    vmax = 5.0

    for ax, term in zip(axes, TERMS):
        mat = build_matrix(term)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(term, fontsize=13)
        ax.set_xticks(range(len(RE_LIST)))
        ax.set_yticks(range(len(RE_LIST)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("test Re")
        ax.set_ylabel("operator Re")

        for i in range(len(RE_LIST)):
            for j in range(len(RE_LIST)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                            fontsize=8, color="black")

    fig.colorbar(im, ax=axes.tolist(), label="Cohen's d", shrink=0.8)
    fig.suptitle("Per-term Cohen's d — op Re (rows) vs test Re (cols)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "cohens_d_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"saved → {OUTPUTS / 'cohens_d_heatmap.png'}")


if __name__ == "__main__":
    main()
