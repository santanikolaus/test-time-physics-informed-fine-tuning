"""Estimates viscosity by least squares from m observed GT frames per IC, pooled over 5-IC pools, Re100 and Re500 val."""
import numpy as np
import torch

from src.pde.ns import NSVorticity

DATA = "/system/user/studentwork/wehofer/data/ns/NS_fine_Re{}_T128_res128_part0.npy"
MS = (3, 5, 9, 17, 65)
DT = 1.0 / 64


def parts(w):
    """Returns (a, l) with a = ω_t + u·∇ω − f and l = ∇²ω on interior frames, so the PDE reads a = ν l."""
    ns = NSVorticity(re=1.0, t_interval=DT * (w.shape[-1] - 1))
    _, (wt, adv, diff) = ns.residual(w)
    return wt + adv - ns.get_forcing(w.shape[1], w.device), -diff


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("| Re | m frames | ν̂ median | ν̂ p5 | ν̂ p95 | ν̂/ν median |\n|---|---|---|---|---|---|")
    for re in (100, 500):
        raw = np.load(DATA.format(re), mmap_mode="r")
        w = torch.tensor(np.ascontiguousarray(raw[240:270, ::2]), dtype=torch.float64).permute(0, 2, 3, 1)
        for m in MS:
            a, l = parts(w[..., :m])
            num, den = (a * l).sum((1, 2, 3)).numpy(), (l * l).sum((1, 2, 3)).numpy()
            est = np.array([num[i].sum() / den[i].sum() for i in (rng.choice(30, 5, replace=False) for _ in range(2000))])
            q = np.quantile(est, (0.5, 0.05, 0.95))
            print(f"| {re} | {m} | {q[0]:.5f} | {q[1]:.5f} | {q[2]:.5f} | {q[0] * re:.2f} |")
