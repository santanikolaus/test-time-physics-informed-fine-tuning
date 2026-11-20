"""E16: no-network physics ceilings (Re ORACLE_RE, default 500): solver horizon per grid, and direct field descent on KFLoss."""
import math
import os
import sys

import numpy as np
import torch

from msc.tta.eval import eval as ev
from src.pde.ns import KFLoss
from src.solver.periodic import NavierStokes2d

RE, RHO = int(os.environ.get("ORACLE_RE", 500)), 0.9
DATA = f"/system/user/studentwork/wehofer/data/ns/NS_fine_Re{RE}_T128_res{{}}_part0.npy"
VAL = range(240, 270)
LOSSES = {"o2": (2, False), "o4e": (4, True)}
DEV = torch.device("cuda")


def resample(w, s):
    """Returns w (B, S, S) spectrally cropped or zero-padded to (B, s, s), amplitude-preserving."""
    S = w.shape[-1]
    fh = torch.fft.fftshift(torch.fft.fft2(w), dim=(-2, -1))
    out = torch.zeros(w.shape[0], s, s, dtype=fh.dtype, device=w.device)
    h, ci, co = min(S, s) // 2, S // 2, s // 2
    out[:, co - h:co + h, co - h:co + h] = fh[:, ci - h:ci + h, ci - h:ci + h]
    return torch.fft.ifft2(torch.fft.ifftshift(out, dim=(-2, -1))).real * (s / S) ** 2


def horizon(p, g):
    """Returns per-sample whole-field correlation horizon (frames, ρ* 0.9) of p vs g, both (N, S, S, T)."""
    S = g.shape[1]
    kinf, n = ev.cheb_bins(S, g.device), S // 2 + 1
    pt, gt, et = (np.stack([ev.band_power_t(x[i:i + 1].float(), kinf, n) for i in range(len(g))])
                  for x in (p, g, p - g))
    return ev.time_to_threshold(ev.corr_curve(pt, gt, et, bands=slice(1, None)), RHO)


def pde(w, order, edges, per_sample=False):
    """Returns the KFLoss pde term of w (B, S, S, T): batch mean, or per sample."""
    loss = KFLoss(re=RE, data_weight=0.0, pde_weight=1.0, ic_weight=0.0, time_order=order, time_edges2=edges)
    if per_sample:
        return np.array([float(loss(x[None, None], x[None])["pde"]) for x in w])
    return loss(w[:, None], w)["pde"]


def solver_horizon():
    """Solves Re RE from the 30 val ICs at 64²–512², scores H and KFLoss pde on the 128² dt 1/64 grid."""
    raw = np.load(DATA.format(256), mmap_mode="r")
    gt = torch.from_numpy(np.load(DATA.format(128), mmap_mode="r")[VAL.start:VAL.stop, ::2]).permute(0, 2, 3, 1).to(DEV)
    ic = torch.from_numpy(np.ascontiguousarray(raw[VAL.start:VAL.stop, 0])).to(DEV, torch.float64)
    print("| field | H mean ± std | censored | pde o2 | pde o4e |\n|---|---|---|---|---|")
    rows = {"GT": gt}
    for s in (64, 128, 256, 512):
        ns = NavierStokes2d(s, s, 2 * math.pi, 2 * math.pi, device=DEV, dtype=torch.float64)
        y = torch.linspace(0, 2 * math.pi, s + 1, dtype=torch.float64, device=DEV)[:-1]
        f = -4 * torch.cos(4.0 * torch.meshgrid(y, y, indexing="ij")[1])
        w, frames = resample(ic, s), []
        for k in range(129):
            if k % 2 == 0:
                frames.append(resample(w, 128).float())
            if k < 128:
                w = ns.advance(w, f, T=1 / 128, Re=RE, adaptive=True)
        rows[f"solver {s}²"] = torch.stack(frames, -1)
    for name, w in rows.items():
        h = horizon(w, gt)
        print(f"| {name} | {h.mean():.2f} ± {h.std():.2f} | {(h == 65).sum()}/30 | "
              f"{pde(w, 2, False, True).mean():.4f} | {pde(w, 4, True, True).mean():.4f} |", flush=True)


def val_fields():
    """Returns {name: (30, S, S, T)} for GT and zero-shot / adapted (E13b p0) / native predictions on val [240:270]."""
    import glob
    from msc.tta.adapt import adapt
    from res_variants import ADAPTED, BASE, CKPTS
    from src.models.kf_fno import kf_forward
    out = {}
    for name, (ckpt, op_re) in CKPTS.items():
        cfg = adapt.load_config(BASE + [f"ckpt={ckpt}", f"op_re={op_re}"])
        model, train_cfg = adapt.build(cfg, DEV)
        if name == "adapted":
            model.load_state_dict(torch.load(glob.glob(ADAPTED.format(""))[0], map_location=DEV)["state_dict"])
        _, held, target_cfg = adapt.build_splits(cfg, train_cfg)
        d, preds, gts = target_cfg.data, [], []
        with torch.no_grad():
            for it in held:
                y = it["y"][None].to(DEV)
                preds.append(kf_forward(model, it["x"][None].to(DEV), y.shape[-1], time_scale=d.time_scale,
                                        temporal_pad=d.temporal_pad, pad_mode=d.pad_mode).squeeze(1))
                gts.append(y)
        out[name], out["GT"] = torch.cat(preds), torch.cat(gts)
    return out


def descent(iters=5000, lr=1e-3, starts=("GT", "adapted", "zero-shot", "native"), every=250):
    """Minimises KFLoss pde directly over frames 1..T−1 (frame 0 = GT IC) from GT, model outputs; logs H, pde, rel_l2."""
    f = val_fields()
    gt = f["GT"]
    for lname, (order, edges) in LOSSES.items():
        for start in starts:
            free = f[start][..., 1:].clone().requires_grad_()
            opt = torch.optim.Adam([free], lr=lr)
            print(f"\n## {lname} from {start}\n| it | pde | H mean | rel_l2 |\n|---|---|---|---|", flush=True)
            for it in range(iters + 1):
                w = torch.cat([gt[..., :1], free], -1)
                loss = pde(w, order, edges)
                if it % every == 0:
                    with torch.no_grad():
                        rl2 = float(((w - gt).flatten(1).norm(dim=1) / gt.flatten(1).norm(dim=1)).mean())
                        print(f"| {it} | {float(loss):.4f} | {horizon(w.detach(), gt).mean():.2f} | {rl2:.4f} |", flush=True)
                opt.zero_grad()
                loss.backward()
                opt.step()


def causal_loss(w, eps):
    """Returns causal-weighted residual (Wang et al. 2022) and the min frame weight; w (B, S, S, T), o2 stencil.

    Args:
      w: vorticity trajectory, channels-last.
      eps: causality strength; frame t weighted by exp(−eps Σ_{s<t} L_s), weights detached.

    Returns:
      (loss, min weight), L_t = ‖Du_t − f‖² / ‖f‖² per sample and frame.
    """
    from src.pde.ns import NSVorticity
    ns = NSVorticity(re=RE)
    du, _ = ns.residual(w)
    f = ns.get_forcing(w.shape[1], w.device)
    lt = ((du - f) ** 2).sum((1, 2)) / (f ** 2).sum()
    cum = torch.cumsum(lt.detach(), -1) - lt.detach()
    wt = torch.exp(-eps * cum)
    return (wt * lt).mean(), float(wt.min())


def causal_descent(iters, lr, eps, start="adapted", every=250):
    """Field descent from a model start under the causal-weighted o2 residual; logs H, plain pde, min weight."""
    f = val_fields()
    gt = f["GT"]
    free = f[start][..., 1:].clone().requires_grad_()
    opt = torch.optim.Adam([free], lr=lr)
    print(f"\n## causal eps {eps} from {start}, lr {lr}\n| it | pde | H mean | rel_l2 | min w |\n|---|---|---|---|---|", flush=True)
    for it in range(iters + 1):
        w = torch.cat([gt[..., :1], free], -1)
        loss, wmin = causal_loss(w, eps)
        if it % every == 0:
            with torch.no_grad():
                rl2 = float(((w - gt).flatten(1).norm(dim=1) / gt.flatten(1).norm(dim=1)).mean())
                print(f"| {it} | {float(pde(w, 2, False)):.4f} | {horizon(w.detach(), gt).mean():.2f} | {rl2:.4f} | {wmin:.2e} |",
                      flush=True)
        opt.zero_grad()
        loss.backward()
        opt.step()


if __name__ == "__main__":
    if sys.argv[1] == "solver":
        solver_horizon()
    elif sys.argv[1] == "causal":
        causal_descent(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    else:
        descent(int(sys.argv[2]), float(sys.argv[3]), sys.argv[4].split(","))
