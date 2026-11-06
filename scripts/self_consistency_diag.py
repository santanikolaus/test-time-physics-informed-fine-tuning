"""Self-inconsistency diagnostic — frozen operator, forward-only, no training.

Measures, pooled over held-out Re500[200:300] (k-unfiltered, full field):

  (1) SELF-INCONSISTENCY by restart point m: relL2( F(u[m])[j] , u[m+j] ) per frame
      — how far the operator drifts from its OWN one-shot tail after restarting from
      its own frame m (u=F(ic)). One curve per m -> see whether/where it diverges.
  (2) PHYSICS RESIDUAL of the one-shot u: total relL2(Du, forcing) per frame, plus
      the three NS terms (d/dt, advection, diffusion) each rel-forcing — to overlay
      against (1) and see if physics moves alongside the inconsistency.
  (3) PHYSICS RESIDUAL OF THE DISAGREEMENT field d = F(u[m]) - u[m:]: total + terms
      — "physics inconsistency on the disagreement" (out of interest).

Restart alignment (chaining premise): F always labels its input as t=0, so
F(u[m])[j] ~ absolute frame m+j only if the operator is ~time-translation-invariant.

Run: CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD python scripts/self_consistency_diag.py --ops op100 op500
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from src.pde.ns import NSVorticity
from msc.tta import setup
from scripts.chain_gate import CKPTS

DATA_RE = 500
HELDOUT = (200, 300)
EPS = 1e-30
OUT = setup.ROOT / "msc" / "tta" / "outputs" / "selfcons"


def _kf(model, ic, T):
    return kf_forward(model, ic, T, time_scale=setup.TIME_SCALE,
                      temporal_pad=setup.TEMPORAL_PAD).squeeze(1)        # (1,S,S,T)


def residual_terms(w, ns) -> dict:
    """w (1,S,S,T) -> per-interior-frame squared-sums (T-2,) of residual + each NS
    term + forcing, summable across instances. Du = wt + adv + diff; residual = Du-f."""
    S, T = w.shape[1], w.shape[-1]
    Du, (wt, adv, diff) = ns.residual(w)
    forcing = ns.get_forcing(S, w.device).expand(1, S, S, T - 2)
    sq = lambda a: (a ** 2).sum(dim=(0, 1, 2)).cpu().numpy()            # (T-2,)
    return {"res": sq(Du - forcing), "wt": sq(wt), "adv": sq(adv),
            "diff": sq(diff), "forcing": sq(forcing)}


def run_op(model, dataset, device, strides, ns) -> dict:
    T = dataset[0]["y"].shape[-1]
    # accumulators (pooled squared sums across instances)
    inc_num = {m: np.zeros(T) for m in strides}        # ||F(u[m])-u[m:]||^2 per abs frame
    inc_den = {m: np.zeros(T) for m in strides}        # ||u[m:]||^2 per abs frame
    one_res = {k: np.zeros(T - 2) for k in ("res", "wt", "adv", "diff", "forcing")}
    dis_res = {m: {k: np.zeros(T - 2) for k in ("res", "wt", "adv", "diff", "forcing")} for m in strides}

    for i in range(len(dataset)):
        ic = dataset[i]["y"][:, :, 0].unsqueeze(0).to(device)          # (1,S,S)
        with torch.no_grad():
            u = _kf(model, ic, T)                                      # one-shot (1,S,S,T)
            for k, v in residual_terms(u, ns).items():
                one_res[k] += v
            for m in strides:
                v_m = _kf(model, u[:, :, :, m], T)                     # restart from own frame m
                seg = v_m[..., 0:T - m]                                # abs frames m..T-1
                u_tail = u[..., m:T]
                d = (seg - u_tail)
                inc_num[m][m:T] += (d ** 2).sum(dim=(0, 1, 2)).cpu().numpy()
                inc_den[m][m:T] += (u_tail ** 2).sum(dim=(0, 1, 2)).cpu().numpy()
                for k, val in residual_terms(d, ns).items():           # residual of the disagreement
                    dis_res[m][k][m:T - 2] += val[:T - m - 2] if (T - m - 2) > 0 else 0.0

    def rel(num, den):
        out = np.sqrt(num / (den + EPS)); out[den == 0] = np.nan; return out

    selfincons = {m: rel(inc_num[m], inc_den[m]).tolist() for m in strides}
    selfincons_aggr = {m: float(np.sqrt(np.nansum(inc_num[m]) / (np.nansum(inc_den[m]) + EPS)))
                       for m in strides}
    f = one_res["forcing"]
    one_res_rel = {k: rel(one_res[k], f).tolist() for k in ("res", "wt", "adv", "diff")}
    dis_res_rel = {m: {k: rel(dis_res[m][k], dis_res[m]["forcing"]).tolist()
                       for k in ("res", "wt", "adv", "diff")} for m in strides}
    return {"strides": strides, "selfincons_t": selfincons, "selfincons_aggr": selfincons_aggr,
            "oneshot_res_t": one_res_rel, "disagree_res_t": dis_res_rel}


def _plot(op, res):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for m in res["strides"]:
        ax[0].plot(res["selfincons_t"][m], label=f"m={m} (aggr {res['selfincons_aggr'][m]:.3f})")
    ax[0].set_title(f"{op} — self-inconsistency  relL2(F(u[m]), u[m:])")
    ax[0].set_xlabel("abs frame"); ax[0].set_ylabel("rel-L2"); ax[0].legend(fontsize=8)

    for k in ("res", "wt", "adv", "diff"):
        ax[1].plot(res["oneshot_res_t"][k], label=k)
    ax[1].set_title(f"{op} — one-shot physics residual (rel forcing)")
    ax[1].set_xlabel("interior frame"); ax[1].set_yscale("log"); ax[1].legend(fontsize=8)

    for m in res["strides"]:
        ax[2].plot(res["disagree_res_t"][m]["res"], label=f"m={m}")
    ax[2].set_title(f"{op} — residual of disagreement (F(u[m])-u[m:])")
    ax[2].set_xlabel("interior frame"); ax[2].set_yscale("log"); ax[2].legend(fontsize=8)

    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / f"selfcons_{op}.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Self-inconsistency + physics diagnostic")
    ap.add_argument("--ops", nargs="+", default=["op100", "op500"])
    ap.add_argument("--strides", type=int, nargs="+", default=[4, 8, 16, 32, 48])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ns = NSVorticity(re=DATA_RE, t_interval=setup.T_INTERVAL)
    heldout = KFDataset(str(setup.data_path(DATA_RE)), n_samples=HELDOUT[1] - HELDOUT[0],
                        offset=HELDOUT[0], sub_t=setup.SUB_T)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"self-consistency diag  strides={args.strides}  heldout={HELDOUT} n={len(heldout)} "
          f"device={device}\n", flush=True)
    for op in args.ops:
        model = setup.load_model(CKPTS[op], device)
        res = run_op(model, heldout, device, args.strides, ns)
        _plot(op, res)
        print(f"== {op} ==  self-inconsistency aggr by m: " +
              "  ".join(f"m{m}={res['selfincons_aggr'][m]:.3f}" for m in args.strides), flush=True)
        (OUT / f"selfcons_{op}.json").write_text(json.dumps(res, indent=2))
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
