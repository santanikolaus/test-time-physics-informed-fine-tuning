"""Dose-response gate — is self-inconsistency a STEERABLE lever for GT late error?

The truth gate showed self-inconsistency CORRELATES with GT late error. Correlation
is not causation: this gate actively MINIMISES self-inconsistency (label-free) and
asks whether GT late error follows DOWN (steerable -> real lever) or stays flat /
rises while the dose drops (bystander -> self-consistency is closed).

Objective (label-free; ν oracle 1/500, no GT in the loss):
  u = F(ic); v = F(u[m])
  L = consist_w * relL2(v[...,:T-m], u[...,m:])   # self-inconsistency (the DOSE)
    + ic_w     * relL2(u[...,0], ic)              # pin frame 0 (anti-trivial-drift)
    + pde_w    * relL2(cheb_lowpass(Du,7), forcing)  # k≤7 physics (anti-collapse)
Tracked on held-out vs step: self-inconsistency (DOSE) and k≤7 GT late (RESPONSE).
Read: response drops with dose -> steerable. dose drops, response flat/up -> closed.

Run: CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD python scripts/dose_response_gate.py --ops op100
"""
import argparse
import copy
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from neuralop import LpLoss
from torch.utils.data import DataLoader, Subset

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from src.pde.ns import NSVorticity, cheb_lowpass
from msc.tta import setup, eval as ev
from scripts.chain_gate import CKPTS

DATA_RE = 500
ADAPT, HELDOUT = (0, 200), (200, 300)
WALL, BANDK_PHYS = 0.58, 0.556      # refs: physics-only wall / E_bandK k≤7-physics late
OUT = setup.ROOT / "msc" / "tta" / "outputs" / "dose_response"


def _kf(model, x, T):
    return kf_forward(model, x, T, time_scale=setup.TIME_SCALE,
                      temporal_pad=setup.TEMPORAL_PAD).squeeze(1)


def pde_k7(u, ns, lp):
    """Label-free k≤7 NS residual of trajectory u (B,S,S,T)."""
    S, T = u.shape[1], u.shape[-1]
    Du, _ = ns.residual(u)
    forcing = ns.get_forcing(S, u.device).expand(u.shape[0], S, S, T - 2)
    return lp.rel(cheb_lowpass(Du, 7), cheb_lowpass(forcing, 7))


def consist_loss(model, ic, target, m, ns, lp, cw, icw, pw) -> dict:
    """Self-inconsistency + IC anchor + k≤7 physics. Label-free (target only for IC@0).

    Single grad-forward (two stacked grad-forwards OOM at 128²/12GB): the one-shot u
    carries grad; the restart v=F(u[m]) is a FROZEN target. We pull the one-shot tail
    toward its own restart — minimising self-inconsistency by moving the one-shot
    (the quantity the GT-late response measures). The dose is the symmetric
    relL2(F(u[m]),u[m:]) measured separately in heldout_dose."""
    T = target.shape[-1]
    u = _kf(model, ic, T)                                   # one-shot, GRAD
    with torch.no_grad():
        v = _kf(model, u[:, :, :, m].detach(), T)          # restart from own mid-field, FROZEN
    consist = lp.rel(u[..., m:], v[..., :T - m])           # pull one-shot tail -> own restart
    ic_l = lp.rel(u[..., 0], target[..., 0])
    pde = pde_k7(u, ns, lp) if pw > 0 else torch.zeros((), device=ic.device)
    return {"loss": cw * consist + icw * ic_l + pw * pde, "consist": consist, "ic": ic_l, "pde": pde}


@torch.no_grad()
def heldout_dose(model, ds, m, lp, device) -> float:
    """Mean held-out self-inconsistency relL2(F(u[m]), u[m:]) — the DOSE we drive down."""
    vals = []
    for i in range(len(ds)):
        ic = ds[i]["y"][:, :, 0].unsqueeze(0).to(device)
        T = ds[i]["y"].shape[-1]
        u = _kf(model, ic, T)
        v = _kf(model, u[:, :, :, m], T)
        vals.append(float(lp.rel(v[..., :T - m], u[..., m:])))
    return float(np.mean(vals))


def run_probe(op, a, device):
    pool = KFDataset(str(setup.data_path(DATA_RE)), n_samples=a.pool_n, offset=ADAPT[0], sub_t=setup.SUB_T)
    heldout = KFDataset(str(setup.data_path(DATA_RE)), n_samples=HELDOUT[1] - HELDOUT[0],
                        offset=HELDOUT[0], sub_t=setup.SUB_T)
    dose_sub = Subset(heldout, list(range(0, len(heldout), 3)))     # ~34 instances for the dose
    model = copy.deepcopy(setup.load_model(CKPTS[op], device))
    setup.enable_gradient_checkpointing(model)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ns, lp = NSVorticity(re=DATA_RE, t_interval=setup.T_INTERVAL), LpLoss(d=3, p=2, reduction="mean")
    loader = DataLoader(pool, batch_size=1, shuffle=True)

    def evaluate():
        model.eval()
        d = ev.band_eval(model, heldout, device, op_re=DATA_RE, test_re=DATA_RE)
        dose = heldout_dose(model, dose_sub, a.m, lp, device)
        model.train()
        return {"late": d["late"], "early": d["early"], "aggr": d["err_k7"], "dose": dose}

    hist = {k: [] for k in ("step", "late", "early", "aggr", "dose", "consist")}
    e = evaluate(); hist["step"].append(0); hist["consist"].append(np.nan)
    for k in ("late", "early", "aggr", "dose"):
        hist[k].append(e[k])
    print(f"  step 0 | GT_late={e['late']:.4f} (response)  self_incons={e['dose']:.4f} (dose)  "
          f"early={e['early']:.4f}", flush=True)

    step, done = 0, False
    while not done:
        for batch in loader:
            ic, target = batch["x"].to(device), batch["y"].to(device)
            out = consist_loss(model, ic, target, a.m, ns, lp, a.consist_weight, a.ic_weight, a.pde_weight)
            opt.zero_grad(); out["loss"].backward(); opt.step()
            step += 1
            if step % a.probe_every == 0 or step >= a.steps:
                e = evaluate()
                hist["step"].append(step); hist["consist"].append(float(out["consist"]))
                for k in ("late", "early", "aggr", "dose"):
                    hist[k].append(e[k])
                print(f"  step {step} | GT_late={e['late']:.4f}  self_incons={e['dose']:.4f}  "
                      f"early={e['early']:.4f}  (train consist={float(out['consist']):.4f})", flush=True)
            if step >= a.steps:
                done = True; break
    return {"op": op, "m": a.m, "lr": a.lr, "weights": [a.consist_weight, a.ic_weight, a.pde_weight],
            "history": {k: [float(x) for x in v] for k, v in hist.items()}}


def _plot(res):
    h = res["history"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(h["step"], h["late"], "o-", label="GT late (response)")
    ax[0].plot(h["step"], h["dose"], "s-", label="self-inconsistency (dose)")
    ax[0].plot(h["step"], h["early"], "^-", alpha=0.5, label="GT early (guard)")
    ax[0].axhline(WALL, color="gray", ls="--", lw=0.8, label=f"wall {WALL}")
    ax[0].axhline(BANDK_PHYS, color="purple", ls=":", lw=0.8, label=f"k≤7-physics {BANDK_PHYS}")
    ax[0].set_xlabel("adapt step"); ax[0].set_ylabel("rel-L2"); ax[0].legend(fontsize=8)
    ax[1].plot(h["dose"], h["late"], "o-")
    ax[1].set_xlabel("self-inconsistency (dose)"); ax[1].set_ylabel("GT late (response)")
    ax[1].set_title("response vs dose — flat ⇒ bystander, down-sloping ⇒ steerable")
    for a_ in ax:
        a_.grid(True, alpha=0.3)
    fig.suptitle(f"{res['op']} — dose-response (m={res['m']})")
    fig.tight_layout(); fig.savefig(OUT / f"dose_response_{res['op']}.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Dose-response gate for self-inconsistency")
    ap.add_argument("--ops", nargs="+", default=["op100", "op500"])
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--probe_every", type=int, default=200)
    ap.add_argument("--pool_n", type=int, default=160)
    ap.add_argument("--consist_weight", type=float, default=1.0)
    ap.add_argument("--ic_weight", type=float, default=5.0)
    ap.add_argument("--pde_weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"dose-response  m={a.m} lr={a.lr:g} steps={a.steps} weights=consist{a.consist_weight}/"
          f"ic{a.ic_weight}/pde{a.pde_weight} device={device}\n", flush=True)
    summary = {}
    for op in a.ops:
        print(f"== {op} ==", flush=True)
        res = run_probe(op, a, device)
        _plot(res)
        h = res["history"]
        summary[op] = {"late_start": h["late"][0], "late_final": h["late"][-1], "late_min": min(h["late"]),
                       "dose_start": h["dose"][0], "dose_final": h["dose"][-1]}
        print(f"  -> dose {h['dose'][0]:.4f}->{h['dose'][-1]:.4f}   GT_late {h['late'][0]:.4f}"
              f"->{h['late'][-1]:.4f} (min {min(h['late']):.4f})\n", flush=True)
        (OUT / f"dose_response_{op}.json").write_text(json.dumps(res, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
