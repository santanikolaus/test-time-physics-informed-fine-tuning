"""Oracle-distillation probe — go/no-go for the self-consistency direction.

Question: can this operator be TAUGHT to predict a better late second half at
all? We hand it the best possible teacher — the ORACLE restart (forward from the
TRUE mid-trajectory frame, our proven ~0.24-0.35 late upper bound) — and distil
it into the one-shot tail:

    pred    = F(ic)                          # one-shot full trajectory (needs grad)
    teacher = F(true_frame@m)  [no_grad]     # oracle restart = good 2nd half
    L = distill_w * relL2(pred[...,m:], teacher[...,:T-m]) + ic_w * relL2(pred[...,0], ic)

The teacher uses GT (the true frame@m) -> this is a THROWAWAY diagnostic, not a
deployable method. It is the UPPER BOUND on any self-consistency scheme (whose
teacher is the model's own, worse, restart). Read-out on held-out Re500[200:300]:
  - one-shot late k<=7 drops toward ~0.24-0.35  -> operator CAN absorb a good
    restart target -> GREEN LIGHT for self-consistency training.
  - late stays at the ~0.58 wall even with this perfect teacher -> distillation
    into the tail is blocked -> self-consistency (weaker teacher) cannot work -> STOP.

Run: CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD python scripts/oracle_distill_probe.py --ops op100
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
from torch.utils.data import DataLoader

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from msc.tta import setup, eval as ev

DATA_RE = 500
ADAPT = (0, 200)            # pool window; pool = first pool_n, stride-1 (decorrelated)
HELDOUT = (200, 300)        # locked eval split (matches the chaining/E_rangeN runs)
WALL, ORACLE_REF = 0.58, (0.24, 0.35)     # late wall / oracle ceiling band (plot refs)
OUT = setup.ROOT / "msc" / "tta" / "outputs" / "oracle_distill"

CKPTS = {"op100": "pretrain-kol/pvqq97sq/checkpoints/best.ckpt",
         "op300": "pretrain-kol/1iix0n42/checkpoints/best.ckpt",
         "op500": "pretrain-kol/38o0kj3y/checkpoints/best.ckpt"}


def distill_loss(model, ic, target, m: int, lp, ic_weight: float, distill_weight: float,
                 data_weight: float = 0.0) -> dict:
    """One-shot tail distilled toward the (stop-grad) oracle restart from frame m, plus
    optional full-trajectory data supervision. ic (B,S,S), target (B,S,S,T).
    Returns {loss, distill, ic, data}."""
    T = target.shape[-1]
    kf = lambda x: kf_forward(model, x, T, time_scale=setup.TIME_SCALE,
                              temporal_pad=setup.TEMPORAL_PAD).squeeze(1)   # (B,S,S,T)
    pred = kf(ic)
    with torch.no_grad():
        teacher = kf(target[:, :, :, m])                  # oracle restart from TRUE frame@m
    teacher_tail = teacher[..., 0:T - m].detach()         # absolute frames m..T-1
    distill = lp.rel(pred[..., m:T], teacher_tail)
    ic_l = lp.rel(pred[..., 0], target[..., 0])
    data = lp.rel(pred, target)                           # full-trajectory GT supervision
    loss = distill_weight * distill + ic_weight * ic_l + data_weight * data
    return {"loss": loss, "distill": distill, "ic": ic_l, "data": data}


def run_probe(op: str, a, device) -> dict:
    pool = KFDataset(str(setup.data_path(DATA_RE)), n_samples=a.pool_n, offset=ADAPT[0], sub_t=setup.SUB_T)
    heldout = KFDataset(str(setup.data_path(DATA_RE)), n_samples=HELDOUT[1] - HELDOUT[0],
                        offset=HELDOUT[0], sub_t=setup.SUB_T)
    model = copy.deepcopy(setup.load_model(CKPTS[op], device))
    setup.enable_gradient_checkpointing(model)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lp = LpLoss(d=3, p=2, reduction="mean")
    loader = DataLoader(pool, batch_size=1, shuffle=True)

    def evaluate():
        model.eval()
        d = ev.band_eval(model, heldout, device, op_re=DATA_RE, test_re=DATA_RE)
        model.train()
        return {"early": d["early"], "late": d["late"], "aggr": d["err_k7"]}

    hist = {"step": [], "late": [], "early": [], "aggr": [], "distill": [], "ic": [], "data": []}
    e0 = evaluate()
    hist["step"].append(0)
    for k in ("distill", "ic", "data"):
        hist[k].append(np.nan)
    for k in ("early", "late", "aggr"):
        hist[k].append(e0[k])
    print(f"  step 0 | held-out late={e0['late']:.4f} early={e0['early']:.4f} aggr={e0['aggr']:.4f}", flush=True)

    step, done = 0, False
    while not done:
        for batch in loader:
            ic, target = batch["x"].to(device), batch["y"].to(device)
            out = distill_loss(model, ic, target, a.m, lp, a.ic_weight, a.distill_weight, a.data_weight)
            opt.zero_grad(); out["loss"].backward(); opt.step()
            step += 1
            if step % a.probe_every == 0 or step >= a.steps:
                e = evaluate()
                hist["step"].append(step)
                for k in ("early", "late", "aggr"):
                    hist[k].append(e[k])
                for k in ("distill", "ic", "data"):
                    hist[k].append(float(out[k]))
                print(f"  step {step} | held-out late={e['late']:.4f} early={e['early']:.4f} "
                      f"aggr={e['aggr']:.4f}  (distill={float(out['distill']):.4f} "
                      f"data={float(out['data']):.4f})", flush=True)
            if step >= a.steps:
                done = True; break

    late = np.array(hist["late"])
    return {"op": op, "m": a.m, "lr": a.lr, "steps": a.steps, "pool_n": a.pool_n,
            "ic_weight": a.ic_weight, "distill_weight": a.distill_weight, "data_weight": a.data_weight,
            "late_start": float(late[0]), "late_min": float(late.min()),
            "late_final": float(late[-1]), "late_argmin_step": int(hist["step"][int(late.argmin())]),
            "history": {k: [float(x) for x in v] for k, v in hist.items()}}


def _plot(res):
    h = res["history"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["step"], h["late"], "o-", ms=3, label="held-out late k≤7")
    ax.plot(h["step"], h["early"], "s-", ms=3, alpha=0.6, label="early (guard)")
    ax.axhline(WALL, color="gray", ls="--", lw=0.8, label=f"wall {WALL}")
    ax.axhspan(ORACLE_REF[0], ORACLE_REF[1], color="green", alpha=0.1, label="oracle band")
    ax.set_xlabel("adapt step"); ax.set_ylabel("rel-L2")
    ax.set_title(f"{res['op']} — oracle-distill (m={res['m']}, lr={res['lr']:g}, dw={res['data_weight']:g})")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"distill_{res['op']}_m{res['m']}_dw{res['data_weight']:g}.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Oracle-distillation probe")
    ap.add_argument("--ops", nargs="+", default=["op100", "op500"])
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--probe_every", type=int, default=250)
    ap.add_argument("--pool_n", type=int, default=160)
    ap.add_argument("--ic_weight", type=float, default=5.0)
    ap.add_argument("--distill_weight", type=float, default=1.0)
    ap.add_argument("--data_weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Oracle-distill probe  m={a.m} lr={a.lr:g} steps={a.steps} pool_n={a.pool_n} "
          f"ic_w={a.ic_weight} distill_w={a.distill_weight} data_w={a.data_weight} "
          f"seed={a.seed} device={device}\n", flush=True)
    for op in a.ops:
        print(f"== {op} ==", flush=True)
        res = run_probe(op, a, device)
        _plot(res)
        print(f"  -> late {res['late_start']:.4f} (start) -> min {res['late_min']:.4f} "
              f"@step {res['late_argmin_step']} -> final {res['late_final']:.4f}\n", flush=True)
        (OUT / f"distill_{op}_m{a.m}_dw{a.data_weight:g}.json").write_text(json.dumps(res, indent=2))
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
