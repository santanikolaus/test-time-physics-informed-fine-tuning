# TTA Findings — banked evidence (append-only)

Backward-looking ledger. Each entry = a run that happened + what it settled.
Roadmap (forward plan) lives in `tta-roadmap.md`.

## Check #2 — matrix `op{100,200,300,500}×Re{own,500}` + paired sign test (n=40)
- **ID floor rises with Re**: early k≤7 = 0.17/0.20/0.22 (Re100/200/300); op500@500 = 0.25 on-trend → **0.24 = intrinsic amortization floor, not a fixable gap**.
- **early k≤7 = Re-invariant** (forcing k=4 + IC): op300 beats op500 early 30/40 → **early TTA target DEAD**.
- **late + aggr k≤7 = SYSTEMATIC** (not chaos): op500 beats op300 35/40 p=1e-6 (late), 34/40 p=8e-6 (aggr) → learnable model-error. **"late=chaos" exclusion OVERTURNED**.
- **Target relocated → late/aggr k≤7**. baseline op300 late 0.52 / aggr 0.40; ceiling op500 late 0.46 / aggr 0.36 (~12% / ~9% headroom).

## Adaptation run — PRE-REGISTRATION (the run that produced the negative)
Setup: TTDA, frozen op300 → adapt all weights, physics-only (`data=0, pde=1, ic=anchor`, `KFLoss` differentiable — NOT `band_eval`). adapt Re500 [0:100], eval held-out [260:300]; pool ablate 100→30. ν: oracle 1/500 then realistic 1/300.
- Deliverable = E4 curve (residual↓ vs late/aggr k≤7↓ vs step). Three pre-registered readings:
  - residual↓ & late-err↓ → physics gap → WIN.
  - residual↓ & late-err flat → data-fit gap (residual-blind) → clean NEGATIVE, not a bug.
  - late-err↑ → over-reg / high-k smoothing / forgetting.
- Success = paired sign test (adapted vs op300, `per_instance_k7`, n=40): beats ~all 40, p<0.05 — NOT a mean-delta (gap=0.06; ⅓≈0.02 < per-instance spread).
- transductive (adapt+eval [260:300]) alongside; (transductive − held-out) = IC-fit vs transferable-physics.
- alias guard live on adapted û: NO-GO if residual drop >X% across 128/192/256 (frozen ref ~1.27).

## Adaptation run — RESULT (oracle ν, NEGATIVE, banked 2026-06-05)
Ran: op300→Re500, **full-weight, physics-only** (`data=0,pde=1,ic=1`), ν=1/500, lr=1e-4, 300 steps, pool Re500[0:100], eval [260:300]. Code: `adapt.py`+`FullWeightTTA`; npz `scripts/outputs/adapt_oracle_ic1.npz` (ic=1), `adapt_oracle.npz` (ic=5).
- residual FELL 1.34→0.60 **but error ROSE** (aggr 0.43→0.51, late 0.56→0.59) = pre-reg **outcome 3**.
- **KILL-SHOT**: op500 (best aggr 0.39) residual = **1.07 > adapted 0.60** → residual is **NON-IDENTIFYING**; best model is NOT at low residual; no `ic` value fixes this.
- **WHY**: Re500 under-resolved @128² → true soln has irreducibly high residual (op500@1.07) → minimizing residual walks toward smoother-WRONG fields. **Under-resolution poisons the objective, not just the eval.** Closes the resolution thread (rows 1–3).
- overshoot immediate (step20 already < op500's residual, error already rising) → **no label-free stopping window**.
- `ic` bracket (both fail): ic=5 → residual stalls (IC dominates); ic=1 → residual↓ error↑.
- **SCOPE of negative**: full-weight, residual-only, oracle-ν. Realistic ν=1/300 NOT run (oracle already fails ⇒ realistic ≥ as bad).
- **Untested levers**: constrain locus (only *confirms* negative is locus-indep — op500 itself @1.07), consistency / input-space objectives (ν-free).

## Zero-shot gap (op→Re500, locked split [260:300], full-field val_l2; warm floor ~0.037)
| op | val_l2 (n=40) | val_l2 (i280-295) | gap vs floor |
|----|--------------|-------------------|--------------|
| 100 | 0.543 | 0.535 | +0.498 |
| 200 | 0.453 | 0.440 | +0.403 |
| 300 | 0.431 | 0.413 | +0.376 |
| 500 | 0.398 | 0.364 | +0.327 |
- Two ceilings ~10× apart: inductive (best shared op = op500) ≈ 0.40 vs transductive per-instance floor 0.037.
- Pooled TTDA "failed" because the inductive headroom is only ~0.03 and residual overshoots past it.
- Resolution is NOT the bottleneck (per-instance fit reaches 0.037 on 128² data); bottleneck is shared-operator generalization (capacity).
- Caveat: full-field val_l2 includes late chaotic decorrelation; re-confirm in band-resolved k≤7 before locking.

## E1 cell — setup check (warm2 regression) + N=1 preliminary (`runner.py`, 2026-06-09)
Preliminary only — purpose was to validate the client, not a result.
- **GATE PASS**: client (`FullWeightTTA`) reproduces warm2's descent — op100, lr=2.5e-3, ic=5, oracle ν, N=1, 1500 steps → pool val_l2 0.49→**0.09**. Harness + telemetry validated; step-0 held-out 0.52 ≈ bracket op100 0.531.
- N=1: strong transductive fit (pool aggr k≤7 →0.05, *below* op500 ceiling 0.37) but **NEGATIVE inductive transfer** — held-out aggr 0.52→**0.65** (above op100 baseline), monotone with pool fitting. held-out residual flat ~2.0.
- Reads: N=1 flatness is **non-discriminating** (forgetting vs non-identifiability both predict it). Held-out *worsening* = forgetting (full-weight + high lr). pool residual 0.26 < true ~0.5–0.7 = smoother-than-truth (IC anchor held val_l2). → matrix must disentangle forgetting (c) from non-identifiability (b) via the LR axis; target = held-out k≤7 error, never residual.
