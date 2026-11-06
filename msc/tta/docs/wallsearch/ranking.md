---
title: "Late-Time relL2 Wall — Feasibility × Return Ranking"
date: 2026-06-25
---

Ranks the levers in `method_synthesis.md` against what is already run
(`coverage_check.md`). Order is a **sequence**, not a fixed list: Tier-0
results may re-rank Tier-1/2.

---

## Criteria

**Two gates (pass/fail, applied first):**
- **G1 one-shot inference preserved** — AR *training* and cheap *post-hoc*
  per-frame algebra pass; per-instance iterative refinement fails.
- **G2 not banked-dead / adjacent-dead** locally.

**Return R** (mechanism, not citations):
- **R1 mechanistic fit** to the oracle-restart proof — does it improve the
  mid-window state / remove a representational defect? *(highest weight)*
- **R2 evidence quality** — primary-verified, KF-relevant, single-shot-relevant.
- **R3 headroom** — fraction of the oracle gap (0.571→0.230, Re=500) plausibly
  closable.

**Feasibility F:** F1 implementation · F2 compute · F3 clean attribution.

**Sequencing rule:** near-free diagnostics that *re-rank a whole family* run
first (value-of-information), even though the R×F product understates them.
Expensive retrains run last, contingent on what the cheap runs reveal.

---

## The picture (why the sequence looks like this)

The wall is **representational, not chaotic** at low Re (chaos-spread gate,
Re=100), and oracle-restart proves **~57% of the late gap is recoverable**
(0.571→0.230, Re=500). The cheap one-shot low-k cupboard is **exhausted**:
capacity, magnitude band-loss, additive temporal-weighting, late-window
specialist, self-chaining, grid-shrink all banked dead; H⁻¹-train /
Spectral-Refiner-FT / RNO / SGNO / SSP fail G1; streamfunction/Leray redundant
for vorticity form; Newton / H^k / local-kernels mistargeted at high-k
(closure gate: wrong high-k buys 0.06–0.12 ≪ 0.47 wall).

Remaining real headroom: **resolve *where* the wall sits (two near-free
ablations), then at most two retrains.** Not a fifth cheap loss tweak.

Re caveat: the 0.571→0.230 headroom is **Re=500**; "not chaos-dominated" is
**Re=100** (chaos-spread is larger at Re=500). Do not read one conflated Re.

---

## Tier 0 — near-free, run first (value-of-information)

### 0.1 Phase-band sweep completion (§3.4b)
UNCONFIRMED ≠ dead — explicitly excluded from the §6 dead count. Targets k2–5,
the band where the recoverable signal lives. Its unconfirmed −0.022 already
beats magnitude's dead −0.005.
**Yield:** small but the *only* loss lever not yet falsified; promotes to a
real lever only if it clears the >0.006 threshold confirmed. Cheapest action
in the set — finish the half-run sweep before anything else.

### 0.2 PINO per-timestep + per-band ablation of the *existing* residual (§3.1b)
Does the physics-loss gain (8.28→6.04 aggregate) act **late** and **low-k** at
all? No per-timestep breakdown exists.
**Yield:** zero direct effect — it is a gate. It decides whether the entire
loss-objective family, and the finer-grid lever (2.2), can touch the wall.
Highest information-per-GPU-hour in the plan.

---

## Tier 1 — cheap GPU, clean attribution

### 1.1 Temporal architecture fix (§2.1 / §2.2)
FNO2d time-as-channels and/or periodic-pad + layernorm vs the current
FNO3d-temporal-FFT baseline. The FNO3d FFT imposes spurious periodicity over a
non-periodic time axis → aliasing that worsens toward the window edge (late t).
The local baseline **is** that defect, so the A/B is maximally clean and
zero inference cost.
**Yield (moderate, honest R2 weak):** Atif's number is AR-confounded and Cao
reports no isolated ablation, so this earns its slot on mechanism + clean
attribution + low cost, not on a cited magnitude. The mechanism is genuinely
late-t-specific; plausibly removes the part of the wall that is an edge
artefact rather than physics. Run before committing to any retrain.

---

## Tier 2 — expensive / high-ceiling, contingent on Tier 0

### 2.1 Differentiable window-chaining, flavor (b) only (§4.6b)
Split T into windows, backprop through the midpoint handoff, train the
midpoint to be a good IC — the one lever that directly attacks the
oracle-restart structure. Flavor (a) (shorten + chain at test) is gated out
(2-step AR inference); only flavor (b) (full-T one-shot operator, chaining as a
training regularizer) preserves one-shot inference.
**Yield (highest ceiling, biggest risk):** oracle-restart caps the prize at
~57%; if training can lift the model's own midpoint toward GT quality, this is
the single largest available move. **Two open risks lower the expected value:**
(i) self-chaining-own-field is banked dead at inference (model-restart 0.521 vs
oracle 0.230), so the gain is hostage to training actually improving the
midpoint; (ii) flavor (b) deploys one full-T pass whose midpoint is an interior
activation, not a re-ingested IC — it is unproven that a two-half-window
training graph reshapes the one-full-window inference graph. **Resolve this
train-vs-inference mechanism question before spending the compute.** Most
expensive item (backprop through two FNO3d 3D-FFT graphs + recurrent memory +
full retrain).

### 2.2 Finer-grid 256² PINO residual
The code evaluates the residual at data-res 128² (`src/pde/ns.py:280`), not the
finer grid the synthesis attributes to PINO — so enforcing physics at an
upsampled 256² grid is a genuine untested lever.
**Yield (uncertain, possibly mistargeted):** its purpose is enforcing physics at
*unresolved high-k* scales; by the closure-gate logic (wall is low-k) that fit
is suspect. The NS residual couples scales nonlinearly so it is not purely
high-k, but the burden is on 0.2 to show the residual touches low-k late-t.
**Contingent on 0.2** — do not commit compute until the existing-residual
ablation says the physics loss acts where the wall is.

---

## Demoted / gated out (with reason)

- **DPOTc enstrophy correction (§5.3)** — 1–2 global DOF on the same energy
  quantity the dead magnitude band-loss could not move; circulation already
  conserved (|gtdrift|≈4e-8). Near-free but near-zero return.
- **Steep (t/T)^α weighting (§3.8)** — the *steeper* arms were measured *worse*
  (p2 α4 +0.0051, p1 α4 +0.0006). Response surface flat-to-harmful.
- **List solver-in-loop (§3.5)** — G1 failure: deploys a learned solver = AR
  inference. Strongest late-time pointwise number in the literature, but not
  one-shot. Its one-shot residue (strain/spectral losses) collapses into the
  dead §3.4 family.
- **H^k / local-kernels / Newton (§3.3, §2.6a, §5.4)** — mistargeted at high-k;
  closure gate shows the wall is low-k.
- **H⁻¹-train, Spectral-Refiner-FT, RNO, SGNO, SSP, Leray, streamfunction** —
  fail G1 (AR inference / per-instance iterative) or redundant for vorticity
  form.

---

## Open decisions (steer the order)

1. **R1 weighting.** Schema text says R1 dominates; the product formula treats
   R/F equally. Equal → 1.1 leads (cheaper, cleaner). R1-dominant → 2.1 leads
   (only lever on the proven-recoverable midpoint). May be moot once the 2.1
   mechanism question is answered.
2. **Flavor-(b) mechanism.** Confirm whether the windowed training graph
   actually reshapes the one-shot inference midpoint before committing 2.1's
   compute. If fit is moderate (train/infer mismatch), 1.1 wins honestly.
