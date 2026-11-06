---
title: "Late-Time relL2 Wall — Ranking v2 (TTA-aligned, G1-relaxable)"
date: 2026-06-25
supersedes: ranking.md
reviewed_by: claude-subagent (2026-06-25), dispositions logged at end
---

## Purpose & reweighting axes

The 2-week wall sprint is an **intermediary** to the thesis goal: TTA under OOD.
Logic chain: an operator that cannot *model* late steps cannot *adapt* them — you
cannot denoise a signal you never captured. So the late-wall fight is graded by
whether it leaves a **better, adaptable model of the late state**, not only a
lower number.

Reweighting axes (priority order):

- **A1 — late-wall impact.** Expected reduction of late (t/T>0.5) k≤7 relL2 toward
  the oracle-restart floor (0.571→0.230 @256² Re500). Highest weight.
- **A2 — modeling > solving.** A lever that improves the *operator's own* late
  state outranks one that patches outputs with a solver. Solving is allowed **as a
  resort** (a real reduction beats a dead lever) but ranks below modeling at equal
  impact, because TTA adapts a model, not a post-hoc patch.
- **A3 — one-shot deterministic > latent/probabilistic.** Latent/probabilistic
  LAST — driven by **setup cost** (no repo scaffold; standing it up would eat the
  budget cheaper modeling levers deserve first), not by merit alone.
- **A4 — sequencing.** One-shot first; then relax G1 (one-shot inference) toward
  highest yield. Near-free analysis that re-ranks a whole family runs before any
  retrain (value-of-information).

Tiers: **lower tier number = run earlier = higher priority.** Every lever and
diagnostic from method_synthesis.md §1–§7 appears exactly once; status tags from
coverage_check.md.

Anchor facts (banked):
- Oracle-restart: GT midpoint → late 0.230 vs one-shot 0.571 (~57% recoverable).
- Self-chaining own field (inference): 0.521 — barely moves. Post-hoc chaining dead.
- Closure gate: recoverable signal entirely k≤7; wrong high-k buys 0.06–0.12 ≪ 0.47.
- Amp/phase: late phase_frac 0.45–0.52 concentrated k2–5. Wall = low-k phase error.
- Re caveat: 57% headroom is Re=500; "not chaos-dominated" is Re=100.

---

## Tier 0 — near-free, run first (gate everything)

### 0.1 PINO per-timestep + per-band ablation of the existing residual (§3.1b) — zero-train analysis
- **Run:** pure post-hoc analysis of an existing checkpoint; no retrain. Cheapest
  decisive measurement in the plan.
- **Helps:** gates the entire physics-loss family (3.1/1.2, plus §3.2/§3.3 in Tier 6).
  If the 8.28→6.04 gain acts early/broadband the residual cannot touch the late low-k
  wall; if late/low-k, the 256² lever (1.2) is promoted. Zero direct wall effect —
  pure value-of-information.

### 0.2 Phase-band loss sweep completion (§3.4b) — cheap finish-the-sweep training
- **Run:** a stalled training sweep, not pure analysis (flagged per review); cheap to
  finish. Angle-weight arms (best unconfirmed −0.022) are the only loss lever not yet
  falsified.
- **Helps:** the only cheap loss targeting the *proven* defect — phase in k2–5.
  Confirms or kills the low-k-phase-loss hypothesis. Ceiling is small (−0.022 ≪ 0.47):
  validates the diagnosis, does not break the wall.

---

## Tier 1 — one-shot, modeling, cheap GPU (A2✓ A3✓)

### 1.1 Temporal architecture fix — FNO2d time-as-channels / periodic-pad+layernorm (§2.1+§2.2)
- **Run:** the local baseline *is* FNO3d-temporal-FFT, so the A/B is maximally clean,
  zero inference cost, one-shot, modeling. §2.1 and §2.2 are two fixes for the same
  defect — one family, not additive. Runs **in parallel** with Tier 0 (no GPU
  contention with the 2.x path).
- **Helps:** FFT over a non-periodic time axis imposes spurious periodicity → aliasing
  worst at the window edge (late t); late-t-specific mechanism, could remove the
  edge-artefact fraction of the wall.
- **Impact caveat (review-flagged, kept Tier 1 deliberately):** this is the *temporal*
  axis, orthogonal to the proven *spatial* low-k-phase defect — A1 here is uncertain.
  Kept early only because it is cheap, one-shot, modeling, and the **only** lever on an
  axis no diagnostic has touched. **Demote to Tier 4 if 0.1 shows the wall is purely
  low-k-phase** (temporal-edge story then unlikely).

### 1.2 Finer-grid 256² PINO residual (§3.1a finer-grid) — contingent on 0.1
- **Run:** the residual is evaluated at data-res 128² (`src/pde/ns.py:280`), not the
  256² the synthesis attributes to PINO — a genuinely untested, one-shot, modeling
  lever.
- **Helps:** enforces physics at unresolved high-k. By closure-gate logic (wall low-k)
  the fit is suspect, but the NS residual couples scales nonlinearly. Commit only if
  0.1 shows the residual reaches low-k late-t.

---

## Tier 2 — one-shot, modeling, high-ceiling retrain (the one-shot wall-breaker)

### 2.0 Window-chaining mechanism pilot — cheap pilot training run (precedes 2.1)
- **Run:** short flavor-(b) training on a small pool (a real but small train, not a
  zero-cost diagnostic — review-flagged label corrected); measure whether the model's
  own inference midpoint moves from 0.521 toward 0.230.
- **Helps:** cheapest test of the only one-shot wall-breaker's kill-risk. Pass → commit
  2.1. Fail → 2.1 dead, drop to Tier 3.

### 2.1 Differentiable window-chaining, flavor (b) — one-shot training regularizer (§4.6b + §4.4 pushforward)
- **Run:** the one-shot-preserving member of the pushforward/unrolling family. §4.4
  pushforward folds in here; the AR-as-published variants §4.1 and §4.3 are listed
  separately (Tier 6 / 3.5) — *not absorbed* (review fix). Split T into windows,
  backprop the midpoint handoff, train the interior midpoint to be a good IC; deploy as
  one full-T pass. ~1 day rollout-graph setup.
- **Helps:** directly attacks the oracle-restart structure — the only proven path to
  0.230, highest one-shot ceiling. **Modeling → best TTA alignment:** it improves the
  operator's own late state, exactly what an adaptation step acts on.
- **Two risks (the 2.0 pilot exists to resolve them):** (i) self-chaining own field is
  inference-dead (0.521), so the gain is hostage to *training* lifting the midpoint;
  (ii) unproven that a 2-window training graph reshapes the 1-window inference midpoint
  (interior activation, not a re-ingested IC).

---

## Tier 3 — relax G1, highest yield (modeling-AR first, solving as resort)

### 3.1 Differentiable window-chaining, flavor (a) — 2-step AR inference (§4.6a)
- **Run:** shorten window + chain at test; reuses Tier-2 chaining infra. Cheapest
  G1-relaxation. **Modeling, not solving.**
- **Helps:** most direct realization of the proven 57% oracle gain.
- **Conditional (review fix — "highest certainty" retracted):** on the *unmodified*
  model this collapses to dead own-field chaining (0.521). Its ceiling is conditional
  on midpoint quality — i.e. on either a dedicated short-window operator predicting a
  better midpoint, or 2.1 having lifted it. Run after/with 2.1, not as a standalone
  sure thing.

### 3.2 Physics + spectral + strain loss inside a differentiable solver (List, §3.5)
- **Run:** ordered before 3.3 (review fix): training-time, changes the operator weights
  → more modeling-aligned and more TTA-adaptable than a per-instance patch, even though
  its raw number is smaller. Requires a differentiable NS solver.
- **Helps:** clearest late-time *pointwise* field-error reduction in the literature —
  late velocity MSE ~10× best case (~10% worst); solver supplies the late dynamics the
  operator misses. **Solving → resort.** One-shot residue (strain/spectral alone)
  collapses into the dead §3.4 magnitude family, so the gain needs the solver in-loop.

### 3.3 Spectral-Refiner post-hoc H⁻¹ fine-tune (§5.1)
- **Run:** per-instance, iterative, solver-assisted refinement of a trained operator.
  Real setup cost (differentiable solver).
- **Helps:** strongest post-hoc number in the corpus (forced turbulence
  2.08e-2→2.82e-4, ~74×). **Solving + per-instance patch → ranked after 3.2 by A2:** it
  does NOT yield an adaptable late-state model (its own limitation), so for the TTA
  thesis it is an upper-bound probe and a fallback deliverable, not the backbone.

### 3.4 Solver-in-loop K=32 training (Kochkov, §4.2)
- **Run:** only if 3.2 shows the differentiable solver pays off and budget remains.
- **Helps:** improved long-trajectory vorticity correlation at KF Re=1000. **Solving →
  resort.** No controlled one-step-vs-K relL2 table → late-wall payoff unquantified for
  our metric; lower confidence than 3.2.

### 3.5 Multi-Step Penalty MP-FNO (§4.3) — contingency variant of 2.1/3.1
- **Run:** only if 2.1's handoff is unstable; detach every r rollouts + learnable
  discontinuities. Low run-priority by **conditionality** (not an A2 violation — it is
  modeling, but it is a contingency, not a primary lever).
- **Helps:** cuts gradient cost cubic→linear, enforces long-term coherence. No published
  relL2; a training-stability variant of window-chaining, not an independent win.

---

## Tier 4 — one-shot but mistargeted / low-impact (cheap, evidence says small)

### 4.1 Newton-step residual correction (§5.4)
One-shot, deterministic, cheap — but banked non-identifying (residual ↓, late err flat).
Mistargeted for low-k late error. One-shot+deterministic cannot rescue ~zero A1.

### 4.2 Local + global hybrid kernels (Liu-Schiaffini, §2.6a)
One-shot architecture; add local differential/integral kernels to the FNO. Restores
high-freq/local structure (17% AR KF Re=5000). By closure gate the wall is low-k →
mistargeted. Run only if 0.1/0.2 reopen a high-k story.

### 4.3 Positive Sobolev H^k loss (§3.3)
One-line, one-shot. High-pass weight; MNO 24% one-step at Re=40, but PDE-Refiner shows
Sobolev *shortens* rollout. High-k target ≠ low-k wall.

### 4.4 Static temporal upweighting, steep pure-power (t/T)^α (§3.8)
Additive form banked dead; steep pure-power (early weight→0) never tested. One-line.
Steeper additive arms measured worse → response surface flat-to-harmful; pure form not
strictly falsified. Cheapest long-shot, near-zero expected yield.

### 4.5 DPOTc conserved-quantity correction (§5.3a)
One-shot post-hoc scale/shift. 1–2 global DOF on enstrophy/circulation (circulation
already conserved, |gtdrift|≈4e-8). A global rescale cannot fix a phase/structure error.
Moment diagnostic (§5.3b, r=0.44–0.77 vs late err) already done — correlation ≠
correctability.

### 4.6 Streamfunction (§2.3) & Leray/Helmholtz projection (§5.2)
Skip for vorticity form; both enforce divergence-free velocity, already implicit.
Relevant only for recovered velocity, not the late vorticity wall.

---

## Tier 5 — latent / probabilistic / new operator class (last: setup cost, A3)

Deferred by **setup cost**, not merit. Out of the 2-week sprint unless every
deterministic lever stalls.

### 5.1 SSP — spatial-basis / temporal-state separation (§2.5)
Latent Fourier-mode propagation; AR at inference; no repo scaffold. Isolates temporal
evolution (up to 48.9% relL2 on SW/RD, not KF). Capacity is banked dead → latent
compression risks smoothing the very k2–5 phase the wall needs.

### 5.2 SGNO — non-expansive ETD spectral generator (§2.4)
New operator class, AR, ℜ(α)≤0 generator. kolm2d Re=100 absolute error ≈0.558 (most
transfer-relevant single number, but high). AR + new-class setup cost defers it.

### 5.3 RNO — recurrent training (§4.5)
Recursive self-application in [0,T]. Claims O(e^{CT})→O(T), but the paper reports it
*fails on FNO* (works only on MgNO). AR. Low yield, high setup → probably skip.

### 5.4 AFNO — autoregression-free latent flow-matching (§7)
Latent continuous-time flow-matching; generative-adjacent, probabilistic, no KF
pointwise number. Violates A3, largest setup cost. Only if a distributional deliverable
becomes acceptable.

### 5.5 Generative / probabilistic refiners — PDE-Refiner, adv-NO, Gen4Turb (§7)
Diffusion/adversarial refinement; deterministic-scope violation (A3). Highest setup,
distributional output. Skip in sprint; used elsewhere only as ablation evidence.

---

## Tier 6 — banked dead / subsumed / scope-excluded (do not run; listed for completeness)

Levers:
- **Magnitude spectral band loss (§3.4a):** energy-add −0.005, band-equalize above
  baseline. Dead (<0.006). High-k/magnitude, not low-k phase.
- **Additive temporal weighting 1+α(t/T)^p (§3.8):** max −0.006, non-monotone. Dead.
- **Capacity (modes 8→16, h 64→128):** late unchanged/worse. Wall not capacity. Dead.
- **16² grid shrink:** wall survives (0.506 ≈ 0.543). Wall intrinsic. Dead.
- **Late-window specialist (W2–W5):** worse than baseline. Dead.
- **Self-chaining own field (inference, §4.6 model-restart):** 0.521, all strides ≈dead.
  This is *why* 2.1 must act in training.
- **Differentiable unrolling WIG/NOG/ONE as-published (§4.1):** subsumed by 2.1/3.1 in
  our single-shot setup (review fix); only relevant under full many-step AR inference,
  which we are not adopting. ~21% KF Re=300–900 in its own AR regime.
- **Local UNet+spectral hybrid (§2.6b, reversed direction):** tested locally, reached the
  wall (0.478 ≈ 0.472), did not beat it. Dead. (Distinct from §2.6a/4.2, the untested
  FNO+local form.)
- **H⁻¹ negative-Sobolev *training* (§3.2):** paper's own ablation 20× worse from
  scratch. Value lives only in the post-hoc FT (3.3). Dead as a training loss.
- **Transport / optimal-transport loss (§3.6):** statistical; DySLIM dropped Sinkhorn as
  non-discriminative in high-dim; no 2D KF pointwise. Dead for this metric.
- **Dissipativity / stability reg (§3.7):** MNO — does not move stepwise relL2
  (0.1545→0.1543). Prevents blow-up only. Dead for the gap.
- **Pushforward inference-side (§4.4 local oracle-distill / self-consistency):** tested
  negative locally (reverts; self-inconsistency a bystander). Surviving form is 2.1.
- **Late Fusion NO, TNO bundling, scheduled/causal temporal weighting, PENO/HF-scaling
  (§7):** scope/direction mismatch (parameter-extrapolation / zero single-shot benefit /
  upweights early t / unvalidated high-k). Excluded.

Diagnostics (not levers — already fired, frame the ranking):
- **§1.1 Lyapunov horizon, §1.2 0.5·T_L completeness:** chaos-vs-representational framing
  (representational at low Re). Bound, not method.
- **§1.3 invariant-measure metrics, §1.4 aggregate≠phase:** metric cautions; explain the
  literature gap. Not levers.
- **§1.5 spectral bias root cause:** mechanism linking time-axis, loss family, wall (MSE
  neglects low-amplitude modes). Motivation, not a lever.
- **Banked gates** (oracle-restart, chaos-spread, closure, amp/phase, moment-
  identifiability): the premises pointing the arrow at 2.1/3.1. Done; not re-run.

---

## Sprint sequence (2 weeks)

1. **Days 1–2 (one-shot, cheap):** 0.1 + 0.2 (Tier 0); 1.1 temporal A/B in parallel.
   If 0.1 reaches low-k late-t, queue 1.2.
2. **Days 3–6 (one-shot wall-breaker):** build chaining infra (~1 day) → 2.0 pilot →
   if pass, 2.1 retrain.
3. **Days 7–10 (relax G1, highest yield):** 3.1 flavor-(a) chaining (reuses infra,
   modeling), then as resort 3.2 List solver-in-loop and/or 3.3 Spectral-Refiner FT
   (solver setup cost; largest literature numbers).
4. **Days 11–14 (resort tail / write-up):** Tier 4 cheap long-shots if the wall stands.
   **Tier 5 (latent/probabilistic) is deferred past the sprint** — pursue only if every
   deterministic lever stalls (setup cost too large for the budget).

Decision gates: 0.1 gates 1.2; 2.0 gates 2.1; 2.1/3.1 outcome gates whether any Tier-3
solving lever earns its setup. TTA-alignment ranks modeling levers (2.0/2.1, 3.1, 1.1)
above solving (3.2–3.4) at equal late-wall impact, because the thesis adapts a model of
the late state, not a solver patch.

---

## Reviewer dispositions (subagent audit, adjudicated)

Accepted: 0.2 relabeled cheap-training (not pure diagnostic); removed false "absorbs
§4.1/§4.3" from 2.1 and kept once-each (§4.1→Tier 6 subsumed, §4.3→3.5); 2.0 relabeled
cheap pilot (not "near-free"); swapped solver-in-loop (3.2) ahead of Spectral-Refiner
(3.3) per A2 modeling>solving; retracted 3.1 "highest certainty", gated on midpoint
quality; §4.1 unrolling moved to Tier 6 (subsumed by 2.1/3.1); removed Tier 5 from the
sprint table (deferred post-sprint); added §2.6b (UNet+spectral, banked dead) to Tier 6.
Partially accepted: 1.1 — kept Tier 1 (cheap, one-shot, modeling, only lever on the
temporal axis) but added explicit A1-uncertainty caveat and a conditional demote-to-Tier-4
rule if 0.1 proves the wall purely low-k-phase. Rejected: full demotion of 1.1 to Tier 4
— burying a clean cheap one-shot modeling A/B below near-dead levers misweights
information-per-cost.
