---
title: "Coverage Check — method_synthesis.md vs tta-results.md"
date: 2026-06-25
---

Status labels used:
  NOT DONE        — nothing was run that addresses this entry
  DONE-EXACT      — run in the setup the synthesis proposes
  DONE-DEVIATED   — run, but setup deviates from synthesis; deviation described

Where one sub-question is answered and another is not, each receives its
own explicit status. "Partial" is not used anywhere.

---

## §1 — The Wall

### 1.1 Lyapunov predictability horizon
Status of the specific diagnostic the synthesis describes: NOT DONE.
Λ (maximum Lyapunov exponent in 1/convective-time units), T_L, and the
scaling α in λ∼Re^α were not computed locally.

Related experiments exist but answer a different question:
- Gate 1 (2026-06-17, scripts/chaos_spread_gate.py): perturbs IC by
  band-limited k≤7 noise δ∈{0.05,0.10,0.20}, rolls M=8 ensembles via
  the true solver, measures σ_phys growth ratio. Growth ratio =
  σ_phys_late/σ_phys_early = 1.32–1.41 at Re500, 1.13–1.20 at Re300.
  This is a growth ratio over one trajectory window; Λ in convective-
  time units and the α scaling were not computed.
- Gate 2 (2026-06-18, scripts/gate2_phase_spread.py): phase spread
  D_late/D_early = 1.28–1.42 at Re500 256². Same limitation: ratio,
  not Λ.
- Chaos-artifact split (2026-06-12, scripts/chaos_artifact_split.py):
  B_fixable vs C_chaos decomposition per restart frame t_r. Does not
  compute Λ or T_L.
The synthesis cites Atif's Λ≈2.15, T_L≈0.45 for decaying turbulence
Re~7–8k — those specific values and the λ∼Re^α scaling law have not
been reproduced locally. Synthesis marks this "diagnostic bound, not a
method."

### 1.2 The 0.5·T_L completeness threshold
Status: NOT DONE.
No local experiment computes T_L, the enstrophy cascade time, or tests
the 0.5·T_L completeness boundary for forced KF. Synthesis cites a
single decaying-turbulence GAN study (J. Fluid Mech. 2023) and
explicitly flags it as not independently re-verified. No local analogue
of that test has been run. Synthesis marks this "framing, decaying
turbulence, not forced KF."

### 1.3 Migration to invariant-measure metrics
Status: NOT DONE (no action implied by the synthesis).
All local experiments use pointwise relL2 exclusively. No Fourier
spectra, POD spectra, MELR, Wasserstein-1, autocorrelation, or TKE
metrics were computed. Synthesis marks this point as an explanation of
the literature gap ("metrics, not levers"), not a proposed lever. Kept
here because downstream ranking must not conflate "not done" with
"no value" for this entry.

### 1.4 Aggregate score ≠ phase tracking
Status: NOT DONE for the SGNO-specific claim.
Synthesis warns that aggregate GMean100 can mask early phase loss,
citing SGNO's "stable step" at ~step 13–14 in an AR rollout. No SGNO
architecture was run locally and no AR rollout was evaluated with that
specific aggregate metric.

Related but different experiment:
Amplitude/phase decomposition (2026-06-22, scripts/amp_phase_probe.py
--per-band --window 8): decomposes per-mode error into amplitude and
phase for our FNO/UNet models in single-shot inference. Late phase_frac
0.447–0.516, concentrated k2–5. This characterises the error structure
of a single-shot operator; it does not replicate the aggregate-vs-phase-
tracking behaviour of SGNO in AR inference. Different question, different
architecture, different inference mode.

### 1.5 Spectral bias as the shared root cause
Status of RFNO/FNO floor numbers the synthesis cites: NOT DONE.
Synthesis states RFNO/FNO report ~0.5–0.6% relL2 at ν=1e-3 rising to
12–14% at ν=1e-5 — these exact numbers were not reproduced locally.

Related experiments that confirm the Re-scaling trend (different metric):
- Spectral budget (2026-06-10, scripts/spectral_budget.py): NS-residual
  floor relL2(Du,f) at Re100=0.013, Re300=0.125, Re500=0.666,
  Re1000=1.460 on 128² GT. Metric = residual floor of the GT field (not
  prediction relL2), at 128² non-dealiased data, not the ν=1e-3 to
  ν=1e-5 sweep of RFNO. Trend consistent with Re-scaling; metric and
  setup different.
- Capacity gate and horizon-vs-capacity: confirm spectral capacity is
  not the lever; do not measure the RFNO floor scaling.

---

## §2 — Architecture

### 2.1 Time-as-channels vs temporal Fourier dimension
Status: NOT DONE.
Synthesis proposes testing FNO2d (time stacked as channels) vs FNO3d
(temporal FFT) on single-shot forced KF to isolate the temporal
periodicity mismatch. Note: the local baseline IS the FNO3d-temporal-FFT
architecture that this entry proposes replacing — so the comparison is
directly applicable and has not been run.

The following experiments probe a DIFFERENT aspect of temporal structure:
- Temporal-mode budget (2026-06-11): GT projected onto 8 time-modes
  gives late 0.189 ≪ wall 0.46–0.53. Confirms temporal mode COUNT is
  not the bottleneck.
- Temporal payoff gate (2026-06-12, scripts/temporal_payoff_gate.py
  and scripts/temporal_payoff_256.py): truncating to K_t∈{3,4,8,16,24}
  raises model_late at every K_t; cutting time modes removes real
  content, not recoverable model error.
These tests ask whether time-mode CAPACITY is the bottleneck (it is
not). The synthesis question is whether the temporal FFT's wrap-around
PERIODICITY ARTEFACT drives late-window degradation — that mechanism has
not been isolated. The FNO2d vs FNO3d comparison has not been run.

### 2.2 Periodic temporal padding + layernorm (Cao et al. ST-FNO3d)
Status: NOT DONE.
No local experiment modifies the temporal lifting operator to use
periodic padding, nor replaces the global normalizer with layernorm.

### 2.3 Streamfunction parametrization
Status: NOT DONE.
No local experiment predicts ψ and derives u=∇×ψ. Synthesis correctly
notes this is near-redundant for the vorticity-form model.

### 2.4 Non-expansive spectral generator (SGNO / ETD)
Status: NOT DONE.
No local experiment implements an ETD step with ℜ(α)≤0 constraint.
Synthesis notes this is autoregressive at inference.

### 2.5 Spatial-basis / temporal-state separation (SSP)
Status: NOT DONE.
No local experiment implements the analysis→propagation→synthesis
factorization of Lu et al. Synthesis notes autoregressive at inference.

### 2.6 Local + global hybrid kernels
Two distinct proposals; statuses differ.

(a) Adding local differential/integral kernels directly to the FNO
(Liu-Schiaffini et al. ICML 2024), reportedly cutting 5-step KF relL2
0.2360→0.1956 (~17%) at Re=5000 in AR:
Status: NOT DONE. This specific architecture (local kernels inside the
FNO's spectral convolution) was not implemented. The Liu-Schiaffini ~17%
AR gain at Re=5000 was not reproduced.

(b) A related local+global hybrid (UNet backbone + FNO-style spectral
bottleneck mixer) was tested, but is a DIFFERENT architecture:
Status: DONE-DEVIATED.
Spatial-FNO UNet (2026-06-21, scripts/unet_band_probe.py + spatial_diag):
UNet3D (local conv backbone) + SpatialSpectralMixer at bottleneck (1×1
channel bottleneck 512→64, two-corner low-pass k≤8, per-mode channel
mixing, zero-init up-proj). This adds global spectral coupling TO a
local-conv backbone; the synthesis proposes adding local kernels TO an
FNO — architecture direction reversed.
Results: 128² late k≤7 0.478 ≈ FNO op100/500 0.472; 256² late k≤7
0.577 ≈ FNO 0.571. Reaches the FNO wall; does not beat it. Diagonal-only
mixer (130K params, 2026-06-22): late 0.512. Second spectral layer @256²:
late 0.590 (worse). Evaluated single-shot; synthesis sources are AR.
The ~17% Liu-Schiaffini AR result was NOT reproduced and the architecture
is not the same.

---

## §3 — Loss Objectives

### 3.1 PDE-residual physics loss (PINO)
Two components; statuses differ.

(a) PINO as baseline training loss:
Status: DONE-DEVIATED.
PDE residual ‖∂_t u − R(u)‖² is the core loss in all local training runs
(data5+pde1+ic1 recipe throughout). The synthesis cites "8.28%→6.04% at
Re=500 in local banked runs" as already confirmed. Reach gate (2026-06-11):
data+pde 5:1 gives late_min 0.481 vs data-only 0.498.
Deviation to verify: the synthesis describes the PINO residual as
"evaluable at a finer grid than the data (256²×65 vs 64²×33)." Local
training runs use 128² data; whether the PDE residual is evaluated at
upsampled 256² or at data-resolution 128² has not been explicitly
confirmed against the code. If at data resolution, this is DONE-DEVIATED
vs the synthesis's finer-grid description.

(b) Per-timestep breakdown of whether the PINO gain concentrates at
late t (the key open ablation cited in the synthesis):
Status: NOT DONE. No experiment isolates the early vs late contribution
of the physics loss gain.

### 3.2 Negative-Sobolev / H⁻¹ residual weighting
Status: NOT DONE.
No local experiment weights the residual norm by (a+|k|²)⁻¹. Synthesis
downgraded this entry based on Spectral-Refiner ablation (~20× worse
than L²-train+H⁻¹-FT). Correctly excluded.

### 3.3 Positive Sobolev H^k loss
Status: NOT DONE.
No local experiment adds a Sobolev H^k penalty (penalising errors in
spatial derivatives via |k|^{2k} weighting). The spectral band losses
tested (see §3.4) operate differently and are not H^k losses.

### 3.4 Spectral / frequency-banded losses
Synthesis proposes LOGLO-FNO (per-band penalties cutting high-band fRMSE
58–73% on KF Re=5k) and BSP (Binned Spectral Power, penalising per-band
energy deviation). Both "optimize magnitudes/bands, not phases." These
specific methods were NOT implemented.

A related but different local experiment was run; validity caveat applies.

(a) band_energy_loss — confirmed result:
Status: DONE-DEVIATED.
From-scratch op100 Re100 128², 150 ep, data5+pde1+ic1+energy_term,
configs/kf_re100_energyweight.yaml. band_energy_loss = abs per-shell
|E_pred−E_gt|² over k=2–7, weight=1 (arm rai1ohu9). Run id confirmed;
checkpoint retrievable.
Result: late k≤7 0.538 vs baseline 0.543 (−0.005). Dead by the §6
threshold (<0.006).
Deviations: (1) LOGLO is an architecture, not a loss; not implemented.
(2) BSP uses a specific binning scheme; the local energy loss uses
Chebyshev shell binning and |E_pred−E_gt|² — not BSP. (3) Evaluated at
Re=100 single-shot, not Re=5k AR. (4) Target metric is late k≤7 relL2,
not high-band fRMSE. The LOGLO 58–73% result was not reproduced.

(b) band_phase_loss (angle arms) — UNCONFIRMED:
Status: UNCONFIRMED — re-run required. NOT to be treated as DONE.
The 2026-06-23 results table lists angle w1 (0.534), angle w5 (0.522),
angle w10 (0.525), angle w20 (0.521), and combo angle5+energy5 (0.522),
but the explicit validity caveat in tta-results.md states: "only energy
w1 (rai1ohu9) measured — the w5/w20 multirun arms had no retrievable
checkpoint at probe time (sweep still finishing)." All angle/phase arm
numbers are unconfirmed. The maximum improvement of −0.022 cited in the
table is unconfirmed and must not be treated as banked.
Additional note: phase-band loss does not map to synthesis §3.4, which
explicitly describes magnitude-band penalties, not phase alignment. The
phase-angle lever (GT-energy-weighted (1−cosΔφ)) is a local-original
lever with no synthesis entry; its status is open pending re-run.

### 3.5 Physics + spectral loss inside a differentiable solver
Status: NOT DONE.
No local experiment combines pointwise velocity L2 + spectral-energy +
strain-rate losses inside a differentiable NS solver (List et al.
setup).

### 3.6 Transport / optimal-transport loss
Status: NOT DONE.
No local experiment uses Wasserstein/Sinkhorn objectives. Synthesis notes
no 2D KF pointwise result exists in the literature either.

### 3.7 Dissipativity / stability regularization
Status: NOT DONE as an explicit dissipativity loss.
No local experiment implements MNO's outer-shell data augmentation or
SGNO's Lipschitz/contraction constraint. The ic_anchor term (5·ic in
TTA) prevents IC collapse but is not a dissipativity regularizer on the
trajectory. Synthesis notes this prevents blow-up but does not close the
on-attractor pointwise gap.

### 3.8 Static temporal upweighting w(t)∝t^α
Status: DONE-DEVIATED for the mild additive form — banked dead.
The steep pure-power form (t/T)^α with early weight→0 was NOT tested.

Loss-emphasis reweighting (2026-06-22, src/pde/ns.py frame_weights/
time_weighted_rel, configs/kf_re100_timeweight.yaml): adds per-frame
weight w_t = 1+α·(t/T)^p, mean-1 normalized, to the DATA term only.
Sweep: α∈{0,1,2,4} at p=2 and p∈{0.25,0.5,1,1.5} at α=4.
Results: best tw p1.5 α4 late k≤7 0.5375 (−0.0057); best tw p2 α2:
0.5380 (−0.0052). Non-monotone: tw p2 α1 = 0.5477 (+0.0047 above
baseline); tw p2 α4 = 0.5481 (+0.0051 above baseline). Max improvement
<0.006. σ(late)≈0.001 (three seeds). BANKED DEAD for the additive form.

Deviations from the synthesis formulation:
(1) Synthesis specifies w(t)∝(t/T)^α (pure power law, early weight→0
as t→0). Local implementation is 1+α(t/T)^p — an additive offset of 1
ensures early frames receive weight 1, late frames receive 1+α (max
ratio 1:5 at α=4). This is a MILDER profile than the synthesis's pure
(t/T)^α, which drives early weight to 0. The steep pure-power form (no
additive offset, early weight→0) was NOT tested. The synthesis caveat
"whether the exact (t/T)^α form differs from the banked variant is
unresolved" is partially resolved: the mild additive variant is dead;
the steep pure-power variant remains open.
(2) Applied to DATA term only; pde and ic weights unchanged.
(3) The canonical t^α form at p=1 was tested (tw p1 α4: late 0.5438,
+0.0006 vs baseline). Dead.

---

## §4 — Training Protocols

### 4.1 Differentiable multi-step unrolling (WIG / NOG / ONE)
Status: NOT DONE.
No local experiment backpropagates through K unrolled rollout steps
during training. Synthesis cites List et al. ~21% lower long-rollout
error at KF Re=300–900 with curriculum K=[1,2,4]. Not implemented.

### 4.2 Solver-in-loop K=32 training
Status: NOT DONE.
No local experiment embeds the operator in a differentiable NS solver
loop during training. Solver calls are used only for data generation and
the chaos-artifact diagnostic.

### 4.3 Multi-Step Penalty (MP-FNO)
Status: NOT DONE.
No local experiment implements the window-detach + learnable
discontinuity scheme of Chakraborty et al.

### 4.4 Pushforward
Synthesis defines pushforward as: feed the model's own stop-gradient
one-step prediction as IC during training. Two local experiments are
adjacent but NOT canonical pushforward; all statuses differ.

(a) True pushforward (model's own prediction, stop-grad, as IC for
next training step):
Status: NOT DONE. Never implemented.

(b) Oracle-distillation probe (2026-06-12, scripts/oracle_distill_probe.py):
Status: DONE-DEVIATED — NOT pushforward. Teacher is GT, not the model.
Setup: TTA opX→Re500, pool[0:160] stride-1, held-out[200:300] n=100,
5000 steps, lr=3e-4. Loss = 1.0·distill + 5.0·ic. Teacher = F(GT@32)
(GT frame 32 as IC — oracle, NOT the model's own prediction). Labelled
"throwaway upper bound" in tta-results.md.
Results (dw=0): op100 late min 0.542 (@step 1000), final 0.684 (reverts).
op500 late min 0.473 (@step 0), final 0.566.
With data anchor (dw=1.0): op100 late final 0.522 (stable, no reversion);
op500 late final 0.488.
Deviation: teacher is GT (upper-bound). Pushforward uses the model's own
prediction, exposing it to its own error distribution. Synthesis itself
labels this "a local stop-grad oracle-distill analogue tested negative."

(c) Self-consistency training — in the pushforward family (output-side),
result negative. NOT canonical pushforward (input-side).
Status: DONE-DEVIATED.
Dose-response gate (2026-06-12, scripts/dose_response_gate.py): TTA
opX→Re500, pool[0:160] stride-1, held-out[200:300] n=100, 3000 steps,
lr=3e-4. Loss = consist + 5·ic + 1·pde(k≤7). consist = relL2(F(u[32]),
u[32:]) — one-shot tail pulled toward its own frozen restart F(u[32]).
The teacher IS the model's own prediction (stop-grad restart), but used
on the TARGET side (comparing tail to restart), not as IC for the next
training input.
Results: op100 dose 0.246→0.126 (manipulation check OK), GT late
0.678→0.659 (min 0.610, reverts). op500 GT late 0.473→0.503 (worse).
Transductive (2026-06-13, scripts/self_consistency_dose.py): op100
si_late 0.302→0.204, GT late 0.615→0.899 (wrong direction). Self-
inconsistency = BYSTANDER, not steerable.
Deviation: this covers "train on own predictions" from the output
(target) side. Canonical pushforward (input-side, model's prediction as
IC for the next forward pass during training) was NOT tested and is not
covered by these experiments.

### 4.5 Recurrent training (RNO)
Status: NOT DONE.
No local experiment implements recursive self-application of the operator
within [0,T] during training. Synthesis notes this fails on FNO; correctly
excluded.

### 4.6 Differentiable window-chaining
Two flavors; statuses split sharply.

(a) Inference-time chaining (oracle and model restart, frozen operator):
Status: DONE-EXACT — extensively characterized.
Gate A 128² (2026-06-11/12, scripts/chain_gate.py): frozen op100/op500,
oracle (GT) and model (own-field) restart at strides 4/8/16/32,
held-out Re500[200:300] n=100.
  op100 oracle stride=4:  late 0.243 vs one-shot 0.678 (−0.424, 100% win)
  op100 oracle stride=16: late 0.361 (−0.310, 100% win)
  op100 model ALL strides: late worse (stride=32 late 0.711, 11% win)
  op500 oracle stride=4:  late 0.237 vs one-shot 0.473 (−0.227, 100% win)
  op500 model stride=32:  late 0.449 (−0.020, 71% win) — only stride
    where model own-field chaining is net-positive for op500
Chain gate @256² (2026-06-18, scripts/chain_gate_256.py): fno16 oracle
stride=16/24 late 0.230 vs one-shot 0.571 (−0.328, 100% win, n=40).
fno16 model stride=16: late 0.525 (80% win, Δ̄ −0.048).
Closure gate / kc sweep (2026-06-19): recoverable signal entirely in
k≤7 — lo_oracle ≈ full oracle (late 0.230); hi_oracle ≈ model (late
0.521). Solver-closure gate: wrong high-k causes only 0.06–0.12 low-k
divergence ≪ operator wall 0.47.

(b) Training-time differentiable window-chaining (backprop through the
handoff, using window-1 prediction as IC for window-2 during training):
Status: NOT DONE. Neither flavor of training-time chaining has been
implemented. This is the synthesis's primary unrun actionable lever.

---

## §5 — Inference-Time Corrections

### 5.1 Negative-Sobolev a-posteriori fine-tune (Spectral-Refiner)
Status: NOT DONE.
No local experiment fine-tunes a trained operator with the H⁻¹ residual
loss using an auto-diff solver. Synthesis correctly marks this as
breaking one-shot inference (per-instance, iterative, solver-assisted).

### 5.2 Leray / Helmholtz-Hodge divergence-free projection
Status: NOT DONE.
No local experiment projects velocity outputs onto the divergence-free
subspace. Synthesis correctly notes near-redundant for the vorticity-form
model (incompressibility already implicit). Correctly excluded.

### 5.3 Conserved-quantity correction (DPOTc)
Two components; statuses differ.

(a) Per-step magnitude scaling/shift on model outputs (the DPOTc
correction — enforcing enstrophy/circulation conservation post-hoc):
Status: NOT DONE. No correction mechanism was implemented.

(b) Moment-identifiability diagnostic (2026-06-14,
scripts/ic_identifiability.py):
Status: DONE-DEVIATED (diagnostic only; correction NOT implemented).
Setup: per-instance vorticity-moment violation |M_n(pred_late)−M_n(IC)|
for n=1,2,3,4, correlated with late k≤7 relL2, frozen op{100,500} +
adapted op100→Re500, held-out Re500[200:300] n=100.
Results: n=1 |gtdrift|≈4e-8 (circulation conserved ✓). n=2,3,4:
|viol|≫|gtdrift|; Pearson r=0.44–0.77 for n≥2 (all ops, significant).
Confirms which moments are violated and that violations track late error.
The DPOTc correction mechanism was NOT implemented.

### 5.4 Newton-step residual correction
Status: NOT DONE.
No local experiment applies a Newton step δu=−[∂_u R]⁻¹R(u_FNO).
Synthesis correctly notes the residual is non-identifying for late-wall
error (confirmed in E_rangeN: residual ↓, late err flat). Correctly
excluded.

---

## §6 — Banked Local Diagnostics (synthesis §6 cross-check)

All six items confirmed in tta-results.md:

- Oracle-restart gate: fno16 256² oracle stride=16/24 late 0.230 vs
  one-shot 0.571 (~57% reduction, 2026-06-18). op500 128² oracle
  stride=16: late 0.283 vs one-shot 0.473 (2026-06-11). Both confirmed.
  The synthesis notes the source document was inconsistent on whether
  the gate is Re=100 or Re=500; the banked result is Re500 at both
  resolutions.

- Chaos-spread gate: growth ratio 1.32–1.41 at Re500 (Gate 1, 2026-06-17);
  phase spread D_late/D_early = 1.28–1.42 (Gate 2, 2026-06-18). σ_op≪σ_phys
  confirmed; cov_op=4.6–24.6. The synthesis's "λ~0.2–0.3 at Re=100" is an
  informal characterisation of the growth rate, not a computed Lyapunov
  exponent (see §1.1 for that distinction).

- Dead levers. The synthesis §6 states "<0.006" for spectral band loss;
  this refers to the confirmed MAGNITUDE-BAND levers only (band-equalize
  arms all above baseline; energy-add −0.005). The PHASE-BAND (angle-weight)
  arms are separately unconfirmed (see §3.4(b)) — not part of the §6 dead-
  lever count. Confirmed dead:
  (1) Capacity: n_modes 8→16, h 64→128: late k≤7 unchanged or worse
      (capacity gate 2026-06-17; horizon-vs-capacity 2026-06-18).
  (2) 16×16 grid shrink: wall survives at 16² (FNO·spectral late 0.506
      ≈ 128² op100 wall 0.543; 2026-06-24). Wall confirmed intrinsic.
  (3) Spectral band loss (magnitude): energy-add −0.005 (confirmed arm
      rai1ohu9). Band-equalize: all arms above baseline. Dead.
      Phase-angle arms: UNCONFIRMED, open — see §3.4(b).
  (4) Label-free temporal weighting (additive form): max improvement
      <0.006, non-monotone (2026-06-22). Dead for the additive form.
      The steep pure-power (t/T)^α form remains open — see §3.8.
  (5) Late-window specialist: W2–W5 both T=5 and T=10 worse than
      baseline (latewin_eval 2026-06-24). Dead.
  (6) Self-chaining own field: model own-field chaining at 128² op100
      all strides above baseline; op500 only stride=32 marginally
      positive (Δ̄ −0.020, 71% win) — effectively dead (2026-06-11/12).

---

## §7 — Examined and Excluded (synthesis §7 cross-check)

None of the methods in this section were run locally. All confirmed NOT
DONE, consistent with their scope exclusion.

- AFNO (arXiv:2605.25413): not run. Latent-space flow-matching, not a
  deterministic single-shot operator.
- Late Fusion Neural Operators (arXiv:2604.16721): not run. "One-shot"
  refers to parameter-regime extrapolation, not late-time temporal
  prediction.
- Scheduled/causal temporal loss weighting (Coker et al.; Wang et al.
  arXiv:2203.07404): not run as these specific AR methods. The single-
  shot reinterpretation survived as §3.8 and was tested (see §3.8 above).
- TNO temporal bundling (arXiv:2504.20249): not run. Benefit is zero for
  a single-shot operator by construction.
- Generative/probabilistic refiners (PDE-Refiner, adv-NO, Gen4Turb):
  not run. Excluded by deterministic scope.
- Spectral-bias-specific levers (PENO arXiv:2406.04367; High-Frequency
  Scaling arXiv:2503.13695): not run. Tier-2, spectral-target not
  pointwise-relL2-validated on KF. Conceptually within §3.4/§2.6 but
  not separately verified.
- Diagnostics, not levers (CORAL in/out-horizon cliff; RFNO ν-scaling):
  not run as levers. RFNO ν-scaling folded into §1.5 as context.
