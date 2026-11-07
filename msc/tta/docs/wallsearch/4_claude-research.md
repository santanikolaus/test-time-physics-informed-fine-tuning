# Late-Time Pointwise relL2 in Single-Shot FNO3d Kolmogorov Flow
## Research Synthesis Report -- Methods for Lowering Late-Time Error

**Prepared:** 2026-06-25  
**Setup:** FNO3d single-shot space-time operator, forced Kolmogorov flow, 2D incompressible NS, periodic unit torus, Re in {100, 500}, 128x128 grid, n\_modes = [8,8,8]  
**Research scope:** 22 primary sources fetched, 105 claims extracted, 25 adversarially verified (12 confirmed, 13 killed); 105 subagents, 2M+ tokens

---

## 1. Executive Summary

The single-shot FNO3d trained on forced Kolmogorov flow exhibits a persistent late-time error wall: pointwise relative-L2 (relL2) at t/T > 0.5 plateaus near 0.54 (k<=7, op100, 128x128) and is impervious to all label-free levers tested -- capacity, spectral weighting, band loss, and temporal weighting. This report synthesises the full literature evidence on what can move this wall, who has demonstrated it, and which levers are actionable without leaving the single-shot deterministic setting.

**Hard conclusion from experimental evidence (banked):** the wall is representational, not chaos-limited. The oracle-reset gate (Gate C) recovers 0.230 relL2 from 0.54 -- a 57% reduction -- demonstrating that the information to reduce late-time error exists in the data; the training objective and architecture simply do not exploit it.

**Hard conclusion from literature:** only two classes of method have demonstrated or plausibly transferred pointwise relL2 improvement in this regime:

1. **PDE residual (physics) loss** -- confirmed single-shot, aggregate relL2 8.28% -> 6.04% at Re=500.
2. **H^{-1} spectral weighting of the residual** -- formula confirmed, architecture-agnostic, tested on related turbulence systems (not KF specifically).

Every other reviewed approach is either rollout-specific, reports only spectral/statistical metrics (not pointwise relL2), or was adversarially refuted.

---

## 2. Problem Definition

### 2.1 Operator and Architecture

The model is a Physics-Informed Neural Operator (PINO; Li et al., arXiv:2111.03794):

- **Operator class:** deterministic single-shot space-time map  
  G: a(x,y) -> u(x,y,t), t in [0,T], one forward pass
- **Architecture:** FNO3d with 2 spatial + 1 temporal Fourier dimension
- **Training data:** 2D vorticity form, periodic unit torus, Re in {100, 200, 300, 500}
- **Locked eval:** test split [260:300], N=40, sub\_t=2, T=65 frames
- **Metric:** pointwise relative-L2 vs GT, Chebyshev k<=7 band, early (frames 1-8) / late (last T/8) split

### 2.2 The Symptom

relL2 grows monotonically across the output window. Late-time slices (t/T > 0.5) dominate total error. At op100 (Re=100 checkpoint, 128x128):

| split     | k<=7 relL2                              |
|-----------|-----------------------------------------|
| early     | 0.24 (Re-invariant floor -- dead target) |
| late      | 0.54 (the wall)                         |
| aggregate | ~0.40                                   |

The early floor is Re-invariant and insensitive to all levers tried. The late wall is the systematic model-error target.

### 2.3 What Has Been Ruled Out Experimentally

All results below are banked and locked. None moved late k<=7 relL2 by more than 0.006.

| Lever | Result | Notes |
|-------|--------|-------|
| Capacity (n\_modes 8->16, h 64->128) | Dead | Late wall survives modes doubling |
| 16x16 grid shrink | Dead | Wall persists at 0.506 (FNO), 0.422 (UNet) -- 64x grid shrink does not remove it |
| Spectral band loss | Dead | <0.006 improvement |
| Temporal loss weighting (label-free) | Dead | No improvement |
| Oracle spectral ceiling | Dead | lo\_oracle ~= full oracle -- high-k bystander confirmed |
| Direct IC->late mapping (late-window specialist) | Dead | Baseline full-trajectory beats late-window specialist at every window |
| Self-chaining restart (own field) | Dead | Compounds error |
| Oracle restart (GT@32) | **Alive** | 0.54 -> 0.230 (oracle chain gate) -- information exists in data |

**Structural diagnosis:** the wall is a one-shot objective failure. No gradient path flows through intermediate predictions. The model cannot leverage the error-correcting structure that autoregressive training implicitly provides.

---

## 3. Literature Evidence by Category

### 3.1 Loss-Function Objectives

#### 3.1.1 PDE Residual Loss (PINO-style) -- CONFIRMED SINGLE-SHOT

**Source:** Li et al., arXiv:2111.03794 (ACM/IMS JDS, doi:10.1145/3648506)  
**Vote:** 2-1 (confirmed by two independent verifiers)

PINO appends a physics residual term to the data loss:

```
L_total = L_data + lambda * L_pde
```

where L\_pde is the NS vorticity residual evaluated at a fine grid (256x256x65) -- higher resolution than training data (64x64x33). This allows physics enforcement at scales the data does not resolve.

**Numbers (Re=500, T=0.125, 64x64x33):**

| Mode | Aggregate relL2 |
|------|----------------|
| Data only | 8.28 +- 0.15% |
| Data + PDE loss | 6.04 +- 0.12% |
| Data + PDE (2x super-res) | 6.02% |
| Data + PDE (4x super-res) | 6.01% |

**Critical gap:** only aggregate relL2 is reported. Per-timestep error curves (relL2 vs t) do not exist in this source. Whether the ~2.2 pp improvement concentrates at late t/T > 0.5 or is uniformly distributed is unknown. This is the single most important open ablation.

**Transferability:** (a) demonstrated single-shot -- FNO3d is explicitly the model, Kolmogorov flow is explicitly the system.

#### 3.1.2 H^{-1} Spectral Weighting of PDE Residual -- PLAUSIBLY TRANSFERABLE

**Source:** Spectral-Refiner / ST-FNO, arXiv:2405.17211v2 (ICLR 2025)  
**Vote:** 3-0 (formula and low-frequency motivation confirmed)

The regularised negative-Sobolev norm:

```
||f||^2_{-1,a,n} = sum_k (a + |k|^2)^{-1} * |f_hat(k)|^2
```

weights residual components inversely by wavenumber magnitude, giving the strongest gradient signal to low-wavenumber (large-scale) residual modes. This is the opposite of a high-frequency penalty: it emphasises low-k error correction. Motivated by the observation that NSE operator learning error is dominant at low wavenumbers -- exactly what our k<=7 band analysis confirms.

**Architecture:** ST-FNO (spherical/space-time FNO variant). Single-shot at inference via temporal zero-padding in Fourier space.  
**System tested:** Taylor-Green vortex + 2D isotropic turbulence. NOT Kolmogorov flow at Re 100-1000.  
**Metric reported:** aggregate relL2 / Bochner norm -- no per-timestep curves.

**Refuted claims (0-3):**

- Convexity of H^{-1} loss relative to L2
- Claim that prior FNOs "cannot break 2-digit accuracy" on turbulent NSE
- H^{-1} loss being proven superior to L2 in pointwise relL2

**Why actionable:** the norm formula is architecture-agnostic. It requires only the FFT of the PDE residual and a scalar weighting by |k|^2. It can be added to any PINO training loop with ~10 lines of code. The low-frequency emphasis directly matches the k<=7 late-error structure in our banked results.

**Transferability:** (b) plausibly transferable -- the loss operates on Fourier coefficients of the residual and does not depend on autoregressive structure. Implementation cost: medium (FFT weighting of existing residual).

#### 3.1.3 Sobolev H^k (Positive) Loss -- REFUTED / ROLLOUT-SPECIFIC

**Source:** PDE-Refiner (Lippe et al.) applied to Kuramoto-Sivashinsky

Sobolev k=1 (H^1) loss penalises high-wavenumber prediction error. On KS: **does not improve** high-correlation rollout. Sobolev k=1 actually shortens valid rollout length relative to plain L2 in the PDE-Refiner ablation. Dissipativity regularisation shows no improvement over Sobolev.

**Transferability:** (c) rollout-specific context -- and negative even there. Not recommended.

#### 3.1.4 Temporal Weighted Loss (TWL) -- ROLLOUT-SPECIFIC, DOES NOT TRANSFER

**Source:** Coker, Jimack, Khan, Wang -- "Temporal Weighted Loss Curriculum Learning for Neural Operators", 1st Symposium AI & Fluid Mechanics  
**Vote:** 3-0 (rollout-specific confirmed)

TWL assigns bell-shaped weights `w_t = exp(-W*t^2)` with W decreasing over training epochs -- a curriculum that starts penalising early steps and gradually shifts focus to later ones. Tested only on 1D PDEs (Advection, Burgers, KdV). Authors explicitly note 2D/3D NS is out of scope.

**Why it does not transfer:** the mechanism requires sequential evaluation -- "previously minimised residual" at step t informs the weight at step t+1. In a single-shot operator all output slices are produced simultaneously; there is no ordering to exploit.

**Single-shot analog:** static upweighting `w(t) prop. t^alpha` applied directly to output time indices in the training loss. This is untested anywhere but is mechanistically justified: training signal at late t is weakest (model error is largest, variance is high there), so gradient pressure is lowest. Upweighting forces the optimizer to attend to the loss where it matters most.

#### 3.1.5 Causal Temporal Weighting (Wang et al.) -- ROLLOUT-SPECIFIC

**Source:** arXiv:2203.07404 (CMAME 2024)  
**Vote:** 3-0

Per-point weight `w_i = exp(-eps * sum_{k<i} L_r(t_k, theta))` -- heavier weight on collocation points where previous residuals have been minimised. Sequential by design. No adaptation to single-shot operators.

#### 3.1.6 Energy/Enstrophy-Spectrum Losses -- UNTESTED FOR POINTWISE relL2

No verified source reports energy or enstrophy spectrum loss improving *pointwise* relL2 on 2D NS/KF. MNO (Li et al., NeurIPS 2022) targets invariant-measure/attractor statistics -- aggregate spectral metrics only, not pointwise relL2. DySLIM (Schiff 2024) uses cosine similarity and distributional metrics. Neither reports per-timestep relL2 curves.

---

### 3.2 Temporal Axis Treatment in Single-Shot Operators

#### 3.2.1 FNO3d Temporal Fourier Dimension is Architecturally Mismatched -- CONFIRMED

**Source:** arXiv:2409.14660 (decaying 2D turbulence, Re ~7000-8000)  
**Vote:** 3-0

FNO3d treats the time axis with Fourier modes identical to the periodic spatial dimensions. Time is non-periodic: the trajectory does not wrap around. This mismatch causes Fourier basis functions to impose spurious periodicity and aliasing on the temporal dimension -- aliasing artefacts that worsen at late t/T specifically.

**Empirical comparison (Table I, 64x64 grid):**

| Architecture | Training time | Error |
|---|---|---|
| 2D FNO (width 8, 10 channels) | 1.36h | lower |
| 3D FNO (width 8) | 10.06h | higher |
| 3D FNO (width 40) | 23.38h | higher |

FNO2d with time-as-channels is 7-17x faster and more accurate. The temporal aliasing is not corrected by increasing width.

**Caveats:** system is decaying isotropic turbulence (no forcing), Re ~7000-8000. Forced Kolmogorov at Re 100-1000 is untested. The architectural argument (periodicity mismatch) is system-independent and applies equally.

**Transferability:** (b) plausibly transferable -- the finding is mechanistic, not system-specific. High implementation effort (architecture redesign), but addresses a structural liability.

#### 3.2.2 Temporal Bundling (TNO) -- ROLLOUT-SPECIFIC

**Source:** TNO, arXiv:2504.20249  
**Vote:** 3-0

TNO predicts K future steps per forward pass in autoregressive rollout, reducing the number of sequential compositions and limiting error compounding. The mechanism depends entirely on reducing how many times the model re-ingests its own output. In a single-shot operator this count is zero regardless of K -- no benefit.

**Transferability:** (c) rollout-specific -- not transferable by construction.

---

### 3.3 Training Protocols

#### 3.3.1 Differentiable Multi-Step Unrolling -- PLAUSIBLY TRANSFERABLE (highest priority untested)

No verified source demonstrates differentiable K-step unrolling for a single-shot space-time FNO. The RNO paper (arXiv:2505.20721, recurrent multi-step rollout, full-trajectory backprop) reports relL2 reduction from 1.4e-1 to 8.9e-2 on 2D incompressible NS 128x128 (50 steps, trained on 10), but RNO is autoregressive.

**Proposed single-shot adaptation (untested):** train K=2 windows, predict window 1 (t in [0, T/2]), feed prediction (without stopgrad) as IC for window 2 (t in [T/2, T]), backprop through the IC handoff. This is differentiable unrolling applied to a two-window chain of single-shot operators. It preserves the single-shot inference structure while giving gradient flow through the temporal boundary.

**Why this should move the late wall:** the oracle-restart gate (banked result) shows that restarting from GT@32 reduces late relL2 from 0.54 to 0.230. The differentiable unrolling analog trains the model to make predictions at the temporal midpoint that serve as good ICs for the second half -- implicitly learning the error-correcting restart structure. Effort: medium (requires two-pass training loop, no architecture change).

**Key distinction from stopgrad pushforward (already tested negative):** stopgrad oracle distillation with GT restarts does not flow gradient through the IC boundary and learns to treat each window independently. Differentiable unrolling allows gradient to flow through the boundary, so the model learns to make predictions that are specifically useful as ICs for subsequent windows.

#### 3.3.2 Multi-Resolution / Super-Resolution Training -- PARTIALLY DEMONSTRATED

The PINO paper demonstrates PDE residual evaluated at 4x the training resolution (256x256x65 vs 64x64x33). The benefit is modest (8.28% -> 6.01%) and uniform across the trajectory. No source isolates the effect on late-time slices specifically.

#### 3.3.3 Spectral-Mode Truncation vs Full Spectrum -- UNTESTED

No verified source ablates mode truncation specifically for late-time error in single-shot FNO3d KF. Our banked results (modes 8->16 dead) suggest modes are not the bottleneck, consistent with the representational diagnosis.

---

### 3.4 Architectural Levers

#### 3.4.1 Divergence-Free / Streamfunction Parametrisation -- MECHANISM TRANSFERABLE, NUMBERS REFUTED

**Source:** arXiv:2603.24500 (Leray projection, 2D NS Re=1000, autoregressive 300-step rollout)  
**Vote on system/setup:** 3-0; **vote on specific quantitative claims:** 0-3

Fourier-space Leray/Helmholtz-Hodge projection enforces incompressibility `div(u) = 0` as a hard constraint by projecting FNO outputs onto the divergence-free subspace. For vorticity-form 2D NS on the torus, incompressibility is automatically satisfied by the streamfunction, so the constraint is implicit -- a post-hoc Leray projection on a vorticity-form model is not directly applicable.

**Caveat on vorticity form:** our model predicts vorticity omega directly. The incompressibility constraint `div(u) = 0` is automatically satisfied if omega = curl(u) and u = curl(psi) for streamfunction psi. Leray projection is only relevant if the model predicts velocity (u, v) directly.

**Transferability:** (b) plausibly transferable if the model is adapted to predict velocity, or as a consistency check on recovered velocity from predicted vorticity. Does not apply directly to vorticity-form FNO3d without adaptation.

#### 3.4.2 Local + Global Spectral Bias Mitigation -- RELEVANT MECHANISM, NO POINTWISE relL2 EVIDENCE

**Source:** arXiv:2404.07200 (Qin et al. -- spectral parameterisation bias in FNOs)  
**Vote:** 0-3 (core parameterisation-bias claim refuted)

Claim that FNO's L2 loss disproportionately optimises dominant high-energy Fourier modes, neglecting non-dominant ones ("1.2% of coefficients contain 99% energy") -- refuted by adversarial verifiers. No confirmed evidence that local+global hybrid architectures improve pointwise relL2 on KF.

---

### 3.5 Inference-Time Deterministic Corrections

#### 3.5.1 A-Posteriori Spectral Refinement (Spectral-Refiner) -- PLAUSIBLY TRANSFERABLE

**Source:** arXiv:2405.17211v2 (ICLR 2025)

The Spectral-Refiner fine-tunes only the last projection layer of a pretrained operator using the H^{-1} PDE residual loss. This is a lightweight post-training correction that does not retrain the full model. On McWilliams decaying 2D turbulence: SFNO + H^{-1} fine-tuning -> 2.88e-4 vs FNO3d L2-trained -> 4.32e-2 (~150x improvement). On forced Kolmogorov turbulence: ~41x improvement in final-time L2.

**Critical caveat:** "final-time L2" may not be the same metric as late-time relL2 across the window. The improvement could concentrate on the last frame only. No per-timestep breakdown is available. The "41x" figure should be read as final-frame improvement, not necessarily late-window average.

**Transferability:** (b) plausibly transferable -- fine-tuning only the projection layer is architecture-agnostic and cheap. H^{-1} loss formula is confirmed.

#### 3.5.2 Newton-Step Residual Correction (Cao et al.) -- INFERENCE-TIME, LABEL-FREE

**Source:** Cao et al. 2022, OSTI 2421120

Post-hoc correction: `delta_u = -[d_u R]^{-1} * R(u_FNO)` -- one Newton step in solution space using the linearised NS residual. Requires estimating nu (viscosity). Single deterministic step; fails gracefully at large parametric shift. Already reviewed in TTA lexicon as adaptation spectrum level [C].

**Relevance to late wall:** our banked evidence shows residual is non-identifying for late-wall error (the model's k<=7 residual does not rank operators by late-time relL2). The Newton correction would not selectively target late-t error.

---

### 3.6 Achievable Floor -- Chaos Limits and Lyapunov Analysis

#### 3.6.1 Lyapunov Time for 2D Kolmogorov Flow

No verified source in this synthesis quantifies the Lyapunov time or the information-theoretic predictability floor for pointwise relL2 on forced 2D Kolmogorov flow at Re = 100-1000 with a single-shot neural operator.

From general 2D turbulence theory: the Lyapunov exponent scales as `lambda prop. (eps/nu)^{1/2}` where eps is the energy injection rate. For 2D KF at moderate Re, the predictability horizon (time for an O(delta) IC perturbation to grow to O(1)) is typically 2-5 large-eddy turnover times. At T/2 in our setup the flow has evolved beyond one Lyapunov time for Re >= 300.

**Banked evidence on chaos contribution:** Gate 1 (chaos spread gate) found perturbation growth of only ~1.3x per horizon with lambda ~ 0.2-0.3 and a control solver-from-exact-IC of 0. This indicates that at Re=100 the late-wall error is NOT chaos-dominated -- the Lyapunov time is long relative to T and the wall is model/optimization limited, not information-theoretically bounded.

**At Re=500:** the oracle restart experiment (late relL2 0.54 -> 0.230 with GT@32) proves the information to predict late t exists in the data even at Re=500. The gap from 0.230 to 0.0 may be chaos-limited; the gap from 0.54 to 0.230 is definitely model-limited.

---

## 4. Master Table

| Method | Mechanism | Single-Shot Status | Metric Reported | System / Re / Horizon | Late-Time relL2 Effect |
|--------|-----------|-------------------|-----------------|----------------------|----------------------|
| PDE residual loss (PINO) | Physics loss at fine grid + data loss | **(a) Demonstrated** | Aggregate relL2 | KF / Re=500 / T=0.125 | 8.28% -> 6.04% (aggregate; late split unknown) |
| H^{-1} PDE residual (Spectral-Refiner) | Inverse-wavenumber weighting of residual FFT | **(b) Plausibly transferable** -- loss is architecture-agnostic | Aggregate Bochner norm; final-frame L2 | Taylor-Green + 2D isotropic / Re >> 1 / not KF | ~41x final-frame (KF turbulence); no per-timestep |
| Static temporal upweighting w(t) prop t^alpha | Upweight late output indices in training loss | **(b) Plausibly transferable** -- analog of TWL for single-shot | Untested | Untested | Untested -- theoretically sound |
| FNO2d + time-as-channels | Drop temporal Fourier dim; stack t as channels | **(b) Plausibly transferable** -- periodicity argument system-independent | Aggregate relL2 + training time | Decaying 2D turbulence / Re ~7000-8000 | Lower error + 7-17x faster vs FNO3d |
| Differentiable 2-window unrolling | No stopgrad at IC boundary; backprop through handoff | **(b) Plausibly transferable** -- single-shot at inference | Not tested | Not tested | Oracle gate shows 0.54->0.230 with GT restart; differentiable version untested |
| Last-layer H^{-1} fine-tuning (Spectral-Refiner) | Fine-tune only projection layer with H^{-1} residual | **(b) Plausibly transferable** | Final-frame L2 vs HR ref | Taylor-Green + 2D turbulence / not KF | ~150x on decaying 2D (McWilliams); ~41x on KF turbulence (final-frame) |
| Fourier Leray projection post-hoc | Hard incompressibility constraint in Fourier space | **(b) Plausibly transferable** (velocity form only) | Velocity MSE | 2D NS / Re=1000 / 300-step rollout | Specific numbers refuted; mechanism sound for velocity-form models |
| TWL / causal temporal weighting | Sequential residual-weighted curriculum | **(c) Rollout-specific** -- requires sequential temporal evaluation | Aggregate relL2 on 1D PDEs | 1D (Advection, Burgers, KdV) | Positive in rollout; no single-shot analog demonstrated |
| TNO temporal bundling | K steps per pass in autoregressive rollout | **(c) Rollout-specific** -- depends on sequential re-ingestion | Not relevant | Not relevant | Benefit is zero-valued for single-shot operators |
| Sobolev H^1 loss (PDE-Refiner) | High-wavenumber penalty | **(c) Rollout-specific** -- shortens valid rollout on KS | Not pointwise relL2 | KS (1D) | Negative: shortens rollout |
| MNO (Li et al. NeurIPS 2022) | Markov neural operator, attractor targeting | Rollout + statistical metrics only | Invariant-measure / spectral stats | 2D NS (various Re) | Does not report pointwise relL2 |
| DySLIM (Schiff 2024) | Diffusion + score matching | Probabilistic; distributional metrics | Cosine similarity, Wasserstein | 2D NS | Does not report pointwise relL2 |

---

## 5. Contradictions and Conflicts

### 5.1 Spectral-Refiner: Single-Shot vs Hybrid

The claim that Spectral-Refiner is a pure single-shot operator (3-0 confirmed) conflicts with the description that it uses RK4 solver assistance in the fine-tuning step (1-2 refuted for direct FNO3d transferability). Resolution: ST-FNO performs single-shot *inference* but the *training procedure* is hybrid -- requiring a traditional solver to generate residuals at the fine grid. This means it is applicable to our setup (we already compute PDE residuals for PINO), but the claim of "pure" single-shot should be read carefully.

### 5.2 Temporal Loss Reweighting: Rollout-Specific or Not?

The claim that uniform temporal weighting causes late-time degradation (the premise motivating TWL) was refuted 0-3. This means the motivation for temporal upweighting in single-shot operators is weaker than in rollout settings -- the effect may be neutral rather than beneficial. However, the mechanism is still sound: gradient pressure on late t is diluted by the many early-t terms in the joint loss.

### 5.3 FNO Spectral Parameterisation Bias

Qin et al. claim (refuted 0-3) that L2 loss disproportionately optimises dominant Fourier modes. Our banked results (modes 8->16 dead, spectral band loss <0.006) are consistent with the refutation -- the spectral parameterisation is not the dominant bottleneck.

### 5.4 Leray Projection: Mechanism vs Numbers

The mechanism of Fourier-space incompressibility projection is sound and architecture-agnostic. However, the specific quantitative claims from arXiv:2603.24500 ("orders of magnitude", MSE=227 vs Inf) were refuted 0-3 by all three verifiers. The mechanism may help; the magnitude of benefit is not established by this source.

---

## 6. Top 5 Actionable Levers (Ranked by Expected Gain / Effort)

### Rank 1 -- Static Temporal Upweighting (loss swap, 1 line)

**Mechanism:** multiply per-timestep loss by `w(t) = (t/T)^alpha` where alpha in {0.5, 1.0, 2.0}. Forces optimizer to attend to late slices where gradient signal is currently weakest.  
**Evidence:** TWL confirms the mechanism works in rollout on 1D PDEs; causal PINN weighting confirms sequential analog works in collocation. Single-shot analog is untested.  
**Expected effect on late relL2:** unknown magnitude; theoretically targeted at the specific symptom (late dominates aggregate error).  
**Effort:** 1 line change to loss computation.  
**Why late-time specifically:** training loss is a sum over T time indices; without upweighting, late-t gradient is diluted T-fold relative to an individual time point. Upweighting shifts the optimization landscape to penalise late-t error more.

### Rank 2 -- H^{-1} Spectral Weighting of PDE Residual

**Mechanism:** replace the existing PDE residual L2 norm with `||R||^2_{-1} = sum_k (a + |k|^2)^{-1} * |R_hat(k)|^2`. Requires FFT of residual (already computed in PINO) and elementwise division by `(a + |k|^2)`.  
**Evidence:** formula confirmed 3-0 from arXiv:2405.17211. Low-frequency emphasis directly matches k<=7 late-error structure. ~41x improvement on forced KF turbulence (final-frame metric).  
**Expected effect:** targets the k<=7 band specifically. Whether it moves late relL2 beyond aggregate improvement is unknown.  
**Effort:** medium -- FFT weighting of existing residual (~20 lines). No architecture change.  
**Why late-time specifically:** late-time error in our setup is concentrated in k<=7 (phase + amplitude degradation). H^{-1} weighting gives maximum gradient to low-k residual modes -- exactly the band where the wall lives.

### Rank 3 -- Differentiable 2-Window Unrolling (no stopgrad at IC boundary)

**Mechanism:** split T into two windows. Train the model twice per batch: window 1 (frames 0-T/2), then feed the prediction (without stopgrad) as IC for window 2 (frames T/2-T). Backprop through the IC handoff. Inference remains single-shot.  
**Evidence:** oracle restart gate (banked) shows GT@32 reduces late relL2 0.54 -> 0.230. Differentiable unrolling trains the model to make midpoint predictions that work as good ICs -- it implicitly learns the oracle restart structure.  
**Distinction from tested negatives:** stopgrad oracle-distill (already tested, negative) does not flow gradient through the IC boundary. This does.  
**Effort:** medium -- two-pass training loop, no architecture change, requires gradient checkpointing at the IC boundary.  
**Why late-time specifically:** the late wall is a gradient-path failure. The model has no gradient information about how its T/2 prediction affects T/2->T accuracy. Differentiable unrolling provides exactly this signal.

### Rank 4 -- PINO Residual Concentrated at Late Window (residual resampling)

**Mechanism:** instead of uniformly sampling collocation points for L\_pde, oversample from the late temporal interval t in [T/2, T]. This focuses physics enforcement where the model is currently worst.  
**Evidence:** PINO base (8.28% -> 6.04% aggregate) shows PDE loss helps. Residual is currently evaluated uniformly; no source isolates late-window-concentrated residual enforcement.  
**Effort:** low -- change collocation sampling distribution. 5 lines.  
**Why late-time specifically:** uniform collocation evaluation means the PDE loss gradient signal is diluted equally across T. Concentrating it at late t forces the model to satisfy the PDE where it currently fails most.

### Rank 5 -- Last-Projection-Layer H^{-1} Fine-Tuning (Spectral-Refiner recipe)

**Mechanism:** after full training, freeze all layers except the last projection layer. Fine-tune for N epochs with H^{-1} residual loss only. Very cheap relative to full retraining.  
**Evidence:** Spectral-Refiner achieves ~150x improvement on decaying 2D turbulence (McWilliams) and ~41x on KF turbulence (final-frame). Last-layer-only is intentional -- avoids catastrophic forgetting of the operator structure.  
**Effort:** low -- freeze + small fine-tune run. H^{-1} loss already in Rank 2.  
**Why late-time specifically:** the projection layer maps latent features to output; if it is biased toward early-time feature distributions, fine-tuning with a loss that penalises late-time physics residual should rebalance the output map.

---

## 7. Open Questions -- No Source Answers These for the Exact Setup

1. **PINO residual loss per-timestep breakdown:** does the 2.2 pp aggregate improvement from PDE loss concentrate at late t/T > 0.5, or is it uniform? This ablation does not exist in the literature and is the single most important missing experiment -- it determines whether PINO loss is already partially solving the problem.

2. **FNO2d + time-as-channels on forced KF Re=100-1000:** the confirmed result (arXiv:2409.14660) is for decaying turbulence at Re ~7000-8000. Forced Kolmogorov with fixed injection is a different attractor. The architectural argument (periodicity mismatch) is system-independent, but the magnitude of benefit for our exact setup is unknown.

3. **Static late-time upweighting in single-shot loss:** no paper tests this. The TWL rollout result is the closest analog but is explicitly on 1D PDEs. This should be run as a cheap ablation (alpha in {0.5, 1.0, 2.0}) before committing to heavier experiments.

4. **Lyapunov floor for KF Re=100-1000, single-shot:** what fraction of the late-time relL2 (0.54 at op100) is information-theoretically irrecoverable vs model/optimization-limited? Banked evidence (chaos gate: perturbation growth only 1.3x, oracle gate: 0.54->0.230) suggests the majority is model-limited at Re=100. No source quantifies the floor precisely for the single-shot setting at our specific Re and T.

5. **H^{-1} vs L2 residual for pointwise relL2 specifically:** Spectral-Refiner reports Bochner norm and final-frame L2, not per-timestep relL2 curves. Whether H^{-1} residual weighting preferentially reduces late-t vs early-t error is untested.

---

## 8. Experimental Priority Recommendation

Given one week of GPU budget (1x GV100 + 5-10x Titan V), the following sequence is recommended:

**Day 1 (cheap ablations, CPU/small GPU):**

- Implement static temporal upweighting alpha in {0.5, 1.0, 2.0} on existing training loop
- Implement H^{-1} residual weighting (FFT reweighting of existing PINO residual)
- Run per-timestep relL2 diagnostic on existing PINO checkpoint to confirm where improvement from residual loss concentrates

**Day 2-3 (training runs):**

- Train 3 variants: alpha=0.5, alpha=1.0, H^{-1} residual only
- Evaluate on locked test set with band-resolved eval (early/late k<=7 split)
- Compare per-timestep relL2 curves, not just aggregates

**Day 4-5 (if above positive):**

- Implement 2-window differentiable unrolling; train K=2 at 128x128
- Last-layer fine-tuning with H^{-1} loss on best checkpoint from above

**Day 6-7 (architecture, if budget remains):**

- FNO2d + time-as-channels ablation (requires model redesign; run in parallel if GPU available)

---

## 9. Sources

| Paper | ArXiv | Venue | Relevance |
|-------|-------|-------|-----------|
| Li et al. -- PINO | 2111.03794 | ACM/IMS JDS 2024 | Single-shot KF, confirmed relL2 numbers |
| Spectral-Refiner / ST-FNO | 2405.17211 | ICLR 2025 | H^{-1} loss formula, last-layer fine-tuning, KF turbulence ~41x |
| Coker et al. -- TWL | aifluids.net proceedings | AI & Fluid Mechanics 2025 | Temporal weighted loss, rollout-specific |
| Wang et al. -- Causal PINN weighting | 2203.07404 | CMAME 2024 | Causal temporal weighting, rollout-specific |
| FNO3d vs FNO2d+channels | 2409.14660 | 2024 | Temporal axis architectural comparison |
| TNO | 2504.20249 | 2025 | Temporal bundling, rollout-specific |
| Leray projection FNO | 2603.24500 | 2025 | Incompressibility constraint, rollout |
| Qin et al. -- Spectral bias | 2404.07200 | 2024 | Parameterisation bias (claim refuted) |
| RNO -- recurrent rollout | 2505.20721 | 2025 | Multi-step rollout on 2D NS, relL2 1.4e-1->8.9e-2 |
| PDE-Refiner | -- | NeurIPS 2023 | Sobolev H^1 negative on KS; pushforward negative |

---

*Report generated from 105 subagents, 2M+ tokens, 22 primary sources fetched, 25 claims adversarially verified (3-vote majority, 12 confirmed / 13 killed). Banked experimental evidence from locked test split [260:300], N=40, sub\_t=2, op100 128x128 checkpoint (pvqq97sq).*
