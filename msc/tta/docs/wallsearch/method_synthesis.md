---
title: "Late-Time relL2 Wall — Method Synthesis"
date: 2026-06-25
---

**Setup.** Single-shot space-time operator G: a(x,y) → u(x,y,t) over t∈[0,T], one forward pass, forced 2D Kolmogorov flow (vorticity form), Re∈{100…1000}. Target: late-time (t/T>0.5) pointwise relative-L2. Constraint: one-shot *inference*; autoregressive *training* permitted.

Each entry: **source paper(s)** → **idea** → **reported evidence (verify on read)** → **scope for this setup**. Numbers are as reported by the originating work and must be confirmed against the paper before weighting.

---

## 1. The Wall — what limits late-time pointwise tracking

### 1.1 Lyapunov predictability horizon
- Papers: Nastac & Ihme, *Lyapunov exponent as a predictability metric*, Phys. Rev. Fluids 2017; Bae & Lozano-Durán, Caltech 2022; Atif et al., arXiv:2409.14660, 2024.
- Idea: chaotic error grows ~e^{λt}; the Lyapunov exponent scales λ∼Re^α, so the predictability horizon shrinks as Re rises. Atif et al. measure a max Lyapunov exponent Λ≈2.15, Lyapunov time T_L≈0.45 convective units, and deliberately restrict FNO predictions to t<T_L.
- Evidence: α≈0.46 (multifractal)–0.5 (Kolmogorov); Λ≈2.15, T_L≈0.45 (Atif, decaying turbulence Re~7–8k).
- Scope: diagnostic bound, not a method.

### 1.2 The 0.5·T_L completeness threshold
- Paper: *Prediction and control of 2D decaying turbulence using GANs*, J. Fluid Mech. 2023.
- Idea: predictions to lead time 0.5·T_L are "almost complete" (small scales recoverable from the input), whereas at T_L–2T_L they are "inevitably incomplete," because enstrophy cascades to the smallest scale within 0.5·T_L. This identifies the wall with the enstrophy-cascade decorrelation time.
- Evidence: qualitative completeness boundary at 0.5·T_L.
- Scope: marks where pointwise relL2 stops being a meaningful target; decaying turbulence, not forced KF. Load-bearing for the whole "wall" framing yet rests on a single decaying-turbulence GAN study (not independently re-verified) — treat its confidence accordingly.

### 1.3 Migration to invariant-measure metrics (the field's response)
- Papers: Li et al. (MNO), *Learning Dissipative Dynamics in Chaotic Systems*, arXiv:2106.06898, NeurIPS 2022; Schiff et al. (DySLIM), arXiv:2402.04467, ICML 2024.
- Idea: because exact trajectories are unpredictable under positive Lyapunov exponents, these works declare pointwise L2 the wrong target and evaluate distribution consistency on the attractor instead (Fourier/POD spectra, MELR, MELRw, covRMSE, Wasserstein-1, autocorrelation, TKE).
- Evidence: neither reports pointwise relL2 for KF at t/T>0.5; DySLIM-KF improves MELR/Wass1/TCM.
- Scope: explains the literature gap; metrics, not levers.

### 1.4 Aggregate score ≠ phase tracking
- Paper: SGNO, arXiv:2602.18801, 2026.
- Idea: a model can win an aggregate long-horizon score while losing phase accuracy early. SGNO's "stable step" is reached at ~step 13–14 despite a better aggregate GMean100.
- Scope: caution for interpreting any single summary number during weighting.

### 1.5 Spectral bias as the shared root cause of the floor
- Papers: Lippe et al. (PDE-Refiner), arXiv:2308.05732; Qin et al., arXiv:2404.07200 (parameterisation-bias variant, refuted); RFNO residual-based FNO (featool ref, doc 1_ src 34).
- Idea: MSE optimises high-energy low-k structure and neglects low-amplitude high-k modes, producing an oversmoothed high error floor even at small t — the mechanism connecting the time-axis (§2.1), the loss family (§3) and the wall. The floor is strongly Re/ν-dependent: RFNO/FNO report ~0.5–0.6% relL2 at ν=1e-3 rising to 12–14% at ν=1e-5, quantifying how the achievable floor worsens with Re.
- Scope: connective mechanism, not a lever; motivates the spectral losses (§3.4) and local-kernel architectures (§2.6). Qin et al.'s specific "1.2% of coefficients hold 99% energy" parameterisation claim was adversarially refuted, so spectral bias is real but its FNO-parameterisation story is contested.

---

## 2. Architecture

### 2.1 Time-as-channels vs temporal Fourier dimension
- Paper: Atif et al., arXiv:2409.14660, 2024.
- Idea: FNO3d runs an FFT over the non-periodic time axis, imposing spurious periodicity and aliasing that worsens toward the window edges (late t). An FNO2d with time stacked as channels removes the temporal FFT.
- Evidence: time-as-channels gives "significantly lower errors" than FNO3d (decaying 2D turbulence Re~7–8k); not tested on forced KF. Caveat: the 7–17× is a *training-time* ratio (1.36 h vs 10.06–23.38 h), and Atif applies the 2D-FNO *iteratively* (autoregressively) — so the accuracy comparison is partly AR-vs-single-shot, not a clean single-shot win.
- Scope: the architectural argument (temporal periodicity mismatch) applies directly to single-shot; the reported magnitude is confounded by the iterative deployment and should be re-measured single-shot on forced KF.

### 2.2 Periodic temporal padding + layernorm
- Paper: Cao et al. (Spectral-Refiner / ST-FNO3d), arXiv:2405.17211, ICLR 2025.
- Idea: keep the temporal Fourier treatment but pad the time dimension periodically in the lifting operator and replace the hard-coded global normalizer with layernorm, removing the wrap-around artefact.
- Evidence: identified as the central temporal defect; enabled the model's accuracy (no isolated ablation number reported).
- Scope: one-time architectural fix, zero inference cost. Competes with 2.1 (same defect, alternative fix — not additive).

### 2.3 Streamfunction parametrization
- Paper: SciML closure review, CWI ir.cwi.nl/pub/34879 (mechanism; classical).
- Idea: predict the streamfunction ψ and set u=∇×ψ so the velocity field is divergence-free by construction rather than by post-hoc projection.
- Scope: for the vorticity-form model incompressibility is already implicit; explicit ψ-parametrization matters only for recovered velocity. Architectural, one-shot-compatible.

### 2.4 Non-expansive spectral generator (ETD, ℜ(α)≤0)
- Paper: SGNO, arXiv:2602.18801, 2026.
- Idea: model the one-step map as an Exponential-Time-Differencing step in Fourier space with a diagonal generator whose real part is constrained ≤0 (so linear dynamics cannot amplify), plus a gated nonlinear path bounding high-frequency feedback.
- Evidence: median 74.8% GMean100 reduction across APEBench (per-task 13.6–92.9%, **but the 92.9% is disp1d 1-D dispersion, not KF**); on kolm2d the absolute error stays ≈0.558 (per docs 2_/3_; doc 1_ cites 0.586).
- Scope: autoregressive at inference — mechanism portable, operator class is not. Note the kolm2d datapoint is APEBench Kolmogorov at **Re=100** — exactly the target low-Re regime, making it the most transfer-relevant single number in the table despite the high absolute error (~0.558).

### 2.5 Spatial-basis / temporal-state separation (SSP)
- Paper: Lu, Yuan, Shi (SSP), arXiv:2605.10154, 2026.
- Idea: factorize analysis→propagation→synthesis; evolve only a compact, bounded set of latent Fourier modes via a frequency-conditioned linear backbone + nonlinear spectral closure, isolating temporal evolution from spatial reconstruction.
- Evidence: up to 48.9% relL2 reduction under long extrapolation; tested on Shallow-Water / Reaction-Diffusion / PDEBench-NS, **not KF** (the 48.9% is on SW or RD).
- Scope: autoregressive at inference; mechanism portable.

### 2.6 Local + global hybrid kernels
- Papers: Liu-Schiaffini et al., *Neural Operators with Localized Integral and Differential Kernels*, arXiv:2402.16845, ICML 2024; Wu et al. (DINO), arXiv:2509.21196, 2026.
- Idea: augment FNO's global spectral convolution with learned local differential/integral kernels to restore high-frequency/local structure.
- Evidence: +local kernels cut 5-step KF relL2 0.2360→0.1956 (~17%, Re=5000); DINO reports step-99 KF relL2 0.5876 vs FNO 3.1284 (single seed, Re unstated). The related parameterisation-bias rationale (Qin et al., arXiv:2404.07200) was adversarially refuted.
- Scope: autoregressive evaluation in sources.

---

## 3. Loss objectives (training-time)

### 3.1 PDE-residual physics loss (PINO)
- Paper: Li et al. (PINO), arXiv:2111.03794, ACM/IMS JDS 2024.
- Idea: add λ·‖∂_t u − R(u)‖² over [0,T], evaluable at a finer grid than the data (256²×65 vs 64²×33) to enforce physics at unresolved scales.
- Evidence: ~7% lower aggregate relL2 (paper); 8.28%→6.04% at Re=500 in local banked runs. **No per-timestep breakdown exists** — whether the gain concentrates late is the key open ablation.
- Scope: confirmed single-shot; the user's baseline.

### 3.2 Negative-Sobolev / H⁻¹ residual weighting
- Paper: Cao et al. (Spectral-Refiner), arXiv:2405.17211, ICLR 2025.
- Idea: weight the residual norm by (a+|k|²)⁻¹, an inverse-frequency emphasis on low-k (energy-containing) error — the opposite of a high-frequency penalty. (This is *Spectral-Refiner*, deterministic; not to be confused with *PDE-Refiner*, arXiv:2308.05732, a diffusion/probabilistic refiner that is out of scope and used here only for ablation/disambiguation evidence.)
- Evidence (verified against the paper, 2026-06-25): applied as a per-instance post-hoc fine-tune, final-time L2 drops ~10⁻²→2.88×10⁻⁴ on the decaying McWilliams case (Table 3) and 2.08×10⁻²→2.82×10⁻⁴ (L² FT)/3.16×10⁻⁴ (H⁻¹ FT) on the *forced* case (Table 2, ~66–74×). A forced-turbulence result therefore **does exist** (correcting docs 1_/3_); doc 4_'s "~41×" is untraceable to any cell. Forcing is low-wavenumber FNO-2010-style (ν=1e-3), not canonical sin(ny) Kolmogorov. **Training the operator under H⁻¹ from scratch is tested and ~20× worse** — Table 3: H⁻¹-train+FT 5.71×10⁻³ vs L²-train+H⁻¹-FT 2.88×10⁻⁴; the section is titled "Fine-tuning using negative Sobolev norm as loss," i.e. H⁻¹ is a convex *post-hoc* fine-tune, not a from-scratch objective.
- Scope: **downgraded.** H⁻¹ as a one-shot-preserving *training* loss is contraindicated by the originating paper's own ablation (non-convex from scratch). The value lives in the post-hoc fine-tune (§5.1), which breaks one-shot inference. The cheap, in-scope residual lever remains the plain-L² PDE residual (§3.1), not its H⁻¹ reweighting.

### 3.3 Positive Sobolev H^k loss
- Papers: Li et al. (MNO), arXiv:2106.06898, NeurIPS 2022 (KF result); Lippe et al. (PDE-Refiner), arXiv:2308.05732, NeurIPS 2023 (KS negative result).
- Idea: penalize errors in spatial derivatives (a high-pass weight) to capture sharp gradients and the energy cascade.
- Evidence: KF Re=40 one-step relL2 0.0187→0.0143 (~24%, H²) [MNO]. Verified against PDE-Refiner (arXiv:2308.05732, 2026-06-25): Sobolev norm losses (k=0,1,2) on 1D KS *decrease* high-correlation rollout vs the MSE baseline (75.4 s) — the paper states "focusing on high-frequency information leads to a decreased one-step prediction accuracy … such that the rollout time shortens." (Doc 1_'s "79.7 s vs 75.4 s improvement" misattributed the 4×-parameters/ensemble scaling rows to Sobolev; the actual Sobolev rows sit at/below baseline.)
- Scope: one-shot-compatible; weak/short-horizon lever.

### 3.4 Spectral / frequency-banded losses
- Papers: LOGLO-FNO, OpenReview 2025 (no arXiv ID in sources); Binned Spectral Power (BSP), arXiv:2502.00472, 2025.
- Idea: add per-band penalties on the (spatial) frequency spectrum to counter spectral bias. BSP bins spatial frequencies and penalizes per-band energy deviation.
- Evidence: LOGLO cuts high-band fRMSE 58–73% on KF Re=5k; BSP improves long-term spectral stability. Both optimize magnitudes/bands, not phases — spectral metrics, indirect pointwise benefit. BSP is a *spatial*-frequency loss, not time-dependent.
- Scope: one-shot-compatible; spectral target.

### 3.5 Physics + spectral loss inside a differentiable solver
- Paper: List, Chen, Thuerey, *Learned Turbulence Modelling with Differentiable Fluid Solvers*, arXiv:2202.06988, J. Fluid Mech. 949 A25, 2022.
- Idea: combine pointwise velocity L2 + spectral-energy + strain-rate losses in a CNN correction model trained inside a differentiable NS solver.
- Evidence: late velocity-field MSE drops at spatial-mixing Re=500 0.0243→0.0025 (~10×) and temporal-mixing Re=250 1.39×10⁻³→2.15×10⁻⁴ (~6×) — but these are best cases; the same ablation's decaying-isotropic-turbulence case improves only 0.182→0.166 (~10%). The clearest existing late-time *pointwise* field-error reduction, but the gain ranges from ~10% to ~10×; MSE not relL2, solver-hybrid, velocity form.
- Scope: solver-hybrid; closest counter-instance to "no clean late-time pointwise reduction exists."

### 3.6 Transport / optimal-transport loss
- Paper: Jiang et al., *Training neural operators to preserve invariant measures*, arXiv:2306.01187, 2023.
- Idea: match distributions of summary statistics via a Wasserstein/Sinkhorn objective.
- Evidence: demonstrated on Lorenz-96 and 1D Kuramoto–Sivashinsky only; DySLIM drops Sinkhorn on KF as "non-discriminative in high dimension."
- Scope: statistical target; no 2D KF pointwise result.

### 3.7 Dissipativity / stability regularization
- Papers: Li et al. (MNO), arXiv:2106.06898, NeurIPS 2022; SGNO, arXiv:2602.18801, 2026.
- Idea: encourage trajectories to contract back onto the attractor (MNO outer-shell data augmentation; SGNO Lipschitz/contraction bound), lowering the error upper bound.
- Evidence: MNO's own statement — "dissipativity can be encouraged without significantly affecting step-wise relL2"; Re=5000 0.1545→0.1543 (unchanged).
- Scope: prevents blow-up; does not close the on-attractor pointwise gap.

### 3.8 Static temporal upweighting w(t)∝t^α
- Paper: none (single-shot analogue of Coker et al. TWL); proposed as doc 4_'s Rank-1 actionable lever.
- Idea: multiply the per-timestep loss by (t/T)^α so the optimiser attends to late slices, where the joint loss otherwise dilutes the late-t gradient T-fold.
- Evidence: doc 4_ ranked this its #1 cheap lever (untested in the literature) — **but doc 4_ also banked a label-free temporal-weighting variant as dead** (no improvement) in its own gates (§6). The synthesis adopts the banked-null reading: the field's cheapest recommended lever did not move late k≤7 relL2 locally. Whether the exact (t/T)^α form differs from the banked variant is unresolved.
- Scope: one-shot-compatible, one-line change; demoted to low priority by the local null, not definitively closed.

---

## 4. Training protocols (multi-step training, one-shot deployment)

### 4.1 Differentiable multi-step unrolling (WIG / NOG / ONE)
- Paper: List et al., *Differentiability in Unrolled Training of Neural Physics Simulators on Transient Dynamics*, arXiv:2402.12971, 2024.
- Idea: backprop through K rollout steps so the model sees and corrects compounding error — differentiable (WIG) vs stop-gradient (NOG) vs one-step (ONE).
- Evidence: KF (Re 300–900, curriculum K=[1,2,4], 250 inference steps) ~21% lower long-rollout error; for *pure predictors* WIG≈NOG (stop-grad numerically slightly better); benefit Lyapunov-bounded (too-far unrolling → diminishing/negative returns).
- Scope: training protocol; deployed operator can stay one-shot.

### 4.2 Solver-in-loop K=32 training
- Paper: Kochkov et al., *Machine learning–accelerated CFD*, arXiv:2102.01010, PNAS 2021.
- Idea: train a coarse learned solver with cumulative pointwise velocity MSE over 32 unrolled steps inside a differentiable solver.
- Evidence: improved long-trajectory vorticity correlation at KF Re=1000; no controlled one-step-vs-K relL2 table.
- Scope: solver-hybrid training.

### 4.3 Multi-Step Penalty (MP-FNO)
- Paper: Chakraborty et al., *Improved Deep Learning of Chaotic Dynamical Systems with Multistep Penalty Losses*, arXiv:2410.05572, 2024.
- Idea: split the trajectory into windows, detach the graph every r rollouts, insert learnable discontinuities δ_s minimized by a penalty term — cuts gradient cost from cubic to linear while enforcing long-term coherence.
- Evidence: KF Re=1e5; spectrum/DNS-correlation plots only — no relL2, no published r/segment count.
- Scope: training trick, but mechanism entangled with the rollout — in-scope only via a window-chaining analogue, not as-published.

### 4.4 Pushforward
- Paper: Brandstetter et al. (origin); negative results in Lippe et al. (PDE-Refiner) arXiv:2308.05732 and Schiff et al. (DySLIM) arXiv:2402.04467.
- Idea: feed the model its own stop-gradient one-step prediction during training to expose it to its own error distribution; the gradient is cut at the handoff.
- Evidence: reported to **not help** in PDE-Refiner and DySLIM settings; a local stop-grad oracle-distill analogue tested negative.
- Scope: training trick, but mechanism entangled with the rollout — in-scope only via a window-chaining analogue.

### 4.5 Recurrent training (RNO)
- Paper: Ye, Zhang, Wang (RNO), *Recurrent Neural Operators*, arXiv:2505.20721, 2025.
- Idea: train the operator by recursively applying it to its own predictions within [0,T], eliminating exposure bias; claims worst-case error growth drops from exponential O(e^{CT}) to linear O(T).
- Evidence: Allen-Cahn step-50 relL2 cut 89.6%; **fails on FNO** (inconsistent r-FNO vs tf-FNO; works only on MgNO); NS benchmark is generic 2D vorticity (no Re, no forcing).
- Scope: autoregressive at inference; FNO-negative.

### 4.6 Differentiable window-chaining (single-shot adaptation)
- Papers: motivated by List et al. arXiv:2402.12971 + local banked oracle-restart result.
- Idea: split T into two windows, feed window-1 prediction (no stop-grad) as IC for window-2, backprop through the handoff; trains midpoint predictions to be good ICs — implicitly learning the oracle-restart structure.
- Evidence: oracle restart from GT mid-window cuts late relL2 0.54→0.230 locally; the differentiable version is untested.
- Scope: the canonical in-scope adaptation. Two flavors — (a) shorten window + chain at test = mild 2-step AR inference; (b) chaining as a training regularizer on a full-T one-shot operator = true one-shot inference. Inherits recurrent-training memory (~×K) and gradient cost (moderate at K=2, Lyapunov-bounded per window).

---

## 5. Inference-time corrections

### 5.1 Negative-Sobolev a-posteriori fine-tune (Spectral-Refiner)
- Paper: Cao et al., arXiv:2405.17211, ICLR 2025.
- Idea: after training, fine-tune (often only the last projection layer) with the H⁻¹ residual loss, using an auto-diff solver to evaluate the residual.
- Evidence (verified): final-time L2 ~10⁻²→2.88×10⁻⁴ on decaying McWilliams (Table 3) and 2.08×10⁻²→2.82×10⁻⁴ on forced turbulence (Table 2, ~74×); the Re=5000 figure (0.294→0.025) is an OOD generalization in the arXiv appendix (App. D.4, Table 7); the OSTI camera-ready I read (Tables 1–5) contains only the forced and decaying cases. All final-time, per-instance, solver-assisted; no per-timestep curve.
- Scope: **per-instance, inference-time, iterative** — breaks one-shot inference speed. Note this is *not* substitutable by an H⁻¹ training loss: §3.2 shows H⁻¹ training is ~20× worse, so the one-shot-preserving residual lever is the plain-L² PDE loss (§3.1), not H⁻¹.

### 5.2 Leray / Helmholtz-Hodge divergence-free projection
- Papers: Negiar et al., *Towards Enforcing Hard Physics Constraints*, OpenReview 2024; *Project-and-Generate*, arXiv:2603.24500, 2026.
- Idea: project the velocity output onto the divergence-free subspace in Fourier space as a hard constraint.
- Evidence: "most often" lower test L2; divergence →~10⁻⁷ (the quantitative magnitudes in 2603.24500 were adversarially refuted).
- Scope: velocity-form only; for the vorticity-form model incompressibility is already implicit, so near-redundant even after ω→u conversion.

### 5.3 Conserved-quantity correction (DPOTc)
- Paper: Current et al., arXiv:2601.22541, 2026.
- Idea: apply per-step magnitude scaling (density/energy) and shift (momentum) on outputs to enforce conservation without touching the loss.
- Evidence: compressible NS Mach 0.1, 10-step relL2 8.4%→7.8%, growth slowed to 0.4%/step. The incompressible-KF transfer (circulation/enstrophy) is the reviewers' speculation, not in the paper — flagged as an open niche.
- Scope: autoregressive in source; mechanism portable; enstrophy/circulation projection on vorticity form untested.

### 5.4 Newton-step residual correction
- Paper: Cao et al. 2022, OSTI 2421120.
- Idea: one Newton step δu = −[∂_u R]⁻¹R(u_FNO) using the linearized NS residual; needs an estimate of ν.
- Evidence: single deterministic step; the local banked residual is non-identifying for late-wall error, so this would not selectively target late-t.
- Scope: one-shot-compatible but mistargeted for this symptom.

---

## 6. Banked local diagnostics (adjudicating evidence, no paper)

- **Oracle-restart gate:** restarting from ground truth mid-window cuts late relL2 0.54→0.230 (~57%) → the information to predict late-t exists in the data; the wall is representational. (Source doc inconsistent on whether this gate is Re=100 or Re=500.)
- **Chaos-spread gate:** perturbation growth ~1.3×/horizon, λ~0.2–0.3 at Re=100, solver-from-exact-IC control = 0 → late-wall error is not chaos-dominated at low Re.
- **Dead levers (banked):** capacity (modes 8→16, h 64→128), 16×16 grid shrink, spectral band loss (<0.006), label-free temporal weighting, late-window specialist, self-chaining own field — none moved late k≤7 relL2 by >0.006.
- **Implication:** at low Re the wall is model/optimization-limited, steering weight toward training-protocol and loss/inference levers over the fatalistic chaos reading.

---

## 7. Examined and excluded (scope filter)

Methods present in the sources but kept out of the in-scope lever set, with reason — including the by-name "one-shot"/"autoregression-free" candidates a reader will ask about:

- **AFNO — Autoregression-Free Neural Operators** (arXiv:2605.25413): despite the name, it is latent-space continuous-time *flow-matching*, not a deterministic single-shot space-time operator; flow-matching is generative-adjacent (out of scope) and it reports no KF pointwise relL2. (Verified from abstract, 2026-06-25.)
- **Late Fusion Neural Operators** (arXiv:2604.16721): "one-shot" refers to *parameter*-regime extrapolation (disentangling state dynamics from parameter effects via sparse regression), not late-time temporal prediction; tested on advection/Burgers/1D–2D reaction-diffusion — no NS/KF, RMSE not late-time relL2. (Verified from abstract.)
- **Scheduled / causal temporal loss weighting** (Coker et al.; Wang et al., arXiv:2203.07404): AR-only and they upweight *early* t (opposite direction); no single-shot analogue. The single-shot reinterpretation survives as §3.8.
- **TNO temporal bundling** (arXiv:2504.20249): benefit is zero for a single-shot operator by construction (it only reduces rollout re-ingestion count).
- **Generative / probabilistic refiners** — PDE-Refiner (arXiv:2308.05732, diffusion), adv-NO / Gen4Turb, generative diffusion for 3D turbulence: excluded by the deterministic scope; used here only for disambiguation (§3.2) and ablation evidence (§3.3).
- **Spectral-bias-specific levers** — PENO (arXiv:2406.04367), High-Frequency Scaling (arXiv:2503.13695): Tier-2, spectral-target not pointwise-relL2-validated on KF; conceptually within §3.4 / §2.6, not separately verified.
- **Diagnostics, not levers** — CORAL in/out-horizon cliff (arXiv NeurIPS 2023); RFNO ν-scaling of the floor (folded into §1.5).
