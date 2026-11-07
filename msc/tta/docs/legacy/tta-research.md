# TTA Research Lexicon

## TODO
- [x] Cao et al. — residual-based error correction for neural operator accelerated Bayesian inverse problems (osti.gov/biblio/2421120)
- [x] Phys-Instruct — Ultra Fast PDE Solving via Physics Guided Few-step Diffusion (arXiv:2602.03627)
- [x] Score-based Generative Solver for PDE-constrained Inverse Problems with Complex Priors (arXiv:2408.16626v1)
- [x] Inverse Neural Operator for ODE Parameter Optimization (arXiv:2603.11854v1) — ODE parameter inversion, C-FNO + ADM; no forward PDE, no parametric shift, no TTA; skipped
- [x] 69_BEYOND LOSS GUIDANCE: browse PDE refinement related work section for additional papers to map (incl. Bastek et al. 2025 "Physics-informed diffusion models" ICLR 2025, arXiv:2403.14404 — cited there, not yet mapped)
- [x] DiffusionPDE — Huang et al. 2024a; sparse-observation forward/inverse via long reverse diffusion (~1000 steps); local: 5_DiffusionPDE.pdf
- [x] FunDPS — Yao et al. 2025 (arXiv:2505.17004); function-space diffusion posterior sampling, UNO backbone; BEAT; code reference for [D]-spectrum guidance loop
- [x] Zhang & Zou 2025 (arXiv:2505.22391) — residual guidance instabilities during diffusion sampling; FRAME only; argument already covered by PRISMA; skipped
- [x] Yang and Sommer (2023) — vanilla DDPM + U-Net for fluid prediction, no physics loss, no OOD angle; dominated by DiffusionPDE/FunDPS/PRISMA; skipped
- [x] PIETRA — "Physics-Informed Evidential Traversability for OOD Terrain" 2024 (arXiv:2409.03005v1); robotics domain, OOD via normalizing flow density on latent features, hand-crafted physics prior — nothing transfers to PDE operators; skipped
- [x] Physics-Informed Diffusion for Unsupervised Anomaly Detection — arXiv:2508.11528v1 (2025–26); time series domain, ELBO as runtime score not residual, physics only at training; does NOT narrow claim (1); skipped
- [x] Qin et al. 2024 — "Toward a Better Understanding of FNOs from a Spectral Perspective"; Fourier parameterization bias, underweights non-dominant modes; no OOD framing (via PP_TTA sweep)
- [x] Shikhman 2026 — "Diagnosing Failure Modes of Neural Operators Across Diverse PDE Families"; GT-based failure atlas, single viscosity, no runtime score (via PP_TTA sweep)
- [x] NTK-alignment OOD theory — "Leveraging NTK Alignment for Reliable OOD Detection" arXiv:2505.16017v1; spectral OOD theory, supports claim (3) conceptually but no FNO/operator link (via PP_TTA sweep)
- [x] 000_PP_TTA:OOD:EL.pdf — sweep done; candidates extracted above
- [x] GradOrth (Behpour et al. 2023, NeurIPS) — gradient-space OOD via SVD of activation covariances + projection; classification-only but closest structural cousin to GradPCA; needed to bound the prior-art gap precisely (via Perplexity gap sweep)
- [x] MetaPhysiCa — ODE meta-learning + causal structure discovery; requires multi-task training + supervised test-time signal; ODE only, no transfer to PDE/FNO setting; skipped
- [x] LoRA (Hu et al. 2021, arXiv:2106.09685) — low-rank weight-update parameterization ΔW=BA for efficient fine-tuning; not for OOD detection; relevant only if TTA update is reparameterized as low-rank perturbation for efficiency (pairs with Dogoulis norm-bound); entry added

## Adaptation Spectrum

Shared signal throughout: PDE residual ℛ(ω, ν) — unsupervised, label-free.
Axis: how much of the model is touched at test time.

```
[A] Full weight update
    signal:   ℒ_pde drives gradient through all θ
    risk:     catastrophic forgetting under long adaptation
    papers:   PINO instance fine-tuning (Li et al.)

[B] Partial weight update — norm-bounded
    signal:   ℒ_adv + ℒ_visc on lifting/projection layers only; spectral conv frozen
    bound:    min_φ ℒ_TTT s.t. ‖φ‖₂ ≤ ε  →  anti-forgetting by construction
    papers:   Dogoulis 2025 [68_AC Power Flow Surrogates]

[C] Full weight freeze — deterministic output correction
    signal:   one Newton step in solution space: δω = -[δ_u ℛ]⁻¹ · ℛ(ω_FNO, ν̂)
    theory:   quadratic convergence (Newton-Kantorovich); Cao Theorem 1 grounds residual as OOD score
    baseline: first-order gradient approx (bypass linear solve); shows ceiling at high Re shift
    papers:   Cao et al. 2022 [73_Residual-Based Error Correction]
    gap:      requires ν̂ estimate; single deterministic step; fails gracefully at large shift

[D] Full weight freeze — stochastic output refinement
    signal:   FNO output as warm-start prior; diffusion denoising guided by ℒ_pde
    theory:   multi-step stochastic generalization of [C]'s single Newton step (Cao → diffusion)
    papers:   DiffusionPDE [5_DiffusionPDE], PRISMA [69_BEYOND LOSS GUIDANCE], Phys-Instruct [70_Ultra]
    gain:     expressive nonlinear correction; expected to surpass [C] at high Re shift
```

OOD gate (orthogonal, applies to B/C/D):
- trigger: ‖ℛ(ω_FNO, ν̂)‖ > τ — grounded formally by Cao Theorem 1
- calibration: n_modes ↔ detection sensitivity curve — claim (3)

---

## Background Citations
<!-- anchor refs: not worth a full entry; one-liner format: grounds X — cite when writing Y — because Z -->

- Phillips 2014, "Residual-based Discretization Error Estimation for CFD" (vtechworks handle/10919/50647) — grounds: residual as solution-quality scalar is classical CFD practice — cite when: introducing PDE residual as OOD score in related work — because: pre-empts "this is obvious" by showing the idea is textbook in numerical solvers but not yet applied to learned surrogates under parametric shift
- ROM / POD-Galerkin a-posteriori error literature (general) — grounds: residual norms as error surrogates for model switching in reduced-order models — cite when: motivating residual-gated TTA trigger — because: shows the trigger pattern exists in classical simulation but not in neural operator deployment

---

<!--
ENTRY SCHEMA — copy this block for each new paper

## [filename] Title — Authors (Year)

**Summary:** one phrase

**Tags:** STEAL / BEAT / FRAME / GROUND  (all that apply)

**Take:** (3-5 bullets — concrete ideas to borrow; populate only if STEAL tagged)
- 

**Gap:** (2-3 bullets — what it explicitly does NOT do; populate only if FRAME tagged)
- 
-->

---

## [67_tta_taxonomy] A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts — Liang, He, Tan; CASIA (2024) | arXiv:2303.15361

**Summary:** full TTA taxonomy across TTDA / TTBA / OTTA paradigms; classification-domain only, zero PDE content — value is terminological grounding

**Tags:** GROUND

**Lingo:**
- **TTIA** (Test-Time Instance Adaptation): batch size = 1, single unlabeled instance, no source data — exact name for your per-sample physics-residual adaptation (Definition 3)
- **Training-agnostic fine-tuning**: unsupervised objective designed purely at test time, no SSL head added during training — your physics residual loss fits here; contrast with TTT [8] which requires auxiliary head at train time
- **Anti-forgetting regularization**: keep adapted model close to pre-trained weights — norm-bounded update (Dogoulis) and Fisher regularization (EATA) are instances; use this term when describing your ||φ||₂ ≤ ε bound
- **SAR** [301]: minimax form `min_θ max_{||Δ_θ||₂≤ε} H(x; θ+Δ_θ)` — same structure as Dogoulis norm-bounded update applied to entropy instead of physics residual; cite as prior art for the optimization form
- Entire taxonomy assumes discrete label space + entropy/pseudo-label signal — physics residual substituting for entropy as the unsupervised objective is the architectural gap between your work and this entire survey

---

## [68_AC Power Flow Surrogates] Test Time Training for AC Power Flow Surrogates via Physics and Operational Constraint Refinement — Dogoulis, Alizadeh, Kubler, Cordy (2025) | arXiv:2511.22343

**Summary:** norm-bounded perturbation vector drives gradient-based physics-residual refinement of a frozen surrogate at test time, label-free, instance-wise

**Tags:** STEAL, FRAME

**Take:**
- Perturbation-bounded update: solve `min_φ L_TTT(f_{θ+φ}(z†))  s.t. ||φ||₂ ≤ ε` — bounds how far a single test instance can pull weights from the pre-trained solution; direct regularizer against catastrophic forgetting during instance-wise TTA
- Partial parameter split: freeze core operator weights θ_frozen, only update θ_adapt via φ; in FNO the natural equivalent is freeze spectral conv layers, adapt lifting/projection MLPs
- Composite physics loss: separate residual terms (ΔP + ΔQ) plus differentiable constraint penalties (squared ReLU); maps to NS decomposition ℒ_adv + ℒ_visc + ℒ_force — lets you weight terms and diagnose which drives the residual under Re shift
- Differentiable constraint penalty via squared ReLU: `ReLU(violation)²` — smooth, zero-gradient when satisfied, non-zero only when violated; cleaner than hard projections for gradient-based TTA

**Gap:**
- Physics constraint is algebraic (AC power flow = Kirchhoff equilibrium), not a PDE; no time derivative, no spatial differential operator, no viscosity or Reynolds parameter
- No parametric shift analysis — "distribution shift" is unseen grid operating conditions, not a continuous PDE coefficient sweep (no Re/ν sweep, no calibration curve)
- No OOD detection gating — TTT runs unconditionally on every sample; trigger logic absent
- No spectral structure exploited; architecture-agnostic wrapper with no mode-truncation or frequency-domain analysis

---

## [56_Operator Splitting] Test-time Generalization for Physics through Neural Operator Splitting — Serrano, Han, Oyallon, Ho, Morel; Polymathic AI (2026) | arXiv:2602.00884

**Summary:** at test time, beam-search over a pre-trained dictionary of physics operators and compose the best subset via Strang splitting to approximate unseen OOD dynamics — no gradient steps, no weight updates

**Tags:** STEAL, BEAT, FRAME

**Take:**
- Strang splitting for 2D NS: decompose ∂_t ω + u·∇ω = νΔω into f_Euler (inviscid advection, ν=0) ∘ f_Diffusion (heat eq ∂_t ω = νΔω); second-order accurate composition û^{L+1} = f_1^{Δt/2} ∘ f_2^{Δt} ∘ f_1^{Δt/2}(u^L); directly usable as compositional residual decomposition in your ℒ_adv + ℒ_visc split
- Short observation window as test-time signal: L consecutive snapshots of the OOD trajectory suffice to fit the operator composition; maps to using a few observed timesteps to compute your PDE residual before triggering TTA
- Weakest-link composition bound (Table 3): composed operator error dominated by least accurate component — for FNO + diffusion refiner pipeline, improving the weaker module has higher ROI than improving the already-accurate one
- Beam search over operator subsets outperforms uniform sampling at equal compute (Figure 5 power-law); transferable if you search over adaptation hyperparameters (ν̂ grid, n_modes sweep)

**Gap:**
- Test-time signal is data-space NRMSE on observed snapshots, NOT the PDE equation residual — no physics constraint enforced at test time; biggest architectural gap vs your approach
- No OOD detection gating — search runs unconditionally on every test sample regardless of shift magnitude
- NS experiment uses matched viscosity; trains Euler + pure Diffusion separately and composes at test — does NOT sweep Re as a continuous parametric variable or study Re-shift detectability
- Requires DISCO dictionary infrastructure (hypernetwork + U-Net); not portable to a single FNO without retraining the full dictionary
- No gradient updates — cannot adapt to regimes outside the span of dictionary operator combinations

---

## [69_BEYOND LOSS GUIDANCE] Beyond Loss Guidance: Using PDE Residuals as Spectral Attention in Diffusion Neural Operators (PRISMA) — Sawhney, Neog, Khurana, Karpatne; Virginia Tech (2025) | arXiv:2512.01370

**Summary:** bakes the PDE residual into a diffusion neural operator architecturally via per-layer spectral cross-attention (SRA), enabling gradient-free inference that is 15x-250x faster than loss-guided diffusion baselines

**Tags:** STEAL, FRAME, BEAT

**Take:**
- SRA block: at each denoising step compute residual field r, FFT both r and feature map x^l, measure per-mode phase alignment S^l(k) = (1/√C)|Σ_c x̃^l_c(k) r̃^l_c*(k)|, build attention mask A^l(k) = σ(w_gain ⊙ S^l(k)), gate via learned scalar g_res; output is F⁻¹(((1-g_res) + g_res·A^l) ⊙ x̃^l) — the residual steers which frequency modes get corrected without any gradient through the physics operator
- Adaptive gating MLP: g_res ∈ [0,1] learned from (r_avg, c_σ) — scalar residual summary + noise-level embedding; when residual is large/noisy g_res is suppressed, when solution is clean g_res is high; invert this logic for OOD use: large residual = OOD = gate should open, not close — the architecture is right, the polarity needs flipping
- Correct 2D NS one-step residual (Eq. 18): R(x) = (ω̃_{t+Δt} - ω_t)/Δt + v̂_{t+Δt}·∇ω̃_{t+Δt} - νΔω̃_{t+Δt} - f(x); they also explicitly flag that DiffusionPDE's proxy ∇·ω(x,t) is physically invalid in 2D since ω is a scalar field — use this citation to justify your residual formulation
- Residual as (H×W) scalar field per denoising step: PRISMA feeds one scalar per grid point into SRA; **potential extension**: feed a (H×W×3) tensor decomposed into temporal, advection, and viscous channels — SRA then learns separate spectral masks per physical term; under Re shift the νΔω channel dominates, attention routes correction there selectively; this decomposition is in neither PRISMA nor Dogoulis

**Gap:**
- Trains from Gaussian noise, not from a pre-trained operator output; no FNO warm-start prior — the diffusion generates from scratch rather than correcting a ~90%-correct deterministic prediction
- No OOD detection gating — SRA runs unconditionally on every sample regardless of whether the input is in- or out-of-distribution
- NS experiment uses single fixed viscosity (ν = 10⁻³, Re = 1000) — no parametric shift evaluation, no Re sweep, no calibration under viscosity change
- On full-observation forward NS, FNO (0.60% L2) outperforms PRISMA (1.51%) — PRISMA's advantage is sparse and noisy observation settings, not clean forward prediction
- Full diffusion model training required (50k samples, 64M params, 2×A100) — not a lightweight adapter on a pre-trained operator

---

## [5_DiffusionPDE] DiffusionPDE: Generative PDE-Solving Under Partial Observation — Huang, Yang, Wang, Park; Michigan/Stanford (2024) | NeurIPS 2024

**Summary:** joint diffusion model over (a,u) guided at inference by sparse observation loss + PDE residual loss via DPS; designed for 1–3% partial observations, not full-observation forward prediction

**Tags:** STEAL, BEAT, FRAME

**Take:**
- Two-phase guidance schedule (Appendix C): first 80% of denoising steps use L_obs only; last 20% shift weight to L_pde while reducing ζ_obs by 10×; rationale — PDE loss destabilizes early denoising when solution estimate is still noisy; maps directly to TTA curriculum: large residual → trust data, small residual → trust physics
- Guidance weights ζ_pde ~ 10²–10³ across PDE families (Table 2); concrete ballpark for λ_phys in your PINO physics loss tuning
- PINO dominates DiffusionPDE on full-observation forward NS (Table 4): PINO 1.1% vs DiffusionPDE 6.1% — confirms your baseline is already the right anchor for the forward problem
- Appendix D Fig 12: tested at ν = 10⁻⁴ (Re = 10000, 10× shift from training ν = 10⁻³) with ~7–9% error; not framed as OOD — the viscosity shift resilience is present in the data but completely unanalyzed; direct citation target when arguing the gap your work fills

**Gap:**
- PDE guidance is spatial residual aggregated to a scalar: L_pde = (1/m)Σ||f(c_j, τ_k; û_j, a_j)||² — no decomposition into temporal / advection / viscous channels; same gap as Dogoulis
- No OOD detection or gating — guided sampling runs unconditionally regardless of shift magnitude
- Fixed resolution 128×128; tied to native spatial domain, not resolution-agnostic
- NS simplified PDE proxy: uses ∇·w(x,t) as guidance (physically invalid for scalar vorticity, as PRISMA Eq.18 notes); residual formulation weaker than your PINO loss
- Advantage over PINO only under sparse observations; at full observation PINO wins — wrong tool for your setting

---

## [70_Ultra] Ultra Fast PDE Solving via Physics-Guided Few-step Diffusion (Phys-Instruct) — Kong, Wang, Zheng, Luo, Lin; Purdue (2026) | arXiv:2602.03627

**Summary:** physics-guided IKL distillation bakes PDE residual compliance into student weights — zero test-time physics cost, k-step generation matching 200-step diffusion baselines

**Tags:** STEAL, FRAME

**Take:**
- Amortized physics correction (Remark 4.2): physics enters only at distillation time via L_G(θ) = D_IKL + λ_phys · L_phys; at inference the student is gradient-free — contrast with PRISMA (SRA per step) and Dogoulis (per-instance gradient steps); these three form a spectrum: baked-in → architectural → online
- Generator gradient (Eq.10): ∇_θ L_G = IKL_score_term + λ_phys · (∇_x R(x₀))^T ∇_θ g_θ(z;k); chain rule through the PDE operator to the generator weights — exact same chain you'd use in PINO's physics loss, here applied at distillation not training
- ControlNet-style conditional branch: frozen unconditional backbone + trainable conditional copy shares weights; adaptable pattern for attaching a lightweight TTA adapter to a frozen FNO without retraining the base
- PDE residual scalar: R(x) = (1/N)Σ||r(x_i)||² — spatially averaged L2 of discrete PDE error field G_h[x](ξ_i); already in the same form as your PINO physics loss ℒ_pde, directly comparable

**Gap:**
- Physics only at distillation, not inference — no test-time residual signal, no OOD correction; if student was distilled at Re=1000 it cannot self-correct at Re=4000
- No OOD detection, no parametric shift evaluation — NS benchmark fixed at ν = 10⁻³ single viscosity, no Re sweep
- Trains from Gaussian noise; no FNO deterministic warm-start prior — generates solutions de novo rather than correcting a near-correct operator prediction
- Diffusion architecture required; not a wrapper adaptable to a pre-trained FNO without full retraining

---

## [74_SCORE-BASED GENERATIVE SOLVER] A Score-based Generative Solver for PDE-constrained Inverse Problems with Complex Priors — Hong, Bansal, Veroy; TU Eindhoven (2024) | arXiv:2408.16626v1

**Summary:** score-based diffusion over parameter space for Bayesian inverse problems; main contribution is an analytically derived time-varying guidance step size that fixes the divergence DiffusionPDE [5_DiffusionPDE] addresses with an empirical two-phase schedule

**Tags:** GROUND

**Take:**
- Time-varying guidance step size ρ(t) = Δt/(t·σ²_ε) (Eq. 34): derived from the condition that the posterior noising process must remain approximately Gaussian at terminal time T; ρ(t)→0 as t→T suppresses physics guidance when the current state is mostly noise, ρ(t)→1/σ²_ε as t→0 restores full guidance when the state carries solution information — this is the analytical justification for why DiffusionPDE's 80/20 two-phase heuristic works; cite both together when motivating the guidance schedule in your [D] implementation

**Gap:**
- Denoises over parameter space μ (material properties), not solution space ω — the opposite direction through the forward operator; not a template for forward-problem OOD refinement
- No NS or fluid dynamics experiments; solid mechanics only; no operator warm-start prior

---

## [73_Residual-Based Error Correction] Residual-Based Error Correction for Neural Operator Accelerated Infinite-Dimensional Bayesian Inverse Problems — Cao, O'Leary-Roseberry, Jha, Oden, Ghattas; UT Austin (2022) | arXiv:2210.03008

**Summary:** a posteriori error estimation grounds a one-Newton-step output-space correction of a trained neural operator via the PDE residual — provides the formal theoretical chain linking residual magnitude to operator approximation error to posterior quality

**Tags:** GROUND, FRAME

**Take:**
- Theorem 1 (Sec 3.3): posterior error ε_post ≤ c_BIP ‖ε‖_{L^p} where ε = ℱ(m) - ℱ̃_w(m) — formal proof that operator approximation error propagates linearly into Bayesian posterior error; contrapositively: large PDE residual → large approximation error → untrustworthy output; this is the theoretical grounding for claim (1) that the residual is a valid runtime OOD score, not just a heuristic
- Accuracy ceiling (Sec 3.2): neural operators empirically saturate at ~90% accuracy regardless of more training data or network depth; the ceiling is irreducible via training alone — formal motivation for why test-time correction is necessary; cite as the principled argument that TTA is not optional but structurally required under distribution shift
- Newton correction as deterministic limit (Eq. 28): given ℱ̃_w(m), find u_C by solving δ_u ℛ(ℱ̃_w(m), m) u_C = -ℛ(ℱ̃_w(m), m) + δ_u ℛ · ℱ̃_w(m); one step yields quadratic error reduction ‖ℱ(m) - u_C‖ ≤ c‖ℱ(m) - ℱ̃_w(m)‖² — this is the zero-temperature, single-step, deterministic version of what diffusion-based refinement (DiffusionPDE, PRISMA) does stochastically over many steps; Cao et al. provides the variational derivation that makes diffusion refinement theoretically principled rather than architectural intuition

**Gap:**
- Requires the true PDE parameter m at correction time — in their MCMC setting m is always known (it is the MCMC sample); in your OOD forward setting ν_test is unknown; correction therefore requires ν̂ estimation first (M3 dependency) before any output-space refinement can be applied
- Bayesian inverse problem setting only — they infer parameters from observations; your problem is forward prediction under parametric shift; the correction mechanism is reusable but the surrounding inference framework is not
- No parametric OOD evaluation — both numerical examples use fixed PDE families with spatially varying coefficients; no Re/ν sweep, no distribution shift analysis, no OOD framing anywhere

---

## [71_NTK] GradPCA: Leveraging NTK Alignment for Reliable Out-of-Distribution Detection — Seleznova, Chou, Verdun, Kutyniok; LMU/Pittsburgh/Harvard (2025) | arXiv:2505.16017v1

**Summary:** NTK alignment induces a C-dimensional low-rank gradient subspace for ID samples; PCA on gradient class-means scores OOD inputs by projection residual — classification only, but the spectral OOD theory (Thm 4.1/4.5) applies to any feature map including physics residuals

**Tags:** GROUND, FRAME

**Take:**
- Theorem 4.1 (sufficient condition): for any feature map h, if ‖Ph(x)‖₂ < ‖h(x)‖₂ and h is continuous at x, then x is OOD — applies directly if h = PDE residual field; if ID residuals span a subspace of rank < ambient dim, OOD inputs guaranteed to project outside; formal backing for claim (1)
- Theorem 4.5 (necessary condition): spectral OOD detection has non-zero sensitivity only if rank(S(h)) < dim(image of h) — predicts the failure mode: if model generalizes so well that residuals are uniformly distributed in the full space (e.g., pre-training on very wide ν range), the score degrades; calibration curve design must ensure this condition holds
- Feature quality framing: regularity-based detectors (exploit geometry in feature space) improve with structured features; physics residuals are domain-structured features — predicts your residual OOD score outperforms confidence/energy baselines under parametric shift; cite as principled justification for the feature map choice

**Gap:**
- Method (GradPCA) is classification-only: requires C discrete classes → class-mean gradients → block-diagonal NTK → C-dimensional subspace; continuous parametric shift (ν sweep) has no class structure; algorithm does not transfer to regression
- NTK alignment theory underpinning the low-rank claim is specific to discriminative models trained with cross-entropy to convergence — not established for neural operators trained with MSE + physics loss
- No regression experiments, no PDE/physics setting, no operator architecture — all benchmarks are CIFAR/ImageNet image classification
- Gradient-space method requires a backward pass at inference per test input — cost is acceptable but adds overhead not present in a forward-only residual score
- **Gap sweep (Perplexity, 2026-05-22):** no documented instance of GradPCA-style gradient-subspace OOD detection adapted to regression or PDE surrogates; closest adjacent work: GradOrth (classification), MetaPhysiCa (PIML OOD via meta-learning, no subspace projection); the regression transfer requires replacing class-mean gradients with a regime-based analogue (e.g. ν-bin means under physics loss) — theoretical gap, low-priority side direction

---

## [76_Spectral Perspective] Toward a Better Understanding of Fourier Neural Operators from a Spectral Perspective — Qin et al.; Concordia/McGill (2024) | arXiv:2404.07200v2

**Summary:** identifies Fourier parameterization bias — FNO over-concentrates on dominant (high-energy) modes during training, neglecting non-dominant modes; prediction residual energy is consequently flat across Fourier space; provides mechanistic spectral grounding for the B3-band OOD detector and its extension to other PDEs

**Tags:** GROUND, STEAL

**Take:**
- Fourier parameterization bias (Sec 2, Fig 1): Fourier-space kernel parameterization causes training gradient to be dominated by high-energy modes in target data; non-dominant modes receive negligible gradient signal; prediction residual energy is therefore more evenly distributed across Fourier space than target energy (Fig 1c) — mechanistic explanation for why PDE residual accumulates in high-frequency bands when physics shifts
- Two-layer explanation for B3 sensitivity: (i) modes above truncation k (B3: k∈[9,16] with n_modes=8) have no FNO spectral parameters — handled by MLP fallback with general low-frequency bias; (ii) Fourier parameterization bias further neglects non-dominant modes even within truncation; B3 sits in the doubly-blind zone by construction
- Re shift → high-frequency residual spike (Fig 2b/c): at higher Re, turbulent energy migrates into smaller scales (higher modes) which are precisely FNO's neglected modes; NMSE rises sharply there — empirical confirmation that upward Re shift triggers the spectral failure B3 z-score measures
- NMSE-by-frequency-mode diagnostic (Fig 2): decompose prediction error per Fourier mode k; run this on your FNO across Re sweep to show which bands light up and that it tracks B3; one plot = claim (3) spectral evidence
- PDE generality (Table 1, Fig 2a-f): same bias observed across NS, Darcy, shallow water, diffusion-reaction — bias is architecture-driven not PDE-specific; supports claim that B3-band OOD detection generalizes to other operators/PDEs by the same spectral argument

**Gap:**
- No OOD framing — ν=1e-3 and ν=1e-5 treated as independent benchmarks, never as parametric shift; the OOD interpretation of their spectral failure data is entirely absent; gap your thesis fills
- Detector is directional: B3 spike occurs only under upward Re shift (low→high Re, energy migrates into high modes); downward shift (high→low Re) produces no B3 signal since low-Re physics has no energy there — asymmetry to acknowledge
- SpecB-FNO fix is training-time only; not a TTA method

---

## [75_GradOrth] GradOrth: A Simple yet Efficient Out-of-Distribution Detection with Orthogonal Projection of Gradients — Behpour, Doan, Li, He, Gou, Ren; Bosch BCAI (2023) | arXiv:2308.00310v1

**Summary:** OOD score = norm of gradient projected onto the ID activation subspace (SVD of last-layer representations); activation-based subspace requires no class structure — regression-compatible algorithmic template, weaker theory than GradPCA

**Tags:** FRAME, STEAL

**Take:**
- Activation-SVD subspace (Algorithm 1): forward-pass n ID samples → concatenate last-layer activations R^L_ID → SVD → threshold at ε_th to get k-rank subspace S^L; requires no labels, no class structure — directly applicable to FNO: collect last-layer (projection MLP) activations on ID ν samples, SVD, store S^L
- Gradient projection score (Algorithm 2, Eq. 7): O(x_i) = ||P_{S^L}(∇_{θ^L} L(θ^L))|| where P_{S^L}(g) = g S^L (S^L)^T; at test time replace L with ℒ_pde — backward pass on physics residual gives gradient, project onto ID subspace, large projection = ID; large orthogonal residual = OOD
- MSE loss derivation (Eq. 2-3): gradient = Ω x^T where Ω = θx - y; maps directly to PDE setting where Ω = ℛ(ω_FNO, ν̂) — no label required, residual is the error signal

**Gap:**
- All benchmarks classification (CIFAR, ImageNet); no regression, no PDE, no operator experiments
- No formal guarantees: no equivalent of GradPCA Thm 4.1/4.5; empirical only
- Operates in a fundamentally different detection space than spectral residual scoring: current approach (B3 band z-score) detects OOD by measuring residual energy in frequency bands outside FNO's representable range — solution space, spectral domain; GradOrth detects OOD by measuring how far the physics loss pushes model parameters from ID-familiar update directions — weight space, gradient domain; neither subsumes the other; no paper combines both axes for PDE surrogate OOD detection; this is the thesis gap: two orthogonal detection principles, each independently motivated, together naturalizing claim (1) within the related work rather than asserting it from nowhere

---

## [80_Failure Modes] Diagnosing Failure Modes of Neural Operators Across Diverse PDE Families — Shikhman; Georgia Tech (2026) | TMLR 5/2026 | arXiv:2601.11428v7

**Summary:** stress-testing framework across FNO/DeepONet/CNO × 5 PDE families under 5 structured shifts; shows ID accuracy does not predict robustness; for NS specifically, rollout instability dominates parameter shift in relative terms; provides formal bounds and spectral diagnostics

**Tags:** GROUND, STEAL, FRAME

**Take:**
- Degradation factor D = E_stress / E_base (Sec 3.3): baseline-normalized robustness metric; controls for model quality when comparing across Re sweep — more informative than raw error for your n_modes ↔ OOD detectability calibration curve [claim 3]; parallel to [76_Spectral Perspective]'s NMSE-by-mode diagnostic
- Proposition 1 (rollout composition bound, Sec 5.1): ||Φ̂^m - Φ^m|| ≤ ε Σ_{j=0}^{m-1} L^j — one-step error grows exponentially with horizon m when the learned map's Lipschitz constant L > 1; formal proof that static training cannot guarantee long-horizon OOD stability; complements Cao et al. accuracy ceiling [73_Residual-Based Error Correction] as second independent motivation for TTA necessity
- Proposition 2 (resolution error decomposition, Sec 5.1): total error = statistical-learning term + C²_{s,d} N^{-2s} E||G(u)||²_{H^s}; the spectral-tail term grows as the solution becomes less regular (higher Re = less regular); formal link between PDE regularity under Re shift and model error — extends Qin et al. spectral bias argument [76_Spectral Perspective] with a theoretical bound

**Gap:**
- No runtime OOD score — all spectral and rollout diagnostics require ground truth; degradation factor not computable at deployment; your B3 z-score [scripts/detector_roc.py] is the label-free deployable equivalent
- NS parameter shift shows mild relative degradation factor (Fig 2c); rollout dominates — in absolute terms error at high Re is large but the ratio D looks modest; does not contradict your OOD motivation but requires careful framing when citing
- No TTA or adaptation; pure evaluation framework

---

## [36_FNO-expl] Fourier Neural Operators Explained: A Practical Perspective — Duruisseaux, Kossaifi, Anandkumar (2025)

**Summary:** practitioner guide to FNO hyperparameter tuning; establishes n_modes as the primary capacity/generalization knob via Nyquist constraints, spectral bias, and aliasing mechanics

**Tags:** GROUND, STEAL

**Take:**
- Power spectrum diagnostic: examine training data power spectrum, select n_modes as the smallest k retaining most signal energy; this is the principled starting point for your n_modes ↔ OOD detectability calibration curve (p.54)
- Nyquist hard constraint: n_modes ≤ N/2 per dimension (N = grid resolution); violating this folds high-frequency energy back into lower modes during training, corrupting spectral representations (p.14-15, 55)
- Nonlinearity-induced aliasing under resolution shift: if trained with n_modes = N, aliased high-freq energy contaminates lower modes during training; at higher test resolution aliasing vanishes and the model applies wrong transformations — mechanistic explanation for why FNO generalizes differently across Re regimes (p.56)
- iFNO adaptive mode expansion criterion: cumulative spectral energy ratio q(K,P) = Σ_{k=1}^K P_k / Σ_{k=1}^p P_k; activate more modes when q < α — directly usable as a data-driven threshold for your n_modes sweep under Re shift (p.65)

**Gap:**
- Never connects n_modes to OOD detectability — discusses spectral capacity and generalization failure but does not formalize "frequencies you can represent = parametric shifts you can detect"
- No distribution shift or Re/viscosity sweep; no OOD framing anywhere in the paper
- Engineering guide only — no experiments, no benchmarks, no quantitative claims about OOD performance

---

## [82_funDPS] Guided Diffusion Sampling on Function Spaces with Applications to PDEs (FunDPS) — Yao, Mammadov, Berner, Kerrigan, Ye, Azizzadenesheli, Anandkumar; Caltech/NVIDIA/Oxford (2026) | arXiv:2505.17004 | NeurIPS 2025

**Summary:** function-space diffusion posterior sampling — UNO backbone + GRF noise + plug-and-play physics/observation guidance at inference; 10× fewer steps than DiffusionPDE, 32% error reduction; code: github.com/neuraloperator/FunDPS

**Tags:** BEAT

**Code reference:** guidance loop implementation (score update + likelihood gradient at each denoising step using a neural operator backbone) is the cleanest open reference for building a [D]-spectrum baseline; see `FunDPS/sampling.py`; caveat: UNO backbone not drop-in with FNO — retraining required

---

## [86_BatteryTTT] Adapting Amidst Degradation: Cross-Domain Li-ion Battery Health Estimation via Physics-Guided Test-Time Training — Feng, Hu, Li, Zhang; Xiamen/Bristol/HKU (2024) | arXiv:2402.00068v3

**Summary:** one-sample streaming TTT for battery SOH estimation — physics-guided self-supervised loss (masked reconstruction + 1-RC ECM ODE residual) + structural anti-forgetting (frozen encoder, adapting auxiliary head only); eliminates months-long UTD collection by adapting on each individual cycle as it arrives

**Tags:** STEAL, FRAME

**Take:**
- Y-shaped architecture as structural anti-forgetting: freeze encoder f + main task head h; update only auxiliary self-supervised head g via ℒ_PG-SSL; main prediction is always h ∘ f_x(x) — no weights touching the task solution; direct template for FNO [B]-type TTA where spectral conv layers = f, lifting/projection = g (→ [[68_AC Power Flow Surrogates]])
- PPA (Prefix Prompt Adaptation): freeze all model weights; optimize only small soft-prompt prefix p* via physics loss; reduces trainable parameters by ~10³× vs full TTA fine-tuning; practical translation to FNO: norm-bounded update on a tiny parameter subset (→ [[68_AC Power Flow Surrogates]] explicit ‖φ‖₂ ≤ ε as the continuous analogue)
- TTT justification verbatim (Table 1): woDegr needs 8473h UTD collection; SSF-WL needs 2542h; BatteryTTT needs 0h — adaptation on 1 arriving sample eliminates the collection bottleneck; cite this table when arguing label-free TTA is the only viable deployment mode for scarce-data physics domains
- Mask ratio sweet spot 0.7–0.9: masking too little (≤0.6) yields trivial reconstruction, no useful gradient; masking 75%+ forces representation learning — consistent with MAE intuition; directly applicable if a masked-reconstruction auxiliary task is added to FNO TTA

**Gap:**
- Physics constraint is ODE, not PDE: 1-RC ECM = θ₁I + θ₂û + ü = 0, a first-order scalar ODE on a 1D time series; NS residual is a 2D spatial PDE — dimensionally incomparable; the "physics-guided" framing is structural prior art only, not a recipe (→ [[68_AC Power Flow Surrogates]] for closer algebraic-constraint cousin)
- Single-sample sufficiency assumes 4-parameter ECM: one battery cycle fully constrains OCV, R, Rp, Cp; FNO has millions of parameters — 10 gradient steps on 1 NS trajectory is informationally insufficient; small-batch OOD adaptation (multiple trajectories) required in the NS port
- Monotone drift assumption: SOH only decreases; adaptation is always unidirectional; NS Re shift is bidirectional — sign of shift must be detected separately (B3 directional probe from OOD detector) before adaptation direction is known
- No spatial structure: SOH is a global scalar; Y-shaped head split treats battery as pointwise; FNO correction is spatially distributed — masked reconstruction auxiliary has no direct 2D analogue on vorticity fields
- Compute: TTA is 10–100× slower than no-TTA (Table 5: GPT2+ 51ms/sample); feasible for BMS (<500ms requirement) but cost is real; LSTM variant (6.75ms/sample) is the compute-efficient reference point

---

## [85_lora] LoRA: Low-Rank Adaptation of Large Language Models — Hu et al.; Microsoft (2021) | arXiv:2106.09685

**Summary:** freeze pretrained weights W_0, inject ΔW = BA (B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r ≪ min(d,k)); merge at deployment — zero inference latency

**Tags:** STEAL

**Take:**
- ΔW = BA reparameterization reduces TTA trainable parameters from d×k to (d+k)×r; if [B]-type TTA (→ [68_AC Power Flow Surrogates]) updates FNO spectral conv layers (larger matrices), low-rank constraint reduces per-instance adaptation cost and acts as implicit anti-forgetting complementary to Dogoulis's explicit ‖φ‖₂ ≤ ε norm bound; rank r is a second hyperparameter alongside ε for the TTA budget
- Deferred: for FNO lifting/projection layers (small dense matrices) the saving is marginal; relevant only if spectral conv layers are unfrozen
