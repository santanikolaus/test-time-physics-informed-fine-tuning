# OOD Detection Literature

Goal: locate the exact research gap filled by the B3 band-resolved residual detector —
a calibrated, spectral, directional OOD score for FNO surrogates under parametric Re shift.

---

## TODO

- [x] Gopakumar et al. 2025 — "Calibrated Physics-Informed Uncertainty Quantification" (arXiv:2502.04406) — FRAME/GROUND; not a direct competitor; appendix baseline
- [x] Mouli et al. ICML 2024 — "Using UQ to Characterize and Improve OOD Learning for PDEs" — FRAME/GROUND; n-MeRCI metric; ensemble diversity ≠ residual detection
- [x] Raonic et al. ICLR 2026 — "Certificate of Trust: Task-Aware OOD Detection for Scientific AI" — FRAME; p(x) failure motivates residual; add FPR/FDR to metrics
- [ ] Lara Benitez et al. 2023 (arXiv:2301.11509) — OOD risk bounds for neural operators (Helmholtz); theory-first, no runtime detector; GROUND candidate for formal generalization bounds under operator OOD transfer
- [ ] Dual-branch neural operator for OOD detection (ScienceDirect 2024, pii/S0955799724005551) — verify relevance first; only paper explicitly combining neural operator + OOD detection architecture
- [ ] "Spectral bias in physics-informed and operator learning" (arXiv 2026) — **HIGH PRIORITY**; shows low-frequency residual reduction masks missing high-frequency error modes; independent published grounding that aggregate residual is insufficient and band-resolved is necessary; find exact arXiv ID
- [ ] Fanaskov & Oseledets 2022 (arXiv:2205.10573) — pointwise nonlinearity aliasing between FNO spectral conv layers; GROUND for M2 (within-passband near-cutoff distortion); read before citing
- [ ] Kraichnan 1967 (Phys. Fluids 10:1417) + Boffetta & Ecke 2012 (Annu. Rev. Fluid Mech. 44:427) — 2D forced turbulence: energy cascade is inverse, forward cascade is enstrophy E(k)∝k⁻³; GROUND for physics narrative; do not say "energy to high k"
- [ ] Bartolucci & Lanthaler 2024 (arXiv:2405.02221) — formal aliasing-error convergence bounds for FNOs; GROUND for M1 (truncation-induced leakage); abstract only verified — read body before quoting rates
- [ ] Yang et al. 2024 (arXiv:2110.11334) — generalized OOD detection survey; FRAME; standard field-surveyed cite
- [ ] Psaros et al. 2023 (JCP 477:111902) — UQ in scientific ML survey; FRAME; pairs with Yang at OOD × SciML intersection

---

## Detection Landscape

Shared signal: PDE residual ℛ(û, ν) — unsupervised, label-free, computable at inference.
Axis: what part of the residual field is used as score.

```
[A] Aggregate scalar reduction
    score:   ‖ℛ‖² / N — single number, no spatial or spectral structure
    blind:   Re-discriminating signal at k > n_modes buried under low-k noise
    papers:  Gopakumar et al. 2025 (PRE-CP); DiffusionPDE proxy loss; Dogoulis TTT trigger
    gap:     Step 4 shows Δ collapses 2.38→0.17 as train Re rises — aggregate is dead for op≥500

[B] Band-resolved scalar reduction
    score:   E_B3 = Σ_{k=9}^{16} |r̂(k)|² — energy in first octave above n_modes truncation
    z-score: z = (E_B3 − μ_ID) / σ_ID — calibrated against operator's own ID distribution
    signal:  Step 6/8: max|d| ≥ 2.2σ every op, 4.21σ at op1000
    gap (ours): band predicted by n_modes geometry, not tuned to data; directional sign probe

[C] Gradient-space / activation-space reduction
    score:   projection of physics-loss gradient onto ID subspace (GradOrth); NTK-based (GradPCA)
    papers:  GradOrth [75], GradPCA [71]
    gap:     classification-only benchmarks; no PDE/operator experiments
```

Differentiator phrase (load-bearing):
"We do not propose the PDE residual as a reliability signal — that is established
(Gopakumar et al., 2025). We show that where in the spectrum the residual concentrates
is determined by the operator's truncation geometry, which turns an undifferentiated
reliability score into a calibrated, directional regime-shift detector.
The detection band is predicted by n_modes, not tuned to the data."

---

## Background Citations
<!-- foundational anchors — not OOD papers; one-liner format; cite when writing method/related work sections -->

- Li et al. 2021 FNO (arXiv:2010.08895) — Fourier truncation at n_modes is the architectural source of B3; cite when introducing the operator and the band prediction claim
- Li et al. 2021 PINO (arXiv:2111.03794) — PDE residual as physics loss on FNO for Kolmogorov flow; cite when introducing residual signal provenance
- Rahaman et al. 2019 (arXiv:1806.08734) — spectral bias in neural networks toward low frequencies; cite as general spectral bias lineage before the FNO-specific argument

---

## Cross-references from TTA literature
<!-- papers already fully mapped in msc/tta/tta-research.md — one-liner format; do not duplicate -->

- [76_Spectral Perspective] Qin et al. 2024 — Fourier parameterization bias; B3 doubly-blind zone mechanism; GROUND for claim (3)
- [80_Failure Modes] Shikhman 2026 — GT-based failure atlas; no runtime score; FRAME (our B3 z-score is the label-free equivalent)
- [71_NTK] GradPCA (Seleznova et al. 2025) — Theorem 4.1 formal sufficient condition for spectral OOD; GROUND for claim (1)
- [75_GradOrth] Behpour et al. 2023 — activation-SVD subspace OOD; regression-compatible algorithm; FRAME (orthogonal detection axis — weight space vs solution space)
- [73_Residual-Based Error Correction] Cao et al. 2022 — Theorem 1: posterior error ≤ c·‖ℛ‖; formal grounding that residual magnitude = approximation error; GROUND for OOD trigger
- [36_FNO-expl] Duruisseaux et al. 2025 — n_modes sets Nyquist cutoff; B3=[9,16] sits in aliased-energy blind zone above truncation; GROUND for "band predicted by architecture, not tuned to data"
- [68_AC Power Flow Surrogates] Dogoulis et al. 2025 — uses aggregate `‖ℛ‖ > τ` as TTT trigger; published instance of [A] aggregate reduction as OOD gate; FRAME: trigger is blind to Re shift for op≥500 (same collapse as Step 4)

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

## [87_Gopakumar] Calibrated Physics-Informed Uncertainty Quantification — Gopakumar et al. (2025) | arXiv:2502.04406v2 | ICML 2025

**Summary:** PDE residual `|D(f̂(X))|` as conformal nonconformity score → calibrated coverage bounds on neural PDE predictions; data-free (no labels); coverage guaranteed within same deployment regime

**Tags:** GROUND, FRAME

**Take:**
- Theorem A.1: residual is a valid nonconformity score with formal coverage guarantee — cite to pre-empt "why should the residual carry OOD signal?"

**Gap:**
- UQ within fixed regime ≠ regime-shift detection: quantifies physics inconsistency assuming ν_test = ν_train; cannot detect when the regime has changed
- Coverage guarantee requires exchangeability: calibrated at Re_train, tested at Re_test → theorem premise violated by construction
- Aggregate `|D(f̂(X))|` blind to Re shift: same collapse as Step 4 — score doesn't separate regimes for op≥500

**Action:** calibrate CP-PRE at training Re, evaluate on 5×5 op/Re matrix; show coverage breaks and PRE scores don't separate regimes; converts competitor into empirical evidence for the gap

---

## [88_Mouli] Using UQ to Characterize and Improve Out-of-Domain Learning for PDEs — Mouli et al. (ICML 2024) | arXiv:2403.10642

**Summary:** ensemble diversity as OOD signal for parametric shift in NOs; DiverseNO = single FNO with M diverse prediction heads; Operator-ProbConserv projects predictions onto conservation constraints using variance estimates

**Tags:** FRAME, GROUND

**Take:**
- n-MeRCI metric (Eq. 7): `(τ·mean(σ_i) − MAE) / (max(|û_i − u_i|) − MAE)` — measures correlation between uncertainty estimate and prediction error; scale-independent; lower = better; **use to evaluate B3 z-score as uncertainty proxy**: does high z correlate with high prediction error?
- Establishes parametric shift (non-overlapping parameter support) as the canonical OOD setting for NOs — cite as problem framing

**Gap:**
- OOD signal is prediction variance across heads — requires retraining with diversity loss; not post-hoc, not deployable on existing checkpoints
- No physics residual used anywhere in detection or correction
- 1D PDEs only (heat, porous medium, Stefan); Operator-ProbConserv requires known linear conservation constraint — no NS port
- Does not produce a binary OOD decision; produces calibrated variance estimates

---

## [89_certificate] Towards a Certificate of Trust: Task-Aware OOD Detection for Scientific AI — Raonic, Mishra, Lanthaler; ETH Zurich (2026) | arXiv:2509.25080

**Summary:** joint likelihood p(x, y_pred) as OOD certificate via score-based diffusion model trained on (input, prediction) pairs; zero-shot, task-aware; tested on NS and Wave

**Tags:** FRAME, GROUND

**Take:**
- p(x) alone fails for NS (Fig 2 left): NS-Sines has high input likelihood but large errors — proves task-agnostic (input-only) detection insufficient; **cite to motivate residual-based detection**: B3 score uses û (task-aware) not just x, achieves this for free without a diffusion model
- Metrics (Table 1): ACC, FPR, FDR, AUROC — standard OOD detection suite; **add FPR and FDR to your reporting** alongside existing AUC

**Gap:**
- Requires training a separate diffusion model on (x, y) pairs — not post-hoc, data-hungry, architecture-coupled; yours is post-hoc, zero additional training
- NS at fixed ν = 4×10⁻⁴, IC-family shift only — no Re/ν parametric sweep
