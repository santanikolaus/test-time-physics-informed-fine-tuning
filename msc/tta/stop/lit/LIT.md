# LIT: label-free STOP / START for TTA of neural PDE surrogates (KF Re100 -> Re500)

Confidence tags: [H] verified via abstract/local PDF; [M] verified title/venue, content from summary; [L] seen only via aggregator or 2026 preprint, unverified; [R] from recall, unverified.
Local libs: `~/Documents/JKU/thesis/papers` (=J), `~/Downloads/thesis/papers` (=D).

Observed facts any rule must survive:
- F1 misspecified-ν residual min @ step 30; GT improves to 500.
- F2 residual ranking inverts across models.
- F3 large-scale (low-k) dip steps 5-65 -> must not stop inside.
- F4 bad configs (wrong ICs, IC-only, residual-only): loss monotone down, GT up -> loss decrease is not a START/help signal.

---

## 1. Label-free model / checkpoint selection (TTA, UDA)

| Work | Signal | Transfers to this setting? |
|---|---|---|
| Zhao, Liu, Alahi, Lin 2023, ICML, arXiv 2306.03536 "On Pitfalls of TTA" (TTAB) [H] | Pitfall #1: HP/model selection extremely hard, online batch dependency; even w/ labels | Framing only. Cite as motivation. |
| Cygert et al. 2024, arXiv 2407.14231 "Realistic Evaluation of TTA: Unsupervised HP Selection" [M] | Surrogate selection (entropy, consistency, source acc., etc.); SOTA methods degrade vs oracle; only reset-every-step robust | Key negative result: oracle-tuned TTA overstated. Supports reporting oracle vs label-free gap. Most surrogates classification-specific. |
| Musgrave, Belongie, Lim 2021, arXiv 2111.15672 "UDA: A Reality Check" [M] | Validators (IWCV, DEV, SND, entropy...) frequently fail; checkpoint selection explicitly in scope | Framing. |
| Sugiyama et al. 2007 JMLR "Covariate shift adaptation by IWCV" [R]; Shimodaira 2000 IWV [R] | Importance-weighted source risk, w = p_t(x)/p_s(x) | NO. Assumes covariate shift p(y|x) fixed. Re100->Re500 changes the dynamics G itself: p(y|x) shifts by construction. Also support mismatch (Re500 ICs off Re100 attractor) -> unbounded weights. |
| You, Wang, Long, Jordan 2019, ICML "Towards Accurate Model Selection in Deep UDA" (DEV) [R] | IWCV + control variate | NO, same covariate-shift assumption. |
| Saito et al. 2021, ICCV "Tune it the right way" (SND) [R]; entropy, InfoMax; Hu et al. 2023 NeurIPS MixVal [M]; Hu et al. 2024 NeurIPS D&B EnsV [M] | Neighborhood/softmax density, mixup probes, ensemble-as-reference | NO (softmax-based) except EnsV idea: ensemble of candidates as pseudo-reference -> regression analog = checkpoint-ensemble mean as pseudo-target (see §3.5). |
| Garg et al. 2022, ICLR, arXiv 2201.04234 ATC [H] | Threshold on confidence learned on source | NO (confidence). Note thm: estimating target acc. as hard as learning target predictor w/o shift assumptions -> any label-free stop rule needs a structural assumption; ours = invariant measure + known PDE form. |
| Guillory et al. 2021, ICCV DoC [R] | Difference of confidences | NO (classification). |
| Miller et al. 2021 ICML accuracy-on-the-line, 2107.04649 [R]; Baek et al. 2022 NeurIPS agreement-on-the-line 2206.13089 [R] | ID vs OOD acc linear in probit space; agreement mirrors accuracy | Needs model population + acc-like metric. Regression transfer open. |
| Kim, Sun, (Baek), Raghunathan, Kolter 2023 NeurIPSW / 2024 NeurIPS, arXiv 2310.04941 [H] | TTA strengthens AotL -> HP selection w/o OOD labels | Closest "TTA HP selection" result. Regression: replace agreement by correlation between checkpoint predictions; unvalidated. |
| Rosenfeld, Garg 2023 NeurIPS, arXiv 2306.00312 Dis^2 [H] | Error upper bound via disagreement discrepancy | Classification; conceptual: bound, not point estimate. |
| Jiang et al. 2022 ICLR "Assessing generalization via disagreement" 2106.13799 [R] | Disagreement of SGD runs ~ test error (needs calibration) | Regression analog = ensemble spread; Mouli 2024 shows works for NOs (§3.5). |
| arXiv 2608.18330 (2026) regression dynamic ensembling under shift [L] | Plain disagreement weakly correlated (+0.02) with gain | Caution for disagreement-as-stop in regression. |

Bottom line §1: no label-free selector from classification TTA/UDA transfers; covariate-shift validators (IWV/DEV) are invalid under dynamics shift. Remaining transferable ideas: ensemble/agreement (model-population) signals.

## 2. Regression / neural-surrogate TTA: how steps are chosen

- Adachi, Yamaguchi, Kumagai, Hamagami 2025, ICLR, arXiv 2410.03263 SSA [H]. Feature alignment in PCA significant subspace, scalar regression. Fixed epochs/HP; tuning per dataset (code kzkadc/regression-tta). No label-free stop.
- Zimmel, Galletti, Setinek, Brandstetter, Zellinger 2025 NeurIPS, "Towards (Accurate) TTA for Neural Surrogates" (SimRA) [H, local D/J `10_Towards_TTA_Simulation.pdf`]. Representation alignment for multi-dim regression; notes importance weighting not directly usable; no stop rule.
- Zimmel et al. 2026, arXiv 2602.15820 SATTS [H, local `D/10_Zimmel_SATTS.pdf` ~l.415-455]. D-optimal source subset + KL feature alignment + anchor; STOP = IWV on D-optimal source samples, density ratio in latent space (Gaussian), line search over lr, "evaluating after each adaptation step and stopping once further updates no longer improve the objective". Only existing explicit stop rule in neural-surrogate TTA. Invalid here (covariate-shift assumption, §1). Cite as closest prior + reason for new criterion.
- Setinek, Galletti et al. 2025, arXiv 2506.12007 SIMSHIFT [H, local J `12_SIMSHIFT.pdf`]. Benchmarks UDA w/ IWV, DEV, Source-Best, oracle; model selection named as core bottleneck.
- Li, Zheng et al. 2021/2024, arXiv 2111.03794 PINO [H, local `34_PINO.pdf` l.444-487]. Instance-wise fine-tuning: L_pde + alpha L_op (anchor); KF transfer Re100->Re500. Residual at CORRECT target Re; fixed iteration budget, error curves vs iterations reported with GT; no stop criterion. Contrast: our residual is at training ν (misspecified).
- Subramanian et al. 2023 NeurIPS, arXiv 2306.00258 [H, local `95_Subramanian.pdf`]. Supervised fine-tuning transfer; labeled downstream data; no label-free selection.
- arXiv 2603.15431, Medvedev et al. (Fraunhofer IISB), ICLR 2026 Workshop AI & PDEs [H, read 2026-11-06]. Poseidon-T fine-tuned offline on steady 2D Poisson (+ Helmholtz app.); physics-only vs hybrid (data + residual, equal weight); M 1–4096 labels; OOD = source-shape shift only; fixed epochs, no stop/gate; no time-dependent PDE, no coefficient shift, no TTA. Claim: hybrid learns from M=1 where data-only needs ≥32. Distinct from ours.
- arXiv 2601.19091 NOVA (2026) [L]. Data-free physics-informed adaptation via convex closed-form solve -> sidesteps stopping (no iterations).
- Serrano et al. 2026, arXiv 2602.00884, ICML 2026 test-time operator splitting [L]. Fixed weights, composition search; recovers PDE params -> related to ν̂ estimation (§5 C2).
- Cao et al. 2022/JCP 2023, arXiv 2210.03008 [H, local `73_Residual-Based Error Correction.pdf`]. Residual-based correction = one Newton step on linearized PDE; links to a-posteriori modeling-error estimation; quadratic error reduction. Requires correct PDE + well-posed elliptic-like problem; not chaotic.

Gap statement (thesis): no regression/surrogate TTA paper provides a label-free stop rule valid under dynamics (parameter) shift.

## 3. Physics / numerics side

### 3.1 Residual-based a-posteriori bounds and why residual is non-identifying here
- Stability argument [R, standard]: ||u - u_hat||(t) <= C exp(L t) * (||e_0|| + integral ||R(u_hat)||). For chaotic NS, L ~ lambda_max > 0: bound exponentially loose; small residual != small trajectory error. Mishra & Molinaro 2022 IMA JNA, arXiv 2006.16144 [R]; De Ryck, Jagtap, Mishra 2023 (NS PINN error) arXiv 2203.09346 [R]; Hillebrecht & Unger 2022 arXiv 2203.17055 certified a-posteriori PINN [R] (linear ODE/semigroup setting).
- Dual-weighted residual: Becker & Rannacher 2001 Acta Numerica [R]. Goal-oriented error = <residual, adjoint>. Adjoint for chaotic long-time QoI blows up (Lea, Allen, Haine 2000 Tellus [R]; shadowing/LSS Wang 2014 [R]) -> DWR not usable for correlation horizon. Mention only as why.
- Misspecified ν: residual at ν_train has nonzero floor for true Re500 trajectory; minimizer of misspecified residual is Re100-like dynamics -> explains F1 (early min, then optimizer trades residual vs IC-term) [interpretation].
- "How PINNs cheat" NeurIPS 2022 WS (double pendulum) [M]: low-physics-loss low-frequency solutions diverging from truth in chaotic systems.
- Silent Failures in PINNs: parameter poisoning, arXiv 2606.25151 (2026) [L]: misspecified coefficients -> loss within clean baseline, large error; Re100 vs Re90 asymmetric; parameter-sweep defense, none universally works.
- McShannon & Dietrich arXiv 2610.23948 (2026) closed-form audit of parameter misspecification [L]: normalized residual vs recovery error log-log corr 0.72; raw loss can be lower for misspecified model.
- Cross-model incomparability precedent for F2: Raonić, Mishra, Lanthaler 2025/26, arXiv 2509.25080 [H, local `J/89_certificate.pdf` ~l.2261]: diffusion joint-likelihood score correlates with error within a model but "cannot be reliably used for model selection ... comparisons between different models are not interpretable". Same structure as residual-ranking inversion -> any self-output score likely valid within-trajectory (across steps) only.

### 3.2 Invariant-measure / spectrum signals (label-free, ν-agnostic)
- Jiang, Lu, Orlova, Willett 2023 NeurIPS, arXiv 2306.01187 [H]: train NOs to preserve invariant measure (OT on summary stats, or contrastive); RMSE poor signal for chaos; FNO backbone.
- Schiff et al. 2024 ICML, arXiv 2402.04467 DySLIM [H, local `J/91_Dyslim.pdf`]: invariant-measure regularizer (MMD on sampled states) improves short-term and long-term; KF included.
- Lippe et al. 2023 NeurIPS, arXiv 2308.05732 PDE-Refiner [H, local `D,J/90_lippe.pdf`]: neglected low-amplitude high-k modes drive rollout failure; spectrum diagnostics; correlation-horizon metric (time until Pearson < 0.9) = our GT metric origin.
- Relevance: target ICs are samples of Re500 invariant measure -> E(k) of target ICs is a label-free reference. Spectrum of predicted frames (esp. late frames) vs this reference is independent of ν misspecification.

### 3.3 Semigroup / time consistency
- Shikhman 2026, arXiv 2605.26324, ICML 2026 "Semigroup Consistency as a Diagnostic for Learned Physics Simulators" [L/M]: normalized error ||G(u,s+t) - G(G(u,s),t)||; trajectory-level Spearman rho = 0.635 with rollout degradation; 1D heat/Burgers only; as regularizer mixed.
- Chen et al. 2023 Deep-OSG arXiv 2302.03358 [M]: semigroup-embedding losses.
- Feasibility here: model maps IC -> 65 frames; any predicted frame on attractor is a valid IC -> one extra forward per probe.

### 3.4 Budget closure (2D NS, Kolmogorov forcing)
- Stationarity: <f·u> = ν<|ω|^2> (+ drag) energy; <f_ω·ω> = ν<|∇ω|^2> enstrophy. Boffetta & Ecke 2012 Annu Rev Fluid Mech 44:427 [R]; Chandler & Kerswell 2013 JFM (KF input I = dissipation D on average) [R]. Depends on ν -> must use ν̂, not ν_train; with true ν, a closure test is label-free.

### 3.5 Ensembles / disagreement for NOs
- Mouli et al. 2024 ICML, arXiv 2403.10642 [H, local `J/88_Mouli.pdf`]: many UQ methods fail even moderately OOD; ensembles of NOs give uncertainty well-correlated with error; DiverseNO multi-head cheap surrogate.
- Gopakumar et al. 2025 ICML, arXiv 2502.04406 CP-PRE [H, local `J/87_Gopakumar.pdf`]: PDE residual as nonconformity score -> data-free conformal UQ with coverage. Same caveat: residual must be at correct physics.

## 4. Shift detection to START adaptation

- EATA, Niu et al. 2022 ICML, arXiv 2204.02610 [R]: entropy-threshold sample selection + redundancy filter + Fisher anchor. SAR, Niu et al. 2023 ICLR, arXiv 2302.12400 [M]: drop high-entropy/large-grad samples, sharpness-aware, model recovery (reset) when EMA loss collapses. CoTTA, Wang et al. 2022 CVPR, arXiv 2203.13591 [H, local `J/19) Continual Test-Time Domain Adaptation.pdf`]: stochastic restore of source weights. TENT, Wang et al. 2021 ICLR, arXiv 2006.10726 [H, local `J/15_TENT...pdf`].
- Press et al. 2023 NeurIPS RDumb, arXiv 2306.05401 [R]: periodic full reset beats most CTTA; interval hard to pick. RDumb++ arXiv 2601.15544 (2026) [L]: entropy + KL drift score triggers full/soft reset.
- Chakrabarty et al. 2023 ICCVW "A Simple Signal for Domain Shift" [M]: consecutive-batch feature similarity threshold -> reset.
- Physics-residual OOD: phase-2 result (own); arXiv 2608.18004 (2026) flow-matching energy + PDE residual complementary OOD scores [L]; Raonić 2025 task-aware joint-likelihood OOD [H].
- Change-point / two-sample: MMD two-sample test, Gretton et al. 2012 JMLR [R]; Rabanser, Günnemann, Lipton 2019 NeurIPS "Failing loudly" (dimension reduction + two-sample tests) [R].
- Key distinction (F4): "shift present" (detectable: IC spectra/residual OOD) != "adaptation will help". Classification CTTA gates only address the former + collapse; none predicts benefit. Benefit prediction needs an independent label-free proxy (§5) evaluated before/after a short probe adaptation.

## 5. Ranked candidate signals for this setting

C1. Spectral invariant-measure distance (STOP primary; START secondary)
- Def: D_spec(θ) = mean_k |log E_pred,θ(k) - log E_IC(k)| over predicted frames t >= t0 (late frames, statistically stationary) on pool + held-out unlabeled ICs; E_IC from target ICs (Re500 invariant-measure samples). Variant: MMD/OT on (energy, enstrophy, E(k) bins) as in Jiang 2023 / DySLIM.
- Why: correlation horizon fails when wrong scales grow; low-k energy mismatch is exactly the F3 dip, high-k mismatch = PDE-Refiner failure mode. ν-agnostic -> immune to F1. Absolute reference -> comparable across models (F2).
- Failure: matching statistics != pointwise trajectory accuracy (a stats-correct, phase-wrong model scores perfect); 5 ICs -> noisy E(k) at low k (few modes); can be gamed if spectrum used in loss (do not train on it).
- F3 handling: dip registers as D_spec increase at low k -> rule "stop at argmin after D_spec has returned below its step-0 value" or use patience over the dip window.
- Cost: FFT of predictions, negligible.
- Cite: Jiang 2023 (2306.01187), Schiff 2024 (2402.04467), Lippe 2023 (2308.05732).

C2. Residual at fitted viscosity ν̂ (START + STOP)
- Def: ν̂ = argmin_ν ||R_ν(target IC trajectories predicted / or IC pairs)|| via 1D sweep (residual linear in ν -> closed-form least squares: ν̂ = <∂tω + u·∇ω - f, Δω>/<Δω,Δω>). START: |ν̂ - ν_train|/ν_train > τ. STOP: track R_ν̂(θ_t); or use ν̂ in budget closure.
- Why: removes misspecification causing F1; residual at correct ν is the PINO-Re500 setting (which works). Closed-form ν̂ makes it label-free.
- Failure: ν̂ estimated from model predictions is circular (Re100 model output yields ν̂≈ν_train) -> estimate ν̂ from target ICs alone is impossible (single snapshots, no ∂t); requires trajectory -> use adapted-model trajectory with care, or estimate from spectrum dissipation range/Kolmogorov scale of ICs (η=(ν^3/ε)^{1/4}, ε from forcing balance) [interpretation]. Residual still non-identifying under chaos (§3.1) -> use as START/coarse, not fine stop.
- Cost: one residual eval per candidate ν (closed form: one).
- Cite: Silent Failures 2606.25151 [L] (parameter sweep defense); McShannon & Dietrich 2610.23948 [L]; PINO 2111.03794 (Re-transfer with correct residual); Serrano 2602.00884 [L] (param recovery).

C3. Semigroup / time-consistency error (STOP)
- Def: S(θ) = mean_u ||G_θ(u)[t1+t2] - G_θ(G_θ(u)[t1])[t2]|| / ||G_θ(u)[t1+t2]||, t1 ∈ {8,16,24}, on held-out unlabeled ICs.
- Why: chaotic drift + mis-learned dynamics break flow-map composition; error accumulates where correlation horizon ends. No ν required.
- Failure: a model trained to produce smooth/decaying (Re100-like) dynamics can be self-consistent yet wrong (consistent-but-wrong fixed point); evidence only 1D (rho=0.635); input-distribution shift of predicted frames (blurrier than true ICs).
- Cost: 1 extra forward per IC per t1.
- Cite: Shikhman 2026 (2605.26324) [L/M]; Deep-OSG 2302.03358.

C4. Checkpoint/ensemble disagreement (STOP; cross-model)
- Def: spread over K adapted runs (seeds / IC subsets / last-N checkpoints) of predictions on held-out ICs; horizon proxy = first t where inter-run Pearson < 0.9 (mirrors GT metric).
- Why: ensembles track NO error OOD (Mouli 2024); inter-run correlation horizon is a direct, label-free analog of GT correlation horizon; agreement-on-the-line logic (Kim 2024).
- Failure: shared bias (all runs drift same Re100-ward direction) -> agreement high, error high; regression disagreement weakly predictive in 2608.18330 [L]; last-N checkpoints highly correlated -> use seeds/IC-subset bootstraps.
- Cost: K× adaptation (K=3-5).
- Cite: Mouli 2024 (2403.10642); Kim et al. 2310.04941; Jiang 2022 (2106.13799) [R].

C5. Enstrophy/energy budget closure at ν̂ (STOP, sanity)
- Def: B(θ) = |<f_ω·ω>_t - ν̂<|∇ω|^2>_t| / <f_ω·ω>_t over late predicted frames.
- Why: stationary attractor requires injection = dissipation; violation = wrong small-scale content (high-k) -> correlates with fast decorrelation.
- Failure: depends on ν̂ quality (C2); time averages over 65 frames x 5 ICs noisy; satisfied by wrong-but-balanced fields.
- Cost: negligible.
- Cite: Boffetta & Ecke 2012 [R]; Chandler & Kerswell 2013 [R].

C6. START gate: two-sample test on IC statistics + probe-benefit check
- Def: (a) shift: MMD / KS on IC spectra E(k) or enstrophy, target vs source training ICs, permutation p-value; or phase-2 residual OOD score. (b) benefit: run probe adaptation of >= ~80 steps (must clear F3 dip, steps 5-65), accept only if C1 (and C3) improve vs step 0.
- Why: (a) cheap, label-free, directly measures attractor change; (b) addresses F4 (loss monotone but GT worse) with an independent proxy rather than the training loss.
- Failure: (a) detects shift, not helpfulness; 5 ICs -> low power for small ΔRe; (b) inherits C1/C3 failure modes and F3 dip forces probe length > 65 steps.
- Cost: (a) negligible; (b) ~15-20% of full adaptation.
- Cite: Gretton 2012 [R]; Rabanser 2019 [R]; Chakrabarty 2023; RDumb++ 2601.15544 [L]; EATA/SAR (gating precedents).

Recommendation: C1 as primary stop (with dip-aware rule), C3 as tie-breaker/second view, C2 for START (ν̂ shift) and to de-misspecify the residual; validate each by Spearman(signal_t, GT-horizon_t) along the step axis per model and across models (F2 test).
