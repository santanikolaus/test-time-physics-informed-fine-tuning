# IC-Structure Exploitation for Trajectory Branch Selection in a Deterministic One-Shot NS Surrogate

**Date:** 2026-06-28
**Depth:** exhaustive
**Confidence:** ~90% (plateaued; see note)
**Sources:** 50 unique sources across 6 search rounds (Tier-1 dominant)

> **Confidence note.** The 95% threshold nominal to "exhaustive" depth is not reachable here: three of the four verdicts rest partly on *absence of evidence* (no published method does X) plus a mechanistic argument. Non-existence cannot be proven by search. Confidence is honestly reported per sub-question (88–92%) with strong Tier-1 convergence and both cross-cutting contradictions resolved against authoritative sources.

---

## Executive Summary

No published method makes a **deterministic, one-shot** PDE surrogate commit to the correct late-time k2–5 phase branch using only IC-derived structure, and three of the four proposed routes are blocked by the same mechanism rather than by missing engineering. The decisive facts: (i) the late-time turbulent phase lives in the **continuous** part of the Koopman spectrum, which has *no eigenfunction basis* — so there is no IC-persistent slow discrete mode to condition on (Q_a, negative); (ii) contrastive/metric IC embeddings demonstrably help conditioning but only marginally (~3% on Navier–Stokes) and are designed to recover *environment parameters / invariant statistics*, not phase identity (Q_b, weak-negative); (iii) a **physics-residual-only** IC optimization is degenerate — every valid trajectory satisfies the NS residual equally, the spectrum constrains statistics not phase, and adjoint-based inversion within the Lyapunov window is dominated by exactly the small scales that set the branch (Q_d, negative). The **one viable direction is Q_c**: in the *exact thesis configuration* (Re=100, 128² grid, k=4 forcing) two independent 2026 studies show that **large-scale (k≲3) coherent structure in the current state predicts late-time extreme-event behavior up to >4 Lyapunov times** — but only for a state-dependent *subset*, and the demonstrated skill is largely distributional rather than deterministic pointwise. The supported negative for everything else is mechanistic: branch divergence is driven by small-scale IC sensitivity whose IC→branch map is (a) spectrally continuous, (b) physically non-invertible, and (c) collapsed by any deterministic L2 objective onto the conditional mean over branches.

---

## Key Findings

1. **Chaotic on-attractor phase = continuous Koopman spectrum ⇒ no slow discrete eigenfunction to condition on.** "The Koopman operator exhibits a continuous spectrum in many systems of interest (mixing flows, chaotic dynamics)… no basis exists for eigenfunctions" [HAL Koopman spectral review](https://hal.science/hal-05014806v1/file/Koopman_Paper_1.pdf). Discrete predictive Koopman modes appear only for **non-chaotic** Kolmogorov regimes (Re=10 travelling wave, Re=12 RPO) [JFM 2024](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/datadriven-statespace-and-koopman-operator-models-of-coherent-state-dynamics-on-invariant-manifolds/B210580336C4ABCD065102170D63FEA9).

2. **A finite-dimensional Koopman model's prediction horizon is ~1 Lyapunov time regardless of capacity.** On the Arnold-cat negative control, one-step MSE ≈3×10⁻⁵ but error reaches persistence by step ~7; growing model size two orders of magnitude extends the horizon by only ~2 steps [Deep-Koopman-KANDy, arXiv:2605.06000](https://arxiv.org/html/2605.06000v1).

3. **Large-scale (k≲3) IC structure predicts late-time extreme events up to >4 Lyapunov times — in the exact thesis flow.** Re=100, 128², k=4-forced Kolmogorov: removing small scales up to wavelength λc≈0.30L leaves predictability intact (R≥0.84); only removing the large scales destroys it [ML hierarchy, arXiv:2603.13789](https://arxiv.org/html/2603.13789v1). Predictable bursts are preceded ~6 Tλ earlier by a k=±1 jet mode with Ω_y≈3.7Ω_x vs Ω_y≈Ω_x for unpredictable ones [JFM 2024.263](https://doi.org/10.1017/jfm.2024.263).

4. **But that predictability is strongly state-dependent and largely distributional.** Skill spans ~1 to >4 Tλ across events; *most* events lose predictability within 1–2 Tλ, and the long-horizon result was obtained with massive Monte-Carlo ensembles / a diffusion model, not a deterministic surrogate [JFM 2024.263](https://doi.org/10.1017/jfm.2024.263), [arXiv:2603.13789](https://arxiv.org/html/2603.13789v1).

5. **Contrastive/SSL IC representations help conditioning only marginally and target the wrong quantity.** Conditioning an FNO on a pretrained IC representation moved NS one-step MSE 13.4→13.0 (×10⁻³, ~3%), UNet 2.60→2.47, with the GT-parameter upper bound at 2.08 [SSL-Lie-symmetries, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/5c46ae130105fa012da0446126c01d1d-Paper-Conference.pdf). The canonical contrastive-for-chaos method learns features that discriminate *environment parameters* and preserve the *invariant measure* — explicitly abandoning pointwise trajectory fidelity [Jiang–Lu, arXiv:2306.01187](https://arxiv.org/html/2306.01187v3).

6. **Deterministic distance/MSE training over chaotic trajectories provably collapses predictive variance onto the climatological (Fréchet-median) state.** This is the formal statement of the thesis's "L2 minimizer averages over phase branches" [Stochasticity-essential, arXiv:2603.28671](https://arxiv.org/html/2603.28671). Forcing a single-valued map onto one arbitrarily-selected branch provably injects discontinuous, hard-to-learn selectors [Bifurcation Models, arXiv:2605.07277](https://arxiv.org/html/2605.07277v1).

7. **A physics-residual-only IC objective cannot disambiguate branches; adjoint inversion within the window is dominated by the branch-setting small scales.** Adjoint-based DA in 2D turbulence suffers a "spectral catastrophe" — backward-time exponential growth concentrated at high wavenumbers, mean-adjoint SNR decaying exponentially while variance grows [Preconditioned-Adjoint, arXiv:2602.14016](https://arxiv.org/html/2602.14016). The IC inverse problem is non-unique ("adjoint butterfly effect") and restricted to ~1 Lyapunov time [Zaki–Wang, arXiv:2109.10290](https://ar5iv.labs.arxiv.org/html/2109.10290).

8. **CLV/FTLE features *can* predict late-time regime structure — but only in low-dim chaos and they require the trajectory.** CLV alignment angles fed to an MLP predict chaotic regime counts at 0.977 accuracy up to 17.5 Lyapunov times in the 3-D Rikitake dynamo [Brugnago–Gallas–Beims, Chaos 2020](http://inaesp.org/PublicJG/brugnago_gallas_machine_learning_beims_rikitake_Chaos2020.pdf). The IC-only proxy (OTD/FTLE-like precursors estimated from a single snapshot) has been fed to deterministic forecasters of extreme-event *observables* [Dynamics-Informed DL, arXiv:2603.10777](https://doi.org/10.48550/arxiv.2603.10777).

---

## Detailed Analysis

### Sub-question (a) — Koopman slow-manifold conditioning

**Verdict: NEGATIVE for the stated mechanism.** There is no IC-persistent *slow discrete* Koopman mode of chaotic 2D forced NS that discriminates late-time k2–5 phase, because the relevant dynamics live in the continuous spectrum, which by definition has no eigenfunction basis.

The operator-theoretic fact is unambiguous and authoritative: for measure-preserving chaotic/mixing flows the Koopman operator decomposes into a **point spectrum** (almost-periodic, structured, coherent components) and a **continuous spectrum** that "encodes the chaotic component of the dynamics corresponding to the on-attractor dynamics," where "eigenfunctions are replaced by eigenmeasures" [Koopman ODM review](https://mediatum.ub.tum.de/doc/1638226/20uy7tjz43whqextouern7wvu.KODM_arcPP.pdf). The spectral-approximation literature is explicit that "the Koopman operator exhibits a continuous spectrum in many systems of interest (e.g., mixing flows, chaotic dynamics). That is, no basis exists for eigenfunctions" and that DMD/EDMD are "fundamentally geared toward identifying isolated eigenvalues" and "fail to capture continuous or mixed spectra" [HAL review](https://hal.science/hal-05014806v1/file/Koopman_Paper_1.pdf). The canonical pedagogical example of a purely continuous (mixing) spectrum is the Arnold cat map [Springer, Koopman Spectra](https://link.springer.com/rwe/10.1007/978-3-0348-0692-3_124-1).

This is confirmed empirically. Deep-Koopman-KANDy uses the Arnold cat map as a **negative control** precisely because finite-dimensional spectral closure is provably impossible: the learned 1024×1024 propagator has |λ|≤0.958 with no eigenvalue near the unit circle, one-step MSE ≈3×10⁻³³00× below persistence yet error reaches persistence scale by step ~7, and "as model size grows, one-step accuracy improves by two orders of magnitude, but the prediction horizon extends by only ~2 steps, consistent with the Lyapunov barrier" [arXiv:2605.06000](https://arxiv.org/html/2605.06000v1). The authors state the general bound directly: "when the spectrum is purely continuous, no architecture rescues long-horizon prediction beyond the Lyapunov time."

Where Koopman *does* work for Kolmogorov flow is the non-chaotic regime: a data-driven state-space/Koopman model reconstructs Re=10 (travelling-wave) and Re=12 (relative-periodic-orbit) Kolmogorov flow accurately over 500–5000 time units [JFM 2024](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/datadriven-statespace-and-koopman-operator-models-of-coherent-state-dynamics-on-invariant-manifolds/B210580336C4ABCD065102170D63FEA9) — these are quasi-periodic (point-spectrum) regimes. Likewise, turbulent cavity flow at Re=12,000 shows clean oscillatory Koopman modes [Mezić, ARFM 2013](https://mgroup.me.ucsb.edu/sites/default/files/publications/mezic_-_2013_-_analysis_of_fluid_flows_via_spectral_properties_of_the_koopman_operator.pdf) — but those are self-sustained *oscillations* (limit-cycle-like), not aperiodic branch selection. The Koopman eigenfunction value at the IC, φⱼ(x₀), is a legitimate IC-only quantity [Modern Koopman Theory](https://www.lri.fr/~gcharpia/deeppractice/2022/chap_5_biblio/Koopman/Modern_koopman_theory.pdf), but at Re=100–500 the modes that matter for late-time phase are not in the point spectrum.

**The only Koopman-adjacent IC feature with predictive content is proximity to an exact coherent structure (ECS)** — an unstable equilibrium / periodic orbit that is a point-spectrum object embedded in the chaotic attractor. The thesis-configuration predictability paper notes that its predictable bursts arise from base flows that "resemble an unstable equilibrium solution that dominates the quiescent, low-enstrophy dynamics," and that flow may be forecast "when certain unstable equilibrium solutions are shadowed by the dynamics" [JFM 2024.263](https://doi.org/10.1017/jfm.2024.263). ECS are computable from a field via Newton/adjoint methods and have been identified in 2D Kolmogorov turbulence with convolutional autoencoders [Page–Brenner–Kerswell](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/exact-coherent-structures-in-twodimensional-turbulence-identified-with-convolutional-autoencoders/B54D02CA985CA291CB1FFC83279CF772). This redirects Q_a into Q_c (an ECS-proximity *projection*), not a Koopman-eigenfunction channel.

**Cross-reference:** the continuous-spectrum verdict is the spectral statement of the same barrier that defeats Q_d's adjoint inversion (the chaotic small scales) and Q_b's contrastive embeddings (no discrete coordinate to separate). The ECS-proximity escape hatch is exactly the large-scale projection that Q_c finds predictive.

### Sub-question (b) — Contrastive IC pairing

**Verdict: WEAK-NEGATIVE.** Contrastive/metric IC embeddings are a low-risk auxiliary that demonstrably help conditioning, but every published instance targets *environment parameters* or *invariant statistics* — not phase-branch identity — and the measured gains on Navier–Stokes are within the noise of the loss-geometry interventions the thesis has already exhausted.

The most direct evidence is the Lie-symmetry SSL framework: it pretrains an encoder by enforcing similarity between augmented views of a PDE solution, then **adds the frozen representation as a conditioning channel** to a neural operator. On 2D Navier–Stokes the one-step MSE moved from 13.4 to 13.0 (×10⁻³, FNO) and 2.60→2.47 (UNet), against a ground-truth-buoyancy upper bound of 2.08 — i.e., it captured a *small fraction* of the available conditioning signal, and the paper describes the time-stepping gain as "slightly reduces the error," "mixed for Burgers," and notes the FNO benefits least [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/5c46ae130105fa012da0446126c01d1d-Paper-Conference.pdf). Critically, the conditioning target there is a *static equation parameter* (buoyancy/viscosity), which is far easier than a phase branch.

The canonical "contrastive learning for chaos" method makes the target mismatch explicit. Its contrastive encoder is trained so that "sequences from the same trajectory have the same chaotic attractor and so should have similar embeddings; sequences from different trajectories… dissimilar" — i.e., it learns features that **distinguish environment parameters φ** and preserve the **invariant measure**, after the authors establish that "RMSE is a poor signal for training a neural operator [on chaotic dynamics]… any predictive model will quickly and exponentially diverge from the true trajectory in terms of RMSE" [Jiang–Lu, arXiv:2306.01187](https://arxiv.org/html/2306.01187v3). In other words, the field's contrastive machinery is explicitly built to *give up* pointwise/phase fidelity and match statistics — the opposite of branch commitment. Follow-on work confirms the scope and notes it was only demonstrated on low-dimensional attractors, with scaling to Kolmogorov flow unclear [DySLIM, ICML 2024](https://proceedings.mlr.press/v235/schiff24b.html).

A contrastive objective *tailored* to the thesis problem — pairing ICs by their eventual k2–5 phase divergence (a "supervised contrastive" or triplet loss on the late-phase label) — is not represented in the literature. It is conceptually coherent, but two facts cap the expected payoff: (i) the late-phase label is exactly the quantity that is unpredictable for generic states beyond ~1 Tλ (Q_c), so the positive/negative pairs are only well-separated for the predictable subset; and (ii) a contrastive auxiliary still feeds a deterministic decoder whose L2 head will re-collapse onto the conditional mean (cross-reference Finding 6). Expected Δlate is therefore in the same regime as the phase/energy/Sobolev/spectral losses already tested (max Δlate=−0.022 in the thesis).

**Cross-reference:** Q_b inherits Q_a's obstruction — there is no low-dimensional discrete coordinate (point-spectrum eigenfunction) for the embedding to align with; it can only learn the structured/large-scale part, which is Q_c's signal.

### Sub-question (c) — Invariant IC projections

**Verdict: MOST VIABLE, but conditional and partial.** A large-scale / ECS-proximity / coherent-structure projection of the IC is the one feature class with concrete, quantitative evidence of late-time predictive content **in the exact thesis configuration** — and it is the natural way to express FTLE/CLV/slow-manifold ideas in an IC-only, deterministic-conditioning form. The hard caveat is that this predictability is state-dependent (a subset of ICs) and the published demonstrations are distributional.

Two 2026 studies bracket this, both at **Re=100, 128² grid, n_f=4 forcing — the thesis flow**:

- The information-theoretic ensemble study finds that "for some specific large-scale configurations, it is theoretically possible to produce informative predictions of extreme events several Lyapunov times in advance… an order of magnitude longer than variational or machine-learning methods." The discriminating IC feature is explicitly a **low-wavenumber projection**: predictable bursts are preceded ~6 Tλ earlier by a k=±1 jet mode with Ω_y=3.7Ω_x (12.3× in the specific case), versus Ω_y≈Ω_x for unpredictable ones [JFM 2024.263](https://doi.org/10.1017/jfm.2024.263). Reference time scales there: Teto=1.92Tλ, T_forcing=0.28Tλ, injection-dissipation delay=0.66Tλ.

- The ML companion quantifies the scale hierarchy directly via spectral filtering: predictability horizons span "≈1 to >4 Lyapunov times," are "controlled predominantly by large-scale structures," and **removing small scales up to wavelength λc≈0.30L (i.e. k≳3) does not change predictability (R≥0.84); only removing the large scales destroys it** [arXiv:2603.13789](https://arxiv.org/html/2603.13789v1). Mechanistically, "highly predictable extremes are associated with persistent, large-scale coherent structures, whereas the least predictable events coincide with intense, small-scale-dominated bursts," and the separator is the *lifetime* of strain-core/quadrupole packets.

This is the central, somewhat counterintuitive result for the thesis: the predictive information for late time is in the **largest scales (k≲3)**, and the small scales are bystanders *for predictability* even though they carry the enstrophy and seed the divergence. The deterministic surrogate already sees these large scales — but the L2 objective averages them out together with the unpredictable branches (cross-reference Finding 6). Conditioning the surrogate on an explicit large-scale invariant (leading POD/coherent-structure amplitudes, the k=±1 jet anisotropy Ω_y/Ω_x, an ECS-proximity coordinate, or a strain-core/Q-criterion descriptor) is therefore the one direction with a mechanistic reason to help — by giving the network a protected, un-averaged handle on the predictable component.

The dynamical-systems quantities named in the question map onto this as follows:

- **CLV / Lyapunov vectors** are computable for chaotic Kolmogorov flow [Inubushi et al., Phys. Rev. E 85, 016331 (2012)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.85.016331); and CLV *alignment angles* used as MLP inputs predict chaotic regime structure at 0.977 accuracy up to 17.5 Lyapunov times in the 3-D Rikitake dynamo [Brugnago–Gallas–Beims, Chaos 2020](http://inaesp.org/PublicJG/brugnago_gallas_machine_learning_beims_rikitake_Chaos2020.pdf). **Caveat:** CLVs require forward+backward integration of the *trajectory* (not the IC alone), the demonstration is a 3-D ODE not a spatially-extended PDE, and it predicts a scalar regime count, not a spatial phase branch.
- **FTLE fields** are defined from the flow map over [t₀,t] (a trajectory integral) and mark Lagrangian coherent structures [Haller–Sapsis](http://georgehaller.com/reprints/smallestFTLE.pdf); the leading Lyapunov vector / fastest-growing perturbations *localize at the smallest scales* in forced turbulence [Mohan–Fitzsimmons–Moser, PRF 2017](https://link.aps.org/doi/10.1103/PhysRevFluids.2.114606), which is why a raw FTLE field is not an IC-only large-scale predictor.
- The **IC-only proxy** that the literature actually feeds to a *deterministic* forecaster is an OTD/FTLE-like precursor estimated from a single snapshot, used as Transformer input to forecast extreme-event observables [Dynamics-Informed DL, arXiv:2603.10777](https://doi.org/10.48550/arxiv.2603.10777).
- **Slow/inertial-manifold coordinates** exist for these dissipative flows [Linot–Graham, PRE 2020](https://doi.org/10.1103/physreve.101.062209) and underlie the ECS/coherent-structure picture, but no work conditions a deterministic NS surrogate on them to pick a phase branch.

**The honest negative within Q_c:** (i) predictability is state-dependent — for generic ICs the branch is set within ~1–2 Tλ and *no* large-scale feature helps; (ii) the >4 Tλ result is distributional (massive ensembles / diffusion), so it bounds what *any* model could do probabilistically, not what a deterministic one-shot map can commit to; (iii) the cleanest discriminator (coherent-structure *lifetime*) is a temporal quantity, only partially inferable from a single IC snapshot.

**Cross-reference:** Q_c absorbs Q_a's ECS-proximity redirect and is the only place where the Q_b contrastive objective could find a well-separated target (the predictable subset). It also sets the ceiling for Q_d: even with a differentiable operator, you can only refine toward what the large-scale IC structure already determines.

### Sub-question (d) — Test-time IC refinement without ground truth

**Verdict: NEGATIVE.** Optimizing the IC at inference against a **physics-only** objective cannot select the correct phase branch, for a structural reason and an inversion reason.

*Structural (degeneracy).* The NS residual is zero for *every* exact trajectory of the system — all phase branches satisfy the equations equally — so a residual objective provides no gradient toward one branch versus another. An energy-spectrum-matching objective constrains *statistics* (it is satisfied by the whole invariant measure), not phase. An IC-frame reconstruction objective simply returns the given IC. None of the physics-only objectives the question names carries branch information. This is the inference-time dual of Finding 6: a deterministic objective with no branch-discriminating signal collapses to the statistically-typical state. The formal result is that "optimizing any deterministic distance metric over multi-step trajectories inherently forces the model to damp out natural variability, leading to a collapse of predictive variance," with the long-lead optimum being the Fréchet median of the invariant measure [arXiv:2603.28671](https://arxiv.org/html/2603.28671). The "right" response to genuine multiplicity is a set-valued map whose *initialization* (a latent) selects the branch [Bifurcation Models, arXiv:2605.07277](https://arxiv.org/html/2605.07277v1) — but that reintroduces a latent/ensemble, i.e., the excluded probabilistic territory, and even then physics-energy training "lets the model settle on the branch easiest to represent," not the correct one.

*Inversion (adjoint chaos).* If one instead supplies *observations* and runs 4D-Var/adjoint IC recovery against the differentiable surrogate — the closest realizable form of the question — it is fundamentally bounded. The IC inverse problem in chaos is non-unique: "infinitesimally close observations can be reproduced by initial conditions that have large separation" (the "adjoint butterfly effect"), and adjoint accuracy "deteriorates for long integration horizons," restricting assimilation to ~1 Lyapunov time [Zaki–Wang, Phys. Rev. Fluids 6, 100501 (2021)](https://ar5iv.labs.arxiv.org/html/2109.10290). Worse for branch selection specifically: the adjoint gradient w.r.t. the IC undergoes a "spectral catastrophe" — exponential backward-time growth concentrated at *high wavenumbers*, with the ensemble-mean adjoint SNR decaying exponentially while its variance grows — so the small scales that actually set the branch are exactly where the gradient is pure chaotic noise; a 35%-better reconstruction is only achieved by *suppressing* those scales and recovering the large ones [Preconditioned-Adjoint, arXiv:2602.14016](https://arxiv.org/html/2602.14016). The classical statement of this limit predates ML [Pires–Vautard–Talagrand, Tellus 1996](https://tellusjournal.org/articles/10.3402/tellusa.v48i1.11634).

*What PINO test-time optimization actually does.* PINO's "test-time / instance optimization" refines the **solution function for a given PDE instance whose IC is the fixed input/constraint** — it does not search over ICs to recover an unknown branch [PINO, ICML 2022](https://openreview.net/forum?id=dtYnHcmQKeM), [FC-PINO, arXiv:2211.15960](https://arxiv.org/html/2211.15960v3). And as a forward solver its pointwise accuracy degrades steeply with Re (Re=100: 0.7% error; Re=500: 16.8%; Re=1000: 25.5%) [MHD-hybrid, arXiv:2507.02106](https://arxiv.org/html/2507.02106v2), so it offers no deterministic branch recovery in the turbulent regime. Reservoir-computing predictors reach ~8 Lyapunov times on KS but require a *warmup/synchronization segment of the trajectory* — not the IC alone [Pathak et al., reviewed in arXiv:2605.28145](https://arxiv.org/html/2605.28145v1).

**Cross-reference:** Q_d fails for the same reason Q_a does (the branch lives in the chaotic small scales / continuous spectrum) and is capped by Q_c (you can only refine toward what the large-scale IC structure already determines, which is the predictable subset).

---

## Comparison

| Criterion | (a) Koopman slow-mode | (b) Contrastive IC pairing | (c) Invariant IC projection | (d) Physics-only IC refinement |
|---|---|---|---|---|
| Direct prior art for *deterministic branch selection* | ❌ None | ❌ None (env-param/stats only) | ⚠️ Closest (large-scale predictability, OTD precursors) | ❌ None |
| Predictive signal exists in IC? | ❌ Not in chaotic spectrum | ⚠️ Marginal (~3% on NS) | ✅ Large scales k≲3, up to >4 Tλ (subset) | ❌ Residual is branch-degenerate |
| IC-only (no trajectory needed)? | ✅ φⱼ(x₀) but useless here | ✅ | ⚠️ Large-scale modes ✅; CLV/FTLE ❌ (need trajectory) | ✅ (but no signal) |
| Deterministic & one-shot compatible? | ✅ | ✅ | ✅ | ⚠️ Optimization loop at inference |
| Blocking mechanism | Continuous spectrum | L2 head re-collapse | State-dependence; distributional | NS-residual degeneracy + adjoint chaos |
| Expected Δlate | ~0 | small (≈ tested loss terms) | **potentially meaningful for predictable subset** | ~0 |
| Verdict | Negative | Weak-negative | **Viable (conditional)** | Negative |

**What the table reveals.** Three of four routes fail through the *same* root cause expressed in different mathematics — the late-time phase branch is encoded in the chaotic, small-scale, continuous-spectrum part of the dynamics, which is (a) not a discrete mode, (b) not a separable contrastive coordinate, and (d) not recoverable by physics-only inversion. The one survivor, (c), survives precisely because it stops trying to capture that part and instead conditions on the *large-scale, persistent* component that the predictability literature shows is the only thing predictable beyond ~1 Tλ. The most important differentiator is therefore not the model class but **which part of the spectrum the feature targets**: large-scale/coherent ⇒ partial signal; small-scale/chaotic ⇒ none. The surprise is that the thesis's own framing ("late k≤7 phase is not intrinsically IC-insensitive; D_late grows only 1.3–1.4×") is consistent with (c): the flow *is* within the predictable window, but the deterministic L2 objective discards the predictable large-scale signal by averaging over the unpredictable branches — which (c) is designed to prevent and (b) is not.

---

## Contradictions & Debates

- **Is IC structure predictive of late time, or not?** *Optimistic*: large-scale IC structure predicts extreme events up to >4 Tλ [JFM 2024.263](https://doi.org/10.1017/jfm.2024.263); [arXiv:2603.13789](https://arxiv.org/html/2603.13789v1). *Pessimistic*: deterministic prediction collapses; generic states decorrelate within ~1 Tλ; adjoint inversion is ill-posed [arXiv:2603.28671](https://arxiv.org/html/2603.28671); [arXiv:2109.10290](https://ar5iv.labs.arxiv.org/html/2109.10290). **Resolution (strong evidence both sides):** predictability is *state-dependent* and the demonstrated long horizon is *distributional* (ensembles/diffusion). For a deterministic one-shot map, the usable signal is the large-scale projection on the predictable subset; for generic/transitional states the branch is genuinely unrecoverable from the IC. Both are true; they describe different subsets and different (probabilistic vs deterministic) notions of "prediction."

- **Does Koopman work for Kolmogorov flow?** *Yes* at Re=10–12 (point spectrum) [JFM 2024](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/datadriven-statespace-and-koopman-operator-models-of-coherent-state-dynamics-on-invariant-manifolds/B210580336C4ABCD065102170D63FEA9); *no* for chaotic on-attractor phase (continuous spectrum) [HAL review](https://hal.science/hal-05014806v1/file/Koopman_Paper_1.pdf); [arXiv:2605.06000](https://arxiv.org/html/2605.06000v1). **Resolution:** discrete predictive eigenfunctions exist only for non-chaotic/coherent regimes; the thesis regime (Re=100–500, turbulent) is continuous-spectrum, so no slow discrete mode discriminates late-time phase beyond ~1 Tλ.

---

## Uncertainties & Gaps

- ⚠️ **Irreducible negative-by-absence:** no published work conditions a *deterministic one-shot* neural PDE surrogate on an IC-only dynamical projection to commit to a late-time spatial phase branch in 2D NS. The verdicts for (a),(b),(d) combine this absence with mechanistic argument; (c) rests on adjacent positive evidence (predictability of *observables*, CLV-features in low-dim chaos, OTD precursors), not a direct phase-branch demonstration.
- ⚠️ **Step↔Lyapunov-time mapping:** the thesis reports horizons in *steps* (T=128, N_cap≈224–278); the literature reports them in Lyapunov times (Tλ=3.6√f₀ in [arXiv:2603.13789](https://arxiv.org/html/2603.13789v1)). The qualitative correspondence is clear (within-window), but the exact dt↔Tλ conversion for the thesis dataset was not available and should be computed locally before quantitative comparison.
- ⚠️ **OTD-from-snapshot fidelity:** the IC-only OTD/FTLE-like precursor [arXiv:2603.10777] predicts scalar observables; its use as a *spatial* conditioning field for branch selection is unproven.
- ⚠️ **Confidence plateau at ~90%:** further searching will not raise this, because the central claim is a non-existence statement.

---

## Recommendations

### Primary recommendation — pursue (c), narrowly scoped
Condition the deterministic surrogate on an **explicit large-scale / ECS-proximity IC projection**, because it is the only feature class with a mechanistic reason and quantitative precedent to carry late-time signal in the exact thesis flow [JFM 2024.263](https://doi.org/10.1017/jfm.2024.263); [arXiv:2603.13789](https://arxiv.org/html/2603.13789v1). Concretely, as a 5th-channel analogue to the GT oracle but computed from the IC alone:
- leading POD/coherent-structure amplitudes and the k=±1 jet anisotropy Ω_y/Ω_x of the IC;
- an ECS-proximity coordinate (distance to the nearest precomputed unstable equilibrium/RPO, or its convolutional-autoencoder latent [Page–Brenner–Kerswell]);
- a strain-core/Q-criterion descriptor of the IC.

**Decisive pre-test before any training:** verify that these IC projections actually separate the diverging k2–5 phase branches in the *training set* (e.g., mutual information between the IC projection and the t=57..64 phase label; or stratify D_late by the projection). If MI is near zero — which the state-dependence results predict for the generic subset — abandon; the surrogate cannot condition on information that is not there. Expect, at best, improvement on the predictable subset, not a uniform fix; this is the honest ceiling.

### Alternative — (b) only as a cheap auxiliary, with low expectations
A *supervised-contrastive* auxiliary that pulls together ICs sharing the late k2–5 phase branch is conceptually sound and cheap to add, but two facts cap it: the L2 decoder head re-collapses onto the conditional mean [arXiv:2603.28671], and the positive/negative pairs are only well-separated for the predictable subset. Expected Δlate is in the regime of the already-exhausted loss-geometry terms (≤0.022). Worth a single ablation, not a research thrust.

### Not recommended
- **(a) Koopman eigenfunction conditioning** — the discriminating dynamics are in the continuous spectrum; no discrete IC-persistent mode exists for the turbulent regime [HAL review](https://hal.science/hal-05014806v1/file/Koopman_Paper_1.pdf); [arXiv:2605.06000](https://arxiv.org/html/2605.06000v1). (Only the ECS-proximity *projection* survives, which is folded into (c).)
- **(d) physics-residual-only IC refinement** — the NS residual is branch-degenerate; any realizable observation-based 4D-Var is bounded to ~1 Tλ by adjoint chaos and is most ill-conditioned in exactly the small scales that set the branch [arXiv:2602.14016](https://arxiv.org/html/2602.14016); [arXiv:2109.10290](https://ar5iv.labs.arxiv.org/html/2109.10290). This belongs to Q3 (external intermediate state), not to IC-only conditioning.

**Overall:** the well-supported negative holds in the *mechanistic* form requested — within the Lyapunov window, late-time k2–5 branch selection is driven by small-scale, continuous-spectrum IC sensitivity that is (a) not a discrete spectral object, (b) not a separable contrastive coordinate, and (d) not recoverable by physics-only inversion; consequently the deterministic L2 minimizer collapses onto the conditional mean over branches. The *only* exception is the large-scale (k≲3) coherent/ECS component, which carries genuine but state-dependent and largely distributional predictability — making (c) the single direction worth an experiment, and only with the MI pre-test as a kill-switch.

---

## Methodology

- **Depth:** exhaustive. **Rounds:** 6 search rounds (12 `research_search` batches + 6 `web_search` batches) plus targeted full-text extraction of 8 pivotal papers.
- **Total sources:** 50 unique, Tier-1 dominant (arXiv/JFM/Phys. Rev. Fluids/NeurIPS/ICML).
- **Final confidence:** ~90% (Q_a 92, Q_b 88, Q_c 90, Q_d 91). The 95% exhaustive threshold is not attainable for a non-existence claim; both flagged contradictions were resolved with authoritative Tier-1 sources.
- **Multi-hop chains:** Koopman spectrum → continuous-spectrum theorem → Arnold-cat negative control → ECS point-spectrum escape → ECS in 2D turbulence (autoencoders) → large-scale predictability (JFM/ML hierarchy); 4D-Var → adjoint chaos → spectral catastrophe → branch-setting small scales; contrastive-for-chaos → invariant-measure target → deterministic-collapse theorem → set-valued maps.
- **Key challenge:** distinguishing *distributional/ensemble* predictability (which exists up to >4 Tλ) from *deterministic one-shot* branch commitment (which does not, beyond ~1 Tλ for generic states) — this distinction is the crux of the entire question and is now explicitly resolved.

---

## Sources

| # | Title | URL | Date | Credibility |
|---|---|---|---|:--:|
| 1 | Deep-Koopman-KANDy: Dictionary Discovery for Deep-Koopman Operators | https://arxiv.org/html/2605.06000v1 | 2026 | ⭐ Tier 1 |
| 2 | Data-driven state-space & Koopman models on invariant manifolds (Kolmogorov Re 10–12) | https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/datadriven-statespace-and-koopman-operator-models-of-coherent-state-dynamics-on-invariant-manifolds/B210580336C4ABCD065102170D63FEA9 | 2024 | ⭐ Tier 1 |
| 3 | Spectral Approximation of Koopman & Perron–Frobenius operators (continuous spectrum) | https://hal.science/hal-05014806v1/file/Koopman_Paper_1.pdf | 2025 | ⭐ Tier 1 |
| 4 | Koopman Operators: Spectra, Function Spaces, Representations (Springer) | https://link.springer.com/rwe/10.1007/978-3-0348-0692-3_124-1 | 2020 | ⭐ Tier 1 |
| 5 | Analysis of Fluid Flows via Spectral Properties of the Koopman Operator (Mezić, ARFM) | https://mgroup.me.ucsb.edu/sites/default/files/publications/mezic_-_2013_-_analysis_of_fluid_flows_via_spectral_properties_of_the_koopman_operator.pdf | 2013 | ⭐ Tier 1 |
| 6 | Koopman Operator Dynamical Models: Learning, Analysis and Control (point vs continuous) | https://mediatum.ub.tum.de/doc/1638226/20uy7tjz43whqextouern7wvu.KODM_arcPP.pdf | 2021 | ⭐ Tier 1 |
| 7 | Modern Koopman Theory for Dynamical Systems (Brunton et al.) | https://www.lri.fr/~gcharpia/deeppractice/2022/chap_5_biblio/Koopman/Modern_koopman_theory.pdf | 2022 | ⭐ Tier 1 |
| 8 | KoopGen: Koopman generator networks for continuous spectra | https://arxiv.org/html/2602.14011v1 | 2026 | 🔵 Tier 2 |
| 9 | Self-Supervised Learning with Lie Symmetries for PDEs (NeurIPS) | https://proceedings.neurips.cc/paper_files/paper/2023/file/5c46ae130105fa012da0446126c01d1d-Paper-Conference.pdf | 2023 | ⭐ Tier 1 |
| 10 | Training neural operators to preserve invariant measures of chaotic attractors | https://arxiv.org/html/2306.01187v3 | 2023 | ⭐ Tier 1 |
| 11 | DySLIM: Dynamics Stable Learning by Invariant Measure | https://proceedings.mlr.press/v235/schiff24b.html | 2024 | ⭐ Tier 1 |
| 12 | Large-scale patterns set the predictability limit of extreme events in Kolmogorov flow (Re=100, 128²) | https://doi.org/10.1017/jfm.2024.263 | 2024 | ⭐ Tier 1 |
| 13 | Hierarchy of extreme-event predictability in turbulence revealed by ML (Re=100, 128²) | https://arxiv.org/html/2603.13789v1 | 2026 | ⭐ Tier 1 |
| 14 | Dynamics-Informed Deep Learning for Predicting Extreme Events (OTD/FTLE precursors) | https://doi.org/10.48550/arxiv.2603.10777 | 2026 | 🔵 Tier 2 |
| 15 | Machine learning, alignment of CLVs, and predictability (Rikitake) | http://inaesp.org/PublicJG/brugnago_gallas_machine_learning_beims_rikitake_Chaos2020.pdf | 2020 | ⭐ Tier 1 |
| 16 | Covariant Lyapunov analysis of chaotic Kolmogorov flows (PRE 85, 016331) | https://journals.aps.org/pre/abstract/10.1103/PhysRevE.85.016331 | 2012 | ⭐ Tier 1 |
| 17 | RIMS-1770 Covariant Lyapunov Analysis of Navier–Stokes Turbulence | https://www.kurims.kyoto-u.ac.jp/preprint/file/RIMS1770.pdf | 2013 | ⭐ Tier 1 |
| 18 | Scaling of Lyapunov exponents in homogeneous isotropic turbulence (PRF 2, 114606) | https://link.aps.org/doi/10.1103/PhysRevFluids.2.114606 | 2017 | ⭐ Tier 1 |
| 19 | Lagrangian coherent structures & the smallest FTLE (Haller–Sapsis) | http://georgehaller.com/reprints/smallestFTLE.pdf | 2011 | ⭐ Tier 1 |
| 20 | Deep learning to discover/predict dynamics on an inertial manifold (Linot–Graham, PRE 101) | https://doi.org/10.1103/physreve.101.062209 | 2020 | ⭐ Tier 1 |
| 21 | Exact coherent structures in 2D turbulence identified with CAEs | https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/exact-coherent-structures-in-twodimensional-turbulence-identified-with-convolutional-autoencoders/B54D02CA985CA291CB1FFC83279CF772 | 2024 | ⭐ Tier 1 |
| 22 | Bifurcation Models: Learning Set-Valued Solution Maps | https://arxiv.org/html/2605.07277v1 | 2026 | 🔵 Tier 2 |
| 23 | Stochasticity & probabilistic trajectory scoring essential for chaotic closures | https://arxiv.org/html/2603.28671 | 2026 | ⭐ Tier 1 |
| 24 | From limited observations to the state of turbulence (Zaki–Wang, PRF 6, 100501) | https://ar5iv.labs.arxiv.org/html/2109.10290 | 2021 | ⭐ Tier 1 |
| 25 | Preconditioned Adjoint Data Assimilation for 2D decaying turbulence | https://arxiv.org/html/2602.14016 | 2026 | ⭐ Tier 1 |
| 26 | On extending the limits of variational assimilation in chaotic systems (Tellus) | https://tellusjournal.org/articles/10.3402/tellusa.v48i1.11634 | 1996 | ⭐ Tier 1 |
| 27 | Physics-Informed Neural Operator (PINO), ICML | https://openreview.net/forum?id=dtYnHcmQKeM | 2022 | ⭐ Tier 1 |
| 28 | FC-PINO: High-Precision PINOs (test-time/instance optimization) | https://arxiv.org/html/2211.15960v3 | 2023 | ⭐ Tier 1 |
| 29 | Resolving Turbulent MHD: Hybrid Operator-Diffusion (PINO error vs Re) | https://arxiv.org/html/2507.02106v2 | 2025 | 🔵 Tier 2 |
| 30 | Learning Dissipative Dynamics in Chaotic Systems (MNO, NeurIPS) | https://proceedings.neurips.cc/paper_files/paper/2022/file/6ad68277e27b42c60ac228c9859fc1a2-Paper-Conference.pdf | 2022 | ⭐ Tier 1 |
| 31 | Adaptive Reservoir Computing (Pathak warmup/synchronization review) | https://arxiv.org/html/2605.28145v1 | 2026 | 🔵 Tier 2 |
| 32 | Long-term prediction of chaotic systems with ML (reservoir synchronization) | https://chaos1.la.asu.edu/~yclai/papers/PRResearch_2020_FJZWL.pdf | 2020 | ⭐ Tier 1 |
| 33 | Lyapunov spectrum of forced homogeneous isotropic turbulent flows | https://arxiv.org/abs/1910.11804 | 2019 | ⭐ Tier 1 |
| 34 | Lyapunov exponent as metric for predictability of LES (PRF) | https://www.osti.gov/biblio/1544369 | 2017 | ⭐ Tier 1 |
| 35 | Mixture of neural operator experts (model selection / set-valued) | https://arxiv.org/html/2502.04562v1 | 2025 | 🔵 Tier 2 |
| 36 | Conditioning on PDE Parameters to Generalise DL Emulation of Chaotic Dynamics | https://ar5iv.labs.arxiv.org/html/2509.09599 | 2025 | 🔵 Tier 2 |
| 37 | The Multiverse of Dynamic Mode Decomposition Algorithms (Colbrook) | http://www.damtp.cam.ac.uk/user/mjc249/pdfs/DMD_multiverse_colbrook.pdf | 2023 | ⭐ Tier 1 |
| 38 | Roots of Ensemble Forecasting (deterministic predictability limit) | https://journals.ametsoc.org/view/journals/mwre/133/7/mwr2949.1.xml | 2005 | ⭐ Tier 1 |
| 39 | Comparison of LETKF, 3DVAR, 4DVAR (4D-Var errors project on Lyapunov vectors) | https://journals.ametsoc.org/view/journals/mwre/137/2/2008mwr2396.1.xml | 2009 | ⭐ Tier 1 |
| 40 | Neural representation of FTLE fields (FTLE = flow-map integral) | https://link.springer.com/article/10.1007/s44267-023-00014-x | 2023 | 🔵 Tier 2 |
