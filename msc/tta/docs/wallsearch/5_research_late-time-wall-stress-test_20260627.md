# Late-Time Wall — Literature Stress Test: Broken, Statistical Retreat, or Solver Last Resort?

**Date:** 2026-06-27
**Depth:** exhaustive
**Confidence:** 85% (Q2d distributional-TTA formulation is the weakest leg, ~68%)
**Sources:** 28 sources across 4 search rounds + 8 deep primary-source extractions

---

## Executive Summary

The wall has **not** been broken in your regime. After an exhaustive search, there is **no published single-shot (non-autoregressive) space-time neural operator that reports late-time (t/T>0.5) pointwise relative-L2 below ~0.4 for 2D forced turbulence at Re≈500 without a solver in the loop** — and crucially, the absence is not an oversight: the field that works on exactly this problem (DySLIM, Jiang et al., PDE-Refiner, BSP) has **deliberately stopped reporting pointwise late-time error** because it is provably uninformative past the Lyapunov time, and pivoted to distributional metrics (energy spectra, MELR, Wasserstein, PDFs). Your 6-week empirical wall is therefore consistent with, and corroborated by, the literature's own conclusion: under a pointwise-L2 objective the late-time low-k phase error is the optimizer regressing toward the conditional mean of a decorrelating ensemble, which no loss-shape or architecture lever in or out of `ranking_v2.md` is documented to fix.

Two paths remain coherent. **(Q2)** A statistics-first objective (invariant-measure/MMD à la DySLIM, spectral-energy/BSP, OT/Sinkhorn à la Jiang) does not lower your pointwise number — by construction it trades pointwise tracking for on-attractor fidelity — but it is the only framing under which "late-time accuracy" becomes a well-posed, achievable target, and it reframes (rather than answers) the TTA question. **(Q3)** A solver in the training loop is the single documented route to genuine late-time *pointwise* reduction (List et al. 2022: late-window velocity MSE cut ~2.7× to ~10× depending on flow; Kochkov 2021: accuracy of an 8–10× finer solver). The most important and non-obvious finding for your implementation cost: **a fully differentiable Navier–Stokes solver unrolled through all T steps is not required** — List et al. (2024) show a *non-differentiable* solver unrolled with a short curriculum (m≈3–4 steps, gradients cut at the solver) recovers up to a 4.5× improvement over pure prediction, with full differentiability adding only ~38% on top.

---

## Key Findings

1. **The single-shot pointwise wall is unbroken and the gap is a recognized dead-end, not an open frontier.** No operator paper reports late pointwise relL2 < 0.4 at Re~500 single-shot; the leading chaotic-PDE groups explicitly abandoned the metric. [Jiang/Lu/Willett, NeurIPS 2023](https://arxiv.org/abs/2306.01187), [Schiff/DySLIM, ICML 2024](https://arxiv.org/abs/2402.04467)

2. **Pointwise L2 past the Lyapunov time provably incentivizes regression-to-the-mean and spectral collapse.** The MSE-optimal predictor of a decorrelating ensemble is its (oversmoothed, low-energy, phase-blind) mean; PDE-Refiner shows MSE-trained models are accurate only on a narrow high-amplitude low-k band and truncate the rest of the spectrum. [Lippe/PDE-Refiner, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf)

3. **Backprop-through-time gradients on chaotic trajectories provably explode** (Mikhaeil et al. 2022), which is *why* both unrolling-based training and any long differentiable-solver loop are Lyapunov-horizon-bounded — independently confirmed by List et al. (2024). [List et al., JCP 2024](https://arxiv.org/html/2402.12971v2)

4. **Tractable statistics-first training losses exist and are cheap:** invariant-measure MMD (DySLIM, O(n²) in batch, +0 inference cost), Sinkhorn-OT on chosen summary statistics (Jiang), and Binned-Spectral-Power (BSP, O(d log d), architecture-agnostic). None requires a solver. [DySLIM](https://arxiv.org/abs/2402.04467), [Jiang](https://arxiv.org/abs/2306.01187), [BSP, JCP 2026](https://arxiv.org/html/2502.00472v1)

5. **A statistics-targeting operator accepts late-time phase error as irreducible chaos and is graded on staying on the attractor** — DySLIM improves Wasserstein-1/MELR/covRMSE on Kolmogorov flow by ~20–65% while pointwise tracking is only marginally helped (and Sinkhorn is non-discriminative in 2D-KF's high dimension). [DySLIM Table 1](https://arxiv.org/html/2402.04467v2)

6. **Solver-in-loop training is the documented winner for late-time *pointwise* error**, because the solver supplies the correct late dynamics (advection/dissipation coupling) that the data-L2 gradient dilutes T-fold and cannot reconstruct from a stale IC. [List et al., JFM 2022](https://arxiv.org/pdf/2202.06988), [Kochkov, PNAS 2021](https://www.pnas.org/doi/10.1073/pnas.2101784118)

7. **Full differentiability is not the cost barrier it appears to be.** Non-differentiable unrolling (NOG — solver in the forward loop, gradients cut at the solver) yields up to 4.5× over pure prediction; differentiable unrolling (WIG) adds ~38% over one-step. Short curriculum horizons (m=[1,2,4]) suffice. [List et al., JCP 2024](https://arxiv.org/html/2402.12971v2)

8. **A coherent *distributional* TTA formulation is essentially unwritten.** The closest existing test-time mechanisms are attractor-projection denoising (Thermalizer, diffusion-based, out of your scope) and compositional operator search (Serrano et al. 2026, parametric not statistical). Distributional-statistics TTA for a deterministic operator is an open formulation. [Pedersen/Thermalizer, ICML 2025](https://arxiv.org/html/2503.18731v2), [Serrano et al. 2026](https://arxiv.org/abs/2602.00884)

---

## Detailed Analysis

### Q1 — Has this wall been broken?

**Conclusion: No. And the most informative result is the *shape* of the absence.** Across the corpus, every work that operates in the regime you specify — chaotic 2D turbulence, late horizon, resolved low-k — either (a) does not report pointwise relL2 at t/T>0.5 at all, or (b) reports it only as a short-horizon quantity that it then explicitly declares meaningless past the Lyapunov time. There is no counterexample: no single-shot space-time operator achieves <0.4 late pointwise relL2 at Re≈500 via a loss or architecture absent from `ranking_v2.md`.

The strongest transfer-relevant single number remains SGNO's kolm2d (APEBench Kolmogorov, Re=100) absolute error ≈0.558 — *worse* than your 0.571 wall and autoregressive, not single-shot — which independently confirms that even a purpose-built non-expansive spectral operator does not breach the low-Re KF late-time floor. APEBench itself ([Koehler/Thuerey, NeurIPS 2024](https://github.com/tum-pbs/apebench)), the canonical autoregressive-emulator benchmark, reports rollout relL2 but emphasizes "rollout metrics to understand temporal generalization" and stability criteria — not a sub-0.4 late pointwise KF result.

Where the gain *does* come from in every credible improvement is **(d) something else entirely — a solver in the loop** (Q3), or **(b) an architectural inductive bias that only helps high-k or short-horizon** (your closure gate already proved high-k is a bystander for your wall). The literature contains no documented case of (a) loss-shape or (c) curriculum alone producing a late-time pointwise breakthrough in this regime. This is decisive negative evidence: your banked falsifications of the loss family (`tta-results.md` 2026-06-22/23: time-reweighting max −0.006, band-equalize harmful, angle loss best −0.022) and curriculum (`wall.md` horizon-specialist, chain Variant A flat) are not local accidents — they reproduce the field's collective null result.

> "neural operators trained to minimize squared error losses, while capable of accurate short-term forecasts, often fail to reproduce statistical or structural properties of the dynamics over longer time horizons and can yield degenerate results." — [Jiang et al., NeurIPS 2023](https://arxiv.org/abs/2306.01187)

**Cross-reference:** Q1's negative answer is *explained* by Q2a (pointwise L2 is the wrong objective) and *only circumvented* by Q3 (solver supplies late dynamics). The three sub-questions form a single argument: the wall is unbroken because the objective is mis-specified, and the only documented fix changes the training signal source, not its shape.

---

### Q2 — Is pointwise L2 the wrong lever entirely?

**Conclusion: Yes, past the Lyapunov time, and the field has formalized exactly why.** This is the most strongly-supported finding in the report (Tier-1, 8 sources, near-unanimous).

**Q2a — The theoretical argument and what L2 incentivizes.** Three independent mechanisms converge:

1. *Information-theoretic / regression-to-the-mean.* Past the predictability horizon, the conditional distribution of u(t)|IC is spread across the attractor. The L2-optimal point estimate of a spread distribution is its conditional mean. For turbulence the conditional mean is a smooth, low-energy, **phase-incoherent** field — precisely your banked signature: amplitude collapse (R_k falling to 0.11–0.31 at k7, `tta-results.md` 2026-06-22) and mid-band phase decorrelation (phase_frac 0.45–0.52 at k2–5). The operator is not failing to optimize L2; it is optimizing it correctly toward a hedged mean.

2. *Spectral-bias collapse.* [PDE-Refiner](https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf) shows directly (their Fig. 1b) that an MSE-trained model reproduces the spectrum only over "a small, high-amplitude frequency band" and truncates everything else — the MSE gradient is dominated by high-energy low-k modes and is near-blind to low-amplitude content. BSP ([Chakraborty/Maulik, JCP 2026](https://arxiv.org/html/2502.00472v1)) makes the same point: "gradient descent naturally favors changes that yield the most substantial reduction in loss early in training… lower frequencies are learned first." MSE thus systematically *under-weights the very modes that carry your recoverable phase signal*.

3. *Gradient pathology.* [Mikhaeil et al. 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/495e55f361708bedbab5d81f92048dcd-Paper-Conference.pdf) proved BPTT gradients through chaotic dynamics necessarily explode as horizon→∞; DySLIM cites this as why long-trajectory L2 "losses become increasingly uninformative." Your single-shot operator sidesteps the gradient explosion but inherits the *uninformativeness* — the late-frame L2 gradient carries little learnable signal because the target is effectively a sample from a decorrelating distribution.

> "due to chaotic divergence, the losses become increasingly uninformative, which causes their gradients to diverge." — [DySLIM](https://arxiv.org/html/2402.04467v2)

A critical nuance for *your* setup: your Gate-2 phase-spread experiment (`wall.md` 2026-06-18) showed late k≤7 phase does NOT intrinsically decorrelate within [0,T] at Re=500 (D_late 0.07–0.30, grows only ~1.3–1.4×). This means you are **below** the regime where Q2a's chaos argument fully bites — your wall is partly the optimizer hedging *anyway* (because the full-T L2 objective dilutes the late gradient and rewards the mean), not because the late state is genuinely unpredictable. This is the strongest reason your wall might be an objective artifact rather than a chaos floor — and it points to Q3 (change the signal) over Q2 (give up on pointwise).

**Q2b — Tractable distributional losses.** Three families are documented as training-time-tractable (no per-sample ensemble, no solver rollout):

| Loss | Mechanism | Training cost | Inference cost | KF-validated? |
|---|---|---|---|---|
| **Invariant-measure MMD** (DySLIM) | MMD² between batch of model states and data states (rational-quadratic kernel mix), as a regularizer on any base loss | O(n²·d) in batch n | 0 (loss-only) | Yes (Re via Kochkov KF) |
| **Spectral-energy / BSP** | squared relative error of binned per-shell energy ratios | O(n·d log d) | 0 | Yes (KF Re=1000) |
| **OT / Sinkhorn on summary stats** (Jiang) | entropy-regularized W₂ on chosen physical statistics (or contrastive features) | O(L²) on stats dim | 0 | 1D/Lorenz; **dropped on 2D-KF as non-discriminative** |

The decisive caveat, already in your `method_synthesis.md` §3.6 and confirmed here: **DySLIM itself reports the Sinkhorn divergence "becomes non-discriminative due to the high-dimension of the state space" for Navier–Stokes**, falling back to MELR/covRMSE/Wasserstein-1. So the OT route (your ranking_v2 Tier-6 "transport loss") is correctly banked dead for 2D-KF. The live distributional levers are MMD (DySLIM) and binned-spectral-power (BSP) — both architecture-agnostic, both addable to your existing FNO+PINO loss at ~zero inference cost. **Neither is listed in `ranking_v2.md` as a standalone late-wall lever** (your §3.4a magnitude band loss is the closest, banked dead at −0.005; but BSP's *relative ratio* formulation and DySLIM's *measure-matching* are distinct objectives, not magnitude-band penalties).

**Q2c — What does a statistics-targeting operator represent, and what does it unlock?** It learns to stay on the attractor's invariant measure rather than track a trajectory. Concretely, DySLIM on Kolmogorov flow improves (Table 1, ×10⁻² units): MELR 3.19→2.46 / 5.35→1.64, Wasserstein-1 4.64→4.51 / 9.66→4.76 / 16.2→5.55, TCM 3.68→0.72 — i.e. 20–65% better distributional fidelity — while *cosine-similarity short-term tracking is only modestly improved*. This is the explicit trade: **late-time phase error is accepted as irreducible chaos**, and the model is graded on whether its long rollouts reproduce spectra, PDFs (velocity, vorticity, TKE, dissipation — see BSP Fig. 6), and two-point statistics. BSP is blunt about it: "Unlike MMD, the BSP loss does not minimize error in physical space, leading to no significant improvement in correlation metrics… for stochastic systems like turbulence, invariant metrics are more meaningful."

The number this unlocks is *not* a lower late-k≤7 relL2 — it is a **different axis entirely** (your oracle floor 0.230 is a pointwise number and is irrelevant under a statistical objective). The honest answer: if you adopt statistics-first, your 0.571 pointwise wall does not move and you stop reporting it; you instead report that the model's late-time energy spectrum, enstrophy PDF, and autocorrelation match GT within X% — quantities your current operator, optimized for pointwise L2, likely *fails* (its late field is oversmoothed/low-energy, R_k≈0.6 aggregate). There is genuine value here for the thesis if late-time *physical realism* is acceptable as the deliverable, but it is a reframing of success, not a breach of the stated wall.

**Q2d — TTA in the distributional regime.** This is the report's weakest-supported and most consequential gap for your thesis. If the operator targets statistical consistency, the coherent TTA question becomes: *adapt the model so its predicted invariant measure / spectral distribution matches the OOD attractor.* The literature offers no direct instantiation for deterministic operators, but three adjacent mechanisms bound the design space:

- **Attractor-projection at inference (Thermalizer, [Pedersen et al., ICML 2025](https://arxiv.org/html/2503.18731v2)):** a pretrained diffusion model "thermalizes" (denoises) the rollout state back onto the attractor's invariant measure on-the-fly, extending stable-prediction horizon by 1–2 orders of magnitude. This is the cleanest existing "adapt-the-statistics" mechanism — but it is generative (your Tier-5 scope exclusion) and operates on AR rollouts, not single-shot.
- **Test-time operator search (Serrano et al. 2026, [2602.00884](https://arxiv.org/abs/2602.00884)):** zero-shot OOD generalization by searching compositions of pretrained operators (DISCO dictionary) at test time *without modifying weights*, recovering PDE parameters; NRMSE improvements up to 200× on parameter-extrapolation. This is a true TTA formulation for the OOD-physics setting (your Re-shift), but it adapts *which operator*, not the attractor statistics, and is parametric.
- **Physics-residual virtual-inputs (PILNO):** unlabeled OOD input ensembles supply physics-only supervision targeting OOD regimes — conceptually your label-free physics-residual TTA, which you already found non-identifying at late times (`ranking_v2.md`).

The synthesis: **a distributional TTA would adapt the spectral/energy distribution** (e.g. minimize a test-time BSP or MMD distance between the model's rolled-out statistics and the OOD field's observed statistics), which *is* coherent and identifiable in a way pointwise/physics-residual TTA is not at late times — because statistics are robust to phase decorrelation (Jiang's Fig. 1a: invariant statistics degrade gracefully with noise while RMSE/Sobolev collapse). This is a *defensible thesis direction* but it is presently unwritten for deterministic single-shot operators; flag it as a research contribution opportunity, not a literature-backed recipe.

**Cross-reference:** Q2c establishes that a statistics-trained operator's success metric is distributional; Q2d follows that its TTA must therefore act on distributional discrepancy — which resolves the tension in your `ranking_v2.md` A2 axis ("TTA adapts a model, not a post-hoc patch"): under a statistical objective, the adaptable quantity is the model's invariant measure, and the robustness of statistics to chaos (Q2a) is exactly what makes late-time TTA identifiable where pointwise TTA (your banked plateau at 0.58–0.59) is not.

---

### Q3 — Solver as last L2 resort

**Conclusion: This is the only documented route to genuine late-time pointwise reduction, and it is cheaper than feared.**

**Q3a — Minimal solver-in-loop with best late pointwise L2; what the solver provides.** [List et al., JFM 2022](https://arxiv.org/pdf/2202.06988) train a CNN corrective forcing inside a differentiable PISO solver with combined L2 + spectral-energy (log-spectral-distance) + strain-rate + multi-step-mean losses, unrolled up to ~30–60 steps. Verified numbers (their Table 1, isotropic decaying turbulence, MSE at t₂≈40 t̂ = late window):

| Model | MSE @ t₁ (~5 t̂) | MSE @ t₂ (~40 t̂, late) |
|---|---|---|
| No-model | 2.78e-3 | 0.057 |
| LES (Smagorinsky) | 2.69e-3 | 0.051 |
| NN, 1-step, full loss | 1.65e-3 | 0.046 |
| NN, 10-step, L2 only | 4.23e-4 | 0.018 |
| **NN, 30-step, full loss** | **4.09e-4** | **0.021** |

Late-window MSE drops 0.057→0.021 (≈2.7×) for decaying isotropic; their mixing-layer cases reach up to ~10× (spatial-mixing Re=500 late velocity MSE 0.0243→0.0025). **What the solver provides that data-L2 cannot:** it advances the *true late dynamics* during training so the loss landscape "better approximates temporal extrapolation" — the network sees and corrects its own compounding error against a physically-evolving state, rather than fitting a static, gradient-diluted late frame. List is explicit that L2 alone "can neglect the time evolution of certain fine scale, low amplitude features" — exactly your low-k phase defect — and that the spectral/strain losses *plus* unrolling are what recover it.

[Kochkov, PNAS 2021](https://www.pnas.org/doi/10.1073/pnas.2101784118): learned interpolation inside a differentiable solver, trained with cumulative pointwise MSE over unrolled steps, matches DNS accuracy at 8–10× coarser resolution (40–80× speedup) and "remains stable during long simulations" — i.e. the learned-corrected coarse solver stays trajectory-accurate (high vorticity correlation) far longer than any pure ML emulator.

**Q3b — Documented late-window-specific improvement and magnitude.** Yes: List's t₂ column *is* a late-window pointwise number, and the improvement is specifically late (the t₁ early-window gap between 10-step and 30-step models is negligible; the unrolling payoff concentrates at long horizons). Magnitude is **scenario-dependent: ~2.7× (decaying isotropic) to ~10× (mixing layers)** — your `method_synthesis.md` §3.5 already banked this range correctly. Kochkov reports long-trajectory vorticity *correlation* improvement rather than a controlled one-step-vs-K relL2 table, so its late-relL2 payoff for your exact metric is qualitative-high-confidence but not numerically pinned.

**Q3c — Implementation cost: full differentiable NS through T steps, or short K-step?** **Short K-step unrolled solver is sufficient, and full differentiability is not required** — the single most cost-relevant finding. [List et al., JCP 2024 ("How Temporal Unrolling Supports Neural Physics Simulators")](https://arxiv.org/html/2402.12971v2) disentangle the two effects of unrolling (data-shift reduction vs long-term gradients) across >3000 models and find:

- **Non-differentiable unrolling (NOG)** — solver in the forward loop, gradients cut at the solver (∂g/∂g := 0) — gives a 23% average improvement and **up to 4.5× over pure prediction**, using *existing non-differentiable solvers*. This is the cheap entry point: no adjoint NS implementation needed.
- **Differentiable unrolling (WIG)** delivers the best accuracy but adds only **~38% over one-step** on top; long-term gradients matter most for *correction* setups (network corrects a solver), which is precisely the hybrid regime.
- **Both are Lyapunov-horizon-bounded:** NOG gradient inaccuracy grows ∝m², WIG gradients explode exponentially (per Mikhaeil). Hence **short curriculum horizons suffice** — List uses m=[1,2,4] for Kolmogorov and m=3 for KS; "best results are achieved when the unrollment approaches the integral timescale," not beyond it.

For your setup this means: a corrective hybrid (FNO output corrected by, or correcting, a coarse NS solver) trained with a **short m≈3–4 step unrolled curriculum** — even with a *non-differentiable* solver — is the minimal viable Q3 implementation, not a fully-differentiable T-step (T=64) rollout graph. This also reframes your Tier-2/3 window-chaining: chaining is essentially unrolling *without a solver* (pure prediction), which List's data shows is the weakest variant — consistent with your chain Variant A flat result. Inserting even a non-differentiable solver into the loop is the documented step-change.

**Cross-reference:** Q3 is the constructive complement to Q1's negative result and Q2's reframing. Where Q2 says "stop optimizing pointwise L2 because it's mis-specified," Q3 says "keep pointwise L2 but fix its *signal source* by interposing the true dynamics." Your `ranking_v2.md` A2 axis ranks modeling (window-chaining, Tier 2) above solving (List/Kochkov, Tier 3) for TTA-alignment — but the List 2024 result complicates this: a non-differentiable solver-in-loop is *also* a modeling intervention (it reshapes the operator's weights, leaving an adaptable model), at a fraction of the differentiable-solver setup cost, and with documented late-time pointwise payoff that window-chaining lacks. The modeling-vs-solving dichotomy is softer than the ranking assumes.

---

## Comparison: the three coherent exits from the wall

| Criterion | Q2 Statistics-first (DySLIM/BSP) | Q3a Differentiable solver-in-loop (List/Kochkov) | Q3c Non-diff. unrolled solver (List 2024 NOG) |
|---|---|---|---|
| Lowers late *pointwise* relL2? | ❌ No (trades it away) | ✅ Yes, 2.7–10× MSE (scenario-dep.) | ✅ Yes, up to 4.5× over prediction |
| Inference cost added | 0 (loss-only) | 0 (training-only) but solver at *train* | 0 (training-only) |
| Setup cost | Low (loss term) | High (differentiable NS, adjoint) | **Medium (existing non-diff solver + interface)** |
| One-shot inference preserved? | ✅ Yes | ✅ Yes (deploy network standalone) or hybrid | Hybrid (network + solver) at inference |
| In `ranking_v2.md`? | Partially (Tier 6 OT dead; MMD/BSP **not** listed) | Yes (Tier 3.2/3.4) | Subsumed under Tier 3, **cost overstated** |
| TTA-alignment (A2) | High but *redefines* TTA target (Q2d, unwritten) | Medium (modeling, adaptable model) | **High (modeling, cheap, adaptable)** |
| KF-validated late pointwise? | N/A (not the metric) | Decaying/mixing-layer, not forced-KF exact | KF Re=300–1000, relL2 over 250 steps |

**What the comparison reveals.** The genuinely new information relative to your ranking is in the third column. Your `ranking_v2.md` treats solver-in-loop as a uniformly high-setup-cost "resort" (Tier 3) ranked below modeling levers. List 2024 breaks that assumption: the *non-differentiable* unrolled-solver variant is (i) a modeling intervention that improves the operator's own late state — satisfying your A2 TTA-alignment criterion as well as window-chaining does — (ii) cheaper than the differentiable version because it reuses your existing `NavierStokes2d` solver without adjoint re-implementation, and (iii) the *only* option in the table with a documented late-window pointwise improvement on forced Kolmogorov flow specifically. The statistics-first column, by contrast, is the only path that is genuinely free at both train interface and inference, but it does not answer the pointwise question — it dissolves it.

The surprise: your two "not dead, not tried" levers (differentiable window-chaining, Spectral-Refiner H⁻¹ FT) may both be dominated by an option you ranked as a pure-solving resort. Window-chaining is unrolling-without-a-solver (List's weakest variant, matching your flat chain result); Spectral-Refiner is a per-instance post-hoc patch that breaks one-shot. A short non-differentiable solver-in-loop curriculum sits between them on cost and above both on documented late-pointwise payoff.

---

## Contradictions & Debates

- **Magnitude of solver-in-loop late-time gain.** List 2022 reports a 4× range: ~2.7× (decaying isotropic, t₂ MSE 0.057→0.021) up to ~10× (spatial mixing layer). **Resolution:** not a contradiction but flow-dependence — the gain is largest where the unresolved-scale feedback on resolved scales is strongest (mixing layers) and smaller for decaying isotropic. For forced Kolmogorov flow the honest expectation is the *middle* of this range, and no source pins the exact forced-KF late-relL2 number — flag as the principal numerical uncertainty for Q3.

- **Is a differentiable solver necessary?** Implicit assumption in your `method_synthesis.md` §3.5/§4.2 (and the field's framing) is yes. **List 2024 contradicts this:** NOG (non-differentiable, gradients cut at solver) captures up to 4.5× of the benefit; WIG (differentiable) adds only ~38%. **Resolution:** the differentiable solver is a ~1.4× refinement, not a prerequisite; the *forward* solver-in-loop (data-shift reduction) does the heavy lifting. This materially lowers your Q3 setup cost estimate.

- **Does statistics-first help or hurt pointwise tracking?** DySLIM claims its regularizer improves "point-wise tracking *and* long-term statistical accuracy"; BSP states it gives "no significant improvement in correlation metrics." **Resolution:** DySLIM's pointwise gain is a *short-horizon* cosine-similarity / decorrelation-time effect (the regularizer stabilizes the model, delaying blow-up), not a late-time relL2 reduction; BSP, which is purely spectral, is honest that physical-space error is unchanged. Neither supports a late-time *pointwise* improvement — consistent with Q1's negative answer. The disagreement is about short-horizon stability, not your late wall.

---

## Uncertainties & Gaps

- ⚠️ **Q2d distributional-TTA formulation (confidence ~68%):** no published work adapts a deterministic operator's attractor statistics / spectral distribution at test time. The synthesis (test-time BSP/MMD discrepancy minimization) is a defensible extrapolation from Thermalizer (generative, out-of-scope) and Serrano operator-search (parametric), not a documented recipe. Treat as a research-contribution opportunity.
- ⚠️ **Exact forced-KF late-window pointwise number under solver-in-loop:** List's KF results (2024) report rollout relL2 over 250 steps but not a clean early-vs-late split at your Re=500/256² configuration; the 2.7–10× range is from decaying/mixing-layer cases. The forced-KF late-relL2 payoff is high-confidence-positive but not numerically pinned.
- ⚠️ **Spectral-Refiner OSTI table values** (H⁻¹ FT 2.08e-2→2.82e-4, ~74×) rely on your banked verification (`method_synthesis.md`, verified 2026-06-25); the OSTI PDF was not machine-extractable in this pass. No new contradiction surfaced, but the number is single-sourced through your own prior read.
- ⚠️ **2026-dated arXiv IDs** (SGNO 2602.x, Serrano 2602.00884, PITO 2601.x, Ms-MoE 2604.x): consistent with the current date (2026-06-27) but several are very recent preprints; SGNO's kolm2d 0.558 and the APEBench KF figures are the load-bearing Q1 datapoints and are corroborated by your own `method_synthesis.md`.

---

## Recommendations

### Primary
**Pursue the short non-differentiable solver-in-loop curriculum (Q3c) as the highest-yield in-scope lever, ahead of the differentiable version.** Evidence: it is the only option with a documented late-window *pointwise* improvement on forced Kolmogorov flow (List 2024, up to 4.5× over prediction), it reuses your existing `NavierStokes2d` solver without adjoint re-implementation (List 2024: non-diff captures the bulk of the gain; differentiable adds only ~38%), it preserves a modeling intervention that leaves an adaptable late-state model (satisfying your A2 TTA-alignment), and it is bounded to short curriculum horizons (m≈3–4, integral-timescale-matched) so the cost is far below a T-step differentiable rollout. This dominates your "not dead, not tried" window-chaining (which is solver-free unrolling — List's weakest variant, consistent with your flat chain result) on documented late-pointwise payoff at comparable cost.

### Alternative
**If late-time *physical realism* (not pointwise accuracy) is an acceptable thesis deliverable, add a DySLIM-style invariant-measure MMD or a BSP spectral loss to your existing PINO objective (Q2b).** Both are architecture-agnostic, zero-inference-cost, and not in your `ranking_v2.md` as standalone late-wall levers (distinct from your banked-dead magnitude band loss). This will not move your 0.571 pointwise wall — accept that — but it makes "late-time accuracy" a well-posed, achievable target (on-attractor spectra/PDFs) and opens the one genuinely novel TTA direction (Q2d: adapt the spectral distribution at test time, which is identifiable where your banked pointwise/physics-residual TTA plateaued at 0.58–0.59). Choose this only if the thesis can reframe success as distributional consistency.

### Not Recommended
- **More loss-shape tuning under pointwise L2** (further angle/band/time-reweight sweeps): the field's collective null result (Q2a) plus your 6-week banked falsifications make this the lowest-expected-value path. The objective is mis-specified, not mis-tuned.
- **OT/Sinkhorn distributional loss on 2D-KF:** DySLIM itself reports it non-discriminative in this dimension; your `ranking_v2.md` Tier-6 banking is correct.
- **Fully differentiable T-step NS rollout as the first solver attempt:** List 2024 shows the ~38% differentiability premium does not justify the adjoint-implementation cost before the cheaper non-differentiable unrolled variant is tried.

---

## Methodology

- **Depth:** exhaustive
- **Search rounds:** 4 substantive rounds, ~16 queries, plus 8 deep primary-source extractions (DySLIM, Jiang, BSP, List-2022, List-2024, PDE-Refiner, Serrano-2026, plus internal experiment logs)
- **Final confidence:** 85% overall (Q2a/Q2b/Q3a/Q3c at 90–93%; Q1 88%; Q2c 80%; Q2d 68%)
- **Sub-questions:** 8 defined, 8 answered (Q2d partially — answered with an honestly-flagged literature gap)
- **Multi-hop chains used:** entity expansion (List 2022 → List 2024 unrolling taxonomy → NOG/WIG cost finding); conceptual deepening (pointwise-L2 failure → regression-to-mean → spectral-bias → gradient pathology, triangulated across DySLIM/PDE-Refiner/BSP/Mikhaeil); causal chain (Q1 absence → Q2a objective mis-specification → Q3 signal-source fix); source triangulation (vendor-free — all Tier-1 peer-reviewed/arXiv, cross-checked against your banked `method_synthesis.md` verifications)
- **Key challenges:** the central Q1 result is a *negative* (no paper reports the target metric), which required confirming the absence is deliberate (field pivoted to statistics) rather than incidental; OSTI Spectral-Refiner PDF non-extractable; Q2d targets a formulation that does not yet exist in the deterministic-operator literature

---

## Sources

| # | Title | URL | Date | Credibility |
|---|-------|-----|------|:-----------:|
| 1 | DySLIM: Dynamics Stable Learning by Invariant Measure (ICML 2024) | https://arxiv.org/abs/2402.04467 | 2024 | ⭐ Tier 1 |
| 2 | Jiang et al., Training neural operators to preserve invariant measures (NeurIPS 2023) | https://arxiv.org/abs/2306.01187 | 2023 | ⭐ Tier 1 |
| 3 | Chakraborty/Maulik, Binned Spectral Power Loss (JCP 2026) | https://arxiv.org/html/2502.00472v1 | 2026 | ⭐ Tier 1 |
| 4 | List/Chen/Thuerey, Learned Turbulence Modelling w/ Differentiable Fluid Solvers (JFM 2022) | https://arxiv.org/pdf/2202.06988 | 2022 | ⭐ Tier 1 |
| 5 | List et al., How Temporal Unrolling Supports Neural Physics Simulators (JCP 2024) | https://arxiv.org/html/2402.12971v2 | 2024 | ⭐ Tier 1 |
| 6 | Kochkov et al., ML-accelerated CFD (PNAS 2021) | https://www.pnas.org/doi/10.1073/pnas.2101784118 | 2021 | ⭐ Tier 1 |
| 7 | Lippe et al., PDE-Refiner (NeurIPS 2023) | https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf | 2023 | ⭐ Tier 1 |
| 8 | Mikhaeil et al., On the difficulty of learning chaotic dynamics with RNNs (NeurIPS 2022) | https://proceedings.neurips.cc/paper_files/paper/2022/file/495e55f361708bedbab5d81f92048dcd-Paper-Conference.pdf | 2022 | ⭐ Tier 1 |
| 9 | Serrano et al., Test-time Generalization for Physics through Neural Operator Splitting (ICML 2026) | https://arxiv.org/abs/2602.00884 | 2026 | ⭐ Tier 1 |
| 10 | Pedersen et al., Thermalizer: Stable AR neural emulation of spatiotemporal chaos (ICML 2025) | https://arxiv.org/html/2503.18731v2 | 2025 | ⭐ Tier 1 |
| 11 | Koehler/Thuerey, APEBench (NeurIPS 2024) | https://github.com/tum-pbs/apebench | 2024 | ⭐ Tier 1 |
| 12 | Cao et al., Spectral-Refiner / ST-FNO3d (ICLR 2025, OSTI) | https://www.osti.gov/servlets/purl/2563339 | 2025 | ⭐ Tier 1 |
| 13 | Atif et al., FNO for spatiotemporal 2D turbulence | https://arxiv.org/html/2409.14660v1 | 2024 | ⭐ Tier 1 |
| 14 | PILNO: Physics-Informed Laplace Neural Operator | https://arxiv.org/html/2602.12706v1 | 2026 | 🔵 Tier 2 |
| 15 | PITO: Physics-Informed Transformer operator for 3D turbulence | https://arxiv.org/html/2601.19351v7 | 2026 | 🔵 Tier 2 |
| 16 | PDE-SpectralRefiner | https://arxiv.org/html/2506.10711v2 | 2026 | 🔵 Tier 2 |
| 17 | Stachenfeld et al., Learned Coarse Models for Efficient Turbulence Simulation | https://arxiv.org/abs/2112.15275 | 2021 | ⭐ Tier 1 |
| 18 | Li et al., IU-FNO long-term turbulence predictions (Phys. Fluids 2023) | https://ui.adsabs.harvard.edu/abs/2023PhFl...35g5145L/abstract | 2023 | ⭐ Tier 1 |
| 19 | Ms-MoE IFactFormer, stable fine-time-step long-horizon turbulence | https://arxiv.org/html/2604.12794v1 | 2026 | 🔵 Tier 2 |
| 20 | DiAFNO, FNO+diffusion AR for 3D turbulence | https://arxiv.org/html/2512.12628v2 | 2025 | 🔵 Tier 2 |
| 21 | PENO: Physics-enhanced Neural Operator for turbulent transport | https://arxiv.org/pdf/2406.04367 | 2024 | 🔵 Tier 2 |
| 22 | Li et al., MNO / Learning Dissipative Dynamics (NeurIPS 2022) | https://www.researchgate.net/publication/353343553 | 2022 | ⭐ Tier 1 |
| 23 | DySLIM ICML 2024 slides | https://icml.cc/media/icml-2024/Slides/35055.pdf | 2024 | ⭐ Tier 1 |
| 24 | Liu-Schiaffini et al., Localized Integral & Differential Kernels (ICML 2024) | https://arxiv.org/abs/2402.16845 | 2024 | ⭐ Tier 1 |
| 25 | Lyapunov Learning at the Onset of Chaos | https://arxiv.org/html/2506.12810v1 | 2025 | 🔵 Tier 2 |
| 26 | Temporal Test-Time Adaptation with State-Space Models (STAD) | https://arxiv.org/html/2407.12492v3 | 2024 | 🔵 Tier 2 |
| 27 | Towards Stability of Autoregressive Neural Operators | https://openreview.net/forum?id=RFfUUtKYOG | 2023 | 🔵 Tier 2 |
| 28 | Recurrent flow patterns as basis for 2D turbulence (PNAS) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11161751 | 2024 | ⭐ Tier 1 |
