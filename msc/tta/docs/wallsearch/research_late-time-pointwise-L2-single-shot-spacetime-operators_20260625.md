# Late-Time Pointwise relL2 in Single-Shot Space-Time Neural Operators for 2D Kolmogorov/Navier-Stokes

**Date:** 2026-06-25
**Depth:** exhaustive
**Confidence:** 87% (honest cap — the central finding is a negative existence claim, whose confidence is structurally bounded; see Methodology)
**Sources:** 48 unique sources across 7 search rounds (incl. 6 open-ended discovery searches with no seed arXiv IDs)
**Scope lock:** deterministic single-shot space-time map 𝒢: a(x,y) → u(x,y,t) over t∈[0,T], one forward pass, NO autoregressive rollout, NO diffusion/probabilistic models. Architectures: FNO3d (2 space + 1 time), U-Net. System: 2D incompressible Navier-Stokes / Kolmogorov forced flow (vorticity form), moderate Re.

---

## Executive Summary

**Your prior hypothesis is essentially correct, with one important qualification.** No paper has cleanly reduced *deterministic pointwise late-time relative-L2* (t/T>0.5) for a *single-shot space-time* FNO3d/U-Net on chaotic 2D Kolmogorov/Navier-Stokes, and no paper even *reports* a per-time-slice relL2 breakdown with a targeted lever for that regime — this is a genuine field-level gap confirmed across 48 sources, not a search miss. The field has demonstrably migrated to spectral / invariant-measure / correlation-time metrics (PDE-Refiner reports correlation-time, MNO/DySLIM/Jiang report invariant-measure statistics, and a 2026 Nature Communications paper states outright that "pointwise L2 error is a poor metric for multi-scale systems").

The qualification: **one single-shot space-time result does lower pointwise relL2 by ~2 orders of magnitude — Spectral-Refiner's ST-FNO3d (ICLR 2025) drops vorticity relL2 from 1.31×10⁻² to ~3×10⁻⁴ via a post-hoc convex negative-Sobolev/L2 fine-tuning step — but only on weakly-turbulent and Taylor-Green toy cases, NOT in the strongly-chaotic KF regime, and it requires an auto-differentiable solver at inference (a hybrid, equation-aware refinement, still deterministic).** This single result is the strongest evidence that your late wall is partly model/optimization-limited at low effective Re, while a separate body of chaos-theory evidence (Lyapunov bounds; the "0.5·T_L completeness" result) shows it becomes fundamentally chaos-limited beyond ~0.5 Lyapunov times. Most loss-function and unrolled-training levers you listed are autoregressive-rollout mechanisms whose stated benefit is *long-time statistics or correlation horizon, not pointwise late-time relL2*, and several do not even apply to a single-shot operator.

---

## Key Findings

1. **No clean single-shot late-time pointwise-relL2 reduction exists on chaotic 2D KF** — no paper reports a t/T>0.5 relL2 time-slice breakdown plus a training/architecture lever that lowers it specifically; confirmed across the seed list and beyond-seed discovery searches. [Spectral-Refiner](https://arxiv.org/html/2405.17211v2) admits "no NO-only-based operator learning approach can break the barrier of even 2-digit accuracy" in the Bochner norm on 2D NSE.

2. **The field migrated away from pointwise L2 for turbulence — explicitly.** A 2026 Nature Communications paper states pointwise L2 "overweighs low-wavenumber (high-energy) errors and underweighs high-wavenumber discrepancies, enabling smoothed predictions to achieve low errors." [Oommen et al.](https://www.nature.com/articles/s41467-026-70145-4) MNO, DySLIM, Jiang-OT, PDE-Refiner all report spectral/invariant-measure/correlation metrics instead.

3. **Single-shot space-time error is roughly *flat* in time (high floor, mild growth), not a steep late wall.** FNO3d on 2D turbulence shows errors that "begin with large values and increase marginally as time progresses" — qualitatively different from the exponential late-wall of autoregressive rollout. [Atif et al. 2024](https://arxiv.org/html/2409.14660v1)

4. **The single demonstrated single-shot relL2 reduction is post-hoc convex spectral refinement (Spectral-Refiner), ~10⁻² → ~3×10⁻⁴, but only on weakly-turbulent/TG cases and using an auto-diff solver at inference.** [Cao et al. ICLR 2025](https://arxiv.org/html/2405.17211v2)

5. **Time-as-Fourier-dimension is *worse* than time-as-channels** — the temporal periodicity/aliasing mismatch in FNO3d's spectral treatment of the (non-periodic) time axis is a concrete, fixable defect. [Atif et al.](https://arxiv.org/html/2409.14660v1); ST-FNO's periodic temporal padding + layernorm fixes the hard-coded temporal lifting. [Cao et al.](https://www.osti.gov/servlets/purl/2563339)

6. **Hard divergence-free (Leray/Helmholtz) projection at inference lowers test L2 on incompressible NS** "most often" and applies to a single-shot operator's velocity output with near-zero cost. [Negiar et al.](https://openreview.net/pdf/7a772c3b05e62ae8956a63349267c06c8a3b56fa); [Project-and-Generate 2026](https://arxiv.org/html/2603.24500v1)

7. **Most loss/unrolled levers are autoregressive mechanisms.** Pushforward, curriculum, MNO-dissipativity, DySLIM-measure-matching, scheduled temporal weighting, conserved-quantity correction (DPOTc) all target rollout error accumulation, long-time statistics, or correlation horizon — not single-shot pointwise late-time relL2. Several have no analogue in a one-pass operator.

8. **The late wall is chaos-limited beyond ~0.5·T_L, model-limited below it.** A 2D-turbulence study shows predictions up to 0.5·T_L are "almost complete" (recoverable) while "T_L or 2T_L [is] inevitably incomplete" because enstrophy cascades to the smallest scale within 0.5·T_L. [Cambridge JFM GAN study](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/prediction-and-control-of-twodimensional-decaying-turbulence-using-generative-adversarial-networks/EAE377A4E1F784D135520DB1EE3F25A0); Lyapunov λ ~ Re^½ sets the horizon. [Nastac & Ihme PRF 2017](http://web.stanford.edu/group/ihmegroup/cgi-bin/MatthiasIhme/wp-content/papercite-data/pdf/physrevfluids.2.094606.pdf)

9. **Spectral bias is the shared root cause of the high floor.** Neural operators "consistently neglect components of the spatial frequency spectrum that have low amplitude … ultimately resulting in a noticeable decline in rollout performance." [PDE-Refiner](https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf); F-principle. This connects the loss-function (Q2), time-axis (Q3), and wall (Q6) findings into one mechanism.

---

## Detailed Analysis

### Q1 — Has anyone reported single-shot space-time late-time pointwise relL2 (t/T>0.5) AND a lever that lowers it?

**Conclusion: No clean instance exists in the chaotic KF regime, and no paper even reports the t/T>0.5 time-slice breakdown there. The closest evidence — Spectral-Refiner — does lower single-shot pointwise relL2 by ~2 orders of magnitude, but on weakly-turbulent/Taylor-Green cases and via equation-aware post-hoc refinement, not in the chaotic late-wall regime you are targeting.**

The base reference, **PINO** ([Li et al. 2111.03794](https://arxiv.org/html/2111.03794v3)), *is* a single-shot space-time operator: it learns 𝒢†: a → u(t)∈C((0,T];𝒰) for the dynamical NS system, and its PDE residual loss is the Bochner norm ‖∂ₜu − ℛ(u)‖²_{L²(T;D)} integrated over the *whole* [0,T] window including late time. Its reported metric is relative L2, and its headline result is "on average, the relative error is 7% lower on both transient and Kolmogorov flows" versus data-driven FNO (Kolmogorov at Re 500, ν=1e-3). Crucially, **PINO never decomposes error by time slice** — there is no t/T>0.5 reporting, and the 7% is an aggregate from adding the PDE loss plus instance-wise fine-tuning. So PINO confirms the single-shot setup and a small aggregate gain, but does not address the late wall as such.

**Spectral-Refiner / ST-FNO3d** ([Cao et al., ICLR 2025, 2405.17211](https://arxiv.org/html/2405.17211v2)) is the most important single result for this question. It is an explicitly single-shot trajectory-to-trajectory FNO3d mapping one time window to the next in one pass. On the McWilliams isotropic-turbulence benchmark (train 64², eval 256²), vorticity relL2 goes:

> FNO3d (100 epochs): ‖ε‖_L2 = 1.31×10⁻² → ST-FNO3d (10 epochs + L2 fine-tune): 1.02×10⁻² (post-training) → after post-hoc convex fine-tune: 2.82×10⁻⁴ (L2 FT) / 3.16×10⁻⁴ (H⁻¹ FT). Taylor-Green vortex after fine-tune: 1.24×10⁻⁷. — [Cao et al., Table 2/3](https://www.osti.gov/servlets/purl/2563339)

That is a ~50–100× reduction in pointwise relL2 from a single-shot space-time operator. But three caveats place it outside your exact target: (i) the strong reductions are on **weakly-turbulent / decaying / Taylor-Green** cases, not strongly-chaotic forced KF at moderate-high Re over long T; (ii) the Taylor-Green error is reported "at the final time step" — it touches late time, but TG is not chaotic; (iii) the refinement **uses an auto-differentiable numerical solver at inference** to evaluate the a-posteriori residual (a hybrid, equation-aware step), so it is not a pure data-driven training/architecture change. It remains fully deterministic (no diffusion). The paper's own framing supplies the negative claim for the rest of the field: *"in all end-to-end operator learning benchmarks of NSE, even SOTA NOs still fall short … no NO-only-based operator learning approach can break the barrier of an even 2-digit accuracy"* in the Bochner norm.

The third corroborating piece is **Atif et al.** ([2409.14660](https://arxiv.org/html/2409.14660v1)): for a single-shot 3D FNO (2 space + 1 time, 10→10 snapshots) on 2D turbulence, "errors … begin with large values and increase marginally as time progresses." This is the key structural observation: **a single-shot space-time operator does not produce the steep autoregressive late-wall; it produces a roughly flat, high error floor across the window.** Your "late slices dominate total error" symptom in a single-shot operator therefore reflects a *floor that is mildly tilted upward*, not runaway accumulation — which changes which levers are relevant (it points toward the spectral-bias floor and the time-axis treatment, not toward rollout-stabilization tricks).

**Cross-reference:** the flat-error structure (Q1) and the spectral-bias diagnosis (Q2, Q6) are the same phenomenon viewed two ways; and the only lever shown to break the floor (Spectral-Refiner) is a Sobolev-norm refinement (Q2) enabled by the temporal-padding ST-FNO architecture (Q3).

### Q2 — Loss-function levers: pointwise relL2 vs spectral/statistical, and late-time effect?

**Conclusion: Of the four levers, only Sobolev/negative-Sobolev has a documented single-shot pointwise-relL2 reduction (Spectral-Refiner); spectral/frequency-weighted losses report spectral (fRMSE) gains and modest correlation-time gains; OT and dissipativity losses report invariant-measure/statistical metrics and are autoregressive — they do not target pointwise late-time relL2.**

**Sobolev Hᵏ loss.** Two distinct results. (a) Spectral-Refiner's **negative** Sobolev H⁻¹ norm, used as a convex post-hoc fine-tuning loss, is "nothing but an inverse frequency weight in the frequency domain that emphasizes the learning of low frequency information," and drives the residual to 4.59×10⁻⁷ while reaching ~3×10⁻⁴ relL2 — the cleanest single-shot relL2 reduction found ([Cao et al.](https://www.osti.gov/servlets/purl/2563339)). (b) A **positive** Sobolev H¹ loss, in PDE-Refiner's ablation on 1D KS, gives only a modest gain (Sobolev k=1 → 79.7 s high-correlation time vs MSE 75.4 s) and the metric is correlation time, not relL2 ([Lippe et al.](https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf)). A geometric-NO paper (MCL) reports H¹/relL2 both improving with gains amplifying at longer rollout, but on 1D/standard PDEs, autoregressive ([2602.16209](https://arxiv.org/html/2602.16209v1)). Net: positive-Sobolev is a weak lever; negative-Sobolev (frequency-weighted toward the energy-containing modes) is the strong, documented one — and it is single-shot.

**Spectral / frequency-weighted loss.** LOGLO-FNO adds a frequency loss on top of MSE and reports **fRMSE** (frequency-banded RMSE) on 2D Kolmogorov (Re=5k), with the largest gains in the high-frequency band (−58% to −73% high-band fRMSE), but the metric is spectral, evaluated at 1-step/5-step, not single-shot late-time relL2 ([LOGLO-FNO](https://openreview.net/pdf/343ad139b82699e6e171af675eb975f5c84274be.pdf)). PDE-Refiner's central diagnosis is that MSE training is "only accurate for a small, high-amplitude frequency band," motivating frequency-aware modeling — this is the mechanism, but their fix is a diffusion-style refiner (probabilistic, excluded). The 2026 Nature Comm paper makes the deprecation explicit (quoted above), and the adv-NO it proposes cuts energy-spectrum error 15× — but adversarial/generative, excluded.

**Transport / optimal-transport loss.** Jiang et al. ([2306.01187](https://arxiv.org/html/2306.01187v3)) match distributions of summary statistics via a Sinkhorn/Wasserstein loss; the metric is invariant-measure preservation, the training is autoregressive/recurrent, and — per DySLIM's assessment — "the approach was only applied to small systems with low-dimensional attractors, and it is not clear how well it scales to … the Kolmogorov flow." DySLIM ([Schiff et al. 2402.04467](https://arxiv.org/html/2402.04467v1)) uses an MMD measure-matching regularizer; on Kolmogorov it reports MELR, MELRw, covRMSE, pointwise-Wasserstein, TCM, and cosine similarity, and explicitly drops Sinkhorn divergence because "SD becomes non-discriminative due to the high-dimension." It improves short-term cosine similarity and long-term statistics — **but reports no pointwise relL2 for KF and is autoregressive.**

**Dissipativity / Lipschitz / stability regularization.** MNO ([Li et al. NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/6ad68277e27b42c60ac228c9859fc1a2-Paper-Conference.pdf)) hard-codes a dissipative term and reports Fourier spectrum, POD spectrum, pointwise distribution, autocorrelation, TKE, dissipation rate (Kolmogorov up to Re 5000) — invariant-measure statistics, autoregressive. SGNO ([2602.18801](https://arxiv.org/html/2602.18801v1)) derives a Lipschitz/contraction one-step amplification bound and reports nRMSE on APEBench (Kolm-2D: FNO 0.689 → SGNO 0.586) — autoregressive long-horizon, not single-shot pointwise late-time relL2.

**Cross-reference:** every "statistical" lever here (OT, dissipativity, measure-matching) is the field's *response* to the Q6 chaos wall — they concede pointwise prediction and target the attractor instead. This is the migration your hypothesis identified.

### Q3 — Time-axis treatment in single-shot operators

**Conclusion: This is the most under-exploited and most single-shot-relevant lever. Two concrete, documented defects in FNO3d's temporal axis — (i) treating the non-periodic time axis as a periodic Fourier dimension causes aliasing and *higher* error than time-as-channels, and (ii) the hard-coded temporal lifting/normalization — are directly fixable, and the fix (periodic temporal padding + layernorm) is exactly what enables Spectral-Refiner's accuracy.**

The sharpest evidence is Atif et al.'s direct comparison: 2D-FNO-with-temporal-channels yields "significantly lower errors" and shorter training than 3D-FNO, "because the periodic spatial dimension cannot be treated at the same footing as non-periodic temporal dimensions" ([Atif et al.](https://arxiv.org/html/2409.14660v1)). In FNO3d the time axis goes through a real FFT that *assumes periodicity*; the solution trajectory is not periodic in time, so the temporal spectral convolution injects aliasing/wrap-around artifacts. Spectral-Refiner identifies the same hard-coded temporal handling as "the biggest hurdle" and fixes it with **periodic padding along the temporal dimension** in the lifting operator, **layer normalization replacing the hard-coded global normalizer**, and **zero-padding the latent temporal dimension** in the projection layer for temporal super-resolution ([Cao et al.](https://www.osti.gov/servlets/purl/2563339)). Boundary-artefact work confirms that naive padding before an FFT introduces edge oscillations and that smooth/periodic padding mitigates them ([Lund thesis](https://lup.lub.lu.se/student-papers/record/9213930/file/9213931.pdf)).

**Temporal loss weighting (upweight late t).** The directly-named method, Scheduled Temporal Loss Weighting ([NeurIPS ML4PS 2025](https://neurips.cc/virtual/2025/123050)), does the *opposite* of what you propose — it *prioritises early* time steps during early training (curriculum) to avoid overfitting to early dynamics while attending to all steps — and it is an autoregressive method on 1D PDEs (Advection, Burgers, KdV; nRMSE improvements 16–62%). It is evidence that temporal loss reweighting moves the error budget, but it neither targets late-time upweighting nor a single-shot operator. No paper was found that ablates *late-time upweighting* in a single-shot space-time operator — a genuine gap.

**Causal/temporal-bundling architectures.** Temporal bundling (predict a block of future steps in one pass) originates with Brandstetter et al. and is used by the Temporal Neural Operator to "limit cumulative distribution shifts and curb error growth" ([TNO, Sci Rep 2025](https://www.nature.com/articles/s41598-025-16922-5)) — but bundling is a *rollout-efficiency* device (fewer autoregressive steps); within a single fixed window it is just the single-shot output itself. Output-window length T is implicitly the single-shot setting; CORAL/neural-fields shows the in-horizon vs out-of-horizon cliff (FNO NS in-time 5.68×10⁻⁴ → out-of-time 8.95×10⁻³, ~16×), quantifying how badly error jumps once t exceeds the trained window ([NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/df54302388bbc145aacaa1a54a4a5933-Paper-Conference.pdf)).

**Cross-reference:** the temporal-aliasing defect (Q3) is the architectural counterpart to the spectral-bias floor (Q2/Q6); fixing the time axis is what let Spectral-Refiner's Sobolev refinement (Q2) reach 10⁻⁴ (Q1).

### Q4 — Differentiable multi-step / unrolled training: does it lower late-time pointwise relL2 on KF, and transfer to single-shot?

**Conclusion: Unrolled/differentiable training demonstrably improves long-horizon *stability* (and L2 over rollouts) for autoregressive time-steppers on 2D KF, but (a) the benefit is bounded by the Lyapunov time, (b) the reported wins are stability/correlation-horizon rather than a late-time pointwise-relL2 reduction at fixed t, and (c) it does not directly transfer to a single-shot space-time operator, which has no rollout to unroll — the analogue is differentiable chaining of K single-shot windows.**

List et al. ([2402.12971](https://arxiv.org/html/2402.12971v1)) is the definitive study: they compare one-step (ONE), fully-differentiable unrolling (WIG), and non-differentiable unrolling (NOG) on multiple systems including 2D Kolmogorov flow, metric L2. WIG performs best, NOG comes close; both are limited because "the stability of the backpropagation gradients … align[s] with the Lyapunov time of the physical system," and "unrolling for too many steps was associated with diminishing returns … and even showed negative effects." So the effective horizon K is capped by chaos. DySLIM's baselines (curriculum, pushforward) and PDE-Refiner's ablation both show the standard pushforward trick "does not improve performance" on their settings and "cannot learn to include the low-amplitude information for accurate long-term predictions." Conserved-quantity-corrected DPOT (Q5) and SGNO (Q2) push correlation horizon further but remain autoregressive.

The transfer question is structural: a single-shot operator 𝒢: a → u(·,[0,T]) emits the whole window in one pass, so there is nothing to "unroll." The honest analogue is **differentiable chaining of K successive single-shot windows** (window-to-window, à la Spectral-Refiner's trajectory-to-trajectory map, with backprop across the K joins). No paper was found that does exactly this on KF and reports a late-time relL2 reduction — so the lever is *plausible but untested* in your exact setting. The List et al. finding implies any such chaining will be horizon-limited by the Lyapunov time per window.

**Cross-reference:** Q4's Lyapunov-bounded horizon is the same wall quantified in Q6; the unrolled-training literature is effectively an empirical measurement of that bound.

### Q5 — Post-hoc physics correction at inference

**Conclusion: This is the cleanest, lowest-effort, single-shot-compatible lever with a documented relL2 *reduction* — but the strong evidence is for divergence-free (Leray) projection on the velocity form, not yet for enstrophy/circulation projection on the KF vorticity form, which is a genuine open niche.**

Two results matter. (1) **Hard divergence-free (Leray/Helmholtz-Hodge) projection.** Negiar et al. ([OpenReview](https://openreview.net/pdf/7a772c3b05e62ae8956a63349267c06c8a3b56fa)) implement the Leray projection efficiently in Fourier space and show it "most often lead[s] to lower test L2 errors, despite the divergence-free condition not being strongly enforced in the training data," and it can be applied **during or after training** at small cost — i.e., as a pure inference-time post-processing step on a single-shot operator's velocity output. "Project and Generate" ([2603.24500](https://arxiv.org/html/2603.24500v1)) confirms on 2D NS: baselines have divergence O(10²–10⁵), projection drives it to ~2×10⁻⁷, with "improved physical fidelity, enhanced stability under long-term rollout, and superior performance." (2) **Conserved-quantity correction, DPOTc** ([2601.22541](https://arxiv.org/html/2601.22541v1)): on *compressible* NS (PDEBench, Mach 0.1), correcting density/momentum/energy yields relL2 rising only 0.4% per timestep and extends high-correlation duration to 28 timesteps (vs DPOT 25, FNO 13); after 10 autoregressive steps, 7.8% vs 8.3% baseline. DPOTc is autoregressive and compressible, so the *numbers* don't transfer to incompressible KF, but the *mechanism* (post-hoc projection onto a conserved-quantity manifold) is model-agnostic and portable.

**The KF-specific gap:** for incompressible KF in *vorticity* form, the relevant invariants are circulation and (inviscid-limit) enstrophy 𝒵 = ½‖ω‖²_{L²}. Hard enstrophy/circulation projection at inference for a single-shot vorticity operator was *not* found. The closest work enforces enstrophy conservation only as a soft loss-term regulariser within a group-equivariant CNN closure, or enforces divergence-free-ness via a streamfunction parametrization (learn ψ, set u=∇×ψ, div-free by construction) ([SciML closure review](https://ir.cwi.nl/pub/34879/34879.pdf)). So: Leray projection on velocity = documented, low-effort, applies now; enstrophy/circulation projection on vorticity = promising but untested.

**Cross-reference:** post-hoc projection (Q5) attacks the same physical-inconsistency that the soft physics losses (Q2) attack, but as a hard constraint at inference — strictly cheaper than retraining, and orthogonal to the time-axis fix (Q3), so they compose.

### Q6 — The wall: chaos-limited vs model/optimization-limited?

**Conclusion: Both, separated by a threshold near ~0.5·T_L. Below ~0.5 Lyapunov times the late-time floor is model/optimization-limited (spectral bias, temporal aliasing) and is reducible; beyond it, pointwise tracking is fundamentally chaos-limited and no deterministic operator can recover it — which is precisely why the field switched to statistical metrics.**

The fundamental-limit evidence is solid. Bae & Lozano-Durán show error in HIT grows as e^{λt} with λk/|ε| ≈ 0.4, and "its reciprocal is closely related to the predictability horizon" ([Bae 2022](http://www.its.caltech.edu/~jbae/publications/Bae_2022b.pdf)). Nastac & Ihme establish the Lyapunov exponent as a predictability metric and give the Reynolds scaling λ ~ Re^α with α≈0.5 (Kolmogorov) or ≈0.46 (multifractal) — **the horizon shrinks as Re grows** ([PRF 2017](http://web.stanford.edu/group/ihmegroup/cgi-bin/MatthiasIhme/wp-content/papercite-data/pdf/physrevfluids.2.094606.pdf)). For 2D specifically, Atif et al. measure max Lyapunov exponent Λ≈2.15 and Lyapunov time T_L≈0.45 convective units, and *deliberately restrict FNO predictions to t < T_L* ([Atif et al.](https://arxiv.org/html/2409.14660v1)).

The threshold itself comes from a 2D-turbulence prediction study: predictions for lead time up to 0.5·T_L are "almost 'complete' in that the small-scale motions … can be entirely recovered from the input data," whereas "the prediction at T_L or 2T_L is inevitably incomplete," because "the time required for the enstrophy to cascade to the smallest scale … is within 0.5·T_L" ([Cambridge JFM](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/prediction-and-control-of-twodimensional-decaying-turbulence-using-generative-adversarial-networks/EAE377A4E1F784D135520DB1EE3F25A0)). This is the cleanest statement of your wall: it is the enstrophy-cascade decorrelation time.

The model-limited component is equally documented: the high *floor* (even at small t) is spectral bias — FNOs "neglect components of the spatial frequency spectrum that have low amplitude" ([PDE-Refiner](https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf)), and the floor depends strongly on Re/ν (RFNO/FNO: 0.5–0.6% relL2 at ν=1e-3 vs 12–14% at ν=1e-5, [J. Simulation](https://www.featool.com/references/A%20visual%20digital%20twin%20framework%20based%20on%20residual-based%20fourier%20neural%20operator%20online%20simulation.pdf)). At Re~100–1000 over t<T_L, the floor is reducible (Spectral-Refiner reaches 10⁻⁴ on comparable weakly-turbulent cases); the optimization-limited gap dominates there. Past ~0.5·T_L, chaos dominates and pointwise relL2 is not a meaningful target — explaining the migration to spectrum/Wasserstein metrics.

**Cross-reference:** this is the unifying section — Q1's flat floor, Q2's spectral-bias-driven losses, Q3's aliasing, Q4's Lyapunov-bounded unrolling, and Q5's physics constraints all sit on the same chaos-vs-model axis defined here.

---

## Comparison Table: Methods vs the Single-Shot Late-Time relL2 Target

| Method (ref) | Mechanism | Applies to single-shot? | Metric reported | System / Re / horizon | Pointwise late-time relL2 effect |
|---|---|---|---|---|---|
| PINO ([2111.03794](https://arxiv.org/html/2111.03794v3)) | PDE residual loss over [0,T] + instance fine-tune | **Yes** | rel L2 (aggregate) | Kolmogorov Re500 (ν=1e-3); whole window | ~7% lower aggregate; **no time-slice breakdown** |
| Spectral-Refiner / ST-FNO3d ([2405.17211](https://arxiv.org/html/2405.17211v2)) | Post-hoc convex H⁻¹/L2 a-posteriori fine-tune + temporal padding | **Yes** | rel L2 (Bochner) + neg-Sobolev residual | 2D NSE Re O(10³–10⁴), McWilliams/TG; final-time | **1.31×10⁻² → ~3×10⁻⁴** (weakly-turbulent/TG only; uses auto-diff solver) |
| FNO3d time-as-channels vs Fourier ([2409.14660](https://arxiv.org/html/2409.14660v1)) | Treat time as channels, not periodic Fourier dim | **Yes** | rel L2 + vorticity correlation | 2D decaying turb Re 7–8k; t<T_L | Time-as-channels: "significantly lower errors" than 3D-FNO; **untested as t-slice relL2** |
| Leray/Helmholtz projection ([Negiar](https://openreview.net/pdf/7a772c3b05e62ae8956a63349267c06c8a3b56fa); [2603.24500](https://arxiv.org/html/2603.24500v1)) | Hard div-free projection (Fourier), inference or train | **Yes** (velocity form) | test L2 + divergence | 2D incompressible NS | "most often lower test L2"; div → 2×10⁻⁷; **not t-sliced** |
| Sobolev H¹ loss ([PDE-Refiner ablation](https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf)) | Derivative-weighted loss | Yes (loss) | correlation time | 1D KS | Modest (79.7s vs 75.4s); not relL2 |
| Spectral/frequency loss (LOGLO-FNO [link](https://openreview.net/pdf/343ad139b82699e6e171af675eb975f5c84274be.pdf)) | High-freq fRMSE penalty | Yes (loss) | **fRMSE (spectral)** | 2D KF Re5k; 1-/5-step | −58…−73% high-band fRMSE; not pointwise relL2 |
| OT / Sinkhorn (Jiang [2306.01187](https://arxiv.org/html/2306.01187v3)) | Match summary-stat distributions | No (autoregressive) | invariant measure | small chaotic systems; "unclear for KF" | Statistical only |
| DySLIM ([2402.04467](https://arxiv.org/html/2402.04467v1)) | MMD measure-matching reg | No (autoregressive) | MELR, covRMSE, Wass1, cosine | Lorenz/KS/Kolmogorov | Improves stats + short-term cosine; **no relL2 for KF** |
| MNO ([NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/6ad68277e27b42c60ac228c9859fc1a2-Paper-Conference.pdf)) | Dissipativity hard term, one-step | No (autoregressive) | Fourier/POD spectrum, TKE, PDF | Kolmogorov Re≤5000 | Statistical only |
| PDE-Refiner ([2308.05732](https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf)) | Diffusion-style multi-step refine | No (autoregressive + **probabilistic, excluded**) | correlation time | 1D KS, 2D KF | KF corr-time 6.45(FNO)/9.66(UNet)/10.66(Refiner) |
| Unrolled/differentiable (List [2402.12971](https://arxiv.org/html/2402.12971v1)) | Backprop through K rollout steps | No (autoregressive); analogue = chain K windows | L2 over rollout | 2D KF; horizon ≤ T_L | Stability gain, Lyapunov-bounded; **not single-shot t-slice relL2** |
| DPOTc conserved-quantity ([2601.22541](https://arxiv.org/html/2601.22541v1)) | Post-hoc conserved-quantity projection | Mechanism portable; tested autoregressive | rel L2 + correlation | **compressible** NS Mach 0.1 | 0.4%/step; 7.8% vs 8.3% @10 steps |
| Scheduled Temporal Loss Weighting ([NeurIPS ML4PS 2025](https://neurips.cc/virtual/2025/123050)) | Curriculum: upweight **early** t | No (autoregressive) | nRMSE rollout | 1D Adv/Burgers/KdV | 16–62% nRMSE; opposite of late-upweight |
| SGNO ([2602.18801](https://arxiv.org/html/2602.18801v1)) | Lipschitz/contraction spectral bound | No (autoregressive) | nRMSE (APEBench) | Kolm-2D | 0.689→0.586 GMean100 |

**Narrative reading of the table.** The single most important pattern is the *applies-to-single-shot* column: nearly every well-cited "long-horizon" lever (MNO, DySLIM, OT, PDE-Refiner, unrolled training, SGNO, DPOTc, scheduled weighting) is an **autoregressive-rollout** mechanism whose stated benefit is stability, correlation-horizon, or invariant-measure fidelity — *not* single-shot pointwise late-time relL2. The handful that genuinely apply to a one-pass operator (PINO, Spectral-Refiner, time-as-channels, Leray projection) either report only aggregate relL2 (PINO), demonstrate the reduction only on weakly-turbulent/TG cases (Spectral-Refiner), or report a different metric (fRMSE, divergence). The second pattern is the **metric column**: it is overwhelmingly spectral/statistical/correlation, confirming the migration. The most surprising entry is Scheduled Temporal Loss Weighting — the one method literally named "temporal loss weighting" upweights *early* time, the opposite of the intuitive late-time-upweighting lever, and that lever is untested in any single-shot KF operator.

---

## Contradictions & Debates

- **Does Spectral-Refiner contradict the "no clean reduction" claim?** Partially. It *does* lower single-shot pointwise relL2 by ~2 orders ([Cao et al.](https://www.osti.gov/servlets/purl/2563339)), seemingly contradicting the negative claim. **Resolution:** the contradiction dissolves on regime — its strong reductions are on weakly-turbulent/Taylor-Green cases (not chaotic forced KF over long T), it reports final-time not a t/T>0.5 sweep, and it uses an auto-diff solver at inference. The paper's *own* text ("no NO-only approach breaks 2-digit accuracy") asserts the negative claim for the pure data-driven case. Evidence weight: the negative claim holds for *pure deterministic data-driven single-shot in the chaotic regime*; Spectral-Refiner is the equation-aware exception at low effective Re.

- **FNO vs U-Net on 2D long horizon.** PDE-Refiner finds "modern U-Nets outperform FNOs on the 2D domain for long rollouts" (KF corr-time U-Net 9.66 vs FNO 6.45) ([Lippe et al.](https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf)), whereas the original FNO work and others report FNO3d best "when there is sufficient data." **Resolution:** architecture ranking is metric- and data-regime-dependent — FNO's spectral truncation hurts the high-frequency content that dominates long-horizon/late-time correlation, while U-Net's local convolutions retain it; with abundant data and short windows FNO3d wins. For your single-shot late-time target, this favors either a U-Net hybrid or an un-truncated/refined FNO (Spectral-Refiner removes truncation in the fine-tuned layer).

- **Does the projection layer always help relL2?** Negiar et al. say "most often" lower test L2; not universally. **Resolution:** projection enforces a constraint absent from the data; when the operator already nearly satisfies it, gains are marginal; when it violates it, gains are real and free. Low-risk, low-upside-floor.

---

## Uncertainties & Gaps

- ⚠️ **Genuine field-level gap (high confidence it is real):** no paper reports a t/T>0.5 *pointwise relL2 time-slice breakdown* with a targeted lever in the *chaotic* KF regime for a single-shot operator. Confirmed across 7 rounds and 48 sources including beyond-seed discovery searches.
- ⚠️ **Enstrophy/circulation hard-projection at inference** for incompressible KF in *vorticity* form is under-explored; only soft enstrophy regularisation and streamfunction parametrization were found. The Leray (velocity-divergence) result is solid; the vorticity-invariant analogue is an untested niche.
- ⚠️ **Differentiable chaining of K single-shot windows** on KF with a reported late-time relL2 effect was not found — plausible by analogy to List et al., but untested in your exact setting; expect Lyapunov-bounded horizon.
- ⚠️ **Spectral-Refiner at moderate-high Re over long chaotic T**: its 10⁻⁴ results are on weakly-turbulent/TG; behavior at t/T>0.5 in strongly-chaotic KF is unverified.
- ⚠️ Several adjacent strong results (adv-NO 15× spectrum-error reduction; WDNO; diffusion refiners) are **generative/probabilistic and excluded** by your scope.

---

## Recommendations — 3 most promising, low-effort, single-shot-compatible levers (ranked)

These are evidence-backed *directions* to test on your single-shot FNO3d KF baseline; ranked by effort-to-evidence ratio. No implementation given.

### 1. Fix the temporal axis: time-as-channels / periodic-temporal-padding instead of raw temporal Fourier dimension
**Why first:** lowest effort (architectural, no extra loss, no solver), and it attacks a *documented defect specific to single-shot FNO3d*. Atif et al. show time-as-channels gives "significantly lower errors" than 3D-FNO because the time axis is non-periodic and the temporal FFT aliases ([2409.14660](https://arxiv.org/html/2409.14660v1)); Spectral-Refiner's periodic temporal padding + layernorm (replacing the hard-coded global temporal normalizer) is precisely the fix that enabled its accuracy ([Cao et al.](https://www.osti.gov/servlets/purl/2563339)). Test: compare temporal-Fourier vs temporal-channels vs periodic-temporal-padding, reporting relL2 per time slice. This is the single lever most likely to lower the *late slices* specifically, because the aliasing error compounds toward the window edges.

### 2. Negative-Sobolev (frequency-weighted) loss or post-hoc spectral refinement
**Why second:** the only documented single-shot pointwise-relL2 reduction comes from Spectral-Refiner's H⁻¹ a-posteriori loss, which is "an inverse frequency weight emphasizing low-frequency information," taking relL2 from 1.31×10⁻² to ~3×10⁻⁴ ([Cao et al.](https://www.osti.gov/servlets/purl/2563339)). A pure-data variant — a frequency-weighted training loss (negative-Sobolev, or the LOGLO high-band fRMSE penalty that cut high-band error 58–73% on KF Re5k, [LOGLO-FNO](https://openreview.net/pdf/343ad139b82699e6e171af675eb975f5c84274be.pdf)) — directly counters the spectral bias that PDE-Refiner identifies as the root cause of late-horizon decay. Effort is moderate (loss change) to higher (post-hoc refinement needs an auto-diff solver, but you already have a Lightning/PyTorch PDE stack). Expect the gain to be largest at low ν / t<T_L (model-limited regime).

### 3. Post-hoc hard physics projection at inference (Leray on velocity; enstrophy/circulation on vorticity as a research bet)
**Why third:** near-zero training cost, applies to the single-shot output as a final layer. Leray/Helmholtz projection "most often lead[s] to lower test L2" on incompressible NS and drives divergence to ~10⁻⁷ at small cost ([Negiar et al.](https://openreview.net/pdf/7a772c3b05e62ae8956a63349267c06c8a3b56fa); [2603.24500](https://arxiv.org/html/2603.24500v1)); DPOTc shows conserved-quantity projection slows relL2 growth to 0.4%/step (compressible, [2601.22541](https://arxiv.org/html/2601.22541v1)). For your KF vorticity form, the velocity-divergence projection requires converting ω→u→project→ω; the higher-payoff, *untested* bet is a circulation/enstrophy projection at each output time slice — flagged as a genuine open niche, so it doubles as a potential contribution rather than just an engineering fix. Rank third only because the documented relL2 gains are modest and the vorticity-form analogue is unproven.

**What NOT to spend effort on (for this specific target):** autoregressive-rollout stabilizers — pushforward (shown to *not* help, [Lippe](https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf)), curriculum/measure-matching (DySLIM/MNO/Jiang report statistics, not single-shot relL2, and don't apply to a one-pass operator), and chasing pointwise relL2 *past ~0.5·T_L* (chaos-limited; the field abandoned it for good reason — beyond the enstrophy-cascade decorrelation time, switch your evaluation to spectrum/vorticity-correlation/Wasserstein rather than fighting an information-theoretic wall).

---

## Methodology

- **Depth:** exhaustive. **Search rounds:** 7 (Tavily advanced), ~26 distinct queries including 6 mandated open-ended discovery searches with no seed arXiv IDs (to make the negative claim credible beyond the seed list).
- **Sources:** 48 unique, overwhelmingly Tier 1 (arXiv, NeurIPS/ICLR/ICML proceedings, JFM, Nature Comm, PRF). Full-text `web_extract` performed on the five most central papers (PINO, Spectral-Refiner, DySLIM, Atif et al. 2D-turbulence FNO, List et al. unrolled-training) to confirm the *exact metric reported* — central to the deliverable.
- **Multi-hop chains used:** (a) conceptual deepening PINO→FNO3d temporal handling→Spectral-Refiner temporal padding→aliasing root cause; (b) causal chain spectral bias→high error floor→metric migration→invariant-measure losses; (c) entity expansion pushforward→Brandstetter→List et al. unrolled taxonomy→Lyapunov-bounded horizon→chaos wall; (d) temporal progression PINO (2021)→MNO (2022)→Jiang/DySLIM/PDE-Refiner (2023–24)→Spectral-Refiner/APEBench (2024–25)→DPOTc/Project-and-Generate/DRIFT-NET (2026).
- **Sub-questions:** 6 defined, 6 answered, 0 unanswered. Final per-question confidence 83–90%.
- **Final confidence 87%** (matches final checkpoint average). The cap reflects that the core deliverable is a *negative existence claim* — confidence in "nobody has cleanly done X" is structurally bounded even after exhaustive search; the residual 13% is the irreducible possibility of an obscure/very-recent paper outside indexed search.
- **Key challenge:** distinguishing autoregressive-rollout results (the bulk of the "long-horizon" literature) from genuine single-shot space-time results — resolved by reading paper bodies for the operator definition (𝒮: uₜ→uₜ₊₁ vs 𝒢: a→u(·,[0,T])) rather than abstracts.

## Sources

| # | Title | URL | Date | Credibility |
|---|-------|-----|------|:-----------:|
| 1 | PINO: Physics-Informed Neural Operator | https://arxiv.org/html/2111.03794v3 | 2021 | ⭐ Tier 1 |
| 2 | Spectral-Refiner / ST-FNO3d | https://arxiv.org/html/2405.17211v2 | 2024 | ⭐ Tier 1 |
| 3 | Spectral-Refiner (OSTI/ICLR camera-ready w/ tables) | https://www.osti.gov/servlets/purl/2563339 | 2025 | ⭐ Tier 1 |
| 4 | FNO for spatiotemporal 2D turbulence (Atif et al.) | https://arxiv.org/html/2409.14660v1 | 2024 | ⭐ Tier 1 |
| 5 | MNO: Learning Dissipative Dynamics in Chaotic Systems | https://proceedings.neurips.cc/paper_files/paper/2022/file/6ad68277e27b42c60ac228c9859fc1a2-Paper-Conference.pdf | 2022 | ⭐ Tier 1 |
| 6 | DySLIM (Schiff et al.) | https://arxiv.org/html/2402.04467v1 | 2024 | ⭐ Tier 1 |
| 7 | Training NOs to preserve invariant measures (Jiang et al.) | https://arxiv.org/html/2306.01187v3 | 2023 | ⭐ Tier 1 |
| 8 | PDE-Refiner (NeurIPS 2023) | https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf | 2023 | ⭐ Tier 1 |
| 9 | DPOTc: Conserved-Quantity Correction rollouts | https://arxiv.org/html/2601.22541v1 | 2026 | ⭐ Tier 1 |
| 10 | Differentiability in Unrolled Training (List et al.) | https://arxiv.org/html/2402.12971v1 | 2024 | ⭐ Tier 1 |
| 11 | Towards Enforcing Hard Physics Constraints (Leray) | https://openreview.net/pdf/7a772c3b05e62ae8956a63349267c06c8a3b56fa | 2024 | ⭐ Tier 1 |
| 12 | Project and Generate: Divergence-Free NOs | https://arxiv.org/html/2603.24500v1 | 2026 | ⭐ Tier 1 |
| 13 | Bae & Lozano-Durán: LES Lyapunov / numerical-modeling error | http://www.its.caltech.edu/~jbae/publications/Bae_2022b.pdf | 2022 | ⭐ Tier 1 |
| 14 | Nastac & Ihme: Lyapunov exponent predictability metric (PRF) | http://web.stanford.edu/group/ihmegroup/cgi-bin/MatthiasIhme/wp-content/papercite-data/pdf/physrevfluids.2.094606.pdf | 2017 | ⭐ Tier 1 |
| 15 | Prediction/control of 2D decaying turbulence (GAN, JFM) | https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/prediction-and-control-of-twodimensional-decaying-turbulence-using-generative-adversarial-networks/EAE377A4E1F784D135520DB1EE3F25A0 | 2023 | ⭐ Tier 1 |
| 16 | LOGLO-FNO (frequency loss, KF Re5k) | https://openreview.net/pdf/343ad139b82699e6e171af675eb975f5c84274be.pdf | 2025 | ⭐ Tier 1 |
| 17 | Learning turbulent flows w/ generative models (Nature Comm) | https://www.nature.com/articles/s41467-026-70145-4 | 2026 | ⭐ Tier 1 |
| 18 | SGNO: Spectral Generator NO (stability bound) | https://arxiv.org/html/2602.18801v1 | 2026 | ⭐ Tier 1 |
| 19 | APEBench (Köhler et al., NeurIPS 2024) | https://tum-pbs.github.io/apebench-paper | 2024 | ⭐ Tier 1 |
| 20 | Scheduled Temporal Loss Weighting (NeurIPS ML4PS 2025) | https://neurips.cc/virtual/2025/123050 | 2025 | 🔵 Tier 2 |
| 21 | Temporal Neural Operator (TNO, Sci Rep) | https://www.nature.com/articles/s41598-025-16922-5 | 2025 | ⭐ Tier 1 |
| 22 | Operator Learning with Neural Fields (CORAL, NeurIPS) | https://proceedings.neurips.cc/paper_files/paper/2023/file/df54302388bbc145aacaa1a54a4a5933-Paper-Conference.pdf | 2023 | ⭐ Tier 1 |
| 23 | DRIFT-NET (ApeBench Kolmogorov error-vs-time) | https://openreview.net/pdf/2fa600f97ae43b3eb5eb888d921545ddc45e0713.pdf | 2026 | 🔵 Tier 2 |
| 24 | A Training-Free Approach for Stable Neural PDE (AAAI) | https://ojs.aaai.org/index.php/AAAI/article/download/39360/43321 | 2026 | ⭐ Tier 1 |
| 25 | JAWS: Jacobian Regularization long rollout | https://arxiv.org/html/2603.05538v1 | 2026 | 🔵 Tier 2 |
| 26 | Autoregression-Free Neural Operators (AFNO) | https://arxiv.org/html/2605.25413v3 | 2026 | 🔵 Tier 2 |
| 27 | ANCHOR: adaptive numerical correction | https://arxiv.org/html/2512.19643v2 | 2025 | 🔵 Tier 2 |
| 28 | Mitigating Spectral Bias via High-Frequency Scaling | https://arxiv.org/html/2503.13695v1 | 2025 | 🔵 Tier 2 |
| 29 | PENO: Physics-enhanced NO (spectral bias, self-augment) | https://arxiv.org/pdf/2406.04367 | 2024 | 🔵 Tier 2 |
| 30 | Energy-conserving NN for turbulence closure | https://ir.cwi.nl/pub/34121/34121.pdf | 2024 | ⭐ Tier 1 |
| 31 | SciML closure review (hard vs soft constraints) | https://ir.cwi.nl/pub/34879/34879.pdf | 2024 | ⭐ Tier 1 |
| 32 | MEEVC: mass/energy/enstrophy/vorticity conserving solver | https://pure.tue.nl/ws/files/27756799/1604.00257v1.pdf | 2016 | ⭐ Tier 1 |
| 33 | Geometric NOs via Lie-group latent dynamics (MCL, H1) | https://arxiv.org/html/2602.16209v1 | 2026 | 🔵 Tier 2 |
| 34 | Residual-based FNO online simulation (RFNO, Re scaling) | https://www.featool.com/references/A%20visual%20digital%20twin%20framework%20based%20on%20residual-based%20fourier%20neural%20operator%20online%20simulation.pdf | 2024 | 🔵 Tier 2 |
| 35 | Two-dimensional turbulence (Boffetta & Ecke, ARFM) | https://courses.physics.ucsd.edu/2019/Winter/physics116_216/annurev-fluid-120710-101240.pdf | 2012 | ⭐ Tier 1 |
| 36 | Dynamics of freely decaying 2D turbulence (Brachet et al., JFM) | https://www.phys.ens.psl.eu/~brachet/files/Publications_&_Reprints_files/JFM-1988-2D.pdf | 1988 | ⭐ Tier 1 |
| 37 | On universal approximation & error bounds for FNOs (Leray) | https://jmlr.org/papers/volume22/21-0806/21-0806.pdf | 2021 | ⭐ Tier 1 |
| 38 | Lyapunov exp. metric for LES (Cambridge repository) | https://www.repository.cam.ac.uk/items/cb9170a7-c16e-4148-a40a-6af79684eb29 | 2017 | ⭐ Tier 1 |
| 39 | Reducing Numerical Errors with Neural Operators (PBDL) | https://physicsbaseddeeplearning.org/diffphys-code-sol.html | 2024 | 🔵 Tier 2 |
| 40 | MNO repository (Kolmogorov Re40/500/5000 datasets) | https://github.com/neuraloperator/markov_neural_operator | 2022 | 🔵 Tier 2 |
| 41 | PINO physics_informed repo (Kolmogorov Re500 data) | https://github.com/neuraloperator/physics_informed | 2021 | 🔵 Tier 2 |
| 42 | Stable Fine-Time-Step Long-Horizon Turbulence (Ms-MoE) | https://arxiv.org/html/2604.12794v1 | 2026 | 🔵 Tier 2 |
| 43 | Mitigating Boundary Artefacts in FNOs (temporal padding) | https://lup.lub.lu.se/student-papers/record/9213930/file/9213931.pdf | 2024 | 🟡 Tier 3 |
| 44 | Late Fusion NOs (strict one-shot extrapolation) | https://arxiv.org/html/2604.16721v1 | 2026 | 🔵 Tier 2 |
| 45 | Lagrangian operator inference + structure-preserving ML | https://kramer.ucsd.edu/img/pubs/Lagrangian-operator-inference-enhanced-with-Structure-preserving-ML-2024.pdf | 2024 | ⭐ Tier 1 |
| 46 | Gen4Turb project page (adv-NO, spectrum error) | https://vivekoommen.github.io/Gen4Turb | 2026 | 🔵 Tier 2 |
| 47 | Lyapunov exponents in 2D turbulence (Phys. Fluids) | https://lsec.cc.ac.cn/~lyuan/2023pof.pdf | 2023 | ⭐ Tier 1 |
| 48 | Kolmogorov Flow: Seven Decades of History (Re~10³) | https://www.scirp.org/journal/paperinformation?paperid=88548 | 2018 | 🟡 Tier 3 |
