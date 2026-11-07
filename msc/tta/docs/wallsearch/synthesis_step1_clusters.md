# Wall-Search Synthesis — Step 1: Clustering

Date: 2026-06-25
Sources: `1_research_late-time...` (exhaustive, 48 src, negative-claim), `2_gemini_research` (optimistic body + self-auditing verification appendix), `3_gpt` (full-PDF, conservative), `4_claude-research` (tied to local banked experiments).

Doc-prefix = origin. Per idea: main idea / problem solved / how (≤3 sentences).

**Tag semantics (revised — AR-training is in scope, AR-inference is not).** The project constraint is *one-shot inference*, not one-shot training. Multi-step / autoregressive *training* tricks are acceptable as long as the deployed operator emits its window in one pass. So the old `[AR-only]` tag is split:
- `[SS-ok]` = native single-shot, applies directly.
- `[AR-train]` = autoregressive/multi-step *training* mechanism whose deployed operator stays one-shot (or whose value proposition is exactly "train coupled, deploy one-shot"). **In scope.** Cost lands on training resources, not inference.
- `[AR-infer]` = autoregressive at *inference* (learns a one-step map rolled out closed-loop). **Needs re-architecture** to preserve one-shot inference; may still donate a portable mechanism.
- `[SS-analogue]` = proposed but untested single-shot/window-chained adaptation of an AR method.

(The legend below still shows the original `[AR-only]` strings inside cluster entries; the Addendum at the end gives the corrected per-entry reclassification rather than re-editing every line.)

**Partition rule (to remove cluster overlap):** training-time objective → Cluster A (loss). Inference-time correction → Cluster D (post-hoc). Structural change to the network → Cluster E (architecture). H⁻¹/negative-Sobolev spans A and D and is listed in both with the distinction noted.

---

## Cluster 1 — Information-theory description of the wall

The user's framing: Lyapunov limit; field non-identifiability → migration from L2 to statistical/manifold metrics.

- **Lyapunov / enstrophy-cascade horizon** [1_] — Pointwise tracking is fundamentally chaos-limited beyond ~0.5·T_L. The error grows as e^{λt} with λ~Re^½, so the predictability horizon *shrinks* as Re rises; the wall coincides with the enstrophy-cascade decorrelation time (small scales fully cascade within 0.5·T_L). Below 0.5·T_L the late floor is model-limited and reducible, above it no deterministic operator can recover phase. [SS-ok diagnosis]

- **Migration to invariant-measure / statistical metrics** [1_,2_,3_] — Because exact trajectories are unpredictable under positive Lyapunov exponents, MNO and DySLIM declared pointwise L2 the wrong target and report spectra/MELR/Wasserstein/correlation-time instead. The problem they solve is "honest evaluation of a chaotic attractor"; the method is to measure distribution consistency on the manifold, not pointwise distance. A 2026 Nature Comm paper makes the deprecation explicit (L2 overweighs high-energy low-k modes, rewards smoothing). [AR-only / statistical]

- **Spectral bias as the shared root cause of the floor** [1_,2_,4_] — MSE optimises high-energy low-k structure and systematically neglects low-amplitude high-k modes, producing an oversmoothed high error floor even at small t. This is the mechanism linking the loss, time-axis and wall findings into one. Note 4_ reports the related Qin-et-al "1.2% of coefficients hold 99% energy" parameterisation-bias claim was adversarially **refuted** — so spectral bias is real but the FNO-specific parameterisation story is contested. [mechanism]

- **Local banked counter-evidence: the wall is model-limited, not chaos-limited (at low Re)** [4_] — At Re=100 the chaos-spread gate measured perturbation growth of only ~1.3× per horizon (λ~0.2–0.3, solver-from-exact-IC control = 0), and an oracle-restart from GT mid-window cut late relL2 from 0.54 → 0.230 (~57%). This proves the information to predict late-t exists in the data and the operator simply fails to exploit it; the 0.54→0.230 gap is model-limited, the residual 0.230→0 may be chaos-limited. (4_ is internally inconsistent on whether this gate is Re=100 or Re=500 — treat the *number* as load-bearing, the Re as soft.) [SS-ok, decisive]

- **Aggregate metric ≠ phase tracking** [3_] — SGNO's "stable step" threshold is reached at step ~13–14 even though its GMean100 aggregate beats FNO. Better aggregate long-horizon error does not mean accurate pointwise phase tracking across the full rollout — a caution for interpreting any single summary number. [interpretive]

**Cross-doc tension:** 1_ frames the wall as substantially fundamental beyond 0.5·T_L; 2_ argues it is breakable; 4_'s direct experiment says model-limited at low Re. The reconciliation is Re/horizon dependence — exactly the 0.5·T_L threshold 1_ draws.

---

## Cluster 2 — Time-axis treatment

The user's framing: time-as-channel vs Fourier; latent-frequency temporal update; binned spectral power.

- **Time-as-channels instead of temporal Fourier dimension** [1_,4_] — FNO3d runs an FFT over the non-periodic time axis, which assumes periodicity the trajectory does not have and injects aliasing that worsens toward the window edges (late t). Replacing the temporal Fourier dim with time stacked as channels (FNO2d) gives "significantly lower errors" and 7–17× faster training on 2D decaying turbulence. Caveat: demonstrated at Re~7–8k decaying, untested on forced KF at Re 100–1000, but the periodicity argument is system-independent. [SS-ok]

- **Periodic temporal padding + layernorm (Spectral-Refiner fix)** [1_] — Same aliasing defect, different remedy: pad the temporal dimension periodically in the lifting operator and replace the hard-coded global normaliser with layernorm. This is the architectural change that enabled Spectral-Refiner's accuracy; it keeps the temporal Fourier treatment but removes the wrap-around artefact. [SS-ok]

- **Latent-frequency temporal update — SSP (Structured Spectral Propagator)** [2_] — Separates spatial basis from temporal state and evolves only a compact, bounded set of latent Fourier modes through a frequency-conditioned linear backbone + nonlinear spectral closure. Solves "spatial-reconstruction error contaminating temporal evolution" by isolating the recurrent dynamics; reduces relL2 up to 48.9% under long extrapolation. Caveat (2_'s own appendix): tested on Shallow-Water/Reaction-Diffusion/PDEBench-NS, **not KF**, and it is [AR-only].

- **Latent-frequency temporal update — SGNO (ETD generator)** [1_,2_,3_] — Models the one-step map as an Exponential-Time-Differencing step in Fourier space with a diagonal generator constrained to non-positive real part ℜ(α)≤0, so the linear dynamics can never amplify. Solves autoregressive blow-up at the root. Caveat (2_/3_ appendices): 92.9% headline is 1D dispersion; on the actual kolm2d task absolute error stays ~56% (GMean100≈0.558 per 2_/3_; 1_ cites 0.586 — a cross-doc cell discrepancy), and it is [AR-only].

- **Binned Spectral Power (BSP) loss** [2_] — Groups *spatial* frequencies into scalar bins and penalises per-band energy deviation, adaptively reweighting large/small scales to fight spectral bias. **Direct answer to the user's uncertainty: BSP is a spatial-frequency loss, not time-dependent** — it optimises magnitudes not phases, so it improves spectral consistency and stability but does not directly minimise pointwise distance. Belongs conceptually in Cluster A (loss); listed here only because the user raised it. [loss, not time-axis]

- **Static temporal upweighting w(t)∝t^α (single-shot analogue of TWL)** [4_] — In a single-shot loss the late-t terms are diluted T-fold, so gradient pressure is weakest exactly where error is largest. Multiply per-timestep loss by (t/T)^α to force the optimiser onto late slices. Untested anywhere; mechanistically targeted at the symptom but 4_ notes the motivating "uniform weighting degrades late-t" premise was refuted 0-3, so the effect may be neutral. [SS-analogue]

- **Scheduled / causal temporal loss weighting (the named methods)** [1_,4_] — The papers literally called "temporal loss weighting" do the *opposite* of late-upweighting: a curriculum that prioritises *early* steps, and a causal PINN scheme weighting points where prior residuals are already minimised. Both require sequential ordering, so they have no native single-shot analogue. [AR-only]

---

## Cluster 3 — Quasi-autoregressive / unrolled training

The user's framing: pushforward (training trick that disables autograd, theoretically motivates pushing the wall back); multi-step penalty (gradient-accumulating variant); differentiable chaining as the natural extension of the local GT-chaining experiment.

- **Pushforward trick** [1_,4_] — Feed the model its own (stop-gradient) one-step prediction as input to expose it to its own error distribution during training. Solves exposure bias, but the gradient is *cut* at the handoff so the model never learns to make corrections useful downstream. Reported to **not help** in PDE-Refiner's and DySLIM's settings; 4_'s stop-grad oracle-distill analogue tested **negative** locally. [AR-only / SS stop-grad analogue = dead]

- **Differentiable multi-step unrolling (WIG vs NOG vs ONE)** [1_,3_] — Backprop through K rollout steps so the model sees and corrects compounding error with full temporal gradients. Solves error accumulation; List et al. on 2D KF (Re 300–900, K=[1,2,4], 250 steps) get ~21% lower long-rollout error, but the benefit is Lyapunov-bounded ("unrolling too far → diminishing/negative returns"). Caveat (3_): for *pure predictors* WIG≈NOG (stop-grad slightly better); full gradients help mainly in hybrid correction. [AR-only]

- **Multi-Step Penalty (MP-FNO)** [2_,3_] — The user's "multi-step penalty optimization": split the long trajectory into windows, detach the graph every r rollouts, and insert learnable discontinuities δ_s minimised by a penalty term. Solves exploding/vanishing gradients over long chaotic trajectories by cutting the graph at strategic points (linear not cubic cost) while still enforcing long-term coherence. Caveat (3_): KF Re=1e5 results are spectrum/correlation plots only — no relL2, no published r/segment count. [AR-only]

- **Recurrent Neural Operator (RNO) recurrent training** [2_] — Train the operator by recursively applying it to its own predictions within [0,T], eliminating exposure bias and (claim) converting worst-case error growth from exponential O(e^{CT}) to linear O(T). Headline: Allen-Cahn relL2 at step 50 cut 89.6%. **Brand-honest caveat from 2_'s own appendix:** the 89.6% is Allen-Cahn not KF, the NS benchmark is generic 2D vorticity (no Re, no forcing), and recurrent training **fails on FNO** (inconsistent r-FNO vs tf-FNO) — works only on MgNO. [AR-only]

- **Solver-in-loop K=32 unrolling (Kochkov)** [3_] — Train a coarse learned solver with cumulative pointwise MSE over 32 unrolled steps inside a differentiable solver. Improves long-trajectory vorticity correlation at KF Re=1000. Caveat: no controlled one-step-vs-K relL2 table; metric is correlation not relL2. [AR-only, solver-hybrid]

- **Differentiable 2-window chaining (single-shot adaptation)** [1_,4_] — The theoretically-motivated extension of the local GT-chaining experiment: split T into two windows, feed window-1 prediction (no stop-grad) as IC for window-2, backprop through the handoff; inference stays single-shot. Solves the "no gradient path through intermediate predictions" failure that 4_ diagnoses as the wall's cause; the oracle-restart result (0.54→0.230) is the evidence this structure exists in the data. Untested; 1_ flags the same "differentiable chaining of K windows" as plausible-but-unverified, Lyapunov-bounded per window. [SS-analogue, highest-novelty]

---

## Cluster 4 — Post-hoc / inference-time corrections

The user's framing: PDE-Refiner; negative-Sobolev (solver?); divergence-free projection extending to conserved-quantity correction (DPOTc); streamfunction.

- **Disambiguation the user asked for** — *PDE-Refiner* [1_,3_] is a **diffusion-style, probabilistic** iterative refiner (out of the deterministic single-shot scope; it reports KF correlation-time, not relL2). It is **not** the negative-Sobolev method. The negative-Sobolev method is *Spectral-Refiner* (next item), and **yes it needs an auto-differentiable solver** to evaluate the residual.

- **Negative-Sobolev / H⁻¹ a-posteriori fine-tune (Spectral-Refiner)** [1_,3_,4_] — Fine-tune (often only the last projection layer) with the regularised H⁻¹ residual norm Σ_k (a+|k|²)⁻¹|R̂(k)|², an inverse-frequency weight emphasising low-k correction. Solves the spectral-bias floor at the energy-containing scales; drops final-time L2 ~10⁻²→~3×10⁻⁴ on McWilliams isotropic turbulence and ~12× on an OOD Re=5000 case (0.294→0.025) [1_,3_]. Caveats: **no KF result exists** — 1_/3_ report only McWilliams/TG/OOD; the "~41× on KF turbulence" figure traces solely to 4_ and is internally contradicted there ("NOT Kolmogorov flow"), so it is dropped as unverified. Needs an auto-diff solver at inference; demonstrated on weakly-turbulent/TG not chaotic KF; "final-time" may not equal late-*window* average (no per-timestep curve). Also valid as a **training** loss → see Cluster A. [SS-ok, solver-hybrid]

- **Leray / Helmholtz-Hodge divergence-free projection** [1_,4_] — Project the operator's velocity output onto the divergence-free subspace in Fourier space, enforcing div(u)=0 as a hard inference constraint at near-zero cost. "Most often" lowers test L2 and drives divergence to ~10⁻⁷. **Load-bearing caveat (4_):** their model is *vorticity-form*, where incompressibility is already implicit — Leray is only applicable if the model predicts velocity directly, or as a consistency check on recovered velocity. 4_ also notes the strongest quantitative claims of the 2603.24500 source were adversarially refuted. [SS-ok only for velocity form]

- **Conserved-quantity correction (DPOTc)** [1_,2_] — Apply per-step magnitude scaling (density/energy) and shift (momentum) to outputs to enforce conservation laws without touching the loss. Solves energy injection/dissipation drift; on compressible NS (Mach 0.1) slows relL2 growth to 0.4%/step (10-step 8.4%→7.8%). Caveats: compressible NS, [AR-only]; the incompressible-KF transfer (circulation ∮ω dA) is the documents' **own speculation**, not in the paper — and is flagged by 1_ as a genuine open niche (enstrophy/circulation hard-projection on vorticity form is untested). [SS-portable mechanism, untested on KF]

- **Newton-step residual correction** [4_] — One Newton step in solution space δu = −[∂_u R]⁻¹R(u_FNO) using the linearised NS residual; needs an estimate of ν. Single deterministic, label-free correction. 4_ notes its own banked residual is non-identifying for late-wall error, so this would not selectively target late-t. [SS-ok, but mistargeted]

---

## Cluster 5 — Architectural

The user's framing: streamfunction parametrization (placement uncertain); pushforward sits adjacent.

- **Streamfunction parametrisation** [1_,4_] — Predict the streamfunction ψ and set u=∇×ψ (or work in vorticity), making the field divergence-free *by construction* rather than by post-hoc projection. Solves incompressibility as a hard architectural guarantee. For the user's vorticity-form FNO this is partly already implicit — the open value is whether an explicit ψ-parametrisation improves the recovered velocity. [SS-ok, architectural]

- **Non-expansive spectral generator — SGNO** [1_,2_,3_] — (Also in Cluster 2 as a temporal mechanism.) The ℜ(α)≤0 constraint on the diagonal spectral generator is an *architectural* inductive bias guaranteeing the linear carry is non-expansive |exp(λ)|≤1. Listed here as the structural-constraint reading of the same method. [AR-only]

- **Spatial-basis / temporal-state separation — SSP** [2_] — (Also Cluster 2.) The analysis→propagation→synthesis factorisation is an architectural decoupling of spatial reconstruction from temporal evolution. [AR-only]

- **Local + global hybrid kernels** [3_,4_] — Augment FNO with learned local differential/integral kernels (DINO, Liu-Schiaffini) to restore high-frequency/local structure the global spectral convolution misses. 3_: +local kernels cut 5-step KF relL2 17% (Re=5000); DINO reports step-99 KF relL2 0.59 vs FNO 3.13. Caveat: short-horizon (5-step) or single-seed/under-specified (DINO); the Qin-et-al parameterisation-bias rationale was **refuted 0-3** in 4_. [AR-only mostly]

- **Geometric-forgetting / latent-drift architectures** [2_] — Standard residual updates in Euclidean latent space lose access to spatial geometry with depth, causing latent drift and physical-consistency violation (2_'s framing motivating RNO/SGNO/SSP). Mechanism claim only; no isolated pointwise-relL2 evidence. [mechanism]

---

## Cluster A (NEW) — Loss-objective / spectral-bias family

This family was split across the user's clusters (negative-Sobolev under 4, BSP under 2, the rest implicit). Partition rule isolates it: **training-time objectives that reshape the error spectrum.** It is the single most cross-cutting omission.

- **PDE-residual physics loss (PINO)** [1_,4_] — Append λ·‖∂_t u − R(u)‖² over the whole [0,T] window, evaluable at a finer grid than the data (256²×65 vs 64²×33) to enforce physics at unresolved scales. The user's own baseline and the one **confirmed single-shot** lever: 1_ reports ~7% lower aggregate; the specific 8.28%→6.04% aggregate at Re=500 figure is from 4_'s banked table. Critical open ablation (4_): nobody reports whether this 2.2pp gain concentrates at late t/T>0.5 or is uniform — the most important missing experiment. [SS-ok, confirmed]

- **Positive Sobolev H^k loss** [1_,3_,4_] — Penalise errors in spatial derivatives (a high-pass weight) to force capture of sharp gradients and the energy cascade. 3_: on MNO KF at Re=40, H²-loss cuts one-step relL2 0.0187→0.0143 (~24%). Caveats: gain is one-step/short, needs finer resolution to avoid derivative aliasing, and 4_ reports it *shortens* valid rollout on KS — weak/negative late lever. [SS-ok loss, weak]

- **Negative-Sobolev / H⁻¹ as a training loss** [1_,4_] — Same inverse-frequency weight as the Spectral-Refiner fine-tune (Cluster D), but applied during training rather than post-hoc. Architecture-agnostic, ~20 lines on the existing PINO residual; directly targets the low-k band where the local wall lives. Untested as a from-scratch training loss on KF. [SS-ok]

- **Spectral / frequency-banded loss (LOGLO fRMSE, BSP)** [1_,2_] — Add a per-frequency-band penalty on top of MSE to counter spectral bias. LOGLO cuts high-band fRMSE 58–73% on KF Re=5k; BSP improves long-term spectral stability. Both optimise spectral *magnitudes/bands*, not phases — they report fRMSE/spectral metrics, with at best indirect pointwise-relL2 benefit. [SS-ok loss, spectral metric]

- **Physics + spectral loss inside a differentiable solver (List et al. JFM 2022)** [3_] — Combine pointwise velocity L2 + spectral-energy + strain-rate losses in a CNN correction model trained inside a differentiable NS solver. This is the closest existing case of a loss reshaping *late-time pointwise field error* (not just spectra): late velocity-field MSE drops at spatial-mixing Re=500 0.0243→0.0025 (~10×) and temporal-mixing Re=250 1.39e-3→2.15e-4 (~6×). Caveats: solver-hybrid (not a standalone FNO/U-Net), metric is MSE not relL2, velocity form. The strongest counter-instance to "no clean late-time pointwise reduction exists" — but note 3_'s List-et-al-*loss* (2202.06988) is a different paper from the List-et-al-*unrolling* (2402.12971) used in Cluster 3. [solver-hybrid, AR-train]

- **Transport / optimal-transport (Sinkhorn) loss** [1_,3_] — Match distributions of summary statistics via a Wasserstein/Sinkhorn objective. Solves invariant-measure fidelity, not pointwise distance; only demonstrated on Lorenz/KS (Jiang), and DySLIM drops Sinkhorn on KF as "non-discriminative in high dimension." [AR-only, statistical]

- **Dissipativity / Lipschitz / stability regularisation** [1_,2_,3_] — Encourage trajectories to contract back onto the attractor (MNO data-augmentation shell; SGNO Lipschitz bound). Solves blow-up/OOD divergence, lowering the *upper bound* of error. 3_ (MNO's own words): "dissipativity can be encouraged without significantly affecting step-wise relL2" — it does not close the pointwise gap between two on-attractor trajectories. [AR-only, stability]

---

## Step-5 critique of the clustering (offered as refinement, not replacement)

1. **Clusters mix two different axes.** Cluster 1 describes *the problem* (the wall); Clusters 2–5 are *interventions* — but on inconsistent axes (2 = where time lives, 3 = how you train, 4 = when you correct, 5 = what the net is). 5 and 3 leak (pushforward, SGNO, SSP all double-placed) and the loss family had no home. A MECE re-cut is **by intervention locus**: *architecture / training-objective / training-protocol / inference-correction*, with the wall-characterisation (Cluster 1) kept separate as the problem statement that scores all four. This makes every method land in exactly one bin and surfaces the loss family automatically.

2. **The dominant discriminator is orthogonal to all five clusters: AR-only vs single-shot-applicable.** Across 1_/3_/4_ nearly every "breakthrough" (RNO, SGNO, SSP, DPOTc, MP-FNO, unrolled training) is autoregressive; the handful that touch a one-pass operator are PINO, Spectral-Refiner, time-as-channels, Leray, and the proposed differentiable 2-window chaining. Whatever clustering you keep, this tag is what actually decides relevance for the single-shot setup — consider it a second colour over every cluster rather than a cluster itself.

3. **A "diagnostic / adjudication" cluster is implicit and worth naming.** 4_'s banked gates (oracle-restart 0.54→0.230, chaos-spread 1.3×, capacity/grid/band-loss all dead) are not methods to break the wall — they are the evidence that decides *which* cluster is even worth pursuing (model-limited not chaos-limited → Clusters 3/A/D over Cluster 1's fatalism). Keep them as a separate evidentiary layer so step-2 (importance/do-ability) can lean on them.

4. **Suggested consolidation for the next step:** collapse to four interventional buckets + one problem bucket + one evidence bucket: (P) the wall, (I-arch) architecture incl. streamfunction/time-as-channels/SGNO/SSP, (I-obj) loss objectives incl. PINO/Sobolev±/spectral/transport/dissipativity, (I-proto) training protocols incl. pushforward/unroll/MP/recurrent/differentiable-chaining, (I-post) inference corrections incl. Spectral-Refiner-FT/Leray/DPOTc/Newton, (E) banked diagnostics. Each tagged AR-train (in scope) vs AR-infer (needs re-architecture).

---

## Addendum — AR-training scope correction + supervisor-LLM critique

### A.1 Scope correction: AR-training is acceptable, only AR-inference is disqualifying

The project tolerates autoregressive *training* (e.g. pushforward, unrolling) so long as *inference is one-shot* — that coupling is the whole trick. Reclassifying the entries that the body tagged `[AR-only]`:

**Criterion:** a method is `[AR-infer]` iff its *deployed operator is a one-step map rolled out closed-loop*; it is `[AR-train]` iff its multi-step structure lives only in the training loop and the deployed operator can emit its window in one pass.

- **In scope as training tricks `[AR-train]`** (deployed operator stays one-shot, or value = "train coupled, deploy one-shot"): differentiable multi-step unrolling / WIG-NOG (Cluster 3 — cleanest case, a pure training protocol), solver-in-loop K-step training (Cluster 3, if a learned-correction not a rolled-out map), positive/negative-Sobolev, spectral/banded, transport, dissipativity losses (Cluster A — all train-time objectives, already one-shot-compatible). **Borderline:** pushforward and MP-FNO — their mechanism is entangled with the rollout itself (pushforward *is* feeding back one-step predictions; MP-FNO inserts discontinuities *between rollout segments*), so by the strict criterion their published operators are AR-infer; they qualify as `[AR-train]` only via the window-chaining *analogue*, not as-published. Flagged so step 2 does not over-credit their do-ability.
- **Genuinely AR at inference `[AR-infer]`** (learn one-step map, roll out closed-loop — need re-architecture, but may donate a mechanism): RNO recurrent operator, SGNO ETD generator, SSP propagator, DPOTc per-step correction, vanilla MNO. Their *mechanisms* (non-expansive ℜ(α)≤0 generator, spatial/temporal separation, conserved-quantity scaling) are portable into a space-time operator; their *published operator class* is not.
- **The canonical in-scope adaptation `[SS-analogue]`:** differentiable window-chaining (Cluster 3 / 4_). Two flavors with different inference status — see A.4.

Net: nothing in Cluster 3 or Cluster A gets filtered for being "autoregressive." The filter applies only to the AR-infer operator class, and even there the mechanism is salvageable.

### A.2 Critique point 1 — Leray projection (vorticity vs velocity). APPLICABLE, slightly understated.

Correct, and stronger than stated. Leray applies to velocity form only; the model is vorticity form, where div(u)=0 is *already implicit* (recover u from ω via the streamfunction Poisson solve and incompressibility holds by construction). So even after an ω→u conversion, Leray projection is near-redundant — it does not just need conversion, it buys almost nothing afterward. **Verdict: filter Leray for this setup** (demote, not conditionally include). [1_,4_ already flagged the velocity-form restriction; this adds the redundancy argument.]

### A.3 Critique point 2 — separate Spectral-Refiner's two halves. APPLICABLE, adds the most value.

Spectral-Refiner bundles two separable components that must be scored independently:
- **(a) Periodic temporal padding + layernorm** — a one-time *architectural* fix (Cluster 2). Zero inference cost, preserves one-shot. Cheap, keep.
- **(b) H⁻¹ residual fine-tuning** — *per-instance*, iterative optimization using an auto-diff solver *at inference time* (3_: "performed for each evaluated instance"). This breaks one-shot inference speed and arguably the one-shot spirit. It is an inference-cost option, not a free win.

The body listed (a) and (b) separately but did not warn that (b) is per-instance inference-time. **Verdict: take (a) as a near-free architectural lever; treat (b) as a separate, inference-expensive option whose do-ability score must carry the per-instance solver cost.** The same H⁻¹ weighting applied as a *training* loss (Cluster A item) is the one-shot-preserving alternative to (b).

### A.4 Critique point 3 — differentiable chaining is a recurrent training protocol. APPLICABLE, moderate magnitude.

Correct. Differentiable window-chaining inherits recurrent-training costs: activation memory ~×K and gradient instability bounded by the per-window Lyapunov time (1_/List et al.: unrolling too far → diminishing/negative returns). Two mitigations and one caveat:
- Magnitude is moderate at the proposed **K=2** (~×2 memory, gradient path <0.5·T_L, where the wall is still model-limited per 4_'s gates).
- The cost is **training-only**; the deployed model's inference cost is unchanged.
- **Inference-status caveat the body glossed:** if chaining is done by *shortening the window* and chaining at test time, full-horizon inference is a 2-step chain = mild AR-inference (K=2), not strictly one-shot. If chaining is used only as a *training regularizer* on a full-T one-shot operator, inference stays truly one-shot. These are different do-ability/scope profiles and must be distinguished in step 2.

**Verdict: keep differentiable chaining, but penalize training do-ability for memory/gradient cost (moderate at K=2) and resolve which inference flavor is intended.**

### A.5 What the critique did not catch (for the supervisor to weigh)

- The AR-train vs AR-infer distinction (A.1) is more consequential than any of the three points and was absent from the critique.
- Point 2 also implies: the cheap half of Spectral-Refiner (temporal padding) overlaps the time-as-channels lever (Cluster 2) — they are competing fixes for the *same* temporal-aliasing defect, not additive. Step 2 should treat them as alternatives, not a stack.
