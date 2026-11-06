# Deterministic Neural Operators for Chaotic Partial Differential Equations: A Comprehensive Analysis of the Relative $L_2$ Error in Long-Term Predictions

Modeling complex, chaotic physical systems with deterministic neural operators (such as Fourier Neural Operators [FNO] and U-Net variants) is one of the most demanding challenges in the field of Scientific Machine Learning (SciML). Especially for systems such as the two-dimensional Navier-Stokes equations (2D NS) or the Kolmogorov flow (KF), the presence of positive Lyapunov exponents causes even microscopic approximation or discretization errors to grow exponentially over time. In the existing literature and in previous analyses, it has frequently been argued that a deterministic reduction of the pointwise relative $L_2$ error (relL2) for long-term rollouts (for time horizons $t/T > 0.5$) is mathematically impossible, and that research has instead focused exclusively on statistical metrics and invariant measures.

A deep, critical evaluation of the most recent research literature, however, falsifies this generalizing assumption. While early and highly influential baseline models explicitly abandoned pointwise error reduction in favor of statistical fidelity, the latest architectural and algorithmic breakthroughs — among them Recurrent Neural Operators (RNO), Spectral Generator Neural Operators (SGNO) and Structured Spectral Propagators (SSP) — demonstrate that the "Pointwise Wall" (the barrier of the pointwise error) can indeed be broken deterministically. The present investigation analyzes in detail the evaluation protocols of the leading baseline models, the quantitative impact of Sobolev and spectral regularizations, the mathematical effects of differentiable multi-step rollouts, as well as the most recent architectural paradigms that significantly minimize the late pointwise $L_2$ error.

## The Evaluation Protocols of Established Baseline Models: MNO and DySLIM

To understand why a substantial part of the literature reports no improvements in the pointwise relative $L_2$ error at late time steps, the philosophical and mathematical orientation of the most influential baseline works must be examined. Two of the most prominent frameworks in this context are the Markov Neural Operator (MNO) and the Dynamics Stable Learning by Invariant Measure (DySLIM) model. Both publications have substantially influenced the research community to abandon long-term trajectory prediction as a metric and instead foreground the preservation of dynamical invariants.

The Markov Neural Operator (Li et al., NeurIPS 2022) was developed specifically with the goal of learning the underlying solution operator for dissipative, chaotic systems. The fundamental mathematical premise of the MNO is that the exact trajectory of a chaotic system over long periods is in principle unpredictable due to the positive Lyapunov exponents, since small perturbations inevitably accumulate during time integration. The statistical properties of the long-term trajectories of such Markov systems, however, are determined by an invariant measure supported on a global attractor. Because the authors declare pointwise tracking of the trajectory over long horizons to be impossible, the publication reports no pointwise relative $L_2$ error for time horizons of $t/T > 0.5$. Instead, the MNO's evaluation protocol on the 2D Kolmogorov flow (tested up to Reynolds numbers of $Re = 5000$) is based entirely on spectral and distribution-based metrics. Performance is assessed via the Fourier spectrum, the spectrum of the Proper Orthogonal Decomposition (POD), the pointwise distribution of the state variables, the autocorrelation, as well as domain-specific statistics such as the turbulent kinetic energy and the dissipation rate. The MNO thus proves its stability through the repeated application of a one-step operator in order to sample the attractor, but does not claim to deterministically close the pointwise L2 gap of the trajectory.

An analogous paradigm is pursued by the DySLIM framework (Schiff et al., ICML 2024). DySLIM posits that the traditional minimization of the Mean Squared Error (MSE) between reference trajectories and the model's unrolled predictions causes models to overfit to the short-term dynamics. This overfitting leads, at large time horizons, to a so-called trajectory blow-up, in which the state variables diverge to infinity, or to grossly erroneous long-term statistics. As a solution, DySLIM introduces a scaled, measure-matching regularization term based on optimal transport that forces the system onto the invariant measure instead of minimizing the trajectory error.

Consistent with this paradigm, DySLIM's evaluation protocol on the Kolmogorov flow refrains entirely from reporting the pointwise relative $L_2$ error. The quantitative evaluation of the models is instead carried out via highly specific statistical indicators. These indicators include the Mean Energy Log Ratio (MELR), which quantifies the average deviation of the energy at each Fourier mode of the generated snapshots compared to the ground truth. In addition, the weighted variant MELRw is used, which weights modes with higher energy — typically the low-frequency structures — more strongly. To capture the spatial statistical properties, the covRMSE (the root mean square of the Frobenius norm of the covariance matrix) is computed. Furthermore, the 1-Wasserstein distance (often approximated via the Sinkhorn divergence) is used to measure the distance between the pushforward measures of the predicted and the true attractors. Although DySLIM operationalizes a "pointwise tracking" for the low-dimensional Lorenz-63 system via the cosine similarity, this metric merely measures the angle between state vectors in phase space and does not correspond to the relative $L_2$ error of the spatial fields. For the 2D Kolmogorov flow, all reported DySLIM results remain, without exception, distribution-based.

| DySLIM training protocol (KF) | MELR (↓) | MELRw (↓) | covRMSE (↓) | Wass1 (↓) | TCM (↓) |
|---|---|---|---|---|---|
| 1-Step (Base)            | 2.77 ×10⁻² | 0.44 ×10⁻² | 7.93 ×10⁻² | 16.2 ×10⁻² | 5.39 ×10⁻² |
| 1-Step (+ DySLIM)        | 1.84 ×10⁻² | 0.85 ×10⁻² | 7.30 ×10⁻² | 5.55 ×10⁻² | 2.45 ×10⁻² |
| Curriculum (Base)        | 5.35 ×10⁻² | 0.95 ×10⁻² | 8.13 ×10⁻² | 9.66 ×10⁻² | 3.50 ×10⁻² |
| Curriculum (+ DySLIM)    | 1.64 ×10⁻² | 0.45 ×10⁻² | 6.95 ×10⁻² | 4.76 ×10⁻² | 2.83 ×10⁻² |
| Pushforward (Base)       | 3.19 ×10⁻² | 0.53 ×10⁻² | 6.81 ×10⁻² | 4.64 ×10⁻² | 3.68 ×10⁻² |
| Pushforward (+ DySLIM)   | 2.46 ×10⁻² | 0.53 ×10⁻² | 6.69 ×10⁻² | 4.51 ×10⁻² | 0.72 ×10⁻² |

The table makes clear that the improvements due to DySLIM are significant, but occur exclusively in a metric family that does not directly capture the late pointwise error on the attractor.

## The Role of Regularizations: Sobolev Norms, Spectral Losses and Dissipativity

A central question of current research is whether specialized loss functions and regularization methods — such as the Sobolev $H^k$ loss, spectral and transport-based losses, or dissipativity regularization — deterministically lower the pointwise relative $L_2$ error, or whether they merely improve the model's spectral statistics. The literature clearly shows that these techniques mainly constrain the error manifold and enforce physical realism, without, however, fundamentally fixing the phase-accurate pointwise error in long-term rollouts.

The standard MSE loss (Mean Squared Error) evaluates discrepancies in the physical $L_2$ space, which naturally leads to a spectral bias. Since low-frequency, large-scale structures in turbulent flows dominate the absolute energy, neural networks primarily optimize these regions. High-frequency phenomena in the inertial and dissipation range are systematically neglected, which leads to oversmoothed, physically unrealistic predictions that rapidly depart from the true chaotic dynamics in autoregressive steps.

The integration of Sobolev $H^k$ norms into the loss function addresses this problem by explicitly penalizing errors in the spatial derivatives of the flow field. Since derivative operators act as high-pass filters in frequency space, Sobolev training forces the neural operator to accurately capture high-frequency details, sharp gradients and nonlinear interactions that are ignored by the classical MSE. A decisive trade-off, however, manifests itself: higher derivatives induce a stronger irregularity of the solution functions, which requires finer spatial resolutions to avoid aliasing effects. Sobolev training consequently guarantees that the predicted fields appear physically plausible and exhibit correct energy cascades. It does not, however, guarantee that the high-frequency phases at a time horizon of $t/T > 0.5$ match the deterministic ground truth perfectly. The spectral statistics become excellent, but the pointwise agreement in the late stage remains elusive.

The recently introduced Binned Spectral Power (BSP) loss behaves similarly. This frequency-space loss function groups spatial frequency components into scalar bins and specifically penalizes deviations in the energy distribution across different scales. Through the adaptive weighting of errors at large and small scales, the BSP loss substantially mitigates the spectral bias and improves stability in long-term predictions. However, since primarily the spectral magnitudes and not the exact spatial phases are optimized, the main contribution of the BSP loss lies in improving spectral consistency. An indirect improvement of pointwise stability is observable, since the model does not accumulate unphysical energies, but the direct pointwise distance is not minimized by BSP as a primary optimization objective.

Dissipativity regularization, as employed in the MNO framework, pursues a different approach. Dissipative physical systems are characterized by the fact that all bounded initial conditions converge, after finite time, into a compact set — the global attractor. The MNO enforces dissipativity through data augmentation on an outer shell around the attractor as well as through a strict bounding far away from the attractor. If the model predicts a state lying slightly outside the physical manifold, the dissipativity constraint acts as a restorative force that forces the next autoregressive step back onto the attractor. This regularization is fundamental for preventing exponential blow-ups when the model moves into "out-of-distribution" (OOD) states. It lowers the upper bound of the error to a physically admissible level, but in the chaotic regime it is unable to deterministically close the pointwise distance error of two neighboring trajectories that both lie on the attractor. The regularization thus improves spectral invariance and prevents divergence, but does not deterministically lower the late "Pointwise Wall".

## Differentiable Multi-Step Rollouts and Recurrent Training Paradigms

The strongest empirical and theoretical evidence so far that the pointwise relative $L_2$ error can be deterministically reduced stems from the application of differentiable multi-step rollouts (Multi-Step Unrolling) and recurrent training paradigms. The assumption that backpropagation through $K$ temporal windows demonstrably lowers the late relative $L_2$ error is validated by the current literature, since it fundamentally changes the mathematical bounds of error accumulation.

### The Problem of Exposure Bias

When deterministic neural operators are trained via the classical "teacher forcing" procedure, the loss function optimizes the model to predict the state $u(t+\Delta t)$ based on the perfect ground-truth state $u(t)$. In the inference phase, however, the model must operate autoregressively, using its own, already error-laden prediction $\hat{u}(t)$ as input to extrapolate $\hat{u}(t+\Delta t)$. This discrepancy between training and inference induces a systematic distribution shift, the so-called "exposure bias". During inference, the model is confronted with states it never saw during training. Since the operator is not contractive for these OOD states, the errors compound exponentially, which manifests as a rapid trajectory drift, phase shifts and the uncontrolled excitation of high-frequency components.

### Multi-Step Penalty Optimization (MP-FNO)

The Multi-Step Penalty (MP) framework was developed to handle the enormous challenge of optimizing over long chaotic trajectories, in which the loss landscape becomes extremely non-convex and gradients explode or vanish. When applied to Fourier Neural Operators (MP-FNO) and U-Nets, the algorithm subdivides the long training trajectory into non-overlapping windows. After each sequence of $r$ rollouts, learnable local discontinuities $\delta_s$ are inserted, whose magnitude is minimized by a penalty term in the loss function. This approach severs the computational graph at strategic points and lowers the cost of gradient computation from a cubic to a linear scaling with respect to the number of parameters, while the model is nevertheless forced to learn long-term temporal coherence.

In evaluations on the 2D Kolmogorov flow with a Reynolds number of $Re = 10^5$, the MP-FNO framework demonstrates a clear improvement in stability and accuracy over the vanilla FNO. However, the cited literature evaluates this method on KF exclusively via the angle-averaged kinetic energy spectrum and the temporal correlation with Direct Numerical Simulations (DNS). An explicit pointwise relative $L_2$ error is not published for the Kolmogorov flow, which means that while MP-FNO validates the mechanism of differentiable rollout, it fails to provide the specific numerical proof for the relL2 on this benchmark.

### Recurrent Neural Operators (RNO): Overcoming the Exponential Error Bound

A definitive answer to the question of whether differentiable multi-step rollouts demonstrably lower the late relative $L_2$ error is provided by the architecture of Recurrent Neural Operators (RNO). The RNO framework discards the teacher-forcing paradigm entirely. Instead, the operator is trained by applying it recursively to its own predictions within the training horizon $[0, T]$. The model is thus forced during training to simulate and correct its own inference dynamics, which eliminates the exposure bias.

This restructuring provides a fundamental mathematical guarantee: under standard teacher forcing, the maximum error over $N$ steps grows exponentially with the time horizon $T$ (scaling $\mathcal{O}(e^{CT})$). Through the recurrent training in the RNO framework, this worst-case error growth is reduced to a linear dependence on $T$, mathematically bounded by $T(\epsilon + \mathcal{O}(\Delta t))$.

Empirically, this shift from exponential to linear error growth manifests as a massive breakthrough of the pointwise barrier. On highly nonlinear time-dependent PDEs such as the Navier-Stokes equations (in vorticity formulation), the Allen-Cahn and the Cahn-Hilliard equation, recurrently trained Multigrid Neural Operators (r-MgNO) demonstrate significant reductions of the relative $L_2$ error at late time steps. For example, the relative $L_2$ error of the recurrently trained model on the Allen-Cahn benchmark at the 50th rollout step is at an extremely low $6.0 \times 10^{-4}$, which corresponds to a reduction of 89.6% compared to the teacher-forced counterpart (tf-MgNO) at $5.8 \times 10^{-3}$. The RNO accumulates extremely little error even up to 500 autoregressive steps and retains its stability.

These findings unambiguously prove: differentiable multi-step rollouts (specifically in the form of recurrent training over $K$ windows) deterministically and massively lower the pointwise relative $L_2$ error for complex 2D flows, without requiring a switch to probabilistic diffusion models.

## Architectural Breakthroughs for Reducing the Pointwise Relative $L_2$ Error

In addition to innovative training paradigms, deep structural analyses have revealed that geometric and spectral architectural deficiencies in standard FNOs and U-Nets contribute substantially to the late pointwise divergence. Recent literature identifies the so-called "Geometric Forgetting" (the geometric forgetting hypothesis). Since standard operators update the latent state in Euclidean space via residual connections, access to the underlying spatial geometry is lost with increasing network depth, which leads to "latent drift" and to the violation of physical consistencies.

To explicitly lower the long-term relL2, new architectures have been developed that structure the temporal update mechanism directly in frequency space.

| Model innovation | Central mechanism for error reduction | Impact on long-term rollout |
|---|---|---|
| Recurrent Neural Operators (RNO) | Recursive training on its own predictions to eliminate exposure bias. | Transforms exponential error growth ($\mathcal{O}(e^{CT})$) into linear growth ($\mathcal{O}(T)$). |
| Spectral Generator Neural Operator (SGNO) | Non-positive generators ($\Re(\alpha_g) \le 0$) in the ETD update suppress linear error amplification. | Reduction of the relative $L_2$ error by up to 92.9% on APEBench. |
| Structured Spectral Propagator (SSP) | Separation of spatial basis and temporal state; evolution of a compact Fourier set. | Reduction of the relative $L_2$ error by up to 48.9% under extreme long-term extrapolation. |

### Spectral Generator Neural Operators (SGNO)

The Spectral Generator Neural Operator (SGNO) attacks the two main causes of autoregressive instability directly at the root: the linear amplification of one-step errors and the explosive feedback of unresolved high-frequency components.

Instead of learning a generic, unstructured black-box update, SGNO models the one-step map as a structured spectral evolution step based on Exponential Time Differencing (ETD) in Fourier space. The SGNO learns a diagonal spectral generator for the linear dynamics and mathematically enforces that its real part be non-positive ($\Re(\alpha_g) \le 0$). This strict physical constraint guarantees that a repeated iteration of the autoregressive step can under no circumstances artificially amplify the linear dynamics over time. For the nonlinear dynamics, SGNO introduces a gating mechanism that acts as a Gated Forcing Term with complex-valued spectral mixing and bounds the high-frequency feedback exactly at the injection point.

The impact on the pointwise error is striking: evaluated over ten mechanism-matched APEBench tasks comprising dissipative, dispersive and Navier-Stokes-like nonlinear regimes, SGNO reduces the long-term relL2 (GMean100) by a median of 74.8% compared to the strongest non-SGNO baselines. The per-task reductions of the relative $L_2$ error thereby vary between 13.6% and an enormous 92.9%.

### Structured Spectral Propagators (SSP)

Following a conceptually related philosophy, the framework of Structured Spectral Propagators (SSP) reformulates the entire PDE rollout process. The accumulation of autoregressive errors frequently results from the architectural intermixing of the complex spatial reconstruction with the temporal evolution. SSP separates these processes through a design of analysis, propagation and synthesis.

The physical state is first encoded into a shared, time-consistent representation on a spatial basis and projected into a compact propagation state. By isolating the recurrent dynamics from the fine-grained spatial details, the temporal propagator operates exclusively on a strictly bounded set of latent Fourier modes. For this it uses a frequency-conditioned linear backbone as well as a nonlinear spectral closure. This explicit structural inductive bias reduces the dynamical drift dramatically. In comprehensive evaluations, the SSP framework lowers the relative $L_2$ error by up to 48.9% and demonstrates significantly improved stability in temporal extrapolation far beyond the time window supervised during training.

## Post-hoc Corrections and Physical Conservation Laws: The DPOTc Framework

A further deterministic path to lowering the late relL2 consists in correcting the outputs of neural operators after the fact during inference, in order to strictly enforce compliance with physical conservation laws. When autoregressive models extrapolate into the future, tiny approximation errors inevitably lead to the violation of fundamental conserved quantities such as mass, energy and momentum. In chaotic flows, the artificial injection or dissipation of energy and momentum by the network considerably accelerates the drift of the predicted trajectory away from the true physical attractor.

The Conserved Quantity Correction Framework (DPOTc), applied to Auto-Regressive Denoising Operator Transformers (DPOT) and classical FNOs, establishes a model-agnostic technique to guarantee these physical criteria without adjusting the training loss function. The methodology applies, at each inference step, a magnitude scaling correction for density and energy as well as a shift correction for momentum directly to the output tensors.

Evaluated on the compressible Navier-Stokes equations (Mach 0.1), DPOTc successfully suppresses the error accumulation. In an autoregressive 10-step rollout, the baseline DPOT model accumulates a relative $L_2$ error of 8.4%. The application of the post-hoc conserved-quantity correction (DPOTc) deterministically lowers this pointwise relative $L_2$ error to 7.8%.

| Model architecture | Parameters | Avg. Err. | T=1 | T=5 | T=10 |
|---|---|---|---|---|---|
| FNO Baseline                    | 36M  | 0.119 | 0.075 | 0.113 | 0.148 |
| FNO + Correction ($FNO_c$)      | 36M  | 0.098 | 0.063 | 0.094 | 0.122 |
| DPOT Baseline                   | 4.8M | 0.069 | 0.049 | 0.065 | 0.084 |
| DPOT + Correction ($DPOT_c$)    | 4.8M | 0.064 | 0.045 | 0.061 | 0.078 |

Although the absolute error reduction of 0.6 percentage points may seem moderate, it represents an extremely consistent improvement that throttles the growth of the relative error to merely 0.4% per time step, while requiring practically no additional computational effort. Spectral analyses of the DPOTc outputs, however, show that while the model guarantees macroscopic conservation in the low frequencies, fundamental architectural limitations in representing high-frequency data features at late time steps still provoke quantized cutoffs in the frequency spectrum. The principle of physics-based correction in itself, however, is a valid approach to deterministically lowering the pointwise relL2. For incompressible systems (such as KF), this could in principle be transferred to the conservation of circulation ($\oint \omega \, dA \approx 0$).

## Synthesis and Implications for Future Model Architectures

The hypothesis that deterministic neural operators cannot overcome the late pointwise error barrier ("Pointwise Wall") for chaotic partial differential equations is clearly refuted by the current state of science. The turn of prominent research works such as MNO and DySLIM toward invariant-measure metrics (Wass1, MELR, spectra) was a pragmatic reaction to the systematic weaknesses of standard autoregressive models based on simple MSE one-step training. Since classical architectures lack internal geometric restrictions and standard training methodologies generate extreme exposure bias, their long-term predictions inevitably converge toward physically meaningless or diverging states.

Recent methodological interventions, however, quantitatively prove that the pointwise relative $L_2$ error can indeed be substantially lowered without switching to probabilistic diffusion frameworks:

- **Differentiable multi-step rollouts transform error bounds:** the switch from simple teacher forcing to recurrent training over temporal windows (as in RNO) lowers the mathematical growth bound of the error from exponential ($\mathcal{O}(e^{CT})$) to linear ($\mathcal{O}(T)$), which leads to reductions of the relative $L_2$ error by up to 89% at late rollouts.
- **Structured spectral evolution suppresses amplification:** standard updates in the latent space allow high-frequency aliasing to feed back nonlinearly. By explicitly constraining the real part of the linear generator to non-positive values (SGNO) or isolating the temporal propagation in a frequency-conditioned state (SSP), these deterministic architectures achieve massive error reductions of up to 75% in the long-term relL2.
- **Physical post-hoc corrections damp the drift:** while Sobolev norms and BSP losses mainly improve the statistical realism of the energy cascade, the exact enforcement of conservation equations during inference (DPOTc) provides a directly measurable, deterministic reduction of the pointwise trajectory error.

To reach the theoretical optimum of the late pointwise relative $L_2$ error for the 2D Navier-Stokes equations and the Kolmogorov flow, the current literature dictates a synthesis of these three deterministic paradigms. A modern neural operator must use a structured spectral propagator (SGNO/SSP) to mathematically bind high-frequency feedback, must necessarily be trained recurrently over multiple $K$ windows to break down the exposure bias, and must implement strict post-hoc conservation corrections in the autoregression. The deterministic lowering of the "Pointwise Wall" is thus not impossible, but merely requires the departure from unstructured black-box learning.

---

# Verification Appendix (audited 2026-06-25)

Each entry: **finding** first, then **paper + citation + exact location** where it was verified. Numbers below were read from the actual paper bodies (arXiv HTML/ar5iv), not abstracts. Purpose: avoid re-derivation.

## Source papers (arXiv IDs)

| Short name | Title | arXiv | Venue/date |
|---|---|---|---|
| RNO | Recurrent Neural Operators: Stable Long-Term PDE Prediction (Ye, Zhang, Wang) | 2505.20721 | May 2025 |
| SGNO | Spectral Generator Neural Operator for Stable Long-Horizon PDE Rollouts (Li, Jiang, Saleem, Wang, Koniusz, Salim) | 2602.18801 | Feb 2026 |
| SSP | Stable Long-Horizon PDE Forecasting via Latent Structured Spectral Propagators (Lu, Yuan, Shi) | 2605.10154 | May 2026 |
| DPOTc | Benchmarking Long Roll-outs ... Compressible NS with Conserved Quantity Correction (Current, Kumar, Gaitonde, Parthasarathy) | 2601.22541 | Jan 2026 |
| DySLIM | Dynamics Stable Learning by Invariant Measure (Schiff, Wan, Parker, Hoyer, Kuleshov, Sha, Zepeda-Núñez) | 2402.04467 | ICML 2024 |
| MNO | Markov Neural Operator / learning dissipative chaotic dynamics (Li et al.) | — | NeurIPS 2022 |
| BSP | Binned Spectral Power loss for chaotic systems | 2502.00472 | Feb 2025 |
| MP-FNO | Multi-step penalty FNO (Kolmogorov Re=1e5) | 2410.05572 | Oct 2024 |
| PINO | Physics-Informed Neural Operator (anchor, your thesis) | 2111.03794 | 2021 |

## Verified findings (finding → exact source location)

**F1. Recurrent training lowers Allen-Cahn relL2 at n=50 from 5.8e-3 (tf-MgNO) to 6.0e-4 (r-MgNO) ≈ 89.7% reduction.** TRUE.
→ RNO, 2505.20721, §4.2 "Basic Rollout Performance", Table 1 (Allen-Cahn, n=50 column). Models trained to n=10, evaluated to n=50.

**F2. RNO proves worst-case error growth drops from exponential O(e^{CT}) (teacher forcing) to linear O(T) (recurrent).** TRUE but CONDITIONAL ("under the assumption that the training successfully enforces stability").
→ RNO, 2505.20721, §3.3 Theorem 1, Eq. (8)–(9); proof Appendix B.

**F3. RNO is autoregressive time-stepping, NOT single-shot space-time. It explicitly classifies single-shot (FNO-3D, "Spatio-temporal Unified Approach") as a separate category it does not study, noting single-shot "performs well within the training horizon [0,T]".** TRUE — direct relevance: results do NOT transfer to single-shot without new experiments.
→ RNO, 2505.20721, §2.1 (three-category taxonomy) and §1.

**F4. Recurrent training helps MgNO but NOT FNO ("the FNO architecture shows inconsistent outcomes, r-FNO vs tf-FNO").** TRUE — your architecture is FNO; the strongest cited result did not work on FNO.
→ RNO, 2505.20721, §4.2, paragraph after Table 1. Table 1 Navier-Stokes column: r-FNO n=50 = 3.7e-1, tf-FNO n=50 = 2.4e+02 (blow-up); r-MgNO = 8.9e-2 vs tf-MgNO = 1.4e-1 (~36%, std ±5.4e-2).

**F5. RNO's Navier-Stokes benchmark is generic 2D vorticity, 128×128 downsampled, no Reynolds number reported, no Kolmogorov forcing — not turbulent KF.** TRUE.
→ RNO, 2505.20721, Appendix C.4 (NS vorticity form), Appendix D.1 (resolution 128², dt 0.01, 1000 train / 200 test).

**F6. SGNO reduces long-horizon rollout error (GMean100) by median 74.8% vs strongest baseline; per-task 13.6%–92.9%.** TRUE. (median SGNO/baseline ratio 0.252; range 0.071 disp1d → 0.864 advdiff3d.)
→ SGNO, 2602.18801, §5.2, Table 1 and surrounding text.

**F7. SGNO's 92.9% is disp1d (1D dispersion), NOT Kolmogorov. The 74.8% median spans mostly 1D/3D tasks.** TRUE.
→ SGNO, 2602.18801, §5.2 ("ratios ranging from 0.071 on disp1d to 0.864 on advdiff3d"); task list §5.1 / Appendix B.

**F8. On the actual 2D Kolmogorov task (kolm2d), SGNO absolute error stays high: GMean100 ≈ 0.558 (~56% nRMSE) — wall not broken on KF, only beats worse baselines.** TRUE.
→ SGNO, 2602.18801, Appendix F, Table 6 (kolm2d, 64² training: GMean100 = 0.5581).

**F9. SGNO mechanism: ETD update in Fourier space, real nonpositive diagonal generator ℜ(λ)≤0 making spectral carry nonexpansive (|exp(λ)|≤1), plus gated complex-valued correction pathway.** TRUE.
→ SGNO, 2602.18801, §4.1 Eq. (14), §4.2 Eq. (17)–(18), Appendix A Prop A.1.

**F10. SGNO is autoregressive (one-step training, m=1 history, 200-step closed-loop rollout); stability is conditional ("does not constitute an unconditional stability guarantee").** TRUE.
→ SGNO, 2602.18801, §3 (Eqs. 5–7), Appendix A preamble, Appendix E.

**F11. SSP reduces relL2 by up to 48.9% under long-horizon extrapolation; autoregressive, separates spatial basis from temporal propagation on a bounded latent Fourier set.** Headline TRUE.
→ SSP, 2605.10154, abstract + §5. Mechanism: §4 — frequency-conditioned linear backbone Kθ with bounded gate 1+β·tanh(·) and normality penalty, plus nonlinear spectral closure Gψ; residual Euler substeps on retained low-freq modes.

**F11a. GAP CLOSED — SSP does NOT test Kolmogorov flow. Benchmarks are Shallow-Water (SW), Reaction-Diffusion (RD), and PDEBench Navier-Stokes (NS). No KF, no Reynolds number, no t/T>0.5 late-time KF metric.** TRUE.
→ SSP, 2605.10154, §5 intro ("Shallow-Water (SW), Reaction-Diffusion (RD), and Navier-Stokes (NS) benchmarks [PDEBench]") and §5.1. Horizons: SW/RD observe 10 frames → predict 90; NS observe 4 → predict only 12. Metrics: time-averaged relL2, E_max (worst-case pointwise), f_low (low-freq spectral).

**F11b. SSP is autoregressive (encoder–projector–propagator–inverse-projector–decoder, recursively composed; trained by autoregressive rollout), NOT single-shot space-time.** TRUE.
→ SSP, 2605.10154, §4 "Training objective" (recursive r=0..N_roll−1) and §3 factorization 𝒢̂ = 𝒟∘ℛ∘Φ∘𝒫∘ℰ.

**F11c. The 48.9% "up to" figure cannot be pinned to a specific benchmark from the body: Table 1 (per-PDE relL2/E_max/f_low) is a rasterized figure, not machine-readable text. Body confirms SSP is best on SW and RD across all metrics; on NS it wins relL2 and f_low but KNO beats it on E_max. So the 48.9% is on SW or RD (PDEBench), definitively NOT Kolmogorov.** TRUE (qualitative); exact table cell unverified.
→ SSP, 2605.10154, §5.1 + Table 1 (figure). To get the exact per-benchmark number, read the rendered Table 1 image directly.

**F12. DPOTc post-hoc conserved-quantity correction lowers compressible-NS (Mach 0.1) 10-step relL2 from 8.4% (DPOT) / 8.3% (DPOTm) to 7.8% (DPOTc).** TRUE.
→ DPOTc, 2601.22541, §4.2, Table 1 (Mach 0.1, T=10): FNO 0.148, FNOc 0.122, DPOT 0.084, DPOTm 0.083, DPOTc 0.078. Avg-Err col: DPOT 0.069 → DPOTc 0.064.

**F13. DPOTc is compressible NS (PDEBench, Mach 0.1, inviscid η=ζ=1e-8), 128² grid, autoregressive (2-step history). The incompressible-KF circulation transfer is NOT in the paper — it is the document's own speculation.** TRUE.
→ DPOTc, 2601.22541, §4 (Eqs. 19–21, dataset), §4.1 (history, grid). Circulation extension: absent.

**F14. DPOTc late-time error grows ~0.4%/timestep; FNO shows quantized frequency cutoffs tied to mode count; all architectures fail to represent high frequencies at late time.** TRUE.
→ DPOTc, 2601.22541, §4.4 (0.4%/step, correlation durations 28/25/13 steps), §4.5 + Fig. 6 (quantized cutoffs).

**F15. DySLIM reports NO pointwise relL2 on Kolmogorov; only MELR/MELRw/covRMSE/Wass1/TCM. Cosine-similarity "tracking" is on Lorenz-63, not spatial-field relL2.** TRUE.
→ DySLIM, 2402.04467, §experiments (Lorenz-63, Kuramoto-Sivashinsky, 2D KF) and results tables. Reproduced DySLIM-KF table is in the body above.

**F16. MNO declares long-horizon pointwise trajectory tracking impossible for chaotic systems; evaluates 2D KF up to Re=5000 on spectral/distributional metrics + dissipativity regularization; no pointwise relL2 for t/T>0.5.** TRUE (consistent with NeurIPS 2022 paper; not re-fetched line-by-line).
→ MNO, Li et al. NeurIPS 2022.

## Bottom-line verdict (do NOT use the body verbatim)

The body of this document accurately transcribes real papers and real numbers — no fabricated citations. BUT its central claim ("the deterministic pointwise wall is broken") is OFF-TARGET for the single-shot FNO3d / U-Net on 2D Kolmogorov setup:

- **G1.** All five "breakthrough" methods (RNO, SGNO, SSP, DPOTc, and the rollout framing) are **autoregressive rollout**, not single-shot space-time. (F3, F10, F13)
- **G2.** No paper reports its headline reduction on **2D Kolmogorov / incompressible-NS pointwise late-time relL2**: RNO 89.6% is Allen-Cahn (F1), SGNO 92.9% is disp1d (F7) with KF absolute error ~56% (F8), DPOTc is compressible NS Mach 0.1 (F12–F13).
- **G3.** The one method with the strongest pointwise result (RNO) explicitly **fails on FNO** (F4).
- **G4.** Therefore the prior exhaustive-run conclusion still stands: no clean deterministic reduction of single-shot late-time relL2 on KF is demonstrated in the literature. This document did not falsify it; it changed the subject.

Transferable HYPOTHESES worth testing on a single-shot FNO3d KF baseline (not yet evidence): (i) recurrent/unrolled training over K windows; (ii) SGNO-style nonpositive spectral-gain constraint; (iii) post-hoc conserved-quantity (circulation/enstrophy) projection at inference.
