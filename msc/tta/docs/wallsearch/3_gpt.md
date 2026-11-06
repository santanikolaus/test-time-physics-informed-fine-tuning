# Deterministic neural operators for late-time 2D Navier–Stokes / Kolmogorov-flow prediction

Date checked: 2026-06-25

## Scope and evidence rule

This note addresses whether deterministic FNO/CNN/U-Net-like models have demonstrated lower late-time pointwise error on two-dimensional Navier–Stokes or Kolmogorov-flow problems.

Every evidence block below is based on the full paper PDF, including the relevant methods, experiments, tables, and appendices—not only its abstract. Each block has:

1. the source;
2. a short exact quotation from that source;
3. the paper locations checked;
4. only the conclusion supported by those locations.

The quoted excerpts are deliberately short. Numerical results are transcribed from the cited tables and are identified as such.

---

## Evidence 1 — MNO: Sobolev losses and dissipativity regularization

**Source**

Zongyi Li et al., *Learning Dissipative Dynamics in Chaotic Systems*, NeurIPS 2022.  
Full paper: <https://arxiv.org/abs/2106.06898>

**Exact quote**

> “We empirically find that dissipativity can be encouraged without significantly affecting the model’s step-wise relative L2 error and the learned statistical properties of the attractor.”

**Locations checked**

- Sections 3 and 4.3.
- Appendix A.3.
- Tables 3, 4, and 5.
- Figures 5, 6, 11, 12, and 13.

**What follows for us**

- MNO is a Markov neural operator: it learns a fixed-time map \(S_h\), then repeatedly composes that map to generate a trajectory.
- Its Kolmogorov-flow experiments use an FNO-based MNO and include \(L^2\), \(H^1\), and \(H^2\) training losses.
- At \(Re=40\), \(h=1\), Table 3 reports one-map testing errors. For MNO, the testing \(L^2\) error is:
  - \(0.0187\) after \(L^2\)-loss training;
  - \(0.0151\) after \(H^1\)-loss training;
  - \(0.0143\) after \(H^2\)-loss training.
- Therefore, this paper does demonstrate a reduction in short-step pointwise relative \(L^2\) error from Sobolev training: \(0.0187\rightarrow0.0143\), approximately 24%.
- That result is not a late-time trajectory-error result. The paper does not report paired trajectory relative \(L^2\) as a function of rollout time or specifically over \(t/T>0.5\).
- Its long-run evaluations use 10,000 compositions and examine the learned attractor, Fourier spectra, pointwise distributions, kinetic-energy and dissipation distributions, and spatial or temporal correlations.
- At \(Re=5000\), Table 5 reports testing \(L^2=0.1545\) without dissipativity regularization and \(0.1543\) with it. This is effectively unchanged.
- The supported interpretation is therefore:
  - Sobolev loss can improve one-step relative \(L^2\);
  - dissipativity regularization prevents blow-up and encourages return to the attractor;
  - MNO does not demonstrate that either mechanism lowers late-half trajectory relative \(L^2\).

---

## Evidence 2 — DySLIM: pointwise tracking and invariant-measure metrics

**Source**

Yair Schiff et al., *DySLIM: Dynamics Stable Learning by Invariant Measure for Chaotic Systems*, ICML 2024.  
Full paper: <https://arxiv.org/abs/2402.04467>

**Exact quotes**

> “The former is measured by a cosine similarity statistic between true and predicted trajectories.”

> “the SD becomes non-discriminative”

**Locations checked**

- Sections 3, 4, 5, and 5.3.
- Appendix C, especially C.1–C.5.
- Appendix D.
- Appendix E.3 and E.3.1.
- Appendix F.3.
- Tables 1, 5, 6, and 7.
- Figures 3, 8, and 9–12.

**What follows for us**

- It is incorrect to say that DySLIM evaluates Kolmogorov flow only with spectral or invariant-measure metrics.
- It also reports trajectory cosine similarity over rollout time. This is a paired pointwise-state tracking metric, but it is an angle/correlation-like metric—not relative \(L^2\).
- The Kolmogorov-flow tabular metrics are:
  - mean energy log ratio, MELR;
  - energy-weighted MELR;
  - covariance RMSE;
  - pointwise Wasserstein-1;
  - time-correlation metric, TCM.
- Sinkhorn divergence is not used as the main Kolmogorov-flow metric because the authors found it non-discriminative in the high-dimensional state space.
- The data protocol is:
  - pseudo-spectral simulation at \(256^2\);
  - downsampling by four in each spatial direction, giving \(64^2\);
  - 128 training trajectories;
  - 32 validation and 32 test trajectories;
  - a 50-time-unit warm-up;
  - 120 further time units sampled every \(0.1\).
- The full paper does not state a viscosity or Reynolds number for this experiment. A Reynolds number should therefore not be assigned to DySLIM’s KF result without additional code-level verification.
- Pushforward training increases its maximum rollout length from 1 to 10; the effective length is sampled uniformly up to that maximum.
- Curriculum training uses a maximum of 5 steps, reduced to as few as 2 at large batch sizes because of memory.
- An appendix tests pushforward training over 40 steps.
- DySLIM improves cosine tracking and long-run statistical metrics in its reported KF experiments.
- It does not report KF trajectory relative \(L^2\), so it cannot directly answer whether it lowers a late-time relL2 wall.

---

## Evidence 3 — Physics and spectral losses can reduce late-time field MSE

**Source**

Björn List, Li-Wei Chen, and Nils Thuerey, *Learned Turbulence Modelling with Differentiable Fluid Solvers: Physics-based Loss-functions and Optimisation Horizons*, Journal of Fluid Mechanics 949, A25, 2022.  
Full paper: <https://arxiv.org/abs/2202.06988>  
DOI: <https://doi.org/10.1017/jfm.2022.738>

**Exact quote**

> “The results indicate that the baseline L2 loss only performs well on short temporal horizons, while its performance deteriorates over longer evaluations.”

**Locations checked**

- Sections 2.3 and 2.4.
- Sections 3–6.
- Appendix F.
- Tables 1, 3, 5, 6, 7, 8, and 9.
- Figures 2–5, 7–10, and 15–20.

**What follows for us**

- This is deterministic 2D incompressible Navier–Stokes work using CNN closure/correction models inside a differentiable numerical solver.
- It is not a standalone black-box FNO or U-Net replacement for the full flow solver.
- Its combined turbulence loss contains:
  - pointwise velocity \(L^2\);
  - spectral-energy loss;
  - strain-rate loss;
  - in the statistically stationary case, a multi-step statistics loss.
- Appendix Table 9 directly evaluates instantaneous velocity-field MSE at a short and a late time.
- For isotropic decaying turbulence, whose Reynolds number evolves from 126 to 296:
  - late velocity MSE with \(L^2\) only: \(0.182\);
  - with \(L^2+\) spectral-energy \(+\) strain loss: \(0.166\).
- For the temporal mixing layer at \(Re=250\):
  - late velocity MSE with \(L^2\) only: \(1.39\times10^{-3}\);
  - with \(L^2+\) spectral-energy \(+\) strain loss: \(2.15\times10^{-4}\).
- For the spatial mixing layer at \(Re=500\):
  - late velocity MSE with \(L^2\) only: \(0.0243\);
  - with all loss components: \(0.0025\).
- Thus, physics/spectral losses do not merely improve spectra in this paper. They also substantially reduce a late-time pointwise velocity-field error.
- The exact metric is MSE, not normalized relative \(L^2\). The result therefore establishes movement of a pointwise error wall, but not the exact relL2 metric in the original question.

---

## Evidence 4 — Fully differentiable multi-step training on Kolmogorov flow

**Source**

Bjoern List et al., *Differentiability in Unrolled Training of Neural Physics Simulators on Transient Dynamics*, arXiv v2, 2024.  
Full paper: <https://arxiv.org/abs/2402.12971>

**Exact quote**

> “Differentiable ones perform best in a comparison among correction networks as well as among prediction setups.”

**Locations checked**

- Sections 2.3–2.5.
- Section 3.3.
- Sections 4.1–4.3.
- Appendix D.2.
- Appendix F, especially Tables 9, 13, 16, and 17.
- Figures 4, 11, 14, 19, 21, and 33–35.

**What follows for us**

- The paper explicitly compares:
  - one-step training, ONE;
  - multi-step unrolling without temporal gradients, NOG;
  - fully differentiable multi-step unrolling, WIG.
- Its Kolmogorov-flow training Reynolds numbers are \(300,400,500,700,800,900\).
- It tests interpolation at \(Re=600\) and extrapolation at \(Re=1000\).
- The spatial training resolution is \(32^2\), obtained by fourfold downsampling from \(128^2\).
- The training curriculum is \(K=[1,2,4]\), with progressively reduced learning rates.
- Errors are accumulated over 250 inference steps, corresponding to five characteristic flow times.
- For a 1M-parameter pure CNN predictor, Appendix Table 13 reports:
  - ONE: \(0.04561\);
  - NOG: \(0.03582\);
  - WIG: \(0.03595\).
- Thus both four-step unrolled variants lower the reported long-rollout field error by approximately 21% relative to one-step training.
- In this pure-prediction comparison, WIG is not materially better than NOG; NOG is numerically slightly lower.
- The hybrid correction models show more consistent improvements from WIG. For example, at 1M parameters, Appendix Table 9 reports \(0.00102\) for ONE and \(0.00080\) for WIG.
- Metric caveat: the paper’s main \(L_{\mathrm{rel}}\) definition uses a kinetic-energy-like pointwise normalization. Some appendix tables call the quantity \(L^2\). It should not automatically be treated as the standard norm ratio
  \[
  \|\hat u-u\|_2/\|u\|_2.
  \]
- Therefore this paper supports the claim that \(K=4\) differentiable or stop-gradient unrolling lowers long-rollout KF field error. It does not provide a clean standard-relL2 result showing that full temporal gradients are superior to stop-gradient unrolling for pure predictors.

---

## Evidence 5 — Kochkov et al.: \(K=32\) solver-in-loop training

**Source**

Dmitrii Kochkov et al., *Machine learning–accelerated computational fluid dynamics*, PNAS 118(21), 2021.  
Full paper: <https://arxiv.org/abs/2102.01010>  
DOI: <https://doi.org/10.1073/pnas.2101784118>

**Exact quote**

> “We unroll the model for 32 time steps when calculating the loss, which we find improves model performance for long time trajectories.”

**Locations checked**

- Methods section on training.
- Results sections for \(Re=1000\), \(Re=4000\), and LES.
- Figures 2–6.
- Appendices A, F, and G.

**What follows for us**

- This is deterministic, differentiable, solver-in-loop training for learned interpolation and learned correction models.
- The principal KF training experiment uses \(Re=1000\), high-resolution \(2048^2\) DNS, and a coarse \(64^2\) learned solver.
- Training uses cumulative pointwise velocity MSE over 32 unrolled steps.
- Evaluation includes paired vorticity correlation with the DNS and long-time energy spectra.
- The paper also tests generalization to \(Re=4000\) and trains an LES case at \(Re=10^5\).
- It supports \(K=32\) as an effective deterministic multi-step training horizon for long-trajectory accuracy.
- It does not provide a controlled one-step-versus-\(K=32\) numerical relL2 table. Its main trajectory metric is vorticity correlation, not relative \(L^2\).

---

## Evidence 6 — MP-FNO does not provide an exact verifiable \(K\)-result

**Source**

Dibyajyoti Chakraborty et al., *Improved Deep Learning of Chaotic Dynamical Systems with Multistep Penalty Losses*, arXiv v1, 2024.  
Full paper: <https://arxiv.org/abs/2410.05572>

**Exact quote**

> “after every r rollouts we detached the computational graph before introducing the discontinuities”

**Locations checked**

- Entire seven-page paper.
- Equations 2–4.
- Figures 1–5.
- Kolmogorov-flow results section.

**What follows for us**

- MP-FNO is not full backpropagation through one uninterrupted long trajectory.
- It backpropagates within segments of length \(r\), then detaches the graph and introduces trainable discontinuities between segments.
- Its KF experiment is stated to use \(Re=10^5\).
- It presents energy-spectrum and DNS-correlation plots for FNO versus MP-FNO.
- The paper does not provide:
  - the final selected \(r\) and number of segments \(s\);
  - a numerical results table;
  - a precise definition of the DNS-correlation quantity;
  - trajectory relative \(L^2\).
- Consequently, this paper cannot support an exact claim such as “\(K=x\) lowers late-time KF relL2 by \(y\).”

---

## Evidence 7 — PDE-Refiner’s KF protocol

**Source**

Phillip Lippe et al., *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers*, NeurIPS 2023.  
Full paper: <https://arxiv.org/abs/2308.05732>

**Exact quote**

> “we report in Table 1 the time until which the average correlation across trajectories falls below 0.8.”

**Locations checked**

- Sections 2–4.3.
- Appendix D.
- Appendix E.3, E.6, and E.8.
- Tables 1, 4, and 6.
- Figures 3, 4, 11, 13, 23, 24, and 26.

**What follows for us**

- The KF experiment uses \(Re=1000\), \(64^2\) data, 128 training trajectories, and 16 test trajectories.
- The primary KF trajectory metric is the duration for which Pearson correlation of vorticity remains above 0.8.
- Table 1 reports:
  - MSE-trained FNO: \(6.451\pm0.105\);
  - MSE-trained U-Net: \(9.663\pm0.117\);
  - PDE-Refiner U-Net: \(10.659\pm0.092\).
- These values are correlation-duration times, not relative \(L^2\).
- The paper examines rollout MSE in detail for the 1D Kuramoto–Sivashinsky experiment, but it does not provide a KF late-time relL2 table.
- PDE-Refiner uses iterative noise/refinement and is diffusion-inspired. It is therefore not a clean example of the strictly deterministic loss modifications posed in the original question.

---

## Evidence 8 — Spectral-Refiner: negative-Sobolev fine-tuning

**Source**

Shuhao Cao et al., *Spectral-Refiner: Accurate Fine-Tuning of Spatiotemporal Fourier Neural Operator for Turbulent Flows*, ICLR 2025.  
Full paper: <https://arxiv.org/abs/2405.17211>

**Exact quote**

> “All error norms are evaluated at the final time step.”

**Locations checked**

- Sections 2–4.2.
- Tables 1–3.
- Appendix C.3.
- Appendix D.2 and D.4.
- Tables 6 and 7.
- Figures 8–13.

**What follows for us**

- This paper uses a trajectory-to-trajectory spatiotemporal FNO and fine-tunes its final spectral layer using a negative-Sobolev residual norm.
- It does not use an ordinary one-step autoregressive FNO protocol.
- Fine-tuning is assisted by a traditional numerical solver and is performed for each evaluated instance.
- For the \(Re=1000\) McWilliams isotropic-turbulence experiment, Table 3 reports final-time \(L^2\):
  - \(3.69\times10^{-2}\) after ordinary \(L^2\) training;
  - \(2.88\times10^{-4}\) after \(H^{-1}\) fine-tuning.
- In the OOD \(Re=5000\) experiment, Table 7 reports:
  - \(0.294\) before fine-tuning;
  - \(0.0250\) afterward.
- These are substantial pointwise final-time \(L^2\) reductions.
- They do not establish that adding a Sobolev loss to a conventional autoregressive FNO lowers a KF late-half rollout wall. They establish that solver-assisted negative-Sobolev per-instance fine-tuning can strongly reduce final-time field error in related 2D turbulence.

---

## Evidence 9 — Local differential/integral neural-operator kernels

**Source**

Miguel Liu-Schiaffini et al., *Neural Operators with Localized Integral and Differential Kernels*, ICML 2024.  
Full paper: <https://arxiv.org/abs/2402.16845>

**Exact quote**

> “For the time-dependent Navier-Stokes and shallow water equations, we also predict the error after 5 autoregressive steps.”

**Locations checked**

- Sections 3 and 4.2.
- Table 4.
- Appendix C.2.
- Figure 3.

**What follows for us**

- The KF experiment uses \(Re=5000\), \(128^2\), temporal step \(\tau=1\), and ordinary squared-\(L^2\) training.
- It augments an FNO with learned local differential and local integral kernels.
- Table 4 reports relative \(L^2\) after one and five autoregressive steps.
- At five steps:
  - FNO: \(0.2360\);
  - FNO plus both local integral and differential kernels: \(0.1956\).
- This is approximately a 17% reduction in five-step relative \(L^2\).
- It is deterministic and directly uses the requested metric.
- However, five steps are not enough to establish a late-time, long-horizon \(t/T>0.5\) result in the sense of a 100-step or longer rollout.

---

## Evidence 10 — DINO: direct 99-step relative-\(L^2\) claim

**Source**

Hao Wu et al., *Differential-Integral Neural Operator for Long-Term Turbulence Forecasting*, arXiv v3 dated 2026-05-19.  
Full paper: <https://arxiv.org/abs/2509.21196>

**Exact quote**

> “We employ a one-step training strategy, where ground-truth data uk is always used as input for the next-step prediction.”

**Locations checked**

- Sections 3.3–4.3.
- Equations 9 and 10.
- Tables 1–3.
- Figures 4–6.
- Appendix B.
- Appendix C.2 and Table 4.

**What follows for us**

- DINO is a deterministic one-step-trained operator with a constrained local differential branch and a global Transformer-based integral branch.
- It is evaluated autoregressively for 99 steps on 2D forced Kolmogorov turbulence at \(128^2\).
- Table 1 labels its evaluation quantity “Relative \(L^2\) Error.”
- At step 99, Table 1 reports:
  - FNO: \(3.1284\);
  - PDE-Refiner baseline: \(1.9954\);
  - DINO: \(0.5876\).
- This is a direct positive result for deterministic late-time field error, and it contradicts an absolute claim that no deterministic operator has moved late-time pointwise error.
- Important limitations found in the full paper:
  - only a fixed random seed of 42 is reported;
  - the KF Reynolds number or viscosity is not stated;
  - the physical time interval between stored frames is not stated;
  - the paper labels the metric relative \(L^2\) but does not provide its normalization formula;
  - detailed baseline reproduction settings are limited.
- It is therefore relevant evidence but not yet a clean, multi-seed, fully specified benchmark from which to infer a universal result.

---

## Evidence 11 — SGNO: 100-step normalized rollout error

**Source**

*SGNO: Spectral Generator Neural Operators for Stable Long Horizon PDE Rollouts*, arXiv:2602.18801, 2026.  
Full paper: <https://arxiv.org/abs/2602.18801>

**Exact quote**

> “Each model is trained as a one-step predictor and evaluated by a 200-step closed-loop rollout.”

**Locations checked**

- Sections 3–5.
- Equations 8 and 9.
- Tables 1–3.
- Appendix B.
- Appendix E.
- Appendix F and Table 6.

**What follows for us**

- SGNO is a deterministic one-step-trained autoregressive neural operator.
- It uses the APEBench Kolmogorov-flow problem, whose default configuration is \(Re=100\), \(\Delta t=0.1\), forcing wavenumber 4, and linear drag \(-0.1\).
- Its main metric is GMean100: the geometric mean of trajectory-normalized RMSE over rollout steps 1–100, followed by a median over independently trained seeds.
- The KF comparison uses 20 independent network seeds.
- Table 1 reports KF GMean100:
  - FNO: \(0.729\);
  - U-Net: \(0.836\);
  - strongest non-SGNO baseline: \(0.729\);
  - SGNO: \(0.556\).
- Thus SGNO reduces a normalized long-rollout pointwise error relative to FNO under a matched protocol.
- The metric aggregates steps 1–100; it does not separately report an average restricted to \(t/T>0.5\).
- The stable-step threshold in Appendix Table 6 is reached at approximately step 13–14, showing that “better long-horizon aggregate error” does not mean accurate phase tracking for the entire rollout.

---

## Evidence 12 — Optimal-transport/contrastive-loss paper is not a 2D KF result

**Source**

Chiyu Jiang et al., *Learning Chaotic Dynamics with Embedded Invariant Measures*, arXiv:2306.01187.  
Full paper: <https://arxiv.org/abs/2306.01187>

**Exact quote**

> “We evaluate our approach on the 1D chaotic Kuramoto–Sivanshinsky (KS) system and a finite-dimensional Lorenz 96 system.”

**Locations checked**

- Sections 3 and 4.
- Sections 4.1 and 4.2.
- Tables 1 and 2.
- Appendices A–C.
- Tables 5, 6, and 9–12.

**What follows for us**

- This paper does not contain a 2D Navier–Stokes or Kolmogorov-flow experiment.
- Its optimal-transport and contrastive losses are evaluated on Lorenz-96 and 1D Kuramoto–Sivashinsky.
- Long-run evaluation emphasizes invariant statistics, energy spectra, Lyapunov quantities, and histogram errors.
- It also reports one-step relative RMSE, but not a 2D KF late-rollout relL2 result.
- It cannot be used as direct evidence that transport loss either does or does not lower the KF pointwise wall.

---

## Final source-grounded conclusions

1. **The absolute statement “no deterministic paper has lowered late-time pointwise error” is false.**  
   The solver-in-loop JFM work reports large late-time velocity-MSE reductions from deterministic multi-step training and physics/spectral losses [Evidence 3]. DINO and SGNO report direct long-horizon normalized field-error improvements for deterministic operators [Evidence 10–11], although both are recent preprints with protocol limitations.

2. **The narrower standalone-FNO loss claim remains mostly unresolved.**  
   MNO shows that Sobolev loss lowers one-step relative \(L^2\), but not late-rollout relL2; dissipativity mainly improves boundedness and attractor return [Evidence 1]. Local differential/integral kernels improve five-step relL2, but the horizon is short [Evidence 9]. Spectral-Refiner obtains major final-time improvements with a materially different solver-assisted, per-instance fine-tuning protocol [Evidence 8].

3. **Spectral or physics-based losses can improve pointwise late-time fields, not only spectra.**  
   This is directly demonstrated by the late velocity-MSE columns in the JFM loss ablation [Evidence 3]. It is not yet the same as a controlled Sobolev-versus-\(L^2\) ablation on a conventional autoregressive KF FNO.

4. **Differentiable multi-step training has demonstrated lower long-rollout field error.**  
   The strongest explicit KF curriculum found is \(K=[1,2,4]\), evaluated over 250 steps and across \(Re=600\) and \(Re=1000\) tests [Evidence 4]. Kochkov et al. use \(K=32\) and report improved long-trajectory correlation [Evidence 5]. The JFM solver-in-loop study finds gains through \(K=30\) or \(60\), with gradient splitting needed for the longest chains [Evidence 3].

5. **Full temporal gradients are not uniformly required for the gain.**  
   In the pure KF predictor comparison, \(K=4\) stop-gradient and fully differentiable training perform almost identically, with stop-gradient numerically slightly better [Evidence 4]. Stronger benefits from full gradients appear in hybrid correction settings.

6. **MNO and DySLIM do not report the requested late-time KF trajectory relL2.**  
   MNO reports one-step norm errors and long-run invariant statistics [Evidence 1]. DySLIM reports pointwise cosine tracking plus spectral/distributional metrics, but no trajectory relL2 [Evidence 2].

7. **MP-FNO cannot currently answer the exact-\(K\), exact-relL2 question.**  
   Its segment length and final rollout configuration are not numerically specified, and its KF results are plots of energy spectrum and DNS correlation rather than a relL2 table [Evidence 6].

## Most defensible thesis statement

Based only on the full papers above:

> Deterministic methods can reduce finite-horizon late-time pointwise error in 2D Navier–Stokes and Kolmogorov-flow prediction. The strongest established evidence comes from differentiable solver-in-loop training, multi-step unrolling, and newer operator architectures. For a conventional standalone autoregressive FNO, there is still no strong controlled result showing that Sobolev, transport, or dissipativity regularization alone lowers a late-half KF trajectory-relative-\(L^2\) wall.

## Explicit exclusions

No claim from DPOTc, the previously mentioned Lyapunov-bound paper, or other papers not fully inspected during this verification is used in this note.
