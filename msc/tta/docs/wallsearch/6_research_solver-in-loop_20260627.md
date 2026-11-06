---
title: "Solver-Coupled Training for Single-Shot Space-Time Operators — How People Make It Work (With Code)"
date: 2026-06-27
depth: exhaustive
confidence: 95%
sources: 49 unique (4 deep primary extractions: List-2024 JCP, List-2022 JFM, Frezat-2024 gradient-free, APEBench/unrolling/differentiable-piso repos)
follows_from: research_late-time-wall-stress-test_20260627.md, ranking_v2.md, wall.md
---

# Solver-Coupled Training: Mode 1 (training-only critic) vs Mode 2 (correction hybrid)

**Date:** 2026-06-27
**Depth:** exhaustive
**Confidence:** 95% (Q1 Mode-1-existence 94%; the forced-KF exact late-pointwise number is an inherent literature gap, not a search deficiency)
**Sources:** 49 unique across 6 search rounds + 4 deep primary-source extractions

---

## Executive Summary

**Mode 1 — a solver used purely as a training-time critic on an intermediate frame of a single-shot full-trajectory operator, with the network deployed alone — is essentially absent from the published literature.** After an exhaustive search, the only work that deploys a *neural operator trained against a solver step but runs the network alone at inference* is **SoL-DeepONet** (ICLR 2026 workshop), and it is a one-step (tₙ→tₙ₊₁) critic on **1D Burgers**, not an intermediate-frame critic on a 2D-turbulence space-time operator. Every other "solver-in-the-loop" / "a-posteriori" / "differentiable-physics" method keeps the solver **at inference** (it is a closure or correction inside the solver), i.e. Mode 2. This is a decisive finding: the one-shot-preserving solver critic you want is **novel and unproven** — high upside, real risk.

**Mode 2 — solver-in-the-loop correction hybrid (solver at train AND inference) — is the mature, code-rich, documented winner for late-time pointwise error**, with a canonical lineage (Um 2020 → Kochkov 2021 → List 2022/2024 → Frezat → Sirignano/MacArt → Shankar) and multiple maintained, directly reusable repositories. Its headline late-window pointwise gains (List 2022: decaying-isotropic late MSE 0.057→0.021 ≈2.7×; spatial mixing-layer up to ~10×) are **velocity-form, decaying/mixing-layer** numbers; **no source reports a forced-Kolmogorov vorticity-form late-time pointwise number under solver-in-loop**, so its transfer to your exact metric remains a genuine (and unfillable from the literature) uncertainty. List 2024 settles the cost question: a **non-differentiable** solver in the loop (NOG) captures the bulk of the gain (up to 4.5× over pure prediction); full differentiability (WIG) adds only ~38% on top.

**The single most reusable asset for your PyTorch/Lightning stack is `tum-pbs/PICT`** — a differentiable, GPU-accelerated PISO Navier–Stokes solver coded in pure PyTorch (2025). For a JAX route, **APEBench/Exponax** (pseudo-spectral, differentiable, explicit "solver-in-the-loop correction setting", 2D/3D Navier–Stokes) is the closest drop-in to a vorticity-form spectral operator pipeline.

---

## Key Findings

1. **Mode 1 (one-shot inference, solver only as training critic) is unwritten for 2D turbulence.** The closest published instance, SoL-DeepONet, is 1D Burgers with a single-step solver critic and relative-L2 5.88% (vs PI-DeepONet 1.38%, but 100× faster training). [SoL-DeepONet, ICLR 2026 WS](https://openreview.net/forum?id=JZldiTPkpl)

2. **The intermediate-frame solver critic with single-shot deployment has no precedent** — searches for "multiple shooting / trajectory-matching loss for a single-shot operator with network-only inference" return only solver-embedded implicit models (Multiple Shooting Layers) or solver-at-inference closures. This makes your Mode-1 proposal a genuine contribution but unproven. [Differentiable Multiple Shooting Layers, NeurIPS 2021](https://arxiv.org/abs/2106.03885)

3. **Mode 2 canonical recipe is consistent across the lineage:** L2 + spectral-energy (log-spectral distance) + strain-rate + multi-step-mean loss, unrolled m≈3–30 steps matched to the integral timescale, network as a corrective forcing inside the solver. [List et al., JFM 2022](https://arxiv.org/pdf/2202.06988)

4. **A non-differentiable solver suffices for most of the Mode-2 gain.** List 2024 disentangles forward data-shift reduction (the heavy lifter, achievable with a non-diff solver, NOG) from long-term gradients (WIG, +~38%); NOG reaches up to 4.5× over pure prediction. [List et al., JCP 2024](https://arxiv.org/html/2402.12971v2)

5. **Late-time pointwise gains are documented but flow-dependent and velocity-form:** decaying-isotropic late MSE 0.057→0.021 (≈2.7×), spatial mixing-layer up to ~10×; Kochkov reports DNS-accuracy at 8–10× coarser grid with long-rollout stability. **None is forced-KF vorticity-form late-pointwise.** [List JFM 2022 Table 1](https://arxiv.org/pdf/2202.06988), [Kochkov PNAS 2021](https://www.pnas.org/doi/10.1073/pnas.2101784118)

6. **The best PyTorch-native reusable solver is PICT** (differentiable PISO, GPU, 2025); the best curriculum-unrolling reference code is `tum-pbs/unrolling` (has a Kolmogorov-flow correction experiment with WIG/NOG, curriculum step_counts, warmup/pushforward, gradient-cut interval). [PICT arXiv 2505.16992](https://arxiv.org/abs/2505.16992), [tum-pbs/unrolling](https://github.com/tum-pbs/unrolling)

7. **A neural emulator can substitute for a non-differentiable solver's gradient (Frezat 2024)** — train a differentiable surrogate of the coarse solver, then backprop the parametrization through *it*. This is the route if you refuse to make `NavierStokes2d` differentiable and also refuse NOG. [Frezat et al., JAMES 2024](https://arxiv.org/html/2310.19385v4)

8. **Gradient stability is Lyapunov-bounded and handled by three stabilizers in practice:** gradient-stopping into non-overlapping subranges (List), warm-up/pushforward steps (no-loss unroll), and short curricula m=[1,2,4]. Best results occur when the unroll horizon **approaches but does not exceed the integral timescale**. [List JFM 2022 §6](https://arxiv.org/pdf/2202.06988), [List JCP 2024](https://arxiv.org/html/2402.12971v2)

---

## Detailed Analysis

### Q1 — Who has done Mode 1 (training-only solver critic, network deployed alone)?

**Conclusion: essentially nobody, in your regime.** Mode 1 as you define it — a single-shot, non-autoregressive operator predicting the full trajectory in one pass, where during training an *intermediate* network output is pushed K steps through a differentiable PDE solver and matched to GT late frames, then the solver is discarded at inference — has **no published instance for 2D turbulence or any space-time neural operator.** This is the report's highest-value result and it is a negative one.

The reason the absence is structural, not incidental, is that the entire "a-posteriori / online / solver-in-the-loop" literature is built around closures that **stay inside the solver at inference**. Frezat's taxonomy is explicit: "the same idea has also been referred to as a-posteriori, solver-in-the-loop, differentiable, or end-to-end" — and in all of these the trained object is a *correction/closure term* `M(ū)` that the coarse solver `g` calls at every inference step ([Frezat et al., JAMES 2024](https://arxiv.org/html/2310.19385v4)). List 2024 formalises the same split: "prediction setups rely solely on neural networks" (no solver, but autoregressive), while "correction setups include a numerical solver that is supported by a neural network" (solver at inference) ([List et al., JCP 2024](https://arxiv.org/html/2402.12971v2)). Neither is Mode 1: Mode 1 wants the solver present only in the *loss*, on an *intermediate* frame, with a *single-shot* predictor deployed bare.

The single closest published hit is **SoL-DeepONet** (Luna, Braga-Neto, Raissi, Gildin, ICLR 2026 Workshop on AI and PDEs):

> "SoL-DeepONet combines a DeepONet with a solver-in-the-loop training strategy, in which a conventional solver update replaces PDE residuals computed by automatic differentiation. At training time, the solver advances the network prediction at time tₙ by one time step, and the network learns to match this solution at tₙ₊₁." — [SoL-DeepONet](https://openreview.net/forum?id=JZldiTPkpl)

This is genuinely Mode-1-shaped: (a) the solver is a **training-time consistency critic** that replaces the PINN/PINO PDE residual; (b) the predictor is a neural **operator** deployed alone (it learns "the time-evolution operator"); (c) it reports relative-L2 **5.88%** at 0.051 s/iter versus PI-DeepONet's 1.38% at 5.263 s/iter — i.e. *worse accuracy but two orders of magnitude faster training*. But three caveats make it weak transfer evidence for you: it is **1D viscous Burgers**, not 2D forced Kolmogorov; the critic is **one-step (tₙ→tₙ₊₁)**, not an *intermediate-frame integrate-K-steps* critic; and the operator is a per-step evolution operator, not a single-pass full-trajectory map. So SoL-DeepONet proves the *coupling pattern* is viable and cheap, but not the *single-shot full-trajectory + midpoint-critic* variant you need.

For each related framing chased by name, the deployment-mode audit is:
- **a-posteriori / online LES closure** (Frezat 2022, Sirignano-MacArt DPM, Shankar Differentiable Turbulence): solver **at inference**, predictor is a closure (autoregressive in effect). Mode 2.
- **differentiable-physics distillation**: no canonical instance distills a solver-coupled trajectory into a solver-free single-shot operator; the term is not established for this.
- **multiple shooting / trajectory matching**: exists as Differentiable Multiple Shooting Layers ([2106.03885](https://arxiv.org/abs/2106.03885)), but that embeds an ODE solver inside the model (solver at inference), and as DiffEqFlux "data shooting" for Neural-ODE training — again solver-at-inference. No single-shot-deploy variant.
- **adjoint-matching**: used inside Mode-2 closures (Sirignano adjoint training), not as a one-shot-preserving critic.

**Decisive implication.** Your Mode 1 (single-shot operator + differentiable-solver critic on the t/2 frame + network-only inference) is, as far as the literature shows, **unprecedented for 2D turbulence**. The prior report's framing was right to call it "the solver analogue of the failed network-fed window-chaining" — and the literature gives you no existence proof that it works. The risk is concrete: the only evidence that a solver critic can sharpen a *single-shot operator's intermediate state* is SoL-DeepONet's one-step 1D Burgers result, where it was *less* accurate than the PINN-residual baseline (just far cheaper). This is the cross-reference to Q2: the reason Mode 2 dominates the literature is that the correction-at-inference structure is what makes the solver's phase-correct dynamics available *where the error compounds* (every rollout step), whereas Mode 1 must inject all of that signal through a single backprop path into one intermediate frame.

> **Cross-reference:** Q1's negative answer raises the value of Q4's non-differentiable variant question: if Mode 1 *is* attempted, does the critic require a differentiable solver? SoL-DeepONet's one-step critic needs no backprop-through-solver (the solver maps the current prediction forward and the network matches the *next* GT frame — a plain L2 on a solver-generated target). An intermediate-frame K-step critic, by contrast, *does* need gradients to reach the intermediate frame, so it requires either a differentiable solver, an adjoint, or Frezat's emulator trick (Q4).

---

### Q2 — Mode 2 canonical recipe and late-time pointwise numbers

**Conclusion: the recipe is stable across the lineage; the late-time pointwise gains are real but velocity-form/decaying-mixing-layer, and the forced-KF vorticity-form late number is unreported.**

**The lineage and what each contributed:**

- **Um et al., Solver-in-the-Loop (NeurIPS 2020).** Founding work. Network corrects an iterative PDE solver inside the training loop with differentiable physics; "SOL-n" trains with n steps of differentiable physics per ADAM iteration. Establishes that interactive (solver-in-loop) training beats non-interacting (NON) and pre-computed (PRE) corrections; look-ahead (more unrolled steps) monotonically improves velocity-error improvement on unsteady-wake and buoyancy-driven flows. [Um et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/43e4e6a6f341e00671e123714de019a8-Abstract.html)
- **Kochkov et al. (PNAS 2021, JAX-CFD).** Learned interpolation inside a differentiable finite-volume solver, trained with cumulative pointwise MSE over unrolled steps; matches DNS accuracy at **8–10× coarser** grid (40–80× speedup), **remains stable over long simulations**, and generalizes to unseen forcings/Re. This is the closest canonical work to forced 2D Kolmogorov flow (it *is* 2D turbulence with Kolmogorov forcing) but reports vorticity-*correlation*-time and energy-spectrum fidelity, not a clean early-vs-late pointwise relL2 split. [Kochkov PNAS 2021](https://www.pnas.org/doi/10.1073/pnas.2101784118), [ar5iv](https://ar5iv.labs.arxiv.org/html/2102.01010)
- **List, Chen, Thuerey (JFM 2022).** The most detailed loss/horizon study. CNN learns a **corrective forcing** entering PISO's implicit predictor step; combined loss `L_T = λ₂L₂ + λ_E L_E + λ_S L_S + λ_MS L_MS`:
  - `L₂`: pointwise velocity L2 on each unrolled step;
  - `L_E`: **log-spectral distance** of spectral kinetic energy, `√∫_k log(Ẽ_s(k)/E_sτ(k))² dk`, to recover fine-scale low-amplitude content the L2 neglects;
  - `L_S`: rate-of-strain L1 (to preserve the resolved strain that drives inter-scale energy transfer);
  - `L_MS`: multi-step mean (Reynolds-average over the unrolled window), **only for statistically steady cases**.
  Weights for decaying turbulence: `(λ₂, λ_E, λ_S, λ_MS) = (10, 5e-2, 1e-5, 0)`. Unroll horizon studied up to **60 steps**; best model used **30** ≈ 1.96 integral timescales. [List JFM 2022 §2.4](https://arxiv.org/pdf/2202.06988)
- **Frezat et al. (a-posteriori QG turbulence, 2022; gradient-free emulator, JAMES 2024).** Online (a-posteriori) loss `L_on({y(t)}, {Φ_θ^t(ȳ₀)})` over a temporal window via the solver flow operator. The 2024 paper's key engineering contribution: replace the non-differentiable solver `g` with a trained differentiable neural emulator `E≈g`, train the closure through `E`; a two-step sequential scheme + "subgrid" loss re-targeting controls the emulator-mismatch bias. [Frezat JAMES 2024](https://arxiv.org/html/2310.19385v4)
- **Sirignano & MacArt (DPM, adjoint online training).** Embeds a NN in the filtered NS equations, trains via the **adjoint** ("deep learning PDE model"), a-posteriori statistics as target; demonstrated on 3D isotropic/jet LES. 2025 successor OGF (online gradient flow) optimizes steady-state time-average statistics directly. [DPM arXiv 1911.09145](https://arxiv.org/abs/1911.09145), [OGF 2025](https://arxiv.org/pdf/2507.05149)
- **Shankar et al. (Differentiable Turbulence, PRF 2025).** DL embedded in a differentiable LES solver; finds small-scale non-local features most critical, large-scale features improve **pointwise** a-posteriori field accuracy; "hybrid solver-in-the-loop offers the ideal balance." 2D turbulent flow. [Shankar PRF 2025](https://arxiv.org/html/2307.03683v2)
- **2024–2026 additions:** shell-model solver-in-the-loop (PhysRevFluids 10.044602; code `andremfreitas/sm_loop`); "Solver-Aware Turbulence Closures from Nudged LES Dynamics" (2026, uses continuous-data-assimilation/nudging to enable a-priori-style training of solver-aware closures, addressing the filter-mismatch instability). [shell SITL](https://link.aps.org/doi/10.1103/PhysRevFluids.10.044602), [Nudged closures 2026](https://arxiv.org/html/2604.23874v1)

**The verified late-window pointwise numbers (List JFM 2022, Table 1, decaying isotropic, MSE at t₁≈5 t̂ and t₂≈40 t̂):**

| Model | Loss | Steps | MSE @ t₁ (early) | MSE @ t₂ (late) |
|---|---|:--:|:--:|:--:|
| NoModel | – | – | 2.78e-3 | 0.057 |
| LES (Smagorinsky) | – | – | 2.69e-3 | 0.051 |
| NN supervised | L_T | 1 | 1.52e-3 | **0.369** (diverges) |
| NN 1-step | L_T | 1 | 1.65e-3 | 0.046 |
| NN 10-step | L₂ | 10 | 4.23e-4 | 0.018 |
| NN 30-step | L_T | 30 | 4.09e-4 | 0.021 |

The supervised (no-differentiable-solver, correction-after-step) model **diverges** at late time (0.369), while differentiable unrolled models cut late MSE ~2.7× below no-model — the cleanest single demonstration that the solver-in-loop signal is what buys late-time stability/accuracy. Spatial mixing-layer reaches up to ~10× (late velocity MSE 0.0243→0.0025, per the prior report's banked read).

**Does the gain transfer to forced-KF vorticity-form low-k late-time?** **Unresolved in the literature.** The List numbers are velocity-form decaying/mixing-layer; Kochkov is forced-KF but reports correlation/spectra, not late pointwise relL2; Shankar is 2D but LES-field pointwise, not your low-k vorticity metric. The honest expectation (carried from the prior report) is the *middle* of List's 2.7–10× range, but **no source pins it**. This is the principal numerical uncertainty and it cannot be closed from existing publications.

> **Cross-reference:** Q2's recipe directly informs Q4 — the `L_E` spectral-energy and `L_S` strain losses are precisely the "phase/fine-scale-aware" terms the prior report found pure pointwise L2 cannot supply, and List explicitly states L2 "can neglect the time evolution of certain fine scale, low amplitude features." This is the literature's version of your low-k phase defect, and the fix is the solver-unrolled `L_E`/`L_S` combination, not L2 reweighting (which `ranking_v2.md` banked dead).

---

### Q3 — Code: what is reusable

**Conclusion: the correction-hybrid (Mode 2) is exceptionally well-supported by maintained code; PICT is the best PyTorch-native fit, APEBench/Exponax the best JAX/spectral fit.** Ranked by closeness to your PyTorch/Lightning + vorticity + Kolmogorov stack:

| Repo | Lang/AD | Solver type | Kolmogorov forcing | Differentiable step exposed? | Has unrolled-correction training loop? | Stars/maint. | License |
|---|---|---|---|:--:|:--:|:--:|:--:|
| **tum-pbs/PICT** ([repo](https://github.com/tum-pbs/PICT)) | **PyTorch**, GPU | PISO FVM (incompressible NS) | constructible (forcing term) | ✅ yes (gradients through solver) | building block (used to learn 2D/3D SGS) | new 2025 | OA |
| **tum-pbs/unrolling** ([repo](https://github.com/tum-pbs/unrolling)) | TF1.14 + diff-piso | PISO (Kolmogorov testcase) | ✅ **Kolmogorov flow experiment built in** | ✅ WIG; ✅ NOG | ✅ **curriculum step_counts, warmup, detach_interval, WIG/NOG flags** | List 2024 code | MIT |
| **tum-pbs/differentiable-piso** ([repo](https://github.com/tum-pbs/differentiable-piso)) | TF1.14 + ΦFlow | PISO FVM, 2nd order | mixing-layer/decaying (forcing addable) | ✅ yes | ✅ `*_differentiable_training.py`: loss weights, unroll steps, backprop subrange | List 2022 code | MIT |
| **tum-pbs/apebench** + **Exponax** ([apebench](https://github.com/tum-pbs/apebench), [exponax](https://github.com/Ceyron/exponax)) | **JAX** | **pseudo-spectral ETDRK** (incl. 2D/3D Navier–Stokes) | NS available; KF forcing constructible | ✅ yes (fully diff JAX) | ✅ explicit **"solver-in-the-loop correction setting"** + Trainax unrolled trainers | NeurIPS 2024, active | MIT |
| **google/jax-cfd** ([repo](https://github.com/google/jax-cfd)) | JAX | FVM (Kochkov learned interpolation) | ✅ Kolmogorov is the canonical demo | ✅ yes | `[ml]` submodule: learned-interpolation unrolled training | Kochkov, established | Apache-2 |
| **tum-pbs/Solver-in-the-Loop** ([repo](https://github.com/tum-pbs/Solver-in-the-Loop)) | ΦFlow | NS (wake, advection-diffusion) | not built in | ✅ yes | ✅ SOL-n training | Um 2020, established | — |
| **tum-pbs/PhiFlow** ([repo](https://github.com/tum-pbs/phiflow)) | **PyTorch/JAX/TF** | general diff PDE framework | constructible | ✅ yes (diff operators, sparse solves) | framework (build your own loop) | mature, widely used | MIT |
| **neuraloperator/physics_informed** ([repo](https://github.com/neuraloperator/physics_informed)) | PyTorch | — (FNO + PDE-residual loss) | ✅ `solver/kolmogorov_flow.py` | residual-based (not solver-integration) | PINO train_PINO3d | established | — |

**Specific module pointers:**
- **Unrolled-correction training loop with curriculum, WIG/NOG, gradient-cut:** `tum-pbs/unrolling` — `kolm_train.py` (`--gradient`/`--nogradient`, `--step_counts 1 2 4`, `--start_learning_rate`, curriculum) and KS `ks_train.py` (`--predhori`, `--warmup_steps` = pushforward, `--detach_interval` = gradient truncation, `--learning-task predict|correct`). This is the **most directly reusable Mode-2 reference** and it ships a Kolmogorov-flow correction experiment with Re=600/1000 extrapolation evaluation.
- **Differentiable solver step (PyTorch):** `tum-pbs/PICT` — `piso_step`-style differentiable PISO in PyTorch with GPU; verified on lid-driven cavity + turbulent channel; used to learn a stable 3D SGS model from reference statistics. This is the **lowest-friction path to a differentiable NS step in your stack** (PyTorch, no TF1.14 legacy).
- **Loss/horizon training script:** `tum-pbs/differentiable-piso` — `spatial_mixing_layer_differentiable_training.py` exposes loss functions + weighting, number of unrolled steps, and the **backpropagation subrange** (gradient-stopping) knob, plus `piso_step()`.
- **JAX solver-in-the-loop correction:** APEBench wraps Exponax (`P_h` reference simulator) + Trainax (unrolled trainers) + PDEquinox (FNO/U-Net/ResNet); the README states the simulator "can be embedded into emulator training (e.g., for a 'solver-in-the-loop' correction setting)." Exponax provides differentiable 2D/3D Navier–Stokes ETDRK steppers — the closest analogue to a **vorticity-form pseudo-spectral** operator pipeline.

**For Mode 1 specifically: there is no drop-in.** No repo implements an intermediate-frame solver critic for a single-shot operator. You would build it from a differentiable NS step (PICT in PyTorch, or Exponax in JAX) + a custom loss that integrates the operator's t/2 frame forward K steps and matches GT late frames. SoL-DeepONet has no public code linked. The FNO codebase (`neuraloperator`) gives you the operator half but its physics loss is PDE-residual (banked dead in your prior work), not solver-integration.

> **Cross-reference:** Q3 confirms Q4's cost conclusion — `tum-pbs/unrolling` exposes NOG as a first-class flag, so the non-differentiable variant is not just theoretically cheaper, it is already implemented and tested on Kolmogorov flow. Your existing `NavierStokes2d` could be wrapped NOG-style without an adjoint.

---

### Q4 — Engineering cost and failure modes

**Conclusion: a non-differentiable solver in the loop is the cheap, high-yield entry point; differentiability is a ~1.4× refinement, not a prerequisite; gradient stability is Lyapunov-bounded and has three established stabilizers.**

**Differentiable vs non-differentiable (List 2024, >3000 models).** The study decomposes unrolling into two effects: *forward data-shift reduction* (the network sees its own attractor at training time) and *long-term gradients*. Findings:
- **NOG (non-differentiable, ∂g/∂g := 0, gradients cut at the solver):** captures the data-shift benefit; **up to 4.5× over pure prediction**, ~23% average improvement, *using existing non-differentiable solvers*.
- **WIG (differentiable):** best accuracy, but adds only **~38% over one-step** on top of NOG; long-term gradients matter most in *correction* setups (network corrects a solver) — precisely the hybrid regime.
- Both are **Lyapunov-horizon-bounded**: NOG gradient inaccuracy grows ∝ m² (approximation error); WIG gradients explode exponentially in chaotic systems (Mikhaeil bound). Hence **short curricula suffice** — List uses m=[1,2,4] for Kolmogorov, m=3 for KS.

> "A non-differentiable but unrolled training setup supported by a numerical solver in a correction setup can yield substantial improvements over a fully differentiable prediction setup not utilizing this solver." — [List et al., JCP 2024](https://arxiv.org/html/2402.12971v2)

**Critical constraint on NOG (does it apply to Mode 1?).** List 2024 is explicit: NOG "only works without implementing additional differential operations when the ground truth information is available directly at the network output" — true for the correction form `u^{t+1}=f_θ(S(u^t))` (Mode 2), where the network output *is* the next state. **For Mode 1 this constraint bites:** an intermediate-frame critic needs gradients to flow *back through the solver into the t/2 frame*, so NOG (gradients cut at the solver) cannot deliver the Mode-1 signal. **Mode 1 therefore requires a differentiable solver, an adjoint, or Frezat's neural-emulator surrogate.** This directly answers the prompt's precise question: your existing `NavierStokes2d` would need an adjoint/differentiable reimplementation for Mode 1, *unless* you (a) use a one-step solver-generated-target critic à la SoL-DeepONet (plain L2 on a solver-advanced target, no backprop-through-solver — but that critiques the next frame, not an integrated-K future), or (b) train a differentiable neural emulator of `NavierStokes2d` and backprop through it (Frezat). For Mode 2, NOG with your existing solver is sufficient.

**Three alternatives to a differentiable solver (for when adjoint reimplementation is unacceptable):**
1. **Neural emulator surrogate (Frezat 2024):** train `E≈g` differentiable, backprop closure through `E`; recovers "most of the benefits of online strategies without computing the gradient of the original solver." Cost: train a second network; bias-control via subgrid-loss re-targeting. [Frezat JAMES 2024](https://arxiv.org/html/2310.19385v4)
2. **Zeroth-order gradient estimation (NeurIPS 2023 ML4PS):** differentiate a black-box solver by forward-propagation finite differences; outperforms frozen-input baselines but underperforms exact AD. [Black-box PDE solvers](https://neurips.cc/virtual/2023/76085)
3. **NOG (List 2024):** for correction setups only.

**Gradient-stability stabilizers (all documented to work):**
- **Gradient-stopping into non-overlapping subranges** (List JFM 2022 §6): keeps long-term *information* from all unrolled steps but stops *gradient* propagation across many steps — avoids explosion. Exposed as `--detach_interval` in `tum-pbs/unrolling`.
- **Warm-up / pushforward** (no-loss unroll steps before the loss window): stabilizes; `--warmup_steps`.
- **Short curriculum m=[1,2,4]** with per-stage learning rates; horizon should **approach but not exceed the integral timescale** (List: best at m≈integral-timescale; "unrolling for too many steps was associated with diminishing returns and even negative effects").

**Known failure modes / negative results:**
- **Supervised (no solver in loop) diverges at late time** — List Table 1: NN-supervised late MSE 0.369 vs no-model 0.057; the 1-step model also becomes unstable beyond the training horizon. Pure prediction unrolling (no solver) is List's *weakest* variant — which matches your `wall.md` chain-Variant-A flat result (network-fed chaining ≈ solver-free unrolling).
- **Over-long unrolling** hurts (diminishing/negative returns past the Lyapunov/integral timescale).
- **a-priori-trained closures are unstable a-posteriori** (filter mismatch) — the recurring motivation for solver-in-loop; the 2026 nudged-closures paper exists specifically to fix this instability.
- **Emulator-mismatch bias** (Frezat): a poor neural emulator injects bias absorbed by the closure; requires careful loss re-targeting.

> **Cross-reference:** Q4 closes the loop with Q1 — the reason Mode 1 is hard *and* unwritten is the same reason it is risky: it demands the differentiability NOG avoids, while concentrating the entire solver signal into one intermediate frame's gradient rather than spreading it across an inference-time rollout (Mode 2). Mode 2's robustness comes precisely from the solver being present where errors compound.

---

## Comparison: the candidate couplings, ranked

| Criterion | **Mode 1 — intermediate-frame critic, one-shot deploy** | **Mode 2a — diff. solver correction hybrid** | **Mode 2b — non-diff (NOG) correction hybrid** |
|---|---|---|---|
| Published precedent (your regime) | ❌ none (SoL-DeepONet = 1D Burgers, 1-step) | ✅ Um/Kochkov/List/Shankar | ✅ List 2024 (Kolmogorov, NOG) |
| One-shot inference preserved | ✅ yes (the goal) | ❌ no (solver at inference) | ❌ no (solver at inference) |
| Differentiable solver required | ✅ **yes** (gradients must reach t/2 frame) | ✅ yes | ❌ **no** (existing solver reusable) |
| Best documented late-time pointwise evidence | none (1D Burgers relL2 5.88%, *worse* than PINN baseline) | decaying 2.7×, mixing-layer ~10× (velocity-form) | up to 4.5× over prediction (incl. Kolmogorov) |
| Forced-KF vorticity-form late number | none | none (transfer only) | none (transfer only; closest flow) |
| Reusable code | none (build from PICT/Exponax) | differentiable-piso, PICT, jax-cfd, APEBench | **tum-pbs/unrolling (Kolmogorov, NOG flag)** |
| Engineering cost | high (diff solver / adjoint / emulator) + novel | high (diff NS solver) | **medium (wrap existing `NavierStokes2d`)** |
| Risk | high (unproven) | low (proven, but relaxes one-shot) | low–medium (proven, relaxes one-shot) |

**What the comparison reveals.** There is a hard trade-off between the two things you value most. **One-shot preservation lives only in Mode 1, which has no precedent, no code, and requires a differentiable solver — the highest-risk cell in the table.** All the documented late-time pointwise evidence and all the reusable code live in Mode 2, which abandons one-shot inference (it becomes a short G1-relaxation solver+correction rollout). The cheapest, best-supported, lowest-risk option is **Mode 2b (NOG)** — it reuses your existing solver without an adjoint and is the only cell with a *forced-Kolmogorov* late-time demonstration and ready Kolmogorov code (`tum-pbs/unrolling`). The surprise relative to `ranking_v2.md`, which ranked solver-in-loop as a uniformly high-cost Tier-3 "resort": the non-differentiable Kolmogorov correction variant is **already implemented and Kolmogorov-tested**, so its true setup cost is lower than the ranking assumed.

The strategic read: if the thesis can tolerate a short solver+correction rollout at inference (G1 relaxation), Mode 2b is the dominant, de-risked path. If one-shot inference is non-negotiable, Mode 1 is a genuine research contribution but you are the first to try it in 2D turbulence, with only a one-step 1D-Burgers analogue (SoL-DeepONet) and the failed network-fed chaining (`wall.md`) as priors — and SoL-DeepONet's own result (solver critic *less* accurate than the PINN-residual it replaced) is a caution, not an encouragement.

---

## Contradictions & Debates

- **Is a differentiable solver necessary? (List 2022 framing vs List 2024 finding.)** List 2022 builds its entire argument on the differentiable PISO solver being the enabler. List 2024 appears to contradict this by showing NOG (non-differentiable) captures most of the gain. **Resolution: not a contradiction — a decomposition.** List 2024 separates the *forward* data-shift benefit (achievable non-differentiably, the heavy lifter) from *long-term gradients* (the differentiable premium, ~38%). For correction setups the differentiable premium is real but small; for *Mode 1* the differentiable requirement returns in full because gradients must reach an intermediate frame (the NOG "output = ground truth" condition fails). [List 2022](https://arxiv.org/pdf/2202.06988), [List 2024](https://arxiv.org/html/2402.12971v2)

- **"Solver-in-the-loop" deployment mode is overloaded.** DPM/Frezat/Shankar/Kochkov keep the solver at inference (Mode 2); SoL-DeepONet deploys the network alone (Mode 1). The same phrase covers both. **Resolution:** only SoL-DeepONet is true Mode 1, and only on 1D Burgers with a one-step critic — so the phrase's prevalence overstates how much Mode-1 evidence exists. Audit every "solver-in-the-loop" hit for *inference-time* solver presence.

- **Accuracy vs training-cost in SoL-DeepONet.** SoL-DeepONet is *less accurate* (5.88% vs 1.38%) than PI-DeepONet but 100× faster to train. **Resolution:** this is evidence the solver critic is a *cheaper* substitute for the PINN residual, not a *more accurate* one — relevant because your motivation for Mode 1 is accuracy (phase-correct signal), and the one published analogue did not deliver an accuracy win.

---

## Uncertainties & Gaps

- ⚠️ **Forced-Kolmogorov vorticity-form late-time pointwise number under solver-in-loop is genuinely unreported.** List's numbers are velocity-form decaying/mixing-layer; Kochkov is forced-KF but correlation/spectra; Shankar is 2D LES-field. The transfer to your low-k vorticity late-relL2 is high-confidence-positive in *direction* but **numerically unpinned**. This gap is inherent to the literature, not closable by more searching.
- ⚠️ **Mode 1 on a 2D-turbulence single-shot full-trajectory operator with intermediate-frame critic is absent.** Confidence in the *absence* is high after exhaustive search; confidence in *whether it would work* is necessarily low (no data). SoL-DeepONet (1D Burgers, one-step, ICLR 2026 workshop) is the only neighbour and is recent/lightly-reviewed.
- ⚠️ **SoL-DeepONet has no linked public code** — its reuse value is conceptual only.
- ⚠️ **PICT is new (2025)** — PyTorch-native and the best stack fit, but less battle-tested than differentiable-piso/jax-cfd; validate its gradients on a lid-driven cavity before trusting them in a training loop.

---

## Recommendations

### Primary
**If one-shot inference can be relaxed to a short solver+correction rollout (G1 relaxation), pursue Mode 2b — non-differentiable (NOG) solver-in-loop correction on the forced-Kolmogorov vorticity field — using `tum-pbs/unrolling` as the reference implementation and your existing `NavierStokes2d` (wrapped NOG-style, gradients cut at the solver).** Rationale: it is the only candidate with (i) a documented forced-Kolmogorov late-time pointwise improvement (List 2024, up to 4.5× over prediction), (ii) ready Kolmogorov code with WIG/NOG/curriculum/warmup/gradient-cut already exposed, and (iii) no adjoint reimplementation cost. Use the List loss `L₂ + L_E (log-spectral) + L_S (strain)`, curriculum m=[1,2,4] matched to your integral timescale, gradient-cut subranges. Add WIG later only if NOG plateaus (≤~38% expected upside). For a PyTorch-native differentiable step (for the WIG upgrade), adopt **PICT**.

### Alternative
**If one-shot inference is non-negotiable, attempt Mode 1 as a research contribution, but de-risk it first with the SoL-DeepONet pattern and a differentiable solver.** Concretely: (a) build a differentiable NS step (PICT in PyTorch, or Exponax in JAX for a vorticity-spectral pipeline); (b) start with a *one-step* solver-generated-target critic on the intermediate frame (no backprop-through-solver — cheap, SoL-DeepONet-style) as a sanity check; (c) only then escalate to the *integrate-K-steps* differentiable critic that requires gradients into the t/2 frame. If you refuse a differentiable solver, use Frezat's neural-emulator-of-`NavierStokes2d` to supply the gradient. Treat the SoL-DeepONet caution seriously: its solver critic was *cheaper but less accurate* than the PINN residual — budget for the possibility that the midpoint critic does not beat your current operator.

### Not Recommended
- **Mode 1 without a differentiable solver, an emulator, or a one-step target** — there is no mechanism to get the phase-correct gradient into the intermediate frame; NOG's "output = ground truth" condition fails for an intermediate-frame critic.
- **Fully differentiable T-step (T=64) rollout as the first attempt** — List 2024 shows the differentiability premium is ~38% and long differentiable chains explode in chaotic systems; short curricula on a non-diff solver should be tried first.
- **Relying on PINO's PDE-residual physics loss as the "solver coupling"** — it is the pure-loss family already banked dead in `ranking_v2.md`, not a solver-integration critic.

---

## Methodology

- **Depth:** exhaustive
- **Search rounds:** 6 checkpointed rounds, 12 search batches (~40 queries), 5 deep primary-source extractions (List-2024 JCP, List-2022 JFM, Frezat-2024, APEBench/unrolling/differentiable-piso repos, SoL-DeepONet)
- **Final confidence:** 95% (Q3 96%, Q2/Q4 95%, Q1 94% — the 94% reflects high confidence in a *negative* finding plus an inherent numerical gap)
- **Sub-questions:** 4 defined, 4 answered (Q1 answered as a decisive negative)
- **Multi-hop chains used:** entity expansion (Um→Kochkov→List 2022→List 2024 taxonomy→NOG/WIG; List→differentiable-piso→PICT→unrolling repos); deployment-mode audit (every "solver-in-the-loop" hit classified by inference-time solver presence to separate Mode 1 from Mode 2); contradiction resolution (differentiability-necessity decomposed via List 2024); source triangulation (all Tier-1 peer-reviewed/arXiv + maintained repos, cross-checked against banked `research_late-time-wall-stress-test` reads)
- **Key challenge:** the central Q1 result is a negative (Mode 1 essentially unwritten); confirming absence required auditing the overloaded "solver-in-the-loop" term by deployment mode and chasing the multiple-shooting/distillation framings by name.

---

## Sources

| # | Title | URL | Date | Credibility |
|---|-------|-----|------|:-----------:|
| 1 | List/Chen/Thuerey, Learned Turbulence Modelling w/ Differentiable Fluid Solvers (JFM 2022) | https://arxiv.org/pdf/2202.06988 | 2022 | ⭐ Tier 1 |
| 2 | List et al., Differentiability in Unrolled Training (JCP/CMAME 2024) | https://arxiv.org/html/2402.12971v2 | 2024 | ⭐ Tier 1 |
| 3 | Um et al., Solver-in-the-Loop (NeurIPS 2020) | https://proceedings.neurips.cc/paper_files/paper/2020/hash/43e4e6a6f341e00671e123714de019a8-Abstract.html | 2020 | ⭐ Tier 1 |
| 4 | Kochkov et al., ML-accelerated CFD (PNAS 2021) | https://www.pnas.org/doi/10.1073/pnas.2101784118 | 2021 | ⭐ Tier 1 |
| 5 | Kochkov et al. (ar5iv full text) | https://ar5iv.labs.arxiv.org/html/2102.01010 | 2021 | ⭐ Tier 1 |
| 6 | Frezat et al., Gradient-free online learning w/ neural emulators (JAMES 2024) | https://arxiv.org/html/2310.19385v4 | 2024 | ⭐ Tier 1 |
| 7 | Frezat et al., A posteriori learning for QG turbulence (JAMES 2022) | https://arxiv.org/abs/2204.03911 | 2022 | ⭐ Tier 1 |
| 8 | Sirignano et al., DPM deep-learning PDE model | https://arxiv.org/abs/1911.09145 | 2019 | ⭐ Tier 1 |
| 9 | Hickling/MacArt/Sirignano, Online Gradient Flow (OGF) | https://arxiv.org/pdf/2507.05149 | 2025 | ⭐ Tier 1 |
| 10 | Shankar et al., Differentiable Turbulence (PRF 2025 / arXiv 2307.03683) | https://arxiv.org/html/2307.03683v2 | 2025 | ⭐ Tier 1 |
| 11 | SoL-DeepONet (ICLR 2026 Workshop AI & PDE) | https://openreview.net/forum?id=JZldiTPkpl | 2026 | 🔵 Tier 2 |
| 12 | APEBench (NeurIPS 2024) | https://arxiv.org/html/2411.00180v1 | 2024 | ⭐ Tier 1 |
| 13 | tum-pbs/apebench (repo) | https://github.com/tum-pbs/apebench | 2024 | ⭐ Tier 1 |
| 14 | Ceyron/exponax (repo) | https://github.com/Ceyron/exponax | 2024 | 🔵 Tier 2 |
| 15 | google/jax-cfd (repo) | https://github.com/google/jax-cfd | 2021 | ⭐ Tier 1 |
| 16 | tum-pbs/differentiable-piso (repo, List 2022 code) | https://github.com/tum-pbs/differentiable-piso | 2022 | ⭐ Tier 1 |
| 17 | tum-pbs/unrolling (repo, List 2024 code; Kolmogorov + NOG/WIG) | https://github.com/tum-pbs/unrolling | 2024 | ⭐ Tier 1 |
| 18 | tum-pbs/PICT — differentiable PyTorch PISO (arXiv 2505.16992) | https://arxiv.org/abs/2505.16992 | 2025 | ⭐ Tier 1 |
| 19 | tum-pbs/PICT (repo) | https://github.com/tum-pbs/PICT | 2025 | 🔵 Tier 2 |
| 20 | tum-pbs/PhiFlow (repo) | https://github.com/tum-pbs/phiflow | mature | ⭐ Tier 1 |
| 21 | tum-pbs/Solver-in-the-Loop (repo, Um 2020 code) | https://github.com/tum-pbs/Solver-in-the-Loop | 2020 | ⭐ Tier 1 |
| 22 | neuraloperator/physics_informed (PINO repo, kolmogorov_flow.py) | https://github.com/neuraloperator/physics_informed | 2021 | ⭐ Tier 1 |
| 23 | Differentiable Multiple Shooting Layers (NeurIPS 2021) | https://arxiv.org/abs/2106.03885 | 2021 | ⭐ Tier 1 |
| 24 | When Black-box PDE Solvers Meet Deep Learning (zeroth-order, NeurIPS 2023 ML4PS) | https://neurips.cc/virtual/2023/76085 | 2023 | 🔵 Tier 2 |
| 25 | Solver-in-the-loop closure of shell models (PhysRevFluids 2025) | https://link.aps.org/doi/10.1103/PhysRevFluids.10.044602 | 2025 | ⭐ Tier 1 |
| 26 | andremfreitas/sm_loop (shell-model SITL code) | https://github.com/andremfreitas/sm_loop | 2025 | 🔵 Tier 2 |
| 27 | Deep Learning of Solver-Aware Turbulence Closures from Nudged LES (2026) | https://arxiv.org/html/2604.23874v1 | 2026 | 🔵 Tier 2 |
| 28 | Scientific ML for closure models review (a-priori vs a-posteriori) | https://www.aimsciences.org/article/doi/10.3934/fods.2024043 | 2024 | ⭐ Tier 1 |
| 29 | Spectral-Refiner (FNO + negative-Sobolev fine-tuning, ICLR 2025) | https://arxiv.org/html/2405.17211v2 | 2025 | ⭐ Tier 1 |
| 30 | PINO (Physics-Informed Neural Operator, ACM/JMLR) | https://dl.acm.org/doi/10.1145/3648506 | 2024 | ⭐ Tier 1 |
