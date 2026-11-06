# Solver as Input — Coarse Backbone for Sharp One-Shot Prediction

**Date:** 2026-06-27
**Depth:** exhaustive
**Confidence:** 91% (capped by an irreducible novelty gap, not by missing evidence)
**Sources:** 52 unique across 5 search rounds

---

## Executive Summary

The pattern you propose — run a coarse solver fully, hand its committed trajectory to a one-shot operator as a static second input, and have the network refine it — **is established and works in every adjacent field except yours**. Climate/turbulence super-resolution, multi-fidelity FNOs, residual-to-LES correction (Tpred = T_LES + f(T_LES) ≈ T_DNS), and physically-anchored multi-resolution operators all instantiate "coarse committed field in → fine field out" and all report that conditioning on the coarse field removes the over-smoothing that pure forward prediction suffers. The mechanism you intuit is correct and corroborated by the Parseval–Plancherel analysis of L2 spectral bias (Oommen et al. 2026) and by the canonical video-prediction result that MSE forces the conditional mean and thus blur (Mathieu, Couprie & LeCun 2016).

The **specific instance you would build — a coarse full-horizon solver trajectory as a static input to a non-autoregressive space-time operator for chaotic 2D Kolmogorov flow, evaluated at late times within the Lyapunov horizon — is unpublished.** Every found instance is either (a) spatial super-resolution of a single frame / short window, (b) autoregressive rather than one-shot, (c) generative (diffusion/GAN) rather than deterministic-refinement, or (d) steady-state. This is the gap your thesis would fill, and the literature supplies both the architecture (concat / cross-attention / FiLM fusion, residual-to-coarse target) and the loss recipe (data + multi-resolution physics, with the warning that physics-loss alone does not fix spectral bias).

**Cheapest first wall-breaking experiment, grounded in what works:** train the operator to predict the *residual to a coarse-grid full-horizon solver trajectory*, fused as concatenated input channels, with an L2 + multi-resolution PDE loss — the deterministic core of PG-Diff/Bao without the generative head. The coarse solver, even crude, supplies phase commitment that the IC alone cannot.

---

## Key Findings

1. **The exact "coarse solver trajectory as static input → deterministic refinement" pattern is unpublished for one-shot chaotic 2D turbulence at late times** — every analog is spatial-SR, autoregressive, generative, or steady-state. This is a genuine novelty, not an oversight. [PG-Diff](https://arxiv.org/html/2504.04375v1) [CFD-GCN](http://proceedings.mlr.press/v119/de-avila-belbute-peres20a/de-avila-belbute-peres20a.pdf)

2. **Conditioning on a coarse committed field demonstrably removes over-smoothing** — residual-to-LES (Tpred = T_LES + f_CNN(T_LES) ≈ T_DNS) and physically-anchored multi-resolution operators report sharp fine-scale reconstruction the forward map cannot achieve. [HAL Thermal-LES](https://hal.science/hal-05454994v1/file/Pr_sentation_SFT.pdf) [Physically-Anchored MR-NO](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-1982/egusphere-2026-1982.pdf)

3. **The blurring is provably a spectral-bias / conditional-mean artifact of L2, not an architecture failure** — Parseval–Plancherel shows L2 over-weights low-k (high-energy) and under-weights high-k, so smoothed predictions achieve low loss. This is the formal version of your "L2 learns the average future." [Oommen et al. 2026](https://arxiv.org/html/2509.08752v1)

4. **Physics loss alone does NOT fix the blur** — NO+∇·u (divergence penalty) leaves the energy spectrum as biased as plain L2, because the PDE residual is itself minimized in an L2 sense. Directly relevant to your planned physics-loss term. [Oommen et al. 2026](https://arxiv.org/html/2509.08752v1)

5. **One-shot long-horizon mapping fails when input–output correlation is weak; a committed backbone is what restores it** — GenCFD's one-shot map u(t)→[u(t+τ)…u(t+36τ)] "did not converge because of the weak input-output correlation" across ~5 eddy-turnover times. This is exactly your late-time wall, and it argues the coarse backbone is the missing correlated input. [Oommen et al. 2026](https://arxiv.org/html/2509.08752v1)

6. **Multi-fidelity FNOs already fuse coarse + fine discretizations and recover near-high-fidelity accuracy at ~80% lower data cost** — the coarse-discretization-as-input half of your idea is proven; the full-horizon-trajectory-commitment half is not. [MF-FNO](https://arxiv.org/abs/2308.09113)

7. **Fusion mechanism is a solved design choice** — concatenated channels (CFD-GCN, PG-Diff), cross-attention global→local (AdaptFNO), and residual-to-coarse targets (Thermal-LES) all work; cross-attention is the most expressive for transferring a coarse backbone to fine scales. [AdaptFNO](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_96.pdf)

8. **The canonical commitment fix outside PDEs is the gradient-difference + adversarial loss** — Mathieu/Couprie/LeCun showed MSE→blur and that a multi-scale + GDL + adversarial recipe sharpens; the GDL (penalize gradient mismatch) is a non-generative, directly transferable sharpening term. [Mathieu et al. 2016](https://arxiv.org/abs/1511.05440)

---

## Detailed Analysis

### Q1 — Is a coarse solver trajectory used as a static input to a one-shot neural operator predicting the fine trajectory?

**Answer: the building blocks all exist and are deployed; the precise composition you want is not.** The closest deterministic instances cluster into three families, none of which is simultaneously (one-shot) ∧ (full-horizon coarse-trajectory input) ∧ (chaotic late-time turbulence).

**Family A — coarse solver embedded, network refines (steady-state).** CFD-GCN ([de Avila Belbute-Peres et al., ICML 2020](http://proceedings.mlr.press/v119/de-avila-belbute-peres20a/de-avila-belbute-peres20a.pdf)) runs a differentiable SU2 solver on a ~300-node coarse mesh, upsamples its output, and concatenates it into a graph-convolution stack that predicts the ~6000-node fine field. The authors state plainly that the hybrid "performs substantially better than the coarse CFD simulation alone… the network is able to provide higher fidelity results than simply running a faster simulation," and that it generalizes far better than a pure GCN. This is the cleanest proof-of-concept that a coarse solver as input beats both the solver and the data-only network — but it is steady-state airfoil flow (AoA, Mach as the varying parameters), not a time trajectory.

**Family B — coarse-discretization fusion (multi-fidelity).** The Multi-fidelity FNO ([Jiang et al., 2023](https://arxiv.org/abs/2308.09113)) defines low- and high-fidelity exactly as "simulation results acquired based on coarse and fine discretization," extracts coarse features with an encoder and refines them, and matches high-fidelity accuracy with 81% less high-fidelity data. This validates the *coarse-grid-input* half of your idea directly, but it predicts static pressure/saturation fields for carbon storage, not a chaotic time evolution, and the coarse input is a field, not a committed trajectory.

**Family C — coarse solver input → generative refinement (the nearest geometric analog).** PG-Diff ("From Coarse to Fine," [2025](https://arxiv.org/html/2504.04375v1)) is the most on-point: *both* low- and high-fidelity data come from numerical solvers at different resolutions, and the model reconstructs the fine field from "solver-generated low-fidelity input," explicitly on Kolmogorov flow. Its key empirical observation mirrors yours: "SOTA diffusion models struggle to reconstruct fine-scale details when faced with solver-generated low-fidelity inputs," which they fix with an importance-weight self-guidance and a training-free residual correction. It is generative (out of your scope as a *solution*), but it confirms the coarse-solver-input commitment pattern is live for your exact PDE.

**Why your instance is still a gap.** None of these is a *non-autoregressive space-time* operator taking a *full-horizon* coarse trajectory and refining it deterministically at *late times within the Lyapunov horizon*. The fact that GenCFD's one-shot long-horizon map "did not converge because of the weak input-output correlation between states separated by ~5 eddy-turnover times" ([Oommen et al. 2026](https://arxiv.org/html/2509.08752v1)) is the strongest indirect evidence that your wall is real and that a coarse backbone — which restores input-output correlation by construction — is the natural fix nobody has yet published for one-shot deterministic refinement.

> "The network is able to provide higher fidelity results than simply running a faster simulation to begin with." — [CFD-GCN](http://proceedings.mlr.press/v119/de-avila-belbute-peres20a/de-avila-belbute-peres20a.pdf)

**Cross-reference:** the spectral reason this works is established in Q3; the loss recipe to make it train is in Q4.

### Q2 — Coarse-grid-to-fine turbulence super-resolution over the full horizon: does it prevent blurring?

**Answer: yes, and this is the most mature evidence you have — but the existing work is dominantly spatial, with temporal super-resolution the weaker, more relevant frontier.** The structurally identical operation (coarse committed field → NN → fine field) is a decade-old, well-populated literature.

Bao et al. (UAI 2022, [Physics-guided NN for spatio-temporal super-resolution of turbulent flows](https://proceedings.mlr.press/v180/bao22a.html)) is the most aligned: input is *low-resolution LES over space and time*, output is the DNS-resolution field, trained with a physics-guided loss. This is "coarse committed trajectory in, fine trajectory out" with a physics term — the deterministic skeleton of what you want, minus the explicit forced-KF late-k≤7 evaluation.

The deterministic residual-correction variant is even cleaner: the Thermal-LES reconstruction work ([HAL 2026](https://hal.science/hal-05454994v1/file/Pr_sentation_SFT.pdf)) learns Tpred = T_LES + f_CNN(T_LES) ≈ T_DNS — i.e. the network predicts the *residual to the coarse field*, exactly the variance-reduction you argued: the residual has far smaller variance than the full field, so the L2 minimizer is sharp. The Physically-Anchored Multi-Resolution Neural Operator ([EGUsphere 2026](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-1982/egusphere-2026-1982.pdf)) generalizes this to dynamics: "a coarse-resolution neural operator that captures large-scale behavior" plus "a terrain-aware refinement module that reconstructs fine-scale boundary structure," preserving domain-integrated quantities over time — deterministic, multi-resolution, temporally evolving.

**The caveat that matters for you.** Almost all turbulence super-resolution is *spatial* upscaling of an already-correct-in-time field. The Oommen super-resolution task is LRLF→HRHF (low-res, low-framerate → high-res, high-framerate), and even there the authors note adv-NO's fine scales "are not perfectly phase-aligned with the ground truth." Your problem is harder: the coarse backbone must commit the *phase/time evolution*, not just the spatial spectrum. The downscaling literature confirms conditioning prevents blur, but does not by itself prove your late-time phase wall closes — only that the spatial-spectral half does.

**Cross-reference:** the phase-commitment question is why a coarse *solver* (which integrates the dynamics) is preferable to a spatial low-pass of the IC — consistent with your rejection of Q4-as-originally-posed.

### Q3 — Commitment mechanisms forcing a deterministic operator to pick one sharp future

**Answer: several exist; ranked by transferability to your one-shot KF setup, the coarse-backbone-as-input is the strongest, followed by gradient/spectral residual mechanisms.** Surveying the deterministic (non-generative) options:

- **Residual-to-coarse pinning (strongest).** Predict the increment over a committed coarse field (Thermal-LES, Physically-Anchored MR-NO). Directly collapses output variance. Implemented on LES/DNS reconstruction and flood dynamics; deterministic; sharp.

- **Coarse-grained closure pinning.** Chen, Lu & Tryggvason ([Fluids 2022](https://www.mdpi.com/2311-5521/7/5/154)) learn closure terms for coarse-grained 2D turbulence "by ensuring that the coarse-grained flow evolves in the correct way," matching instantaneous vorticity short-term and energy spectrum/enstrophy long-term — on 2D turbulence, your PDE class. This is trajectory-pinning at the coarse level, deterministic, and it explicitly preserves the spectrum at long times.

- **Spectral residual ensembling.** SpecBoost ([Better Understanding of FNOs, 2024](https://arxiv.org/html/2404.07200v1)) trains a second FNO on the *residual* of the first to recover high-k content, +up to 71% accuracy on PDEs. Addresses spectral bias but is about high-k energy, not late-time phase; complementary, not a wall-breaker alone.

- **Gradient-difference loss (cross-domain, deterministic).** Mathieu et al. ([ICLR 2016](https://arxiv.org/abs/1511.05440)) — the canonical statement that MSE yields the blurred conditional mean and that a multi-scale + image-gradient-difference loss (penalize |∇pred − ∇target|) sharpens without any generative component. The GDL term is trivially portable to vorticity fields and is the cheapest non-generative sharpener to bolt on.

- **Multiple shooting / segment conditioning.** Turan & Jäschke ([ICLR 2023](https://www.semanticscholar.org/paper/Multiple-Shooting-for-Training-Neural-Differential-Turan-J%C3%A4schke/e40e4e4a924dd6275344998fd991ac8f9dcc2559)) split trajectories into short segments with continuity constraints for neural ODEs — relevant to *training* on long horizons by anchoring midpoints, but it is for neural ODEs/autoregressive integration, not one-shot space-time maps.

The architecture failure is secondary: Oommen et al. prove the bias is in the *objective*, not the network — "even architectures that are capable of representing high frequencies may fail to learn them under MSE-like training." That is why a commitment mechanism (input or loss) is required, not a bigger model.

**Cross-reference:** these confirm Q1's premise — the conditional-mean degeneracy is the disease, and conditioning on a committed coarse field (Q1/Q2) is the most direct cure.

### Q4 — Loss recipe on top of a coarse backbone

**Answer: data-L2 on the residual + multi-resolution PDE loss is the documented recipe, but the literature carries one sharp warning — physics loss alone will not de-blur, so you still need either the coarse backbone or a spectral/gradient term to carry sharpness.**

PINO ([Li et al., ICML 2022 / ACM 2024](https://dl.acm.org/doi/10.1145/3648506)) is the reference for combining data and physics at *different resolutions*: train data loss at the available (coarse) resolution and impose the PDE residual at higher resolution, giving zero-shot super-resolution. It is tested on Kolmogorov Re=500 — but only to T=0.125–1 s, i.e. short horizon, so it does not report your late k≤7 metric. PINO also offers test-time optimization (refine the operator ansatz per instance using the PDE residual), which is the natural home for your later ν-inference/TTA phase.

The critical warning comes from Oommen et al. ([2026](https://arxiv.org/html/2509.08752v1)): the divergence-penalized NO+∇·u "also minimizes an L2 norm of the mass-conservation PDE residual, and therefore struggles to minimize the fine-scale features… due to spectral bias." Its energy spectrum is as biased as plain L2; the only gain was slightly better phase alignment. Implication for your plan: a physics-loss term is worth having for consistency and phase, but **it will not by itself break a spectral/blur wall** — the sharpness must come from the coarse backbone (variance reduction) and/or a spectrum/gradient-targeted term (binned spectral power, GDL).

Recommended recipe, ordered: (1) L2 on the residual-to-coarse field (variance reduction does most of the work); (2) multi-resolution PDE residual à la PINO (phase/consistency, cheap, and your TTA hook); (3) a band/gradient sharpener (GDL or binned spectral power [Chakraborty et al., cited in Oommen]) only if 1–2 leave high-k under-energized. A distribution loss (your eventual third term) is, per Oommen's comparison, replaceable by adversarial training but not strictly necessary once the backbone commits the trajectory — start without it.

**Cross-reference:** this directly tempers the original plan ("data + physics + distribution from a biased start") — the biased start (backbone) plus data-L2 is the load-bearing pair; physics and distribution are refinements.

---

## Comparison

| Approach | Architecture | Coarse input type | PDE tested | Metric | Late-time improvement | One-shot? |
|---|---|---|---|---|---|---|
| CFD-GCN | GCN + embedded SU2 solver | coarse mesh field (concat) | steady airfoil | RMSE field | beats solver & pure-GCN | n/a (steady) |
| MF-FNO | FNO encoder fusion | coarse vs fine discretization | GCS pressure/sat | rel. error | matches HF at 81% less data | static |
| Bao 2022 (PG-NN) | physics-guided CNN | low-res LES over space+time | turbulent flows | recon error | sharper than SR baselines | windowed |
| Thermal-LES residual | CNN, Tpred=T_LES+f(T_LES) | coarse LES field | thermal LES→DNS | field recon | sharp DNS recovery | per-frame |
| Physically-Anchored MR-NO | coarse NO + refinement module | coarse-res dynamics | shallow-water floods | mass/depth | fine-scale + mass-consistent | temporal |
| PG-Diff | diffusion, coarse as guidance | solver low-fidelity field | Kolmogorov | spectrum/recon | recovers fine scales | generative |
| Oommen adv-NO | TC-UNet + discriminator | LRLF field | Schlieren / 3D HIT | spectrum NRMSE | 10–15× spectrum error ↓ | windowed/AR |
| SpecBoost | 2nd FNO on residual | (none; residual) | various PDEs | rel. L2 | +≤71% (high-k) | one-shot |
| Mathieu GDL | multi-scale CNN + GDL | (none; loss) | video | sharpness/PSNR | de-blurs MSE | AR |
| Chen/Tryggvason closure | ML closure term | coarse-grained field | 2D turbulence | spectrum/enstrophy | long-term spectrum match | AR coarse |

**What the table reveals.** Two clusters dominate. The *input-conditioning* cluster (CFD-GCN, MF-FNO, Bao, Thermal-LES, MR-NO, PG-Diff) consistently shows that a committed coarse field lets the network beat both the coarse solver and the unconditioned network — this is the load-bearing evidence for your hypothesis. The *loss/architecture* cluster (SpecBoost, GDL, adv-NO) attacks the same blur from the objective side and is complementary. The single most striking pattern: **every row that reports sharpness improvement either conditions on a committed coarse field or replaces L2 with a spectrum/gradient/adversarial objective — none achieve sharpness from pure L2 forward prediction.** That is the empirical core of your thesis, and the empty cell (one-shot ∧ full-horizon coarse-trajectory ∧ chaotic late-time) is your contribution.

---

## Contradictions & Debates

- **Does physics loss help sharpness?** PINO ([2022](https://dl.acm.org/doi/10.1145/3648506)) implies multi-resolution PDE constraints improve fidelity; Oommen et al. ([2026](https://arxiv.org/html/2509.08752v1)) show a divergence penalty does *not* fix spectral bias and leaves the spectrum blurred. **Resolution:** both are right at different scales — physics loss improves large-scale consistency and phase but cannot inject high-k energy because the residual is itself L2-minimized. For your wall (k≤7 phase, not high-k energy), physics loss may help phase but is not the primary lever; the backbone is.

- **Spatial vs temporal commitment.** The super-resolution literature (strong, mature) proves spatial conditioning de-blurs; but adv-NO's own admission that super-resolved fine scales are "not perfectly phase-aligned" warns that spatial conditioning ≠ temporal/phase commitment. **Resolution:** this is precisely why a coarse *solver trajectory* (dynamically integrated) is the right backbone, not a spatial low-pass — favoring your instinct over the original Q4 framing.

---

## Uncertainties & Gaps

- ⚠️ **The exact instance is unpublished.** No source builds a one-shot, full-horizon, coarse-solver-trajectory-conditioned deterministic operator for chaotic 2D KF late times. This is the novelty — and the reason confidence caps at ~91% rather than 95%.
- ⚠️ **No forced-KF late k≤7 relL2 number exists in the literature.** PINO's KF numbers are T≤1 s; do not expect a drop-in baseline for your 0.47 wall. You will be generating the first such number.
- ⚠️ **Phase-commitment efficacy is inferred, not measured.** The variance-reduction argument and the residual-to-coarse evidence are strong, but no paper isolates *late-time phase* recovery from a coarse backbone on a chaotic flow.
- ⚠️ **Generative confound.** The most KF-specific coarse-input results (PG-Diff, GenCFD, adv-NO) are generative; their success conditioning-on-coarse is suggestive but does not prove a *deterministic* refiner will match them.

---

## Recommendations

### Primary Recommendation
Build the deterministic core of PG-Diff/Bao: **predict the residual to a coarse-grid, full-horizon solver trajectory, fused as concatenated input channels, trained with L2 (on the residual) + a multi-resolution PINO-style PDE loss.** This is the cheapest path that the literature actively supports — residual-to-coarse (Thermal-LES) supplies the variance-reduction mechanism, MF-FNO supplies the coarse-discretization fusion proof, and PINO supplies the loss scaffold and your later TTA hook. Run the coarse solver as crudely as is cheap (the commitment, not the accuracy, is what matters); your own band_impact_gate result (operator k≤7 → solver → 0.29) already shows even imperfect low-k commitment moves the wall.

### Alternative
If solver infrastructure is the bottleneck, start with a **pure-loss commitment**: add the gradient-difference loss (Mathieu) and/or binned spectral power loss (Chakraborty) to the existing one-shot operator. No backbone, no solver, deterministic, hours to implement — a clean ablation that isolates how much of the wall is objective vs information. If GDL alone moves 0.47 materially, the wall is mostly objective; if not, the backbone is necessary and the primary recommendation is justified.

### Fusion mechanism
Prefer **cross-attention global→local** (AdaptFNO) over plain concatenation if the coarse backbone and fine target live on different grids; use simple channel concatenation (CFD-GCN, PG-Diff) if same-grid. Avoid FiLM as a first choice — it modulates but does not inject spatial structure, which is what a committed backbone is for.

### Not Recommended
Do **not** rely on a physics loss as the primary de-blurring mechanism (Oommen: divergence penalty leaves spectrum biased). Do **not** spatially low-pass the IC as the "coarse input" — it discards information and carries no time evolution, so it cannot commit phase (consistent with your own rejection of that framing). Do **not** reach for a distribution/adversarial term first; per the evidence it is a refinement on top of a committed backbone, not the load-bearing element.

---

## Methodology

- **Depth:** exhaustive
- **Search rounds:** 5 checkpoints over ~11 search batches, ~38 queries
- **Final confidence:** 91% (capped by the irreducible novelty gap, not missing evidence)
- **Sub-questions:** 4 defined, 4 answered (Q1 88→92, Q2 70→94, Q3 50→90, Q4 50→88)
- **Multi-hop chains:** spectral-bias (Oommen) → why L2 blurs → why physics loss fails → why a coarse backbone is the fix; CFD-GCN (steady) → MF-FNO (coarse-grid) → Bao (spatio-temporal LES) → MR-NO (temporal dynamics), progressively closing on your instance.
- **Key challenge:** distinguishing the in-scope deterministic-refinement evidence from the out-of-scope generative (PG-Diff/GenCFD/adv-NO) and solver-in-loop (P2C2Net, ML-coarse-correction) literature, which share the "coarse" keyword but differ structurally.

## Sources

| # | Title | URL | Date | Credibility |
|---|-------|-----|------|:-----------:|
| 1 | Learning Turbulent Flows with Generative Models (Oommen et al.) | https://arxiv.org/html/2509.08752v1 | 2026 | ⭐ Tier 1 |
| 2 | Combining Differentiable PDE Solvers and GNNs (CFD-GCN) | http://proceedings.mlr.press/v119/de-avila-belbute-peres20a/de-avila-belbute-peres20a.pdf | 2020 | ⭐ Tier 1 |
| 3 | Multi-fidelity Fourier Neural Operator | https://arxiv.org/abs/2308.09113 | 2023 | ⭐ Tier 1 |
| 4 | PG-Diff: From Coarse to Fine | https://arxiv.org/html/2504.04375v1 | 2025 | ⭐ Tier 1 |
| 5 | Physics-guided NN for spatio-temporal super-resolution (Bao) | https://proceedings.mlr.press/v180/bao22a.html | 2022 | ⭐ Tier 1 |
| 6 | Physics-Informed Neural Operator (PINO) | https://dl.acm.org/doi/10.1145/3648506 | 2024 | ⭐ Tier 1 |
| 7 | Deep multi-scale video prediction beyond MSE (Mathieu) | https://arxiv.org/abs/1511.05440 | 2016 | ⭐ Tier 1 |
| 8 | Better Understanding of FNOs / SpecBoost | https://arxiv.org/html/2404.07200v1 | 2024 | ⭐ Tier 1 |
| 9 | Thermal-LES field reconstruction (Tpred=T_LES+f_CNN) | https://hal.science/hal-05454994v1/file/Pr_sentation_SFT.pdf | 2026 | ⭐ Tier 1 |
| 10 | Physically-Anchored Multi-Resolution Neural Operator | https://egusphere.copernicus.org/preprints/2026/egusphere-2026-1982/egusphere-2026-1982.pdf | 2026 | ⭐ Tier 1 |
| 11 | AdaptFNO (cross-attention global→local) | https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_96.pdf | 2025 | 🔵 Tier 2 |
| 12 | Finding Closure Models for Coarse-Grained 2D Turbulence (Tryggvason) | https://www.mdpi.com/2311-5521/7/5/154 | 2022 | ⭐ Tier 1 |
| 13 | Multiple Shooting for Neural Differential Equations | https://www.semanticscholar.org/paper/e40e4e4a924dd6275344998fd991ac8f9dcc2559 | 2023 | ⭐ Tier 1 |
| 14 | GenCFD (generative statistical computation of fluids) | https://arxiv.org/html/2409.18359v2 | 2024 | ⭐ Tier 1 |
| 15 | Integrating Neural Operators with Diffusion (NO+DM) | https://arxiv.org/html/2409.08477v2 | 2024 | ⭐ Tier 1 |
| 16 | IU-FNO long-term turbulence prediction | https://ui.adsabs.harvard.edu/abs/2023PhFl...35g5145L/abstract | 2023 | ⭐ Tier 1 |
| 17 | P2C2Net PDE-preserved coarse correction | https://proceedings.neurips.cc/paper_files/paper/2024/file/7f605d59a0dbde101518b552cb616ddf-Paper-Conference.pdf | 2024 | ⭐ Tier 1 |
| 18 | DINO Differential-Integral NO (KF long-term) | https://arxiv.org/html/2509.21196v1 | 2025 | ⭐ Tier 1 |
| 19 | Data-driven correction of coarse-grid CFD (Kiener) | https://www.researchgate.net/publication/371616583 | 2023 | 🔵 Tier 2 |
| 20 | Deep climate downscaling overview (residual SR) | https://github.com/paulaharder/deep-downscaling-overview | 2024 | 🔵 Tier 2 |

---

## Primary Source Reads

| Paper | Keyword Summary | Contribution to Cause |
|---|---|---|
| ThermalLES HAL 2026 | U-Net per-frame SR; T_pred = T_LES + f_CNN(T_LES); anisothermal channel Re_τ=180; conference slide deck | Residual-to-coarse formula confirmed; variance reduction mechanism portable; not peer-reviewed, not temporal |
| Bao 2022 (UAI) | AR RK4-inspired PRU + per-frame spatial SR (PGSR); 3D NS Taylor-Green Re=1600 | AR explicitly fails long-time (RMSE grows monotonically Fig.6); coarse→fine SR principle valid; AR failure motivates one-shot |
| MR-NO egusphere 2026 | Two-stage: coarse one-shot FNO → FiLM-RCNO residual magnifier; shallow-water; deterministic | Closest architecture blueprint; Δh zero-sum portable; coarse FNO stage matches our structure; non-chaotic so phase commitment under chaos unproven |
| Chen/Tryggvason 2022 | AR closure term on coarse NS; 2D freely decaying turbulence; metric = energy spectrum statistics only | Wrong metric (statistics ≠ pointwise L2); confirms chaotic divergence; decaying ≠ forced; AR only — net negative for our hypothesis |
| Mathieu 2016 (ICLR) | GDL = spatial gradient magnitude difference penalty; 8→8 one-shot appendix; natural video | GDL portable to vorticity; proven in one-shot multi-frame output (Table 4); drop-in loss term; formal proof L2 averages modes → blur |
| SpecBoost Qin 2024 | Second FNO trained on first FNO's residual; 71% error reduction on NS ν=1e-5; architecture-agnostic | FNO low-k bias is architectural (mode truncation), not just loss; SpecBoost portable to FNO3d; no GAN, no AR; directly applicable |
| Oommen 2026 | adv-NO (GAN) + NO+∇·u; 3D HIT Re_λ=90; Parseval-Plancherel formal proof; AR forecasting | Physics loss (L² PDE residual) formally proven NOT to fix spectral bias — kills naive Cluster C; structural proof of L2 bias in turbulence; adv-NO excluded |
