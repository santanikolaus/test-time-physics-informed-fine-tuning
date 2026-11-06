---
fontsize: 10pt
header-includes:
  - \usepackage{float}
  - \makeatletter\def\fps@figure{H}\makeatother
  - \usepackage{newunicodechar}
  - \newunicodechar{∈}{\ensuremath{\in}}
  - \newunicodechar{σ}{\ensuremath{\sigma}}
  - \newunicodechar{≤}{\ensuremath{\leq}}
  - \newunicodechar{≥}{\ensuremath{\geq}}
  - \newunicodechar{≈}{\ensuremath{\approx}}
  - \newunicodechar{ν}{\ensuremath{\nu}}
  - \newunicodechar{∇}{\ensuremath{\nabla}}
---

### findings summary ood detector 

**0. problem statement**: Build a runtime OOD detector for Re shifts (Re ∈ {100, 200, 300, 500, 1000}) using the NS equation residual as score. The intuition: if a trajectory is OOD, evaluating NS at training viscosity ν_train should yield a larger residual — wrong physics means larger equation error.

**1a. Energy Spectrum:** Re fingerprint exists, lives at k>8

*idea:* Before building a residual-based detector, ask whether Re leaves a discriminative fingerprint in the data at all, and at which scales — this upper-bounds what an aggregate L2 score can capture .

*result:* The ground-truth energy spectrum separates monotonically by Re above k≈10 across six orders of magnitude (Re=100 drops to $10^{-8}$ at k=64; Re=1000 stays at $10^{-2}$). Below k≈8 the curves are nearly overlapping ([Fig. 1](#fig1a)).

*conclusion:* Re-discriminating information exists in the data and is concentrated above>10. A detector that cannot access these modes is working with the wrong part of the spectrum and will have a hard time discriminating.

![](../scripts/outputs_server/energy_spectrum.png){ #fig1a }

*Fig. 1 — Ground-truth energy spectrum (left). Curves separate monotonically by Re above k≈10; below k≈8 they are nearly indistinguishable.*

---

**1b. UMAP k≤8 vs k>8:** FNO's cutoff sits at the information boundary

*idea:* The energy spectrum suggests k>8 is where Re-information lives. Verify whether this spectral separation is actually exploitable for discrimination — and whether k≤8 alone is sufficient.

*result:* Embedding the spectral features split by band tells the story clearly: k≤8 features cannot separate Re≥300 regimes (Re=300/500/1000 interleave into one blob; only Re=100 is isolated). k>8 features give five separated, Re-ordered clusters with little overlap ([Fig. 2](#fig1b)). This is quantified by the Wasserstein distance: MMD(Re=500, Re=1000) at k≤8 = 0.06 vs k>8 = 0.88.

*conclusion:* For Re≥200, discriminative information lives almost entirely above k>8. Since FNO output is bounded at k=8 by construction, any detector scoring only FNO outputs via aggregate L2 is operating in the blind zone for high-Re pairs. This is a structural prediction — not yet an empirical result.

| k ≤ 8 (FNO passband) | k > 8 (FNO discards) |
|:---:|:---:|
| ![UMAP k1-8](../scripts/outputs_server/umap_k1to8_100-200-300-500-1000.png){ width=100% } | ![UMAP k9+](../scripts/outputs_server/umap_k9tofull_100-200-300-500-1000.png){ width=100% } |

[]{#fig1b}
*Fig. 2 — UMAP of energy spectrum features. Left: k≤8 only — Re=100 separates, Re=300–1000 interleaved. Right: k>8 only — five clean clusters, perfectly Re-ordered.*

---

**2. Aggregate pde_loss:** high-Re operators are blind, low-Re are not

*idea:* Having established that Re-information lives above k>8, build baseline for detector: score each trajectory by how well it satisfies the NS equation at training viscosity ν_train. Five operators (Re ∈ {100, 200, 300, 500, 1000}, all n_modes=8), each evaluated against all five test Re values.

*result:* The structural prediction from 1b is confirmed. For low-Re operators the aggregate pde_loss is a strong detector: op100 achieves a dynamic range of 2.38 and Cohen's d > 3σ for far Re pairs. For op500 and op1000 the score collapses to Δ=0.17 — flat across all test Re ([Fig. 3](#fig3)). The asymmetry is sharp: op100 detects op1000 test data at >5σ; op1000 cannot detect op100 test data above 0.4σ. Same physical gap, opposite directions, completely different scores. Importantly, even op500/op1000 show non-zero effect sizes on some pairs (up to 0.63σ and 0.82σ).

*conclusion:* The aggregate score works where k≤8 still carries Re signal and fails where it does not. But the non-zero cells and the strong op100 result show that discriminative information is present in the residual. This raises two questions:

(1) The score is computed in coordinate space with no frequency decomposition — could scoring specific spectral bands directly recover what the aggregate hides?

(2) The aggregate is already stronger than the k≤8 spectral analysis predicts — where does that extra discriminative power come from?

![](../scripts/outputs_server/cohens_d_heatmap.png){ width=72% #fig3 }

*Fig. 3 — Each cell: Cohen's d = how many pooled standard deviations the pde\_loss mean shifts from the in-distribution baseline (diagonal) when the operator is evaluated on a different test Re. The score is the L2 norm of the NS residual summed over the full spatial grid and time — coordinate space, no frequency decomposition. Rows: operator train Re. Columns: test Re. Top rows hot, bottom rows cold — detection collapses as train Re increases.*

---

**3. Band-resolved residual — scoring the right frequencies:**

*idea:* Two mechanisms suggest residual energy above k>8 carries Re signal even from a k≤8-bounded operator.
(1) Nonlinear aliasing: the advection term (u·∇)u is a product in physical space — a convolution in Fourier space — so k≤8 × k≤8 folds energy into B3=[9,16] of the residual.
(2) FNO skip connections: the architecture adds a pointwise linear path (W·u) that is not band-limited — high-k spatial structure passes through this additive connection and appears in the residual, likely preserving Re-discriminating information above the Fourier truncation boundary. The residual field is computed at full resolution regardless of the operator's output bandwidth — all modes are accessible at evaluation time. FFT the residual field and score B3=[9,16] directly instead of the aggregate norm.

*result:* B0 and B1 collapse with train Re — near zero for op500/op1000 across most pairings, confirming the aggregate failure in section 2 was a low-k effect. B3=[9,16] carries signal at every operator including op1000 ([Fig. 4](#fig4)), recovering the same trajectories that gave ~0.4σ under the aggregate score. B4 is competitive in effect size; B3 is the onset band at the n_modes=8 architectural boundary.

*conclusion:* The architecture is not blind at n_modes<=8 — only summing up the result in coordinate space hid it. Scoring B3 in frequency space recovers the signal without any retraining. This answers both questions from section 2.

![](../scripts/outputs_server/banded_cohens_d_per_op_abs.png){ width=62% #fig4 }

*Fig. 4 — Signed Cohen's d per frequency band, one subplot per operator, one line per OOD test Re. B0/B1 collapse with train Re; B3=[9,16] (shaded) and B4 carry signal at all five operators. B3 is the onset at the n_modes=8 architectural boundary.*

---

**Finding: sign inversion — B3 as directional probe**

*result:* The signed Cohen's d in B3 forms a antisymmetric pattern across the full 5×5 operator × test Re grid ([Fig. 5](#fig5)): when test Re > train Re the B3 residual energy increases (positive d); when test Re < train Re it shrinks (negative d). The upper triangle is  positive and the lower triangle is (almost) entirely negative. Two adjacent-regime cells are weak (op500/test1000 +0.30σ, op1000/test500 +0.48σ). The onset is sharp at exactly k=9, the n_modes=8 architectural boundary — confirming B3 is not an arbitrary choice.

*conclusion:* The sign of B3 residual energy tracks which side of the training Re the test data sits on. This turns the detector from a binary OOD flag into a directional probe: not just "is this OOD?" but "is the test flow more or less turbulent than the operator was trained on?" This is the key finding for Phase 2 adaptation.

![](../scripts/outputs_server/per_mode_b3_signed_heatmap.png){ width=72% #fig5 }

*Fig. 5 — Signed Cohen's d at B3=[9,16]. Upper triangle (test Re > train Re): all positive. Lower triangle (test Re < train Re): (almost) all negative. Color saturated at 3σ; cell annotations carry sign.*

**4. Detector performance — AUC:**

*idea:* Translate the B3 signal into a deployable detector. Per trajectory, z-score the B3 energy against the operator's in-distribution baseline (40 trajectories at training Re). Two variants: |z| two-sided (direction-agnostic, threshold on magnitude) and signed z (exploiting the sign inversion from the Finding — positive for test Re > train Re, negative otherwise). Evaluate both with AUC and 95% bootstrap CI.

*result:* Two-sided |z|: AUC ≥ 0.75 for all pairs separated by more than one Re step; far corners (op100/test1000, op1000/test100) reach 1.00 ([Fig. 6](#fig6)). Three adjacent-regime cells fall near chance: op200/test300 (0.53), op300/test200 (0.58), op500/test1000 (0.45) — 95% CI straddles 0.5 in all three. Signed z (thresholding directionally using the sign inversion from the Finding rather than magnitude alone) improves AUC in 19 of 20 pairs. One exception: op1000/test500 drops from 0.45 → 0.39 (ΔAUC = −0.26).

*conclusion:* Two-sided |z| is the deployment-safe choice — it works for all but physically adjacent pairs. The three near-chance cells are not a detector failure: they correspond exactly to the weakest Cohen's d cells from the Finding, where the B3 spectral difference between adjacent Re regimes is genuinely small. The signed variant earns its keep on most pairs but breaks at the one cell where the sign was already unreliable — consistent with the Finding rather than a new failure mode.

| \|z\| two-sided | signed z |
|:---:|:---:|
| ![AUC two-sided](../scripts/outputs_server/detector_auc_twosided.png){ width=100% } | ![AUC signed](../scripts/outputs_server/detector_auc_signed.png){ width=100% } |

[]{#fig6}
*Fig. 6 — AUC heatmaps, 95% bootstrap CI, n=40 per class. Left: |z| two-sided (deployment-safe). Right: signed z (directional). Improvement visible in most cells; op1000/test500 drops 0.45→0.39 (the one cell where sign is unreliable).*

**5a. Ablation — physics loss vs data-only:**

*idea:* The B3 mechanism was described via two architectural pathways — nonlinear aliasing and skip connections. Both are properties of the FNO architecture, not the physics loss. If the signal originates from architecture alone, a data-only model (pde_weight=0) should also show B3 discriminability. Conversely, if the physics loss is load-bearing, removing it should suppress the signal. Op300, n_modes=8; both variants evaluated on the full Re={100,200,300,500,1000} test set.

*result:* B3 signal is present in the data-only model. Sign is correct for all four OOD pairs (d<0 when test Re < train Re, d>0 when test Re > train Re); median |d| ≥ 0.64σ across all pairs, ≥ 1.78σ for Re=500 and Re=1000. Effect sizes match or exceed the PINO counterpart in every pair. The key difference is spectral: PINO shows a flat response in k<9 with a sharp switch-on at the architectural cutoff k=9; the data-only model shows signal rising from k≈5 that bleeds through B1[3,5] and B2[6,8] with no discrete onset ([Fig. 7](#fig7)).

*conclusion:* B3 detectability is an architectural property, not a consequence of the physics loss. The physics loss does not create the signal — it concentrates it: by enforcing NS consistency in the passband, PINO suppresses residual energy below k=9 and makes the k=9 onset structurally sharp. For a data-only FNO the effective detection band broadens to k≥5, but the B3 detector remains valid.

![](../scripts/outputs_server/ablation_dataonly/per_mode_signed_d.png){ width=80% #fig7 }

*Fig. 7 — Signed Cohen's d per wavenumber for op300. Top: data-only (pde\_weight=0). Bottom: PINO. Shaded region: B3=[9,16]; dashed line: n\_modes=8 cutoff. Data-only signal bleeds into k<9; PINO onset is sharp at k=9. Re=300 omitted (in-distribution baseline, d=0 by definition).*

**5b. Ablation — n_modes sweep:**

*idea:* Does OOD discriminability depend on model capacity — does a weaker operator (fewer modes) lose the B3 signal, or does it persist? Secondary: if the detection band is anchored to n_modes, varying n_modes should shift it — what is the scope of validity of B3=[9,16]? Op300 evaluated at nm4, nm8, nm16; nm16 additionally checked cross-Re at train Re=100 and Re=1000.

*result:* nm4 shows near-zero signal in B3 for near-Re pairs (−0.31σ, +0.02σ); moderate for far-Re (+1.59/+1.80σ). No clean onset anywhere. nm16 is stronger than nm8 in every cell (k=12: −2.56σ at Re=100, +2.09/+2.36σ at Re=500/1000); signal rises from k≈5 and persists through B3 and beyond, with no discrete onset at k=17. Cross-Re nm16 sign structure is correct at all three operators, >3σ at far cornerLs ([Fig. 8](#fig8)).

*conclusion:* B3 is valid for n_modes=8. For nm4 it is the wrong band; for nm16 it remains discriminative but undersensitive. The generalised detection band is [n_modes+1, 2×n_modes].

![](../scripts/outputs_server/ablation_nmodes_sweep/per_mode_signed_d.png){ width=80% #fig8 }

*Fig. 8 — Signed Cohen's d per wavenumber, op300, nm4/nm8/nm16. Shaded: B3=[9,16]; dashed: n\_modes cutoff per variant. nm4 flat, no onset; nm8 sharp onset at k=9; nm16 broad signal, no discrete shift to k=17.*

---

**7. ν* — viscosity argmin:**

*idea:* r(ν) = A − νB is quadratic in ν with closed-form minimum $\nu^* = (A \cdot B)/(B \cdot B)$. If the residual shape encodes test physics, ν* tracks true viscosity where the scalar ||r(ν_train)|| is flat.

*result:* op100: ν* monotone in test Re, >1.88σ separation at all off-diagonal pairs. op500/op1000: strong signal only toward test Re=100 (>3σ), non-monotone for adjacent pairs. All operators show systematic bias — ν* is 2–14× above ν_train at the in-distribution diagonal.

*conclusion:* ν* works as a parameter estimator for low-train-Re operators; at op500/op1000 the argmin landscape is too flat. The bias is consistent with the operator imprinting its training viscosity on the residual.

---

**8. What's next:** 

- use OOD detector as adaption trigger and maybe use the direction sign signal.
- any surrogate network output can be FFTd and band scored, the detection in theory is architecture agnostic but definitely needs fresh calibration

---

**9. Formula — Cohen's d:**

Rather than a full distribution distance (Wasserstein, KL, MMD) — which need shape information that 40 samples per group don't reliably support — we use Cohen's d, an effect size that compares group means scaled by the typical noise:

$$d \;=\; \frac{\bar{x}_{\text{OOD}} - \bar{x}_{\text{ID}}}{\sigma_{\text{pooled}}}, \qquad \sigma_{\text{pooled}} \;=\; \sqrt{\tfrac{1}{2}\!\left(\sigma_{\text{ID}}^{2} + \sigma_{\text{OOD}}^{2}\right)}$$

