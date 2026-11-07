# TTA-OOD Roadmap — Kolmogorov Flow

---

## Label-free distributional TTA via backbone statistics (CONTRIBUTION CANDIDATE, 2026-07-10)

**Idea:** adapt operator weights at test time by matching the *statistics* of the prediction to the *statistics* of the coarse-solver trajectory — MMD loss, no GT. The backbone (run from the test IC at adapted ν) is on the OOD attractor in its resolved band even where its exact phase has drifted → trustworthy reference for distributions beyond the frames/shells where it is trustworthy for values.

**DySLIM reference (Schiff et al., ICML 2024, arXiv:2402.04467, local `~/Documents/JKU/thesis/papers/91_Dyslim.pdf`):** supplies loss + license. Their Eq. 13 = MMD regularizer (rational-quadratic kernel mixture) pulling predicted state-distribution toward a reference measure; on KF (conditional-only, λ2=100) it improved MELR/covRMSE/Wass1/TCM 20–65% AND lengthened the decorrelation horizon (TCM 3.68→0.72). They require GT samples for the reference; the substitution GT→backbone samples makes it label-free — the step no paper has written (wallsearch file 5, Q2d gap, confidence ~68% = flagged research opportunity).

**Mechanics:** (1) run coarse solver from OOD IC → coarse traj; (2) lowpass pred + coarse to solver-valid band (k≲5); (3) frames of each = two sample sets → MMD (few lines torch; frames-as-samples workaround for batch=1, DySLIM used batch 64–128); (4) descend operator weights on MMD + IC anchor, same loop as residual TTA. ν moves backbone + objective together (one knob).

**Ladder position (rung 3, same reference signal, progressively weaker assumptions):**
1. residual + IC (existing).
2. per-shell correlation to backbone, k1–5 within-horizon (frame-by-frame, needs backbone phase right; per-shell horizon table 2026-07-10 tta-thesis.md defines the valid region).
3. this — distributional match to backbone (survives backbone phase drift).

**Grading:** band×horizon-resolved eval (pointwise in coherent core k≤7 within-horizon; MELR/W1/spectrum-ratio companions elsewhere). Prereq measurement: cross-Re per-shell horizon table (vanilla + solver16 on Re500 data, existing `--data-path`/`--coarse-path` overrides).

---

## Rung 0 — backbone spectral recalibration (gradient-free, CONTROL for MMD rung, 2026-07-10)

**Data source:** γ/ρ decomposition, tta-thesis.md 2026-07-10 (cont. 2). Cross-vs-native (coarse arm, recoverable-gap pair): k1 gap = amplitude (γ 0.925 vs 0.980, (γ−ρ)² term; k1 ≈ half of in-band energy); k2–4 already closed (ρ agree ≤0.01, cross γ ≥ native); k5–8 gap = late-window phase (majority of aggr gap, NOT touchable by this rung).

**Idea:** post-hoc per-shell energy rescale of û_k toward the backbone's in-band spectrum E(k,t) — a scalar multiplication per shell, no gradients, no adaptation loop, label-free (backbone spectrum = target). Apply ONLY where phase is intact (k1–4/5, cross ρ 0.96–1.00): phase-oracle rule (banked 2026-06) — magnitude correction lowers error at high ρ, raises it at low ρ.

**Role:** cheapest ladder rung AND the mandatory control baseline for rung 3 — MMD-adaptation must beat this rescale, else it merely re-derives the spectrum (reviewer kill-shot). MMD's justification = what it sees beyond per-shell energy (cross-mode structure, PDFs).

**Expected gain:** modest (amplitude slice only, k1-dominated); value = free + cleans the amplitude axis so later rungs are measured on phase.

---

## Eval client — beyond-L2 metrics (NEXT, 2026-07-09)

**Mission:** consolidate scattered eval scripts + add rollout-quality/spectral metrics into one client (`msc/tta/eval.py` + `report.py`) BEFORE further builds; all TTA results and both open gates (refiner, high-k reweight) read through it.

**Metrics (source → integration):**
1. **Banded rel-L2 early/late/aggr** — exists (`report.py`), unchanged anchor. Arena analogue `ScaledLpLoss` (`pdearena/modules/loss.py:58`) — no port needed.
2. **Correlation time per band** — frames until Pearson corr(pred,GT) < threshold (0.8/0.9). Primitive: `pdearena/modules/loss.py:39` `pearson_correlation` (flattens space, returns (B,T)); wrapper `loss.py:92` `PearsonCorrelationScore`; consumed in `pdearena/models/pderefiner.py:104,193-235` (rollout eval loop). Thresholding happens in their paper analysis, NOT in the repo. Port near-verbatim, apply per spectral band.
3. **Per-shell spectrum ratio** E_pred(k)/E_gt(k), time-resolved — smoothing/sharpness diagnostic (separates "sharper" from "righter", cf. modes32 gifs vs k32-64 rel-err 1.19>1). NOT in arena repo (grep for spectrum/fft eval = empty; paper figures produced outside released code) — build from existing band machinery.
4. **Amp/phase split per shell × time** — port `scripts/amp_phase_split.py` into client (per-mode |û−g|² = amp + phase decomposition, banked 2026-06-13).
5. **Per-shell phase-coherence horizon** — frame at which per-shell phase correlation with GT decays below threshold; vanilla vs solver16-conditioned. DECIDES: does the coarse backbone entrain mid-k (k8-31) phase deeper into the horizon → is high-k loss-reweighting live? (modes32 k8-31 flat 0.783 despite 4× mode budget = current counter-evidence.)
6. **Residual/error co-location** — per-frame, per-shell correlation of |NS-residual(pred)| vs |err(pred,GT)| fields; fp32 autocast-off residual wrap. DECIDES: refiner residual-input channel (entry below); doubles as TTA-objective sanity check.

**Pending survey:** DySLIM + Atif (arXiv:2409.14660) eval sections — distribution-level/invariant-measure metrics; reconcile before freezing client spec. Lippe repo contributes exactly one importable idea (2.); spectra are ours to build.

---

## Iterative residual-conditioned one-shot refiner (GATED design, 2026-07-09)

**Idea:** weight-shared FNO3d applied K times per sample; loop over refinement rounds, NEVER over physical time (one-shot constraint intact). Round k input = [gridx, gridy, gridt, ic, coarse, resid(pred_{k-1})]; round 0 residual channel ZEROED (constant `data_channels`, one spec all rounds). Feedback = PDE-residual field ONLY — own-prediction channel deliberately excluded (Gate A model-restart 0% win; Chain Variant A null, banked).

**Motivation:** coarse channel pins the low-k trajectory family (well-posedness: residual=0 + IC ⇒ unique solution; non-identifiability of residual-min was practical, not mathematical); residual-as-INPUT-FEATURE untested — it is a computation (spectral derivatives, k² factors) a band-limited one-pass network cannot form internally. Distinct from banked-dead residual-as-objective TTA and from PDE-Refiner (whose mechanism is the noise schedule/denoising objective — stripped here; deterministic, no scheduler).

**GATE (run first, free once eval client exists):** residual/error co-location (metric 6 above) on solver16 ckpt `mjzrrn6o`. Correlation ≈ 0 → KILL (remaining error = chaos-consistent drift, residual-invisible). Known headroom bound: conditioned model ≈ backbone intrinsic fidelity (solver24: model late_k7 0.1441 vs bare solver 0.1444) — refiner can at most recover reading-loss.

**Training:**
- Loss: full KFLoss data+pde+ic (`src/pde/ns.py:223`) on EVERY round k=0..K-1, uniform weights, summed. Never final-round-only (identity-collapse risk on feedback channel). Equation-only ruled out (dataonly arms worst-late, banked; residual non-identifying; resu_f7≈0.9).
- Backprop: DETACH handoff (pred_{k-1} + its residual); backward per round (grad accumulation), ONE optimizer step per batch → memory stays single-pass for any K (K=3 full unroll of FNO3d@128²×65 exceeds fleet; chain module maxed at 2 passes on 32GB). Full-unroll no-detach = the single ablation (reuse `stop_grad` pattern, extend `src/models/kf_module_chain.py`). PDE-Refiner precedent: trains steps independently, never unrolls through own output.
- Residual channel mechanics: `NSVorticity(re=ν).residual` (`src/pde/ns.py:71`) → (B,S,S,T-2), edge-pad 1 frame/side to T; compute under `torch.autocast(enabled=False)` with `.float()`, cast back to ambient dtype (mirror `kf_module.py:93-96`). ν required at call time — on OOD, ν = the TTA-adapted quantity (couples channel to adaptation; explicit, not incidental). Toggle: residual on 2× upsampled grid (de-aliases quadratic advection; Re500@128² residual ≈ half aliasing, floor 0.666).
- K=2 smoke → K=3.

**Inference:** K forward passes, SAME K as training (no contractivity guarantee — do not extrapolate rounds); recompute residual per round (same fp32 wrap); coarse tensor reused; output = round K. Never select round by residual (non-identifying).

**Baseline/success:** K=0 = solver16 ckpt `mjzrrn6o`, k0-7 aggr 0.2067 (`report.py`, test [270:300) n=30) — refiner with_coarse must beat it. Historical err_k7 numbers (coarse_oracle_gate, [260:300] n=40) retired, cite as historical only.

**Sequencing:** eval client → co-location gate → build only on gate-pass. Does NOT block the TTA ladder (refiner = optional amortized-refinement vs gradient-TTA comparison arm; residual-min TTA on the coarse-conditioned operator is the primary PINN-mechanism × PINO-prior hybrid and tests the same identifiability hypothesis). Δu output-parameterization rung: PARKED (no information added). High-k loss reweighting: blocked on metric 5; within-k0-7 shell-normalized weighting live pending amp-share (metric 4) on `mjzrrn6o` ≥ ~40%.

---

## Future Work — Generative Sibling Layer (latent-space retrieval)

**Idea:** replace the physics-derived sibling generator (NS solver + IC perturbation) with a generative model trained on the k≤7 coarse manifold. Given a query IC, sample nearby points in the latent k≤7 space — effectively generating siblings without running the solver.

**Why this might work:** the k≤7 projected attractor has estimated intrinsic dimension 5–15 (much smaller than the full d_KY ~ 20–150), because the large-scale circulation is more organized than the enstrophy cascade (k>7). A VAE, flow-matching, or diffusion model trained on k≤7 coarse trajectories can learn this manifold. Latent-space perturbation near the query IC → decode → k≤7 coarse trajectory structurally equivalent to a physics sibling, but without solver cost at inference.

**Advantage over pool retrieval:** generative retrieval is not blocked by the curse of dimensionality. You don't need 300k stored trajectories to cover the attractor — you sample from the learned distribution conditioned on the query IC. Scales to Re=500 where pool retrieval is completely infeasible.

**Advantage over physics siblings:** no NS solver at inference. The generative model distills the local attractor structure from training data; at deployment it produces siblings in milliseconds rather than seconds-to-minutes.

**Connection to existing work:** the current sibling conditioning (Path C) is the ground truth for this idea — it proves that phase-matched coarse input is exploitable. A generative layer is the next abstraction: learn to produce that input from data rather than from the solver. The operator model itself does not change; only the coarse input generation pipeline is replaced.

**Not a thesis deliverable — future work / discussion chapter.**

---

## Future Work — Retrieval-Augmented Physics Simulation (production hypothesis)
othsod/tta output conventions), and one stale line in the old one. Given we're already tidying docs for the new journal, want me to fold the two missing blocks into documentation/paths.yaml and mark msc/configs/paths.yaml deprecated (or delete it), so there's one source of truth?

✻ Sautéed for 42s

❯ yes fold into documentation; then delete the other one

✶ Cerebrating…
                                                                          copied 5 chars to clipboard
────────────────────────────────────────────────────────────────────────────────────────── baseline ──
❯
**Finding:** pool retrieval is blocked by the curse of dimensionality, not by a slightly-too-small dataset. The KF attractor at Re=100 has Kaplan-Yorke dimension d_KY ~ 20–35 (number of positive Lyapunov exponents ~ Re^{2/3} ~ 21). With 300 ICs you are nowhere near coverage; the train/val/test split difference in retrieval quality (mean A_k: 0.46 / 0.38 / 0.35) is a sampling artifact from consecutive NS trajectory indexing, not a fundamental property of the splits.

**Samples required to reach A_k > 0.70 for arbitrary query IC:**

| d_KY | N |
|:--:|:--:|
| 20 | ~25k |
| 30 | ~340k |
| 50 | ~10M |

**Production hypothesis:** if the operator is deployed on new ICs continuously, each run generates one coarse trajectory (k≤7 lowpass). Storing at 16×16 (minimum grid that faithfully represents k≤7): ~130 KB per trajectory. 300k trajectories = ~39 GB — one commodity disk. A 16×16 coarse NS solver run is sub-second on CPU; building the library is a side-effect of normal operation, not a separate data collection campaign.

At ~300k accumulated trajectories (d_KY~30): retrieval quality for any query IC, including test, reaches A_k ~ 0.70 — near-sibling grade. The model trained on siblings generalizes; retrieved coarse is structurally equivalent conditioning. The system improves monotonically with usage at zero marginal cost.

**Open question for this path:** does a sibling-trained model accept real-retrieved coarse at inference without fine-tuning? One ablation: train on siblings, eval with retrieved pool coarse vs solver coarse — measures zero-shot transfer of conditioning.

**Not a thesis deliverable — flag as future work in discussion chapter.**

---

## Path C — Synthetic Sibling Auxiliary Input (ACTIVE 2026-06-29)

**Mission:** teach op100 to exploit local k=4-7 IC neighborhood structure at inference — without the GT coarse oracle that made dropout50 work.

**Evidence chain (all banked wall.md 2026-06-29):**
1. GT k≤7 oracle injection → dropout50 err_k7 0.398→0.138 (−65%): k=4-7 is the exploitable band.
2. Random k=4-7 sibling (no dropout) → model ignores it (zero_coarse collapse zdtf3mhp) → oracle must be phase-matched to be used.
3. Shuffle training (coarse_shuffle_p=0.5) → model rejects phase-mismatched coarse (shuffle≈zero, Δ=0.0001) but retains benefit from matched (err_k7=0.245 vs null 0.398).
4. IC sibling divergence (Exp 1): k=4-7 perturbations stay trajectory-close; Spearman(ic_dist, traj_div)≈1.000 at t=8, degrades to 0.81–0.90 at t=128; traj_div DECREASES over time (attractor contracts).
5. Sibling model error proximity (Exp 2): op100 errors correlated across siblings; Spearman(ic_dist, Δerr)=0.49–0.83 at t=8, grows to 0.81–0.91 at t=128; monotone in perturbation level.

**Hypothesis:** at both training and inference, retrieve the k nearest neighbors of the target IC from the training pool (by k=4-7 relL2 distance) and feed their IC fields (k=4-7 coarse) as auxiliary input channels alongside the target IC. The model learns to use real neighboring IC-trajectory pairs to generalize to the new IC — analogous to the coarse oracle but grounded in the actual training distribution rather than GT future state. At inference, the full training pool is available as a retrieval index; no solver, no GT trajectory needed.

**Critical distinction from oracle:** oracle supplies k≤7 TRAJECTORY (all T frames, future info from GT). Retrieval supplies k=4-7 of real training ICs at t=0 only — lateral view from the neighborhood, not future state. Model must learn to interpolate dynamics from neighbors, not read off the trajectory. Geometrically grounded by Exp 1+2: nearby ICs → similar trajectories → similar errors.

**Open questions (sequenced):**

**Step A (current) — what is structurally similar in k=4-7 trajectory space?**
IC-space relL2 proximity is a proxy for trajectory proximity (Exp 1, Spearman≈1 at t=8). But the model exploits phase, not amplitude (wall.md shuffle50: wrong-phase coarse ≈ zero, Δ=0.0001). So relL2 is too crude as a retrieval criterion — it conflates amplitude drift and phase drift. We need to know whether k=4-7 IC-proximity implies k=4-7 PHASE ALIGNMENT along the trajectory.

Measurement: re-run NS solver on orig + siblings (same deterministic seed as Exp 1; distances.npz has only scalar divergences, not trajectories). Compute per-shell phase alignment A_k at each probe frame (8,16,32,64,128):

    û_orig = FFT2(orig_traj[t])
    û_sib  = FFT2(sib_traj[t])
    FOR shell k in 1..K_EVAL:
        modes_k = {m : chebyshev_shell(m) == k}
        energy_weights = |û_orig[modes_k]|²
        Δφ = angle(û_sib[modes_k]) - angle(û_orig[modes_k])
        A_k = sum(energy_weights * cos(Δφ)) / sum(energy_weights)
        # A_k=1: perfect phase alignment, A_k=0: decorrelated, A_k=-1: anti-aligned

A_k is exactly the Phase-operator A/B metric (wall.md) applied to sibling pairs.
Key comparison: amp-perturbed siblings start A_k≈1 at t=0 (phases unchanged, only amplitude scaled) — does this hold through t=128? Phase-perturbed siblings start A_k<1 at t=0 (phases rotated) — does the attractor re-align them over time?
Script: scripts/perturb/sibling_phase_alignment.py; output: /system/user/studentwork/wehofer/perturb/sibling_phase_re100/phase_alignment.npz

**Step B (after A) — training set density in trajectory-phase space:**
Run pairwise NS solver on all 200 training ICs, compute A_k and amplitude relL2 at probe frames for every pair. Map who is a trajectory-phase neighbor of whom. Gates whether IC-space retrieval is sufficient (IC-proximity → trajectory-phase-proximity) or whether a short-horizon solver confirmation is needed.

**Step B result (done 2026-06-29) — training density:**
`scripts/perturb/training_density.py`; output: `/system/user/studentwork/wehofer/perturb/training_density_re100/density.npz`
Sibling double-check: amp=3.70e-08, phs=2.87e-08 (kernel validated). NN distances (k≤7 relL2):
  train NN intra-pool: mean=0.711, p10=0.474, p50=0.688, max=1.211
  test NN → train:     mean=1.075, p10=0.966, p50=1.091, max=1.261
Calibration upper bound: ic_dist=0.337 → A_k≥0.892. Test NN mean 1.075 = 3× outside measured range.
k=4-7 IC-space is extremely sparse at Re100 — NN retrieval from 200-IC pool gives essentially random
phase alignment (A_k≈0), confirmed independently by shuffle_coarse≈zero_coarse (wall.md 2026-06-29).
**Retrieval-augmented Path C BLOCKED at n_train=200. Pivot: sibling-as-coarse.**

**Step C (pivot) — sibling-as-coarse training:**
Use phase-perturbed siblings (known A_k from wall.md) as the 5th-Nth coarse channel instead of retrieved ICs.
Phase alignment is controlled and known: σ=0.3 → A_k≥0.971 at t=0, converges toward 1 by t=128;
σ=0.6 → A_k≈0.833 at t=0, rises to ≈0.974 by t=128 (attractor re-alignment, wall.md 2026-06-29).
Ablation: n_sibs ∈ {1,2,3} × σ ∈ {0.3,0.6} = 6 training runs.
Comparison: null err_k7=0.398 / dropout50 err_k7=0.138 / GT oracle err_k7=0.018.
Locked eval: Re100 128², held-out [260:300] n=40, coarse_oracle_gate.py (three-arm: with/zero/shuffle).

Implementation — 6 steps (in order; step 1 runs on server while 2-5 done locally):

  1. scripts/perturb/materialize_sibling_coarse.py  [NEW — run first, takes hours on GPU]
     - Args: --re, --sigma, --n_siblings 3, --outdir, --device
     - For each IC idx in [0, N): rng=default_rng(idx*1000+42); call perturb_phase n_sib times
       (same seed stream as Exp 1; advancing rng further gives 2nd/3rd distinct sibling)
     - Each sibling: run NavierStokes2d T=128 frames, apply cheb_lowpass(·,7), write to memmap
     - Output: {outdir}/sibling_coarse_re{re}_s{sigma*100:.0f}_sib{k}_part0.npy, k=1..n_sib
       each file shape (N, 129, S, S) float32 — identical format to existing coarse_k7 files
     - Reuse: perturb_phase, _forcing from ic_sibling_divergence; NavierStokes2d; cheb_lowpass
     - Run: σ=0.3 and σ=0.6 in parallel on separate GPUs (6 files total)

  2. src/datasets/kf_dataset.py  [MODIFY]
     - Add coarse_paths: Optional[List[str]] = None (plural, alongside existing coarse_path)
     - When set: load each file with same offset+sub_t, store as self.coarse_list List[Tensor]
     - __getitem__: stack list → batch["coarse"] shape (n_sibs, S, S, T_eff)
     - Existing coarse_path (singular) unchanged — backward compatible

  3. src/datasets/kf_datamodule.py  [MODIFY]
     - Add coarse_paths: Optional[List[str]] = None constructor arg
     - Pass through to both train and val KFDataset calls (one-liner per call site)

  4. src/models/kf_fno.py  [MODIFY — prepare_input only]
     - Detect coarse_traj ndim:
       4-D (B,S,S,T): existing path → unsqueeze(-1) → (B,S,S,T,1)
       5-D (B,n_sibs,S,S,T): permute(0,2,3,4,1) → (B,S,S,T,n_sibs), append directly
     - kf_module.py: NO changes (zeros_like/dropout work shape-agnostically)

  5. configs/data/kf.yaml  [MODIFY]
     - Add field: coarse_paths: null  (alongside existing coarse_path: null)

  6. configs/experiment/ — 6 new configs  [NEW, copy of kf_coarse_re100_k7_dropout.yaml]
     kf_sibling_coarse_re100_s03_n{1,2,3}.yaml  data_channels={5,6,7}  coarse_paths=[sib1..sibN s03]
     kf_sibling_coarse_re100_s06_n{1,2,3}.yaml  data_channels={5,6,7}  coarse_paths=[sib1..sibN s06]
     All: dropout=0.5, data_weight=5, pde_weight=1, ic_weight=1, 150 ep, same hyperparams as dropout50

**Locked eval:** Re100 128², held-out [260:300] n=40, same band_eval gate as all prior entries. Compare to: null 4-ch err_k7=0.398, dropout50 err_k7=0.138.

**Step C results (done 2026-06-30):**
Sibling ablation — val_l2 at ep29 (best across 150 epochs, all 6 runs):

| σ   | n=1    | n=2    | n=3    |
|-----|--------|--------|--------|
| 0.3 | 0.1788 | 0.1579 | 0.1497 |
| 0.6 | 0.2001 | 0.1973 | 0.1966 |

Monotone: lower σ better, more siblings better. Diminishing returns at n: Δ(n=1→2)=0.021, Δ(n=2→3)=0.008 at σ=0.3; essentially flat at σ=0.6.

Band-resolved gate — s30_n3 ckpt k62n2x26, locked test [260:300] n=40:

| mode        | early_k7 | late_k7 | err_k7 | err_full |
|-------------|----------|---------|--------|----------|
| with_coarse | 0.0904   | 0.1109  | 0.1124 | 0.1874   |

late/early=1.23 (null op100=3.10). References: null=0.398, dropout50=0.138, GT oracle=0.018.
Sibling proxy closes 72% of null→dropout50 gap. Remaining gap to oracle: 0.094.

**Step D results (done 2026-06-30) — proxy quality characterization:**
`scripts/sibling_proximity_analysis.py`; lowpass validation 5.61e-06 ✓. Results banked tta-results.md 2026-06-30.

Proximity curve: sibling-to-GT relL2 DECREASES over time (s30: 0.164→0.121; s60: 0.317→0.230) — attractor contraction confirmed. Model receives improving proxy quality along the trajectory.

Dose-response: x-axis (mean proxy relL2) flat within each σ cluster (2% range); y-axis (val_l2) varies 16% within σ=0.3. Diversity (n) drives model gain, not proxy quality (x). σ is the lever.

n-utility: inter-sibling diversity 0.1926 (n=2) → 0.1928 (n=3), flat. n=4 predicted negligible. Δval_l2: 0.021 (n=1→2), 0.008 (n=2→3). Diminishing returns confirmed.

**Step E (done 2026-06-30) — pairwise alignment: is pool retrieval viable at full dataset scale?**
Tore down train/test/val walls. Computed A_k for all 300×300 IC pairs, both IC-level (t=0) and full trajectory, using k=5,6,7 [k=4 excluded: base-flow forcing contamination]. `scripts/pairwise_phase_alignment.py`; `phase_alignment_re100_all300.npz`. Results banked tta-results.md 2026-06-30.

Key findings:
- NN A_k (k=7, t=0): mean 0.593, p90 0.717 — well below σ=0.3 siblings at 0.956
- Fraction k=7 >0.70: 0.07% (59/89700 pairs) — too sparse to train on
- Locked NN decay (k=7): 0.593 at t=0 → 0.277 at t=32 → 0.114 at t=128 (collapses)
- Per-frame-max flat ≈0.60 (different best neighbor each frame — dataset IS diverse, variety ≠ retrievability)
- Attractor contraction exists for siblings (shared perturbation → shared neighborhood) but NOT for arbitrary pool pairs

**Pool retrieval BLOCKED at full 300-IC level.** Extends and confirms Step B finding (200-IC pool). It is a fundamental property of the Re100 attractor — ICs spread across it with no dense clusters in phase space — not a sample size artifact. The flat per-frame-max shows the dataset covers the attractor well, which is precisely why the model generalizes (test split [260:300] err_k7=0.1124 with s30_n3) and why retrieval fails.

**Path C conclusion:** sibling-as-coarse is the viable path. Next decision: push σ-sweep (σ∈{0.1,0.15}) to narrow remaining gap to oracle (current gap: 0.094), or collect err_k7 gate results for remaining 5 checkpoints (s30_n1/n2, s60_n1/n2/n3) to complete the dose-response before committing GPUs to new training runs.

**Step F (planned) — selective conditioning: pool-neighbor when A_k > θ, solver coarse otherwise:**

Distribution finding (banked tta-results.md 2026-06-30): best-neighbor traj A_k (k=7) is heterogeneous across the 200 training ICs. ICs 127–140 form a tight cluster (A_k=0.73–0.84); ICs 100–119 are isolated desert (A_k=0.19–0.27); median=0.44, only 6% of ICs above 0.70.

Implication: for the ~40 ICs with A_k > θ (e.g. θ=0.60–0.70), a retrieved pool neighbor gives near-sibling conditioning. For the ~160 ICs below θ, feeding the neighbor injects noise that dilutes the signal. This may explain why n4 and n5 currently underperform n2 and n3 — additional neighbor ranks pull more low-quality ICs into the batch.

Proposed mechanism: at training time, compute A_k for each IC pair. If best-neighbor A_k > θ: use retrieved coarse (as current n1–n5). If A_k ≤ θ: replace the coarse channel with a short-horizon solver coarse — run NavierStokes2d from the target IC for a cheap low-res coarse trajectory (`cheb_lowpass(·,7)`, same format), substituting a physics-grounded but trajectory-specific signal for the uninformative retrieved neighbor. At inference: same logic — retrieve neighbor and check A_k (pre-computed index); fall back to solver if below threshold.

This gives the model consistent-quality conditioning during training and keeps the coarse channel meaningful for all ICs. Threshold θ is a hyperparameter; candidates: 0.55 (35% ICs get neighbor), 0.60 (16%), 0.70 (6%).

Split-resolved retrieval quality (banked tta-results.md 2026-06-30): train→pool mean=0.464 (cluster peaks 0.84); val→pool mean=0.375, max=0.629; test→pool mean=0.345, max=0.526. The intra-train cluster (ICs 127–140) is a temporally-adjacent NS sampling artifact not present in val/test. At θ=0.60: 0/40 test ICs qualify; at θ=0.50: 1/40. Step F collapses to solver-only at inference on the test split. This reframes the n1–n5 runs: they measure "heterogeneous train-time conditioning averaged over a distribution the model will never see at evaluation time", not "inference-time retrieval benefit". The solver-coarse fallback is the operative path for all test ICs regardless of θ.

Pending: wait for ep=150 results from current n1–n5 runs to establish the train-conditioned upper bound, then design Step F as a controlled solver-coarse experiment.

---

## Goal
ID ops Re100/200/300 → TTA on Re500 (OOD) via PDE-residual min → eval vs GT_Re500. KF, 128², FNO n_modes=[8,8,8].

## Journey (assumption → answer)  · scripts: spectral_analysis · gt_residual_check · alias_check · band_gate
1. high-Re GT under-resolved? → **YES**. E(k) no Nyquist roll-off Re500/1000.
2. GT own-residual type? → dt→0 spatial floor 0 / .008 / .065 / **.44** / **1.06** (Re100–1000); cliff 300→500; Re≤300 resolved, Re≥500 under-resolved.
3. bound w/o running? → residual affine in ν → secant LB Re500≥0.49, Re1000≥1.03 (confirmed 0.67 / 1.46).
4. resolution irrelevant for never-trained OOD? → half: data-free ≠ resolution-free; n_modes does NOT band-limit û (GELU broadband) → needs empirical check.
5. TTA residual signal aliased? → **NO**. grid-invariant 1.27, û smooth ~7 decades. GT under-res = genuine data deficit, bites only as eval ref.
6. signal in band k≤7? → **YES**. err k≤7 = 0.41 (93% of gap), residual(û) 81% in-band, residual(GT) 0.2%.
7. k≤7 gap physics or drift? → **BOTH**. static early 0.24 + drift to 0.53.

## Settled (locked)
- clean ladder = {Re100,200,300}@128²; Re500/1000 spatially under-resolved.
- Re500 OOD **detection** works (residual ≫ ID).
- TTA residual signal alias-free, 81% in-band; GT low-k self-consistent → band-resolved eval valid.
- **early k≤7 = Re-invariant floor 0.24 → DEAD target. late/aggr k≤7 = SYSTEMATIC model-error → real target** (op500 beats op300 35/40, p=1e-6; "late=chaos" overturned — see findings).
- **Phase-3 label-free CONVERGED (negative):** k≤7 late wall is REPRESENTATIONAL, not chaos (Gate 1 `chaos_spread_gate`: perturbation growth only ~1.3×/horizon, λ≈0.2–0.3; control solver-from-exact-IC = 0). All label-free levers DEAD — spectral/moment/low-k-positioning oracle ceilings ≤0.01–0.03; residual non-identifying; self-chaining compounds (Gate A); self-consistency bystander. Only movers: oracle resets (observations / data-assimilation) OR a richer class + resolved data (→ Path B).

## Decision: Option A
128² data + band-resolved eval. Option B (rebuild ladder higher-res + raise n_modes) **shelved** — not needed for band-limited claim.

## Caveats
- "GT low-k clean" = internally self-consistent, NOT proven = true Re500 low-k (no 256² ref). claims scoped k≤7.
- ν fork: 1/300 **realistic** (carries claim) vs 1/500 **oracle** (UB). run both, report both.

## Setup
- ckpts `paths.yaml → pretrain_checkpoints`: op100=pvqq97sq · op200=4em1mfrx · op300=1iix0n42 · op500=38o0kj3y (`.../checkpoints/best.ckpt`).
- run: `CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD /system/apps/studentenv/wehofer/bin/python -u <x>.py`
- split: **LOCKED** — adapt `[0:100]`, held-out `[200:300]`, buffer `[100:200]` (non-adjacent), sub_t=2.

## TTA framing  · survey 67_tta_taxonomy (Liang/He/Tan 2024, arXiv:2303.15361 — structural framing only)
- regime: TTIA (per-IC, reset) / OTTA (online, accumulate — deferred) / TTDA (shared, multi-epoch). **pivot: per-IC fine-tune → scale pool (TTIA→TTDA) to locate generalization onset** (pooled TTDA failed, see findings).
- locus: **BN-family DEAD** (FNO has no norm); baseline = full-weight; locus = open sweep.
- objective: physics residual only (PINO) ≡ Tent entropy-min analogue. future lever: consistency (grid/temporal, ν-free).
- anti-forgetting: only if weights persist (OTTA); baseline none except IC-anchor.
- parked: meta-learning (retrains op), Cao input/output-space frozen-weight.
- validation: label-free TTA has no clean val set → pre-register thresholds, select HP **label-free**, never by GT.

## Plan
Bounds (first): lower = op100 zero-shot, upper = op500 zero-shot (inductive
ceiling); warm floor 0.037 = transductive reference. Brackets achievable gain.
Recompute on the locked eval window (see split note) — old [260:300] numbers
are not comparable to a [100:130] window.

E1–E3 — per-IC fine-tune, scale pool: N=1 → 2 → {3,5,10}, 1.5k steps.
objective = **full residual** (all-k, undecomposed) + IC-anchor (label-free; IC-anchor
only — data-anchor leaks labels). Log per-step, **adapted pool vs held-out as separate
curves**: per-sample residual + eval-loss + aggregate → locate generalization onset.
  pre-reg prediction: adapted → ~0.037 (transductive); held-out → ~no-adapt 0.40,
  may degrade as N→large (= pooled-failure regime). The adapted–held-out gap IS the result;
  flat held-out = clean generalization-boundary finding, not a null.
  matrix: LR {1e-4, 5e-4, 2.5e-3} × IC-weight {0, .., 5}; select label-free.
  ν arm (reported, not tuned): oracle 1/500, realistic 1/300.
  IC-weight = anchor fork {0 = none, >0 = warmstart-style} = physics-vs-reconstruction attribution.

Deferred: OTTA (online, shared-weight, 10-step/IC) · locus sweep — only if E1–E3 motivate.

## Guards (every run)
- alias GO/NO-GO on adapted û (residual drop must not be grid-dependent; ref ~1.27).
- log full residual (all-k) + banded VIEW (k≤7 vs full, NOT partial-operator split)
  + full-field val_l2; all time-resolved (early/late/aggr). band/early/late = analysis lens, not pre-filter.
- never select LR/step by GT.

## Bracket — DONE (zero-shot, Re500 held-out [200:300], k≤7; `bracket.py`)
| op | early | late | aggr | full | res_f7 |
|----|-------|------|------|------|--------|
| 100 | .271 | .678 | .531 | .554 | .67 |
| 200 | .219 | .567 | .428 | .459 | .80 |
| 300 | .234 | .527 | .404 | .438 | .81 |
| 500 | .248 | .473 | .370 | .408 | .90 |
- must-beat = op100/200/300; **ceiling = op500** (late .473 / aggr .370). residual-min ranks op500 WORSE → aspirational, not target.
- late/aggr monotone by source (CONFIRMED); early Re-invariant floor (.22–.27, op100 worst early) → dead.
- error = late-time drift in k≤7; k>7 rel-err≈1 all ops (no skill). op300 matches banked [260:300] (.52/.40) → split-stable.
- **op100 = widest playground** (late headroom .205 vs op300 .054). target = late/aggr, never early.
- caveat: res_f7 = in-band FRACTION, not absolute residual rank → kill-shot (op500 abs residual high) untested here; client must log abs residual.

(Settled findings + banked negative → `tta-findings.md`.)

## Path B — operator-class build (ACTIVE; beat the representation floor)
Why: Gate 1 → wall is representation-dominated (see Settled). ⇒ only a **richer function class + clean (resolved) data** can move it. Physical lever = **closure**: k≤7 dynamics couple to k>7 via the cascade; FNO n_modes=8 is blind to k>7, and 128² high-k is aliased.

Invariants to hold:
- Floors (128², k≤7 late): op500 = **0.473** (best-FNO), full-GT-traj-sup adapt = 0.522, label-free adapt_op100 = 0.580, op100 zero-shot = 0.678.
- Resolution: 128² under-resolved (no Nyquist roll-off; k1 over-energized ~32%, ≈53% of k≤7 energy; k3–7 converge ≤10%). 256² (spectral-downsample of native 512²) validated faithful to 512² (energy retention k≤128 = 1.000000, in-band k≤7 exact, residual floor = 512²) → **256² is the working ceiling** (512² training won't fit 12 GB).
- Data status: Re500 @256² ✓ (VERIFY it has a real train split, not just 100 eval). Re100/200/300 @256² = **TODO generate** (match the Re500 256² pipeline exactly).
- Hardware: student09 = 5× TITAN V **12 GB** (tight: FNO@256 needs batch 1–2 + grad-checkpoint + amp). student06 GPU3 = **24 GB** (slower) — reserve for OOM / UNO@256. **1-step smoke before any multi-hour launch.**

Method: each arm = a SINGLE **op500 in-dist floor-gate** (base-train from scratch; fine-tune only as cheaper fallback — it carries 128² bias). Eval k≤7 on the **256² target**, PAIRED vs op500@128 run on 256² ICs (isolates operator gain from the target-bias fix). Only the winning config earns the full ladder + TTA. Locks: **time-modes stay 8** (temporal sweep dead); eval band stays **k≤7**.

Arms — DECISIVE-FIRST:
1. **A4 [GATE, run first]** FNO @256², modes [16,16,8] — combined res+modes lever vs floor. Pass ⇒ proceed; fail ⇒ wall survives clean data+modes ⇒ consolidate.
2. **A2 [attribute resolution]** FNO @256², modes [8,8,8] — clean target only, no extra band (only if A4 passes).
3. **A3 [control]** FNO @128², modes [16,16,8] — more modes on dirty data; expected ≈null/worse (128² high-k aliased) → proves modes need clean scales.
4. **A5 [architecture]** UNO @256² vs FNO @256² — the real UNO/FNO A/B (only meaningful with clean scales; UNO@128 = confounded null, SKIP).
5. **A1 [loose end, low priority]** residual ∂ₜ CD4/CD6 swap — expected null (residual non-identifying); cheap closure-check, does NOT gate the above.

In-flight (A4 floor-gate, launched 2026-06-17) — HARDWARE FINDING: n16 h128 @256² OOMs on BOTH 12 GB TITAN V and 24 GB P40 (activations/FFT, not params; amp dead — cuFFT needs power-of-2 dims, T=70). Fits: TITAN V → h32 (~6.6 GB), P40 → h64 (~13.9 GB). ⇒ "more modes at full h128 width" infeasible here; A4 split across two feasible widths, and dense higher-modes FNO is memory-pathological ⇒ UNO (param-efficient multiscale) is the proper vehicle.
- Two op500 base-trains on Re500@256² (`NS_fine_Re500_T128_res256_part0.npy`, n_train=200/n_val=60, sub_t=2, batch1, grad_checkpoint=true [hook in `src/train_kf.py`]):
  - **h32**: student09 GPU0, `experiment=kf_re500_256_n16_h32`, wandb `vxsk9mex`, ckpts `pathB-256/vxsk9mex/checkpoints/`, log `msc/tta/runs/ladder_logs/pathB_h32.log`. ~5.4 h total.
  - **h64**: student06 GPU3 (P40, shared node), `experiment=kf_re500_256_n16_h64`, wandb `adb4tfh0`, ckpts `pathB-256/adb4tfh0/checkpoints/`, log `pathB_h64.log`. ~24 h total (slow).
  - val_l2 (full-traj, NOT the gate): h32 0.449, h64 0.477 — informational only.
- Re100/200/300 @256² NOT generated yet (deferred until a gate-passing config; match the Re500 256² pipeline).
- CAPACITY GATE — DONE (2026-06-17, `scripts/k7_modes_gate.py`, banked in tta-results). Late k≤7 on Re500@256²[260:300] n=40, all arms same data (FNO res-flexible): n8h128 anchor 0.528 (128² = 0.472 ✓ reproduces banked floor) → n16 h32/h64 0.589/0.580 (NO late gain, width-invariant on gate) → n32 PENDING (user's GV100 m32 run completes the sweep; eval its best.ckpt with `--arm n32h64=<ckpt>:32:64`). Modes improve EARLY (0.279→~0.19) but gain decays by late. ⇒ capacity (modes+width) is NOT the late lever. UNO is the SAME spatial-representation axis ⇒ also not justified by this evidence.
- Gate 2: phase-spread test — DONE (2026-06-18, `scripts/gate2_phase_spread.py`, 256² faithful control=4e-6, n=20 M=8; numbers banked in tta-results). D_late LOW (δ0.1/0.2/0.3 = 0.073/0.192/0.296), grows ~1.3–1.4× (NOT →1) ⇒ late k≤7 phase LEARNABLE, not chaos. Same-day @256² (all banked): chain_gate (oracle re-anchor 0.528→0.313 op500 / 0.571→0.230 fno16, 100% win; own-field chaining weak→modest), temporal-mode point (DEAD), paired op-quality (late weakly systematic 61/100 p.035).
- MIGRATE-FORK RESOLVED (elimination): late wall NOT chaos (Gate 2 phase + Gate 1 power), NOT capacity (spatial+temporal+arch all dead), NOT representation (78% err in resolved k≤7). Surviving lever = the training OBJECTIVE — HYPOTHESIS not result (low D_late ⇒ map smooth = necessary, not sufficient for a one-shot fit).
- NEXT (pick up here) — OBJECTIVE lever, explicitly NOT self-consistency:
  - self-consistency / self-inconsistency-minimizing objectives BANKED-DEAD (bystander; dose-response 2026-06-12/13: dose↓ but GT-late flat/worse, non-causal; teacher was op's OWN weak restart field = the chain_gate weak-bootstrap). DON'T revive.
  - candidate = LATE-WEIGHTED / horizon-truncated GT-SUPERVISED loss (reweight existing data loss toward late frames; no self-referential teacher ⇒ no bystander risk).
  - DIFFICULTY (chaos-artifact-split 2026-06-12): late err = operator mid-field err × chaos amplification (op500 C 0.345→0.443, grows w/ t_r); C_ctrl=0 ⇒ NOT an irreducible floor (perfect field→0) but a ~1.3× multiplier the late-weighting must beat by shrinking mid-trajectory error.
  - OFF the table: new ARCH (CNO) + more capacity (closed axes).
  - SEQUENCING (user 2026-06-18): late-weighted PARKED until "the unet is up" — clarify which unet (UNO arm already run @256²; reconcile with arch-closed before committing).
  - LADDER: build → ttd-agent → supervisor → git add (.py only).
- Re100/200/300 @256² NOT generated yet (deferred until a gate-passing config).
- OPS NOTE: student06 NFS read of the 10 GB file wedged once (D-state folio_wait); fix = relaunch. Avoid background polling shells (pile up + hang on wedged nodes) — use foreground checks.

Parked within Path B: **Gate 3 — hybrid closure** (cheap NO-TRAIN oracle: cheap coarse `NavierStokes2d` from exact IC, late k≤7 vs GT; cheapest solver < floor = headroom). **NOT pursuing the solver direction now** (user). Also parked: Path A (coherence-horizon + chaining) — closed label-free (Gate A: oracle resets work but need observations; self-chaining compounds; self-consistency bystander).
