# KS as second time-dependent PDE — feasibility spec

Status: proposal, 2026-11-06. Read-only investigation; no code changed, nothing executed except `pip list`/`nvidia-smi` on student05. Every computed number below is a literature value or an estimate; verification is S1's job.

Purpose: answer the "favorable misspecification" objection. In KF Re100→Re500, ν∇²ω is a small residual share (dropping it keeps 85% of the TTA gain), so a source-ν residual is nearly correct on target data. KS must be set up so that the shifted coefficient multiplies a leading-order residual term.

---

## 1. Equation, shift axis, parameters

### 1.1 Form
u_t + u u_x + u_xx + ν u_xxxx = 0, x ∈ [0, L) periodic, zero-mean u (∫u dx is conserved).
- Same form as PDE-Refiner (Lippe et al. 2023, Eq. 3) and the LPSDA generator (Brandstetter et al. 2022), which PDE-Refiner reuses.
- Exponax/APEBench use the same operator with coefficient knobs (`second_order_scale`, `fourth_order_scale`; also a gradient-norm "combustion" form). Exponax docstring: chaos for L ≳ 60 at unit coefficients, 200–500 steps of dt=0.1 transient, mean-zero IC.

### 1.2 Scaling equivalence: one parameter family
x = √ν x', t = ν t', u = u'/√ν maps (ν, L) onto the standard KS with ν=1 and L̃ = L/√ν. Every term scales as ν^{-3/2}.
- ν, L and the u_xx coefficient α are therefore one axis: effective size L̃, plus a unit change (length ∝ √ν, time ∝ ν, amplitude ∝ ν^{-1/2}).
- The u u_x coefficient alone is a pure amplitude rescale. Rejected as trivial.
- **L at fixed coefficients is a null-misspecification shift.** The equation is identical on both sides, so a source-PDE residual is exactly correct on target data. That is the maximally "favorable" case the reviewers object to. Rejected as primary; usable later as a control row (pure attractor shift, zero equation shift).

### 1.3 Recommendation: ν (hyperdiffusion coefficient) at fixed L=64
- **Source ν_s = 1.0 → target ν_t = 0.5**, L = 64. Effective L̃ = 64 → 90.5. Both values sit inside PDE-Refiner's verified parameter-dependent range ν ~ U[0.5, 1.5] at L≈64, N=256.
- Fixed L keeps one grid and one wavenumber set for both regimes. The shift is then a single loss coefficient, the exact analog of `pde_re` (adapt.yaml:12, loop.py:22–34).
- Direction ν↓ mirrors Re↑: more unstable modes, energy moves to higher q, amplitude ×√2, dynamics ~2× faster.

Linear growth σ(q) = q² − ν q⁴, q = 2πn/L = 0.0982 n:

| | ν=1 (source) | ν=0.5 (target) |
|---|---|---|
| neutral q = 1/√ν (mode n) | 1.00 (n≈10) | 1.41 (n≈14) |
| fastest-growing q = 1/√(2ν) (≈ energy peak) | 0.71 (n≈7) | 1.00 (n≈10) |
| max growth σ_max = 1/(4ν) | 0.25 | 0.50 |
| unstable modes n ≥ 1 | 10 | 14 |
| L̃ = L/√ν | 64 | 90.5 |
| λ_max ≈ λ̃(L̃)/ν | ≈0.09 | ≈0.18 |
| Lyapunov time T_λ | ≈11 tu | ≈5.6 tu |

- λ̃(L̃) ≈ 0.08–0.09 for L̃ ≥ 36 (0.043 at L̃=22). **Secondhand**: arXiv:2001.09131 Table I, citing its unidentified "Ref. [67]". Not confirmed to be Edson et al. 2019 (arXiv:1902.09651), whose numbers were not read. S1-G5 re-measures it.
- Linot & Graham 2020 (PRE 101, 062209): L = 22/44/66, inertial-manifold dimension ≈ linear in L. Recalled values 8/18/28 are unverified. Implication: target attractor dimension ≈ 1.4× source.

### 1.4 Why the ν term dominates the residual (the core argument)
- **Energy balance.** Multiply by u, integrate, use periodicity (u u_x conserves energy) and stationarity: ⟨u_x²⟩ = ν⟨u_xx²⟩. The anti-diffusion injection and the hyperdiffusion sink balance *exactly*. The ν term is the only sink, not a correction.
- **Spectral ratio.** |ν q⁴ û| / |q² û| = ν q² is 1/2 at the energy peak and 1 at the neutral mode, so u_xx and ν u_xxxx are the same order.
- **Misspecification term.** On target GT, R_{ν_s} − R_{ν_t} = (ν_s − ν_t) u_xxxx = (Δν/ν_t) · ν_t u_xxxx. For 1→0.5 this is 1.0× the hyperdiffusion term, i.e. leading order. In KF the analog (ν_s−ν_t)∇²ω is O(10⁻²·|∇²ω|) against the advective term.
- **Drop-term arm (ν_loss = 0): measured, not assumed.** Without ν u_xxxx the forward IVP u_t + u u_x + u_xx = 0 is ill-posed (growth ∝ q², unbounded). That does not imply residual minimisation over a band-limited network output fails, so it is a prediction, not a fact.
- **Symmetric bracket.** ν_loss ∈ {0, 1.0} sit at equal |Δν| = 0.5 on opposite sides of ν_t = 0.5.
- **Pre-registration.**
  - Oracle-ν TTA gains.
  - Source-ν (1.0) gains ≪ oracle, and is ≤ 0 in the shift-sensitive band n8–14, because it pushes predictions back toward ν_s dynamics (double damping).
  - ν_loss = 0 also gains ≪ oracle, failing in the opposite direction (missing sink → excess high-n energy).
  - If either off-oracle arm keeps ≥ 85% of the oracle gain, as in KF, the gain is not driven by the shifted coefficient. Either outcome is reportable.
- **Inferred-ν arm.** The residual is linear in ν, so ν̂ = −⟨R₀, u_xxxx⟩/⟨u_xxxx, u_xxxx⟩ with R₀ = u_t + u u_x + u_xx on the prediction. One line; matches the "inferred viscosity" note at adapt.yaml:12.

### 1.5 Grid, time, ensemble

| item | value | rationale |
|---|---|---|
| domain | L = 64 | PDE-Refiner/LPSDA |
| stored resolution | N = 256 (dx = 0.25, q_Nyq = 12.6) | PDE-Refiner; at ν=0.5, dx̃ = dx/√ν = 0.35, same as the common L=22/N=64 setup |
| solver resolution | N = 512, spectral crop → 256 | mirrors KF res256→res128 FFT crop; G1 checks convergence |
| solver step | h = 0.025 (ETDRK4) | Kassam–Trefethen use h = 0.25 at ν=1; time ∝ ν, so ×5 margin at ν=0.5 |
| raw frame spacing | Δt_raw = 0.125, 257 raw frames (0…32 tu) | file (300, 257, 256) fp32 |
| model/residual frames | sub_t = 2 → T_eff = 129 at Δt = 0.25, window 32 tu | = 2.9 T_λ (source), 5.8 T_λ (target); target ρ<0.9 horizon must fall inside the window |
| burn-in | 500 tu, both ν | 45 T_λ source / 90 T_λ target; PDE-Refiner uses only 360·0.2 ≈ 72 tu |
| IC pre-burn-in | Σ_{m=1}^{10} A_m sin(2πℓ_m x/L + φ_m), mean removed (LPSDA/PDE-Refiner family); seed = base + chain index (as generate_pilot.py:53) | independent chains; post-burn-in IC = attractor snapshot |
| chains | 300 independent per ν; train [0:240], val [240:270], test [270:300] | msc/configs/configs.yaml split, reused verbatim |
| precision | fp64 solve, fp32 store | as KF (generate_pilot.py:65) |
| size | 300·257·256·4 B ≈ 79 MB per ν | negligible |

- **Frame spacing vs residual floor.** ns.py:119–120 uses a 2nd-order central difference. Estimated relative error (ωΔt)²/6 at the target energy peak, with ω ~ qU and U ~ 1.8 (unverified): ≈ 3% at Δt = 0.25, ≈ 13% at n≈20.
- G3 gates this: the GT residual floor must stay below the misspecification term (ratio < 0.1).
- Fallbacks if G3 fails: sub_t = 1 (257 frames at 0.125), or a 5-point 4th-order stencil (loses 4 frames, not 2).
- Grid cost: the (x,t) grid is 256 × 129 ≈ 33k points, against 128² × 65 ≈ 1.06M for KF (32× smaller), so 129 frames are affordable.

---

## 2. Solver
- **ETDRK4** (Kassam & Trefethen 2005, SIAM J. Sci. Comput. 26:1214) in torch, batched over chains.
  - Diagonal linear operator L̂(q) = q² − ν q⁴.
  - φ-coefficients via M = 32 complex contour points, complex128.
  - Nonlinear term −½ i q · FFT(u²), 2/3-dealiased at N = 512.
- New file `scripts/generate_ks.py`, about 110 LOC. Pattern follows generate_pilot.py: per-ν output, `--start/--n_samples` sharding, atomic write.
- Cost: (500 + 32)/0.025 ≈ 21.3k steps × ~8 rfft of a (300, 512) fp64 batch.
  - One CPU core: a few minutes per ν.
  - TITAN X Pascal: fp64 at 1/32 rate, still < 1 min.
  - Note `nproc` = 1 on student05 (cgroup). Either device works; use GPU by default for headroom.
- **jax-cfd / exponax: not worth it.**
  - Not installed in the conda env. `pip list` shows torch 2.6.0+cu124, neuraloperator 2.0.0, numpy 2.3.5, scipy 1.15.3; no jax, no exponax.
  - Adding them means a JAX/CUDA stack beside torch 2.6 on Pascal for ~60 LOC of ETDRK4.
  - Use exponax only as an offline cross-check reference (local venv), optional.

---

## 3. Code delta

### 3.1 Design decision: generalise in place (`pde` key + dimension-agnostic spectral helpers), not a parallel `ks/` module
- A parallel module duplicates eval.py (860), loop.py (260), adapt.py (334), locus.py (394) and setup.py (263), ≈ 2.1k LOC. It also forks the metric layer, so KS and KF numbers stop being comparable by construction.
- Generalising touches ~15 call sites. KF behaviour must stay numerically identical, guarded by a regression test (forward_bands on a KF checkpoint before/after, arrays bit-identical).
- Registry = one top-level train-config key `pde: kf|ks` (default kf), resolved in 4 places: KFLitModule loss, setup.load_model/build_dataset (data map), eval residual fn, loop._loss_fn.
- **Parameter naming: key the shift on the loss coefficient, not on Re.** Aliasing re := 1/ν fails twice: the ν_loss = 0 arm needs re = ∞, and the ν_t = 0.7 contingency gives re = 1.43, not an int.
  - `Regime` (setup.py:23–59) stores floats `nu_op`/`nu_test` directly. KF construction keeps passing `1/re`, so KF behaviour is unchanged.
  - Adapt config gains `op_nu/target_nu/pde_nu` (null for KF). `loss_re` (loop.py:22–34) becomes `loss_nu(cfg)`. KSLoss/KSResidual take ν directly.
  - paths.yaml `ks_nu` is keyed by ν as a string ("1.0", "0.5", "0.7").
  - KF keeps its `op_re/target_re/pde_re` keys and maps them to ν once, in one helper. Cost ≈ +40 LOC over the alias hack.

### 3.2 Touch list (file:line → change → LOC)

| # | file:line | KF/2D-specific because | KS change | LOC |
|---|---|---|---|---|
| 1 | new `scripts/generate_ks.py` | — | ETDRK4 generator, §2 | ~110 |
| 2 | new `src/pde/ks.py` | ns.py:57–125 is 2D vorticity + forcing; ns.py:313–323 `lp.rel(Du, forcing)` divides by the forcing norm, which is zero in KS | `KSResidual(nu, t_interval).residual(u)` → (B,S,T−2) + terms (ut, adv, antidiff, hyper). Spectral u_x, u_xx, u_xxxx via rfft with 2/3 mask (k⁴ amplifies output noise ×2.5e4 at Nyquist). CD in time as ns.py:119–120. `KSLoss(nu, t_interval, ...)`: data = lp.rel (d=2); pde = batch-mean of per-sample sqrt(mean(Du²))/res_scale, unsquared like `lp.rel(Du, forcing)`, so the banked pde + 5·ic balance transfers; ic = lp.rel(u[...,0], u0). res_scale = fixed config constant (RMS of u_t on source train GT, computed in S1), never a pred-dependent denominator; eval `res_rms` divides by the same constant | ~80 |
| 3 | src/datasets/kf_dataset.py:51,53,59–63,75–79 | `[:, ::sub_t, :, :]` and `.transpose(0,2,3,1)` hard-code 2 spatial dims | `chunk[:, ::sub_t]`, `np.moveaxis(chunk, 1, -1)`; rest (ic = traj[...,0]) already generic | ~6 |
| 4 | src/models/kf_fno.py:12–29 (`get_grid3d`), 52, 63–71 (`prepare_input`), 294 (`permute(0,4,1,2,3)`) | 2 spatial grids + 5D permute | grid per spatial axis from `ic.ndim-1`; `x.movedim(-1, 1)`; KS input channels [gridx, gridt, ic] = 3. `get_model` with `n_modes=[32,16]` (len 2) gives the (x,t) FNO directly. Name clash: `model_arch: fno2d` (kf_fno.py:186, 212–231) is the KF time-as-channel model; keep `model_arch: fno` for KS | ~20 |
| 5 | src/models/kf_module.py:8, 40–49, 114, 137 | KFLoss hard-wired; `LpLoss(d=3)` | loss chosen by `cfg.pde`; `d = pred.ndim − 2` | ~15 |
| 6 | src/models/kf_unet.py:7–29 (PeriodicConv3d), 100–176 (spatial mixers rfft2) | 3D convs, 2D spatial FFT | UNet over (x,t): Conv2d, circular x / zero t, rfft mixer on x. **Defer**: FNO first; UNet only if the architecture contrast is needed for KS | ~80 (deferred) |
| 7 | configs: new data/ks.yaml, loss/ks_pino.yaml, model/fno_ks.yaml, experiment/ks_pino_nu1_pretrain.yaml, ks_pino_nu05_pretrain.yaml; callbacks → `default` | configs/callbacks/kf.yaml:12 uses KFVisualizerCallback (expects (B,1,S,S,T), kf_visualizer.py:12) | yaml only; `pde: ks`, data_channels 3, n_modes [32,16], hidden 64, temporal_pad 5, sub_t 2, loss nu 1.0. **`loss.t_interval: 32.0` is mandatory**: ns.py:119 sets dt = t_interval/(nt−1), and probe.py:27 feeds `loss.t_interval` to eval. Inheriting kf_pino.yaml's 1.0 gives dt = 1/128 instead of 0.25, a silent 32× u_t inflation | ~70 yaml |
| 8 | msc/configs/paths.yaml `data:` | only `kf_re` map | `ks: <dir>`, `ks_nu: {"1.0": KS_L64_nu1.0_N256_T256.npy, "0.5": KS_L64_nu0.5_N256_T256.npy}`, `ks_checkpoints:` | ~8 yaml |
| 9 | msc/tta/setup.py:23–59 (`Regime`), 77–92 (`data_path_for_re`), 107–130 (`resolve_regime`), 219 (`build_fno_kf`), 242–263 | int-Re regime; reads `kf_re` only | Regime stores floats nu_op/nu_test (§3.1). `data_path_for(pde, param)` branches on `cfg.pde`. `path_re` (62–74) returns None for KS names, so its warning is skipped, harmless. build_fno_kf is already generic via get_model | ~30 |
| 10 | msc/tta/eval/eval.py:26–38 (`cheb_bins`), 41–75 (`band_power*`, `fft2`), 78–94 (`resid_minus_forcing`, NSVorticity), 151–154 (S, n_bands, kinf), 186 (`kf_forward`), 396–426 (`resid_rms` divides by S**4) | 2D shells, fft2, NS residual, 2D Parseval | `kinf` = max over spatial axes of \|n_i\| (reduces to \|n\| in 1D); `fftn` over `dim=range(1, ndim−1)`; residual fn from `pde`; `resid_rms` divisor S^(2d). `corr_curve` (333–364) needs DC = 0: guaranteed by zero-mean data, so assert it in S1. w1_* (659–750) are ravel-based and generic. `cov_rmse` (796) is KF y-axis-specific: skip for KS | ~40 |
| 11 | msc/tta/adapt/loop.py:7–12, 16 (`BANDS` k1-4/k5-7/k8+), 18 (`W1_FRAMES=(4,63)`), 37–47 (`_loss_fn`), 154 (`/F_RMS`), 220 (`kf_forward`) | KF bands, forcing RMS, KFLoss | BANDS → config. KS proposal: n1–7 (source energetic), n8–14 (target-only unstable band, shift-sensitive), n15–40 (dissipative), n41+. W1_FRAMES → (4, 127). F_RMS → per-pde `res_scale`. Loss from registry; `loss_re` → `loss_nu`; adapt.py `describe`/`run_name`/`_save_arrays` meta (128–131, 168, 203–205) print ν | ~40 |
| 12 | msc/tta/adapt/locus.py:23–71 (`mask_fno_shifted`, raises on non-3D at :46), 255–293 (`check_mode_index_map`) | (kx,ky,kt) mode layout | add `fno_shifted_1d` layout over (kx,kt), shell = \|kx\|; register in MODE_LAYOUTS (:119) | ~25 |
| 13 | msc/tta/eval/report.py:33, 509, 519; report_tta.py:10, 198 | `F_RMS = 4/√2` (forcing) | per-pde `res_scale` | ~10 |
| 14 | msc/tta/adapt/configs: adapt.yaml (+ op_nu/target_nu/pde_nu, null for KF); new experiment/ks_fno.yaml (ckpt, op_nu 1.0, target_nu 0.5, pde_nu null\|0\|1.0), locus/modes_ks.yaml | — | yaml | ~20 yaml |
| 15 | tests (tests/pde, tests/tta, tests/solver exist) | — | (a) KS spectral derivatives exact on sin(2πnx/L); (b) ETDRK4: zero-mean preserved, energy-balance identity ⟨u_x²⟩ ≈ ν⟨u_xx²⟩ on a burned-in sample; (c) KSLoss residual of a GT chain ≈ floor; (d) residual dt == stored frame spacing (0.25) from config; (e) KFDataset 1D shapes; (f) KF regression: eval/loss outputs unchanged on a fixed KF tensor | ~140 |

Total: ≈ 515 LOC Python + ≈ 100 yaml, plus 80 deferred (UNet). Unchanged: probe.py, adapt.py (except describe strings), kf_datamodule.py, train_kf.py, the split.

---

## 4. Stationary IC sampling (stop/start criterion prerequisite)
- **Requirement.** The IC ensemble is a sample of the target invariant measure μ_ν. Then E_IC(n) = ⟨|û_n(t₀)|²⟩ over ICs is the label-free reference: ICs are model inputs, so using them is OOD-legal. GT E(n,t) is t-invariant over the window, so a faithful prediction has E_pred(n,t) ≈ E_IC(n) for all t.
- **Procedure.** Random sum-of-sines IC, independent seed per chain → 500 tu burn-in → t₀ snapshot = IC → 32 tu rollout. Zero mean is enforced at IC and conserved; assert |mean| < 1e-10 per frame.
- **Stationarity checks (S1-G4):**
  1. E(n) at frame 0 vs frames 64 and 128 of the ensemble: ratio within bootstrap 95% CI for n ≤ 40.
  2. Burn-in sweep 250/500/1000 tu on 30 chains: E(n) unchanged within CI.
  3. Ensemble E(n) vs time-average of one 10⁴-tu chain (ergodic reference).
  4. Per-mode |û_n|² ≈ exponential distribution (KS-test; uses translation invariance).
- **Estimator noise.** |û_n|² is ~exponential, so relative SE per mode ≈ 1/√N_IC: 6.5% with the 240-IC pool, 18% with the 30 held-out. Pool modes into the §3.2/#11 bands, or ~12 log bands, for the criterion; report the floor as GT-vs-GT split-half (120/120), the same convention as the W1 floor in 5_tta-thesis.md.
- **Translation invariance.** E(n) is shift-invariant, so random x-shifts add no information. Do not augment.

---

## 5. Risks

| risk | effect | mitigation / gate |
|---|---|---|
| CD temporal floor comparable to misspec term | residual signal buried | G3; fallback sub_t=1 or 4th-order stencil |
| k⁴ noise amplification of network output | residual dominated by n > 85 garbage | 2/3-dealias inside KSResidual; analog of `pde_band_kmax` (ns.py:267–269) |
| zero-forcing normalisation | `lp.rel(Du, 0)` → ~1/eps garbage, same failure mode as ns.py:253–255 | fixed `res_scale`; unit test |
| source surrogate horizon > window (one-shot FNO too good at ν=1) | in-distribution ρ horizon censored; OOD gap understated | S3 gate: target-cross censoring ≈ 0. If ρ is still > 0.9 at window end on target, extend raw window to 64 tu (Δt_raw 0.25) |
| OOD gap too large (zero-shot rel_l2 ~1 at 1→0.5: scale shift √2, peak n 7→10) | TTA cannot recover; no signal | S3 gate: zero-shot rel_l2 in n1–14 below the target-native-trained upper bound + margin. Contingency only if it fails: ν_t = 0.7 (L̃ = 76.5, still chaotic) |
| trivial/steady KS solutions (u=0, cellular steady states, travelling waves) | residual-only collapse | u=0 from a nonzero IC violates the u_xx growth term; ic_weight=5 anchor as banked; monitor E_pred(n,t) |
| Galilean mode (u + c with x−ct) | nonzero predicted mean | assert/strip DC in eval; zero-mean data |
| Lyapunov/attractor-dimension numbers unverified | window sizing off | G5 measures λ (twin-trajectory, 10 chains per ν) |
| disk quota | /system/user/studentwork 94% used, 13 GB free | KS needs < 0.2 GB; adapt weight snapshots (85 MB each for KF) are the real hog. Keep save_weights off for KS ladders |
| student05 absent from msc/configs/gpus.yaml (last_updated 2026-07-30) | allocation ambiguity | update gpus.yaml before S2 |
| KF regressions from in-place generalisation | banked numbers drift | test #15e bit-identical before merge |

---

## 6. Staged plan (GPU-h on TITAN X Pascal 12 GB)

**S1 — data gen + sanity (≈0.5 GPU-h; ~2 days engineering, items 1, 2, 3, 15a–d).** Generate ν ∈ {1.0, 0.5}, 300 chains each. Gates:
- **G1 resolution.** E(n) at N=256 vs N=512-crop: < 1% relative difference for n ≤ 85, and E(n_Nyq)/E_peak < 1e-8.
- **G2 term share.** RMS of u_t, u u_x, u_xx, ν u_xxxx and the misspec term (ν_s−ν_t)u_xxxx on target GT. Expect misspec/RMS(u_xx) ≈ 0.5–1. Report side by side with KF Re500: ν∇²ω share from `NSVorticity.residual` → (wt, adv, diff) (ns.py:125), at ν = 1/500 and 1/100. This is the table that answers the reviewers.
- **G3 floor.** RMS(R_{ν_t}(GT)) / RMS((ν_s−ν_t)u_xxxx) < 0.1 at Δt = 0.25.
- **G4 stationarity**, §4.
- **G5 chaos.** λ > 0 both ν, and T_λ within 2× of §1.3.
- **G6 (record only).** Energy-balance identity ⟨u_x²⟩ = ν⟨u_xx²⟩ holds to < 1%.

**S2 — pretrain source surrogate (≈4 GPU-h; items 4, 5, 7, 8).**
- FNO (x,t), n_modes [32,16], hidden 64, 4 layers, PINO loss data 5 / pde 1 / ic 1 (kf_pino.yaml weights), 150 epochs, 240 chains.
- ≈ 0.5–1 GPU-h per run (32× fewer grid points than KF); 3–4 runs to fix modes/res_scale.
- Also train the target-native ν=0.5 model (oracle upper bound), ≈ 1 GPU-h.

**S3 — baseline OOD gap (≈0.5 GPU-h; items 9–13).**
- forward_bands for source@source, source@target, native@target on val.
- Report rel_l2 per KS band, ρ horizon with censoring, γ.
- Gates: risks rows 4–5.

**S4 — TTA reproduce (≈20 GPU-h; items 11, 12, 14).**
- Ladder: objective physics (pde 1 + 5·ic), ν_loss ∈ {0 drop-term, 0.5 oracle, 1.0 source, ν̂ inferred} × locus {full, modes_ks} × regime {n1, n20} × 3 pool_offsets.
- 48 runs at ~15–25 min each (500 steps, probe_every 5 over pool + 30 held-out).
- Primary read: off-oracle/oracle gain ratio per arm in band n8–14. Compare against the KF 85% figure.
- Locked test read on the frozen config only (heldout_split=test).

**S5 — criterion test (≈1 GPU-h + CPU).**
- Label-free signal = band-pooled distance of E_pred(n,t) to E_IC(n) (pool ICs), per probe step.
- Spearman vs held-out GT rel_l2 across steps, same protocol as msc/tta/stop/experiments/E00_landscape/survey.py.
- Report oracle-best step vs criterion-chosen step, with the GT-vs-GT split-half floor.

Total ≈ 26 GPU-h (≈ 7 h wall on 4 GPUs). Engineering ≈ 515 LOC + tests dominates (≈ 1–1.5 weeks).

---

## 7. References
- Lippe, Veeling, Perdikaris, Turner, Brandstetter. PDE-Refiner, NeurIPS 2023, arXiv:2308.05732.
  - Read from PDF: Eq. 3; L ~ U[0.9·64, 1.1·64], N = 256, Δt ~ U[0.18, 0.22]; 2048 train trajectories × 140Δt, 128 test × 640Δt; 360-step warm-up; parameter-dependent KS with ν ~ U[0.5, 1.5] (4096/512 trajectories).
  - PDE-Refiner accurate to ≈ 100 s.
- Brandstetter, Welling, Worrall. Lie point symmetry data augmentation (LPSDA), ICML 2022: KS generator (pseudo-spectral method of lines). The KS data source is LPSDA, not PDEArena.
- Gupta & Brandstetter 2022 (PDEArena): modern U-Net used by PDE-Refiner. Whether PDEArena ships a KS dataset was not verified.
- Koehler, Niedermayr, Westermann, Thuerey. APEBench, NeurIPS 2024, arXiv:2411.00180. Exponax ETDRK reference solvers; KS docstring: L > 60 chaotic, L=60/N≥100/dt=0.1 "good values", 200–500-step transient.
- Bar-Sinai, Hoyer, Hickey, Brenner. PNAS 116 (2019), arXiv:1808.04930. KS as the chaotic benchmark for learned discretisations; long-wavelength sinusoid ICs; domain size not extracted.
- Linot & Graham, PRE 101, 062209 (2020), arXiv:2001.04263. L = 22/44/66; dimension ≈ linear in L. Per-L numbers unverified.
- Edson, Bunder, Mattner, Roberts. ANZIAM J. 61 (2019), arXiv:1902.09651. Lyapunov spectra vs L; only the abstract was read.
- arXiv:2001.09131 Table I (citing its Ref. [67], unidentified): λ = 0.043 (L=22), 0.080 (36), 0.089 (60), 0.088 (100).
- Kassam & Trefethen, SIAM J. Sci. Comput. 26 (2005) 1214. ETDRK4 + contour-integral φ-functions.
