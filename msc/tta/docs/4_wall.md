> **Entry rules.** Append every new entry directly below this block and above all older entries — **newest always on top**. Each entry: 
(1) date `# YYYY-MM-DD`, one header per new day; 
(2) raw experiment description **+ why it was run**, naming the paper; 
(3) raw numbers (table untouched + anchor checks); (4) interpretation **strictly bound to the numbers** — no outlook, next steps, mechanism, pivots, or cross-experiment synthesis.

---

# 2026-07-02

Context-frames pretrain — n_context ∈ {5,10,20,30}, Re=100, 128², from scratch, no coarse, 150 ep, batch=1, same loss/arch/opt as op100 baseline. 4th FNO channel: GT frames 0..n_ctx−1 at true time positions, last frame held constant for prediction horizon; data_channels=4 unchanged. Motivation: phase/amplitude co-degeneration (wall.md) suggests IC-only 4th channel gives zero phase-velocity signal → L2 averaging collapses phase; hypothesis: temporal trajectory in channel 4 populates k_t>0 FNO modes and breaks late-time k≤7 wall. configs/experiment/kf_ctx_re100_n{5,10,20,30}.yaml; student09 GPUs 2/3/4 + student06 GPU 1; wandb pretrain-kol ctx-n{5,10,20,30}-re100.

| n_context | val_l2 ep149 |
|-----------|-------------|
| 5         | 0.393       |
| 10        | 0.3937      |
| 20        | 0.391       |
| 30        | 0.3895      |

Flat dose-response: range 0.389–0.394, Δ=0.004 across n5–n30. n30 shares ~46% easy context frames in val_l2 yet matches n5 (8% context) — deflation cannot explain the near-identical values. No post-training k≤7 band eval run; flat dose-response sufficient to reject context-length as lever.

# 2026-06-29

Sibling phase alignment — per-shell energy-weighted cos(Δφ) (A_k, k=1..7) between sibling and orig trajectories at t=(0,8,16,32,64,128); same 60 Re100 ICs and same deterministic seed (amp-then-phase order) as IC sibling trajectory divergence (Exp 1); amp ε=[0.1,0.3,0.5], phs σ=[0.1,0.3,0.6], n_sib=5; solver_check mean=0.0000. Motivation: IC-space relL2 conflates amplitude and phase drift; model exploits phase not amplitude (wall.md shuffle50); test whether IC-proximity implies phase alignment along the trajectory. scripts/perturb/sibling_phase_alignment.py; /system/user/studentwork/wehofer/perturb/sibling_phase_re100/phase_alignment.npz.

AMP A_k — energy-weighted cos(Δφ) per shell (k=1..7), mean over 60 ICs × 5 siblings:

| | t=0 | t=8 | t=16 | t=32 | t=64 | t=128 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| ε=0.1 k=1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ε=0.1 k=2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ε=0.1 k=3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 |
| ε=0.1 k=4 | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | 0.999 |
| ε=0.1 k=5 | 1.000 | 1.000 | 0.999 | 0.998 | 0.999 | 0.999 |
| ε=0.1 k=6 | 1.000 | 1.000 | 0.999 | 0.998 | 0.998 | 0.998 |
| ε=0.1 k=7 | 1.000 | 1.000 | 0.999 | 0.998 | 0.998 | 0.997 |
| ε=0.3 k=1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ε=0.3 k=2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 0.995 |
| ε=0.3 k=3 | 1.000 | 1.000 | 0.999 | 0.996 | 0.992 | 0.991 |
| ε=0.3 k=4 | 1.000 | 0.998 | 0.995 | 0.991 | 0.993 | 0.994 |
| ε=0.3 k=5 | 1.000 | 0.996 | 0.989 | 0.984 | 0.985 | 0.984 |
| ε=0.3 k=6 | 1.000 | 0.995 | 0.989 | 0.984 | 0.982 | 0.978 |
| ε=0.3 k=7 | 1.000 | 0.995 | 0.989 | 0.984 | 0.981 | 0.971 |
| ε=0.5 k=1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 |
| ε=0.5 k=2 | 1.000 | 1.000 | 1.000 | 0.999 | 0.993 | 0.981 |
| ε=0.5 k=3 | 1.000 | 0.998 | 0.994 | 0.985 | 0.970 | 0.963 |
| ε=0.5 k=4 | 1.000 | 0.988 | 0.973 | 0.961 | 0.964 | 0.975 |
| ε=0.5 k=5 | 1.000 | 0.982 | 0.961 | 0.948 | 0.948 | 0.941 |
| ε=0.5 k=6 | 1.000 | 0.979 | 0.959 | 0.948 | 0.939 | 0.919 |
| ε=0.5 k=7 | 1.000 | 0.981 | 0.966 | 0.953 | 0.938 | 0.892 |

PHS A_k — k=1..7, mean over 60 ICs × 5 siblings:

| | t=0 | t=8 | t=16 | t=32 | t=64 | t=128 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| σ=0.1 k=1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| σ=0.1 k=2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 |
| σ=0.1 k=3 | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | 0.999 |
| σ=0.1 k=4 | 0.995 | 0.995 | 0.996 | 0.997 | 0.998 | 0.999 |
| σ=0.1 k=5 | 0.995 | 0.995 | 0.996 | 0.997 | 0.998 | 0.998 |
| σ=0.1 k=6 | 0.995 | 0.996 | 0.997 | 0.997 | 0.998 | 0.998 |
| σ=0.1 k=7 | 0.995 | 0.996 | 0.997 | 0.998 | 0.998 | 0.997 |
| σ=0.3 k=1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| σ=0.3 k=2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 0.995 |
| σ=0.3 k=3 | 1.000 | 1.000 | 0.998 | 0.995 | 0.991 | 0.991 |
| σ=0.3 k=4 | 0.956 | 0.960 | 0.966 | 0.976 | 0.987 | 0.993 |
| σ=0.3 k=5 | 0.955 | 0.959 | 0.965 | 0.971 | 0.980 | 0.985 |
| σ=0.3 k=6 | 0.956 | 0.961 | 0.968 | 0.975 | 0.977 | 0.980 |
| σ=0.3 k=7 | 0.955 | 0.963 | 0.971 | 0.978 | 0.979 | 0.972 |
| σ=0.6 k=1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 |
| σ=0.6 k=2 | 1.000 | 1.000 | 1.000 | 0.999 | 0.994 | 0.980 |
| σ=0.6 k=3 | 1.000 | 0.999 | 0.995 | 0.982 | 0.966 | 0.961 |
| σ=0.6 k=4 | 0.833 | 0.846 | 0.868 | 0.906 | 0.951 | 0.974 |
| σ=0.6 k=5 | 0.834 | 0.843 | 0.860 | 0.882 | 0.917 | 0.940 |
| σ=0.6 k=6 | 0.841 | 0.856 | 0.874 | 0.892 | 0.903 | 0.919 |
| σ=0.6 k=7 | 0.835 | 0.854 | 0.884 | 0.907 | 0.904 | 0.890 |

Amp-perturbed A_k(t=0)=1.000 for all shells (phases structurally unchanged by perturb_amp). Minimum A_k at t=128: ε=0.1→0.997, ε=0.3→0.971, ε=0.5→0.892 (k=7); values for k=1 remain ≥0.999 at all levels and frames. Amp A_k(k=4) at ε=0.5: 1.000→0.961 (min at t=32) then recovers to 0.975 at t=128. Phs-perturbed k=1,2,3: A_k=1.000 at t=0 and remain ≥0.961 throughout (unperturbed shells, control check). Phs-perturbed k=4-7: A_k monotonically increases from t=0 to t=128 at all levels — σ=0.3 k=4: 0.956→0.993 (+0.037); σ=0.6 k=4: 0.833→0.974 (+0.141); σ=0.6 k=7: 0.835→0.890 (+0.055). Re-alignment rate decreases with shell index (k=4 faster than k=7) at matched σ.

---

IC sibling trajectory divergence — k=4-7 Chebyshev-shell amplitude (lognormal exp(ε·N(0,1)), conjugate-symmetric) and phase (antisymmetric ±θ, θ∼N(0,σ²)) perturbations on 60 Re100 ICs (30 x-jet, 30 y-jet by |jet_ratio|, pool[0:200]), ε=[0.1,0.3,0.5], σ=[0.1,0.3,0.6], n_sib=5; NS solver float64 adaptive dt; probe t=(8,16,32,64,128); metric: k≤7 relL2 sibling vs orig trajectory + within-level Spearman(ic_dist, traj_div); solver_check mean=1.7e-6. Motivation: establish whether k=4-7 IC perturbations preserve physical proximity along NS trajectories as a precondition for synthetic-sibling auxiliary-input training. scripts/perturb/ic_sibling_divergence.py; /system/user/studentwork/wehofer/perturb/ic_sibling_re100/distances.npz.

| | ic_dist | t=8 | t=16 | t=32 | t=64 | t=128 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| amp ε=0.1 | 0.055 | 0.054 | 0.052 | 0.050 | 0.046 | 0.043 |
| amp ε=0.3 | 0.176 | 0.172 | 0.167 | 0.159 | 0.146 | 0.136 |
| amp ε=0.5 | 0.337 | 0.330 | 0.321 | 0.305 | 0.279 | 0.261 |
| phs σ=0.1 | 0.056 | 0.055 | 0.054 | 0.051 | 0.047 | 0.044 |
| phs σ=0.3 | 0.165 | 0.161 | 0.157 | 0.149 | 0.137 | 0.127 |
| phs σ=0.6 | 0.319 | 0.312 | 0.304 | 0.289 | 0.264 | 0.243 |

Spearman(ic_dist, traj_div) within level:

| | t=8 | t=16 | t=32 | t=64 | t=128 |
|---|:--:|:--:|:--:|:--:|:--:|
| amp ε=0.1 | 1.000 | 0.998 | 0.990 | 0.966 | 0.852 |
| amp ε=0.3 | 1.000 | 0.998 | 0.992 | 0.976 | 0.862 |
| amp ε=0.5 | 1.000 | 0.998 | 0.992 | 0.979 | 0.904 |
| phs σ=0.1 | 0.999 | 0.998 | 0.991 | 0.969 | 0.838 |
| phs σ=0.3 | 0.999 | 0.997 | 0.989 | 0.962 | 0.843 |
| phs σ=0.6 | 0.999 | 0.998 | 0.991 | 0.965 | 0.809 |

ic_dist ≈ traj_div at t=8 (amp ε=0.3: ic_dist=0.176, traj=0.172, ratio=0.98). Traj_div decreases t=8→t=128 at all levels (amp ε=0.3: 0.172→0.136, −21%): Re100 attractor contracts sibling trajectories. Spearman ≈1.000 at t=8 for all levels; degrades to 0.81–0.90 at t=128. Amplitude and phase perturbations produce equivalent trajectory proximity at matched ic_dist (phs σ=0.3 ic_dist=0.165 traj_div(t=8)=0.161 vs amp ε=0.3 ic_dist=0.176 traj_div(t=8)=0.172).

Sibling model error proximity — same 60 Re100 ICs and k=4-7 siblings (deterministic seed regeneration, bit-exact contract amp=0.0 phs=0.0) through op100 (pvqq97sq, 4-ch, 128²); each IC scored against its own NS-solver trajectory (Option A); metric: per-shell k=1..7 relL2(PINO(IC), solve(IC)), delta_err=|err(sib)−err(orig)| averaged over k=1..7; within-level Spearman(ic_dist, delta_err); probe t=(8,16,32,64,128); orig_err mean=0.048 std=0.050; solver_check mean=1.7e-6. Motivation: test whether op100 error landscape is smooth w.r.t. k=4-7 IC proximity. scripts/perturb/sibling_error_proximity.py; /system/user/studentwork/wehofer/perturb/sibling_error_re100/error_proximity.npz.

| | t=8 | t=16 | t=32 | t=64 | t=128 | mean Δerr |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| amp ε=0.1 | 0.485 | 0.396 | 0.419 | 0.613 | 0.808 | 0.022 |
| amp ε=0.3 | 0.785 | 0.701 | 0.697 | 0.830 | 0.878 | 0.107 |
| amp ε=0.5 | 0.832 | 0.754 | 0.768 | 0.880 | 0.907 | 0.212 |
| phs σ=0.1 | 0.502 | 0.464 | 0.493 | 0.662 | 0.815 | 0.022 |
| phs σ=0.3 | 0.598 | 0.522 | 0.551 | 0.695 | 0.789 | 0.101 |
| phs σ=0.6 | 0.584 | 0.566 | 0.611 | 0.756 | 0.795 | 0.221 |

Spearman increases t=8→t=128 at all levels (amp ε=0.1: 0.485→0.808; ε=0.5: 0.832→0.907) — opposite direction to trajectory divergence. delta_err monotone in perturbation level: amp ε=0.1/0.3/0.5 mean=0.022/0.107/0.212; phs σ=0.1/0.3/0.6 mean=0.022/0.101/0.221. Phase Spearman weaker than amplitude at matched ic_dist (σ=0.3 t=128: 0.789 vs ε=0.3 t=128: 0.878). Smallest level (ε=0.1/σ=0.1, ic_dist≈0.055) achieves t=128 Spearman=0.808/0.815 with mean Δerr=0.022.

---

k7 oracle with coarse_shuffle_p=0.5 (shuffle50) — does replacing the matched k≤7 coarse channel with a random other sample's coarse at 50% of training steps produce a model that (a) rejects phase-mismatched coarse (shuffle_coarse ≈ zero_coarse) and (b) still benefits from phase-matched coarse (with_coarse < null)? Motivation: distinguish whether the oracle model exploits low-k phase alignment or merely spectral energy statistics. Three-arm eval added to `msc/tta/eval.py` and `scripts/coarse_oracle_gate.py`: shuffle_coarse draws random j≠i from dataset, seed=42, deterministic derangement. Same hyperparams as oracle baseline: Re100 128², n_train=200, n_val=60, sub_t=2, 150 ep, milestones=[25,50,75,100], γ=0.5, lr=1e-3, data_weight=5, pde_weight=1, ic_weight=1, hidden_channels=128. ckpt 8xtandil (best.ckpt). Evaluated via `scripts/coarse_oracle_gate.py`, locked Re100 [260:300] n=40.

| ckpt | mode | late_k7 | early_k7 | err_k7 | err_full |
|---|:--:|:--:|:--:|:--:|:--:|
| shuffle50 k7 (8xtandil) | with_coarse | 0.3014 | 0.1242 | 0.2450 | 0.2841 |
| shuffle50 k7 (8xtandil) | shuffle_coarse | 0.5332 | 0.1738 | 0.3985 | 0.4211 |
| shuffle50 k7 (8xtandil) | zero_coarse | 0.5335 | 0.1738 | 0.3986 | 0.4210 |

Anchors: null 4ch err_k7=0.398 (l49ucwcs, banked); dropout50 zero_coarse=0.400 consistent with shuffle50 zero_coarse=0.399.

shuffle_coarse err_k7=0.3985 ≈ zero_coarse err_k7=0.3986 (Δ=0.0001): phase-mismatched coarse is numerically indistinguishable from zero coarse — model rejects wrong-phase backbone. with_coarse err_k7=0.245 vs null 0.398 (−38%): phase-matched coarse remains beneficial after shuffle training. with_coarse err_k7=0.245 vs no-dropout oracle (zdtf3mhp) 0.018 (13.6×): shuffle training substantially reduces oracle ceiling relative to matched-phase training.

---

Coarse NS solver as 5th channel, no shuffle (solver24) — does a free-running NS solver at 24² (Nyquist k=12, dealiasing cutoff ~k≤8), spectrally zero-padded to 128², as the 5th FNO channel yield a trainable improvement over the null 4ch baseline? Motivation: replace the GT k≤7 oracle (unavailable at inference) with a computationally cheap coarse solver; assess ceiling degradation relative to oracle. Coarse file materialized via `scripts/materialize_coarse_solver.py` (spectral FFT-crop IC to 24², solve NS for T=129 frames, spectral zero-pad to 128²). Config kf_coarse_re100_solver24: Re100 128², n_train=200, n_val=60, batch_size=1, sub_t=2, 150 ep, milestones=[25,50,75,100], γ=0.5, lr=1e-3, data_weight=5, pde_weight=1, ic_weight=1, hidden_channels=128, no shuffle. ckpt i32w35z2 (intermediate best.ckpt, training ongoing). Evaluated via `scripts/coarse_oracle_gate.py`, locked Re100 [260:300] n=40.

| ckpt | mode | late_k7 | early_k7 | err_k7 | err_full |
|---|:--:|:--:|:--:|:--:|:--:|
| solver24 no-shuffle (i32w35z2) | with_coarse | 0.1441 | 0.0759 | 0.1213 | 0.1625 |
| solver24 no-shuffle (i32w35z2) | zero_coarse | 0.9749 | 0.9097 | 0.9646 | 0.9647 |

Anchors: null 4ch err_k7=0.398; oracle k7 err_k7=0.018; coarse backbone gate late k≤7=0.144 (same 24² solver data, no operator).

with_coarse err_k7=0.121 vs null 0.398 (−70%); vs k7 oracle 0.018 (6.7× gap). zero_coarse err_k7=0.965 — collapses without coarse; same failure mode as the no-dropout oracle (zdtf3mhp). Checkpoint is intermediate; final numbers subject to change at training completion.

---

# 2026-06-28

Coarse channel dropout training — does training FNO3d (5ch, coarse_k7) with the coarse channel randomly zeroed at probability p_drop ∈ {0.5, 0.8} yield a model that (a) still benefits from the coarse channel at inference and (b) gracefully degrades when the channel is absent? Motivation: oracle model zdtf3mhp collapses to zero_coarse err_k7>1.0 — the model has no fallback path when coarse is missing; channel dropout as a structural fix. Two runs: dropout50 (p=0.5, hjsruqwr, student07 GPU4 GV100) and dropout80 (p=0.8, agbqpod3, student06 GPU1 V100). Same hyperparams as oracle: Re100 128², n_train=200, n_val=60, batch_size=1, 150 ep, milestones=[25,50,75,100], γ=0.5, lr=1e-3, data_weight=5, pde_weight=1, ic_weight=1, hidden_channels=128. Evaluated via `scripts/coarse_oracle_gate.py`, locked Re100 [260:300] n=40.

| ckpt | mode | late_k7 | early_k7 | err_k7 | err_full |
|---|:--:|:--:|:--:|:--:|:--:|
| null 4ch (l49ucwcs) | — | — | — | 0.398 | — |
| k7 oracle no-dropout (zdtf3mhp) | with_coarse | — | — | 0.018 | — |
| k7 oracle no-dropout (zdtf3mhp) | zero_coarse | — | — | >1.0 | >1.0 |
| dropout50 p=0.5 (hjsruqwr) | with_coarse | 0.149 | 0.100 | 0.138 | 0.202 |
| dropout50 p=0.5 (hjsruqwr) | zero_coarse | 0.531 | 0.174 | 0.400 | 0.422 |
| dropout80 p=0.8 (agbqpod3) | with_coarse | 0.192 | 0.111 | 0.167 | 0.230 |
| dropout80 p=0.8 (agbqpod3) | zero_coarse | 0.555 | 0.184 | 0.413 | 0.436 |

Anchors: null 4ch err_k7=0.398 consistent with banked 0.409 (pvqq97sq) and oracle entry above. zero_coarse on dropout models ≈ null confirms fallback recovery.

dropout50 with_coarse err_k7=0.138 vs null 0.398 (−65%); vs oracle 0.018 (7.7× gap remains). dropout50 zero_coarse err_k7=0.400 ≈ null 0.398 (Δ=+0.002): fallback path fully restored — dropout training eliminates the zero_coarse collapse seen in the oracle model. dropout80 with_coarse err_k7=0.167 vs null 0.398 (−58%): weaker coarse utilisation than dropout50, consistent with 80% masking impairing the coarse-dependent path during training. dropout80 zero_coarse err_k7=0.413 ≈ null (Δ=+0.015): fallback also preserved. Both dropout models trade oracle ceiling for fallback robustness; dropout50 achieves a better with_coarse/zero_coarse balance.

---

IC jet anisotropy gate — does the ratio Ω_y/Ω_x of k_⊥=(±1,0) to k_∥=(0,±1) enstrophy in the t=0 IC predict late-time phase dispersion D_late, and does it predict well enough to use anisotropy as a sample-selection gate for a k=±1 conditioning avenue? Motivation: JFM 2024.263 and arXiv:2603.13789 link organized perpendicular-jet ICs (high Ω_y/Ω_x) to more predictable phase branches at Re=100 128² n_f=4. Kill-switch: Spearman ρ(log ratio, D_late) with |ρ|>0.2 AND top-Q D_late < bot-Q D_late − 0.02. D_late = energy-weighted circular phase dispersion over k≤7 modes, late-window frames, M=4 perturbed true-solver members per sample. Evaluated via `scripts/ic_jet_anisotropy_gate.py`, n=100 samples per Re, offset=0, δ=0.1. Re=100 on student06; Re=500 on student07.

| Re | S | n | ρ(log ratio, D_late) | p-value | D_late top-Q | D_late bot-Q | margin | MI (bits) | signal |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 100 | 128 | 100 | −0.409 | 2.37e-05 | 0.0314 | 0.0500 | 0.0186 | 0.547 | NO |
| 500 | 256 | 100 | −0.244 | 0.0146 | 0.0615 | 0.0711 | 0.0096 | 0.450 | NO |

Anchors: null wall err_k7=0.398 (banked); D_late range at Re=500 consistent with gate2_phase_spread (banked 0.07–0.30 over held-out split).

|ρ| > 0.2 at both Re (p<0.05): negative correlation exists — high-jet ICs have lower D_late. Kill-switch fails on the margin condition at both Re: Re=100 margin 0.0186 < 0.02; Re=500 margin 0.0096 < 0.02. D_late is globally compressed at Re=100 (0.031–0.050 range) and at Re=500 within this offset=0 split (0.062–0.071). Ratio quartiles: Re=100 Q1=0.743 median=1.234 Q3=2.442; Re=500 Q1=0.642 median=0.954 Q3=1.373. Absolute quartile separation is insufficient for conditioning despite a statistically significant rank correlation at both Re.

---

Coarse backbone gate — free-running NS solver at 24² from spectral-cropped GT ICs vs GT k≤7, n=16 holdout chains [200:216], no operator. Motivation: empirical test of whether a bare classical solver at k≤7-resolving coarse grid stays correlated with GT k≤7 within T=65 frames (inside Lyapunov horizon N_cap≈278/224), as a precondition for using the coarse solver as a cheap FNO backbone. Evaluated via `scripts/coarse_backbone_gate.py`, 24² grid (Nyquist=12, 2/3-dealiasing cutoff≈8), spectral FFT-crop IC, same Re/dt/forcing as GT generation.

| Re | run | early k≤7 relL2 | late k≤7 relL2 |
|---|---|:--:|:--:|
| 100 | ctrl (full-res solver, true IC) | 0.0000 | 0.0000 |
| 100 | coarse 24² solver | 0.0069 | 0.1444 |
| 500 | ctrl (full-res solver, true IC) | 0.0000 | 0.0000 |
| 500 | coarse 24² solver | 0.0059 | 0.1387 |

Anchors: ctrl=0.000 both Re confirms solver/dt/forcing match GT generation. Null wall=0.398, oracle=0.018 (banked).

Coarse late 0.144 (Re100) / 0.139 (Re500): both below wall (0.398), both above oracle (0.018). No blowup on either Re. Re500 marginally better than Re100 (Δ=0.006). Backbone is correlated with GT k≤7 at late time; gap to oracle (≈8×) is what FNO correction would need to close.

---

Cross-resolution oracle inference sweep — how does err_k7 degrade as the coarse oracle is progressively restricted to k≤kmax (kmax ∈ {1,3,4,5,6,7,15}), tested on both the k7-trained (zdtf3mhp) and k15-trained (mp4dxh5c) 5ch models? Motivation: locate the minimum spectral resolution of backbone signal needed to break the k≤7 wall; establishes whether the cliff is sharp or gradual and at which kmax the oracle loses utility relative to the null baseline (err_k7=0.398). Evaluated via `scripts/coarse_oracle_gate.py`, locked Re100 [260:300] n=40, op_re=100.

| coarse kmax → model | late_k7 | early_k7 | err_k7 | err_full |
|---|:--:|:--:|:--:|:--:|
| k15 → k15 (native) | — | — | 0.018 | — |
| k7 → k15 | 0.0187 | 0.0187 | 0.0188 | 0.1734 |
| k6 → k15 | 0.1370 | 0.1293 | 0.1360 | 0.2188 |
| k5 → k15 | 0.2271 | 0.2155 | 0.2261 | 0.2824 |
| k4 → k15 | 0.3445 | 0.3313 | 0.3408 | 0.3780 |
| k3 → k15 | 0.5597 | 0.5404 | 0.5542 | 0.5728 |
| k1 → k15 | 0.7454 | 0.7345 | 0.7436 | 0.7526 |
| zero → k15 | 1.0031 | 0.9917 | 1.0021 | 1.0018 |
| k7 → k7 (native) | — | — | 0.018 | — |
| k7 → k7 | 0.0187 | 0.0187 | 0.0188 | 0.1734 |
| k6 → k7 | 0.1460 | 0.1003 | 0.1383 | 0.2038 |
| k5 → k7 | 0.2221 | 0.1399 | 0.2064 | 0.2553 |
| k4 → k7 | 0.3315 | 0.2057 | 0.3033 | 0.3375 |
| k3 → k7 | 0.5685 | 0.3957 | 0.5267 | 0.5432 |
| k1 → k7 | 0.7597 | 0.6069 | 0.7213 | 0.7281 |
| zero → k7 | 0.9892 | 0.8053 | 0.9425 | 0.9424 |

Anchor: zero_coarse k7-model 0.9425 / k15-model 1.002 reproduce banked ablation. Null 4ch baseline err_k7=0.398 (banked).

k4 crosses below null baseline on both models (k7-model 0.303, k15-model 0.341 vs 0.398): k4 is the minimum useful oracle. k3 collapses above null baseline (0.527/0.554 > 0.398): k3 carries no net gain over the null model. Cliff is between k3 and k4 — not gradual. k5 gives roughly half the null baseline error (0.206/0.226); k6 reduces it by 2/3 (0.138/0.136). k7→k15 and k7→k7 are indistinguishable from native oracle (Δ<0.001): the k8–k15 content trained into the k15 model is dispensable at inference. Both models track each other closely across the sweep (max Δ err_k7 = 0.017 at k4); neither generalises better to mismatched coarse resolution.

---

Oracle coarse-trajectory training — does feeding the GT k≤7 (or k≤15) Chebyshev-lowpass trajectory as a 5th input channel to FNO3d break the k≤7 late-time wall? Motivation: test whether the wall is a capacity/architecture limit or a consequence of missing low-k structural information at inference. Three runs: (A) 4ch null baseline (l49ucwcs, no coarse channel, identical architecture); (B) 5ch k7 oracle (zdtf3mhp, coarse_k7 file); (C) 5ch k15 oracle (mp4dxh5c, coarse_k15 file). All Re100 128², n_train=200, n_val=60, batch_size=1, sub_t=2, 150 ep, milestones=[25,50,75,100], γ=0.5, lr=1e-3, data_weight=5, pde_weight=1, ic_weight=1, hidden_channels=128, lifting_channel_ratio=0. `cheb_lowpass(field, kmax)` zeroes modes max(|kx|,|ky|)>kmax. Evaluated via `scripts/coarse_oracle_gate.py` on locked Re100 [260:300] n=40. wandb project pretrain-kol.

| ckpt | mode | err_k7 | err_full |
|---|:--:|:--:|:--:|
| null 4ch (l49ucwcs) | — (no coarse ch.) | 0.398 | — |
| k7 oracle (zdtf3mhp) | with_coarse | 0.018 | — |
| k7 oracle (zdtf3mhp) | zero_coarse | >1.0 | >1.0 |
| k15 oracle (mp4dxh5c) | with_coarse | 0.018 | 0.173 |
| k15 oracle (mp4dxh5c) | zero_coarse | 1.002 | 1.002 |

Anchor: banked op100 wall err_k7=0.409 (pvqq97sq); null 4ch err_k7=0.398 consistent.

Oracle with_coarse err_k7=0.018 vs op100 wall 0.409 (Δ=−0.391, −96%): GT lowpass oracle breaks the wall. k7 and k15 oracles reach identical err_k7=0.018 — spectral richness of the oracle beyond k7 does not improve in-band accuracy. zero_coarse on both 5ch models >1.0: worse than a constant predictor; the model does not fall back to the null path when the coarse channel is zeroed. null 4ch err_k7=0.398 vs banked op100 0.409 (Δ=−0.011): consistent, null baseline reproduces within noise.

---

# 2026-06-27

Chain Variant A cold-start — does removing the warm-start initialization change the k≤7 result, testing whether op500 pretrain weights trap chain training in an ineffective local minimum? Same architecture and chain loss as warm-start (k3a39qgi): W1=full-T PINO forward; mid=pred1[:,0,:,:,T//2] (no detach); chain_loss=LpLoss(pred2[:,0,...,:T//2], target[...,T//2:]); total=W1_loss + 1.0×chain_loss. No pretrain checkpoint (random init), 100 epochs, milestones=[50], γ=0.5, Re500 128², data_weight=5, pde_weight=1, ic_weight=1, batch_size=1, lr=1e-3. Quadro GV100 32GB (student07 GPU4), wandb s5lkfjg1, project chain-wall-pilot. Evaluated via `scripts/band_gate.py --op_re 500 --test_re 500 --ckpt chain-wall-pilot/s5lkfjg1/checkpoints/best.ckpt`, locked held-out Re500 [260:300] n=40, sub_t=2, time_scale=1.0, temporal_pad=5.

| variant | early (t=1..8) | late (t=57..64) | k≤7 aggr |
|---|:--:|:--:|:--:|
| op500 full-T (38o0kj3y) | 0.2527 | 0.4723 | 0.3714 |
| chain warm-start (k3a39qgi) | 0.2456 | 0.4772 | 0.3704 |
| chain cold-start (s5lkfjg1) | 0.2452 | 0.4742 | 0.3684 |

resu_f7=0.880, resgt_f7=0.002. val_l2=0.3986 epoch 99.

Anchor: resu_f7=0.880 consistent with warm-start (0.861); resgt_f7=0.002 reproduces across all runs. op500 baseline k≤7 aggr 0.3714 unchanged.

Cold-start k≤7 aggr 0.3684 vs warm-start 0.3704 (Δ=−0.002) vs baseline 0.3714 (Δ=−0.003 from baseline). Early frames 0.2452 vs warm-start 0.2456 (Δ=−0.000): initialization does not affect early IC propagation quality. val_l2 flatlined at 0.399 in both chain variants — same convergence floor regardless of warm vs cold init. Late error 0.4742 vs warm-start 0.4772 (Δ=−0.003): marginal late-time improvement does not change aggregate result.

---

Band-masked data loss sweep — does restricting the supervised data loss to k≤kmax (Chebyshev L∞ lowpass hard-mask applied to both numerator and denominator of the relative L2) lower the held-out late k≤7 wall? Hypothesis: k>kmax modes are near-random phase at late time, injecting noisy gradient that prevents the model from minimising the in-band loss. Six arms: kmax ∈ {4,5,6,7} warm-started from op500 (38o0kj3y, late=0.4723), plus k4-cold (random init, 300 epochs, milestones=[50,100,150,200,250]) and k4-dataonly (pde_weight=0, ic_weight=0, warm-start). All warm-start arms: 150 epochs, milestones=[25,50,75,100], γ=0.5, lr=1e-3, Re500 128², data_weight=5 (dataonly: 1), pde_weight=1 (dataonly: 0), ic_weight=1 (dataonly: 0), batch_size=1. Checkpoint selected on val_l2_band (k≤kmax lowpass-matched validation metric). wandb project bandmask-kol: icg0ibec (k4w), mfo7m3v5 (k5w), jn3rtiza (k6w), q14y0bk8 (k7w), 01tgs39h (k4cold), 0ut5glfh (k4do). Evaluated via `scripts/band_gate.py --op_re 500 --test_re 500`, locked held-out Re500 [260:300] n=40, sub_t=2, time_scale=1.0, temporal_pad=5.

| variant | early (t=1..8) | late (t=57..64) | k≤7 aggr |
|---|:--:|:--:|:--:|
| op500 full-T (38o0kj3y) | 0.2527 | 0.4723 | 0.3714 |
| mask-k4 warm (icg0ibec) | 0.2583 | 0.4742 | 0.3774 |
| mask-k5 warm (mfo7m3v5) | 0.2830 | 0.4914 | 0.3962 |
| mask-k6 warm (jn3rtiza) | 0.2684 | 0.4874 | 0.3886 |
| mask-k7 warm (q14y0bk8) | 0.2693 | 0.4817 | 0.3855 |
| mask-k4 cold (01tgs39h) | 0.2575 | 0.4774 | 0.3757 |
| mask-k4 dataonly (0ut5glfh) | 0.2478 | 0.4912 | 0.3864 |

resu_f7: 0.929/0.917/0.908/0.900/0.939/0.904 (k4w/k5w/k6w/k7w/k4cold/k4do). resgt_f7=0.002 all runs. Best-checkpoint val_l2_band (validation split, band-matched): 0.3036 @ep4 (k4w), 0.3374 @ep4 (k5w), 0.3565 @ep9 (k6w), 0.3661 @ep4 (k7w), 0.2991 @ep54 (k4cold), 0.3115 @ep9 (k4do).

Anchor: resgt_f7=0.002 reproduces across all 6 runs. op500 baseline k≤7 aggr 0.3714 unchanged.

All 6 arms ≥ op500 baseline on late k≤7 (Δ_late: k4w +0.002, k5w +0.019, k6w +0.015, k7w +0.009, k4cold +0.005, k4do +0.019). k≤7 aggr all above baseline (0.3714): range 0.3757–0.3962. Narrower mask less degraded on late: k4 (0.4742) < k7 (0.4817). k4-cold late 0.4774 vs k4-warm 0.4742 (Δ=+0.003): warm-start marginally better on late despite shorter training. k4-dataonly worst on late (0.4912): removing pde+ic terms degrades held-out late beyond all other arms. early: k4w and k4cold within ±0.006 of baseline (0.2527); k5w degrades most (+0.028). Best-checkpoint val_l2_band @ep4 for k4w/k5w/k7w (val metric converges in ≤9 epochs for all warm-start arms). resu_f7 0.900–0.939 consistent with banked operator residual range.

---

# 2026-06-26

Ablation B (periodic pad) + Ablation C (FNO2d time-as-channel) vs Re100 baseline — do (B) replacing zero-fill temporal padding with periodic wrapping (Cao et al. arXiv:2405.17211 §2.2) or (C) replacing FNO3d with FNO2d where T=65 frames are stacked as output channels eliminating the temporal FFT axis (Atif et al. arXiv:2409.14660 §2.1) improve val_l2 over the op100 baseline? Both ablations identical to baseline (pvqq97sq) in data/loss/opt (Re100, n_train=200, n_val=60, batch=1, sub_t=2, 150 epochs, lr=1e-3, milestones=[25,50,75,100], γ=0.5, data_weight=5, pde_weight=1, ic_weight=1) and model width (hidden_channels=128, lifting_channel_ratio=0). B: pad_mode=periodic, FNO3d, student09 GPU2, wandb d7pezvlb. C: model_arch=fno2d, n_modes=[8,8], out_channels=65, student09 GPU1, wandb 1tjkzlbl. Metric: val_l2 = global relative L2 over all T frames and k, validation split (not band-gated). Baseline val_l2 logged every 10 epochs; ablations every 50 epochs (kf callback).

| run | ep49 | ep99 | ep149 | best |
|---|:--:|:--:|:--:|:--:|
| baseline FNO3d (pvqq97sq) | 0.3911 | 0.3915 | 0.3928 | 0.3902 @ep69 |
| B periodicpad (d7pezvlb) | 0.3974 | 0.3957 | 0.3969 | 0.3957 @ep99 |
| C fno2d (1tjkzlbl) | 0.4336 | 0.4362 | 0.4402 | 0.4336 @ep49 |

Anchor: baseline ep49=0.3911 consistent with full logged curve (best 0.3902 @ep69, prior run verified).

B best=0.3957 vs baseline 0.3902: Δ=+0.0055 (+1.4%); periodic padding does not improve val_l2. C best=0.4336 vs baseline 0.3902: Δ=+0.0434 (+11.1%); FNO2d time-as-channel substantially worse than FNO3d. C val_l2 is monotonically increasing ep49→ep149 (0.4336→0.4402), the only run where val_l2 rises after ep49.

---

Chain Variant A — does training with a second forward pass from the predicted midpoint (gradient flowing through the handoff frame) improve k≤7 late-time accuracy over the standard PINO operator? KFLitModuleChain subclasses KFLitModule, overrides training_step only: W1 = full-T PINO forward pass; mid = pred1[:, 0, :, :, T//2] (0-based, no detach); W2 = second full-T forward from mid; chain_loss = LpLoss(pred2[:, 0, ..., :T//2], target[..., T//2:]) (relative L2, d=3). Total loss = W1_loss + 1.0 × chain_loss. Warm-started from op500 (38o0kj3y), 50 epochs, Re500 128², data_weight=5, pde_weight=1, ic_weight=1, batch_size=4, lr=1e-3. Quadro GV100 32GB (student07 GPU4). wandb k3a39qgi, project chain-wall-pilot. Evaluated via `scripts/band_gate.py --op_re 500 --test_re 500 --ckpt chain-wall-pilot/k3a39qgi/checkpoints/best.ckpt`, locked held-out Re500 [260:300] n=40, sub_t=2, time_scale=1.0, temporal_pad=5. Output: `scripts/outputs/band_gate_chain_variantA.npz`.

| variant | early (t=1..8) | frame 16 | frame 32 | late (t=57..64) | k≤7 aggr |
|---|:--:|:--:|:--:|:--:|:--:|
| op500 full-T (38o0kj3y) | 0.2527 | 0.2991 | 0.3673 | 0.4723 | 0.3714 |
| chain Variant A (k3a39qgi) | 0.2456 | 0.2887 | 0.3588 | 0.4772 | 0.3704 |

Full k≤7 err_t curve (t=0..64): 0.2180, 0.2271, 0.2324, 0.2386, 0.2445, 0.2491, 0.2535, 0.2577, 0.2616, 0.2653, 0.2691, 0.2729, 0.2767, 0.2806, 0.2846, 0.2887, 0.2929, 0.2970, 0.3013, 0.3056, 0.3099, 0.3143, 0.3186, 0.3230, 0.3273, 0.3317, 0.3361, 0.3406, 0.3451, 0.3496, 0.3542, 0.3588, 0.3634, 0.3680, 0.3726, 0.3772, 0.3818, 0.3863, 0.3907, 0.3950, 0.3992, 0.4034, 0.4076, 0.4117, 0.4159, 0.4200, 0.4241, 0.4282, 0.4323, 0.4363, 0.4403, 0.4442, 0.4481, 0.4519, 0.4555, 0.4590, 0.4625, 0.4658, 0.4691, 0.4724, 0.4756, 0.4788, 0.4820, 0.4853, 0.4887.

Anchor: chain eval run with same script/split as banked op500 baseline; op500 k≤7 aggr 0.3714 reproduced in prior runs. val_l2 flatlined at 0.399 from epoch 32 through epoch 49. Early frames improved (Δ=−0.007 to −0.010 at frames 16, 32). Late frames marginally worse (Δ=+0.005 at t=57..64). k≤7 aggr 0.3704 vs 0.3714 (Δ=−0.001). err_t curve is monotonically increasing and approximately linear across all 65 frames with no inflection at the handoff frame (t=32).

---

Horizon-specialist midpoint gate — does restricting the training horizon to T/2 or T/4 frames improve k≤7 quality at the handoff frame relative to the full-T operator? Gate for the chaining direction: if specialists yield a better IC at frame m, chain training gains a head-start; if not, any improvement from chaining must come from the gradient path alone. Two specialists warm-started from op500 (38o0kj3y), 100 epochs Re500 128², data_weight=5, pde_weight=1, ic_weight=1: T32 (`data_t_hi=32`, `t_interval=0.484375`, wandb ht99jatc) and T16 (`data_t_hi=16`, `t_interval=0.234375`, wandb cyvyt8sx). Evaluated via `scripts/band_gate.py --op_re 500 --test_re 500`, T=65 full inference, locked held-out Re500 [260:300] n=40, sub_t=2, time_scale=1.0, temporal_pad=5. Outputs: `scripts/outputs/band_gate_T{16,32}_specialist.npz`. op500 full-T baseline re-run with identical protocol (same script, same split).

| variant | early (t=1..8) | frame 16 | frame 32 | late (t=57..64) | k≤7 aggr |
|---|:--:|:--:|:--:|:--:|:--:|
| op500 full-T (38o0kj3y) | 0.2527 | 0.2991 | 0.3673 | 0.4723 | 0.3714 |
| T32 specialist (ht99jatc) | 0.2809 | 0.3392 | 0.4757 | 0.6379 | 0.4808 |
| T16 specialist (cyvyt8sx) | 0.2315 | 0.3610 | 0.5730 | 0.7533 | 0.5598 |

Anchor: op500 baseline k≤7 aggr 0.3714 reproduces banked err_k7 (prior run 0.3714, same split). T32 worse than full-T at frame 32 (0.476 vs 0.367, Δ=+0.109). T16 worse than full-T at frame 16 (0.361 vs 0.299, Δ=+0.062). Both specialists uniformly worse than full-T at every frame ≥16. T16 marginally better at early (0.232 vs 0.253, Δ=−0.021) but worse from frame 16 onward. Shorter training horizon does not improve midpoint IC quality; full-T operator is the stronger handoff at both horizons.

---

Lyapunov predictability horizon — measured the maximal Lyapunov exponent λ(Re) of forced 2D Kolmogorov flow [Nastac, Labahn, Magri, Ihme, *Lyapunov exponent as a metric for assessing the dynamic content and predictability of large-eddy simulations*, Phys. Rev. Fluids **2**, 094606 (2017); Alg. 1, Eqs. 5–6] to decide whether the late k≤7 rel-L2 wall is the chaotic predictability limit or lies below it. Perturb a held-out IC by ε‖ic‖ (ε=1e-7, band-limited k≤7 direction), roll reference+perturbed through the true NS solver (`NavierStokes2d`, fixed substep frozen from IC CFL → identical schedule, no adaptive drift), roll 8× past the T=1 data window to expose the linear region, fit λ = slope of ln‖δ(t)‖ over the max-R² window (≥1 e-fold dynamic-range gate); t_p=1/λ (one e-fold), N_cap=t_p/dt (dt=T/128). n=30 held-out ICs (locked split, offset 260), 1 direction each. λ_full=full-field separation rate; g_k≤7=k≤7-band rate; A=within-window amplification ‖δ(T)‖/‖δ(0)‖; T(turn)=window in eddy-turnovers (τ_eddy=1/ω_rms). Solver-only (no operator). `scripts/lyapunov_gate.py --spectrum band --roll_mult 8 --eps 1e-7 --n 30`.

| Re | λ_full | g_k≤7 | N_cap full | N_cap k≤7 | A_full(win) | A_k≤7(win) | T(turn) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 100@128² | 0.461 ± 0.220 | 0.450 ± 0.222 | 278 | 284 | 1.157 | 0.997 | 2.92 |
| 500@256² | 0.572 ± 0.219 | 0.666 ± 0.233 | 224 | 192 | 2.211 | 1.484 | 4.12 |

n=30, fits 30/30 both Re, SEM≈0.04. Anchor: dt-converged (λ identical at dt_factor 1.0/0.5/0.25; shifts to 0.95 only at 4.0, above CFL); ε-tangent (Re500 prior run λ stable 1e-3→1e-7); units sane (T=2.9–4.1 turnovers).

λ rises Re100→Re500 (λ_full 0.461→0.572; g_k≤7 0.450→0.666, ~3.7σ on SEM). N_cap = 278 (Re100) / 224 (Re500) full-field, 284 / 192 k≤7 — all exceed the 128-step data window. Within-window amplification < e at both Re (A_full 1.16 / 2.21; A_k≤7 1.00 / 1.48) ⇒ < one e-fold over [0,T] ⇒ t_p > T. A_k≤7=0.997 at Re100 ⇒ no net within-window growth in the k≤7 band. Operator late-wall onset ≈ step 64 lies 3–4× below N_cap. Prior n=4 run gave the opposite λ ordering (Re100 0.689 > Re500 0.638) — reversed and resolved at n=30.

---

Band-impact gate — which spectral bands in the IC at t_r are causally responsible for the k≤7 late wall? 16 held-out instances each Re, op one-shot prediction taken at t_r=16, 11 IC variants constructed by algebraically splicing GT bands (lo_kc{1,2,3,5,7}: GT k≤kc + pred k>kc; hi_kc{7,16}: pred k≤kc + GT k>kc; mix_lo3_hi8: GT k≤3 + pred k=4..7 + GT k>7; mix_lo7_hi42: GT k≤7 + pred k=8..42 + GT k>42), each variant handed to true NS solver (correct Re, f=−4cos(4y), adaptive dt), integrated to T=65, k≤7 late rel-L2 pooled. op500@Re500 (38o0kj3y) and op100@Re100 (pvqq97sq). S=128, T=65, dt=0.01562. `scripts/band_impact_gate.py --t_r 16 --n 16`. Outputs: `scripts/outputs/band_impact_re{100,500}_tr16.json`.

| variant | Re500 | Re100 |
|---|:--:|:--:|
| full_pred | 0.2907 | 0.3918 |
| full_gt | 0.0000 | 0.0000 |
| lo_kc1 | 0.2632 | 0.2804 |
| lo_kc2 | 0.2391 | 0.2299 |
| lo_kc3 | 0.2112 | 0.1926 |
| lo_kc5 | 0.1287 | 0.1168 |
| lo_kc7 | 0.0727 | 0.0495 |
| hi_kc7 | 0.2871 | 0.3894 |
| hi_kc16 | 0.2906 | 0.3918 |
| mix_lo3_hi8 | 0.1998 | 0.1883 |
| mix_lo7_hi42 | 0.0727 | 0.0495 |

---

Band-impact gate — per-band per-time breakdown for 4 probe variants (full_pred, full_gt, lo_kc7, mix_lo3_hi8). Same run parameters as above; `run()` now also accumulates `err_pt (n_bands, T)` and `gt_pt (n_bands, T)` pooled across instances, saved as `band_impact_re{re}_tr16_{op}.npz`. Per-band rel-L2 computed as `sqrt(err_pt_k / gt_pt_k)` per frame; early = frames t_r..t_r+nlate-1, late = last nlate=8 frames.

Re500 per-shell rel-L2 (k1..k7):
| variant | k1 early | k2 | k3 | k4 | k5 | k6 | k7 | k1 late | k2 | k3 | k4 | k5 | k6 | k7 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| full_pred | 0.060 | 0.199 | 0.302 | 0.457 | 0.678 | 0.867 | 0.939 | 0.077 | 0.293 | 0.383 | 0.462 | 0.683 | 0.806 | 0.863 |
| full_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lo_kc7 | 0.000 | 0.000 | 0.001 | 0.005 | 0.021 | 0.080 | 0.254 | 0.003 | 0.034 | 0.062 | 0.100 | 0.200 | 0.285 | 0.360 |
| mix_lo3_hi8 | 0.001 | 0.010 | 0.057 | 0.451 | 0.674 | 0.869 | 0.887 | 0.020 | 0.169 | 0.246 | 0.334 | 0.503 | 0.624 | 0.668 |

Re100 per-shell rel-L2 (k1..k7):
| variant | k1 early | k2 | k3 | k4 | k5 | k6 | k7 | k1 late | k2 | k3 | k4 | k5 | k6 | k7 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| full_pred | 0.117 | 0.272 | 0.325 | 0.297 | 0.537 | 0.695 | 0.730 | 0.130 | 0.404 | 0.505 | 0.411 | 0.592 | 0.700 | 0.774 |
| full_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lo_kc7 | 0.000 | 0.001 | 0.002 | 0.004 | 0.014 | 0.037 | 0.117 | 0.002 | 0.016 | 0.038 | 0.043 | 0.086 | 0.113 | 0.173 |
| mix_lo3_hi8 | 0.001 | 0.008 | 0.039 | 0.287 | 0.517 | 0.683 | 0.739 | 0.015 | 0.104 | 0.236 | 0.204 | 0.309 | 0.377 | 0.481 |

full_gt=0 both Re: solver/forcing/Re/dt valid. lo_kc7 anchor: 0.073/0.050 within expected range. hi_kc7≈hi_kc16≈full_pred (Re500: 0.287/0.291/0.291; Re100: 0.389/0.392/0.392): correcting any amount of high-k (k>7) in the IC gives no benefit — high-k is a bystander. lo_kc sweep monotonically decreasing: k≤7 content at t_r is causally responsible; largest single step is lo_kc5→lo_kc7 (Re500: 0.129→0.073 Δ=0.056; Re100: 0.117→0.050 Δ=0.067) — k=6,7 shells are the highest-impact pair. mix_lo7_hi42 = lo_kc7 exactly (both Re): once k≤7 is correct the inertial range (k=8..42) is causally irrelevant. mix_lo3_hi8 (0.200/0.188) ≈ lo_kc3 (0.211/0.193) within 0.011: since the two variants differ only in k>7 (GT vs pred) and produce nearly the same number, this independently confirms k>7 is a bystander. mix_lo3_hi8 (0.200) vs lo_kc7 (0.073): substituting pred's k=4..7 (forcing band) back costs +0.127 in Re500 — the forcing band in the pred IC is strongly causal; GT k=4..7 is not a free correction.
