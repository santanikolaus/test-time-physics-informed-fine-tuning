# TTA-OOD Thesis Journal

---

## 1. Document structure

- §2 — entry rules (how to log new experiments)
- §3 — dataset specification 
- §4 — links (server access, GPU allocation, path config)

Entries append below §4, newest on top, per the rule in §2.

---

## 2. How to make new entries

Append every new entry directly below this block and above all older entries — newest always on top.

Each entry has four parts:

1. **Date header** `# YYYY-MM-DD` — one per new day; same-day follow-ups get `(cont.)`.
2. **Raw description** — what was run + why, naming the paper/motivation if applicable.
3. **Raw numbers** — table untouched, plus anchor/sanity checks (e.g. reproduces prior banked number, control=0, load-guard match).
4. **Interpretation** — raw interpretation of what the numbers mean, No outlook, no next steps, no mechanism speculation, pivots, or broader cross-experiment synthesis. ---

## 3. Dataset specification

Fixed split for all entries below this point. 

| range | role | n |
|---|---|---|
| [0:240] | train (supervised pretraining) | 240 |
| [240:270] | eval (checkpoint selection, ν-inference validation, label-free-stopping sanity checks) | 30 |
| [270:300] | test (locked; GT kill-switches and final comparisons only, single read) | 30 |



## 4. Links

- `documentation/setup/server.md` — SSH / server access.
- `msc/configs/gpus.yaml` — GPU allocation.
- `documentation/paths.yaml` — checkpoint and data paths.

---

# 2026-07-24 (cont. 5)

Recurrent U-Net at Re500 (bundle + recipe) — in-distribution. Run sl3fp15e (best val_l2=0.237): same bundle/recipe as the Re100 base entry (residual, time_history=1, circular, EMA 0.995; residual-space MSE; AdamW lr1e-4 warmup5→cosine, batch 8, 150 ep) with output_factor=0.2488 (Re500 one-step residual std). `msc.tta.report`, Re500 test [270:300], default bands.

| band | rel_l2 | ρ | γ | corr>0.8 |
|---|:--:|:--:|:--:|:--:|
| k0-7 | 0.208 | 0.978 | 0.986 | 65.0/65 (cens 30/30) |
| k8-64 | 0.735 | 0.730 | 1.002 | 32.7/65 (cens 0/30) |

W1 all 0.0074 / late 0.0188 (floor 0.045); covRMSE all 0.031 / late 0.093 (floor 0.406).

Interpretation. Bundle transfers to Re500, strong in-dist: dominant band never decorrelates (k0-7 30/30 censored), ρ 0.978, on-attractor (W1/covRMSE far below floor). Harder than Re100 (val_l2 0.046→0.237, ρ 0.996→0.978, k8-64 horizon 63.7→32.7 — fine scales lose predictability ~2× sooner). Beats the raw Re500 coarse solver in-dist (cont. 3: W1 0.0074 vs 0.024, corr 30/30 vs 27/30, ρ 0.978 vs 0.942) — a well-trained in-dist learned model DOES beat the physics solver; the "network redundant/harmful" result (cont. 2/3) was specific to the one-shot FNO+coarse applied OOD, not learned models in-dist. γ(k8-64)=1.00: recurrent gets high-band energy right (vs FNO deficit) but phase-decorrelated (ρ 0.73) — spectrally correct, phase unpredictable.

---

# 2026-07-24 (cont. 4)

ν-inference from first frames, unknown-ν case — answers the surviving question of (cont. 3): can ν be recovered at test time without labels? New `msc/tta/infer_nu.py` (88f5663), entry point `infer_nu(frames, dt, method)`, two label-free estimators: **residual** — NS residual linear in ν (`Du = wt + adv − ν·wlap` vs forcing) → ν̂ closed-form LS, no solver run; **rollout** — golden-section over log-Re, candidates scored by `CoarseSolver(re, coarse_res, S=128)` rollout from frame 0 vs observed frames. Run `run_infer_nu.py` (47f64fa) on `NS_fine_Re{100,300,500}_T128_part0.npy`, 5 frames, sub_t=1 → dt=1/128, S=128, CPU. Unlabeled ≠ IC-only; ν is a rate, invisible in one frame (≥3 for central-difference wt).

RAW ν̂ vs true Re (traj 0, coarse_res=32, 30 iters):

| true_Re | method | Re_hat | nu_hat | rel_err | obj |
|---|:--:|:--:|:--:|:--:|:--:|
| 100 | residual | 100.1 | 0.00999 | 0.1% | 0.0047 |
| 100 | rollout | 90.8 | 0.01101 | 9.2% | 0.0414 |
| 300 | residual | 300.8 | 0.00332 | 0.3% | 0.1013 |
| 300 | rollout | 254.6 | 0.00393 | 15.1% | 0.0801 |
| 500 | residual | 497.7 | 0.00201 | 0.5% | 0.3501 |
| 500 | rollout | 255.6 | 0.00391 | 48.9% | 0.1072 |

Repeat traj 7 (coarse_res=32): residual 100.0 / 300.4 / 497.8 (0.0% / 0.1% / 0.4%); rollout 91.0 / 198.7 / 242.8 (9.0% / 33.8% / 51.4%). Rollout at coarse_res=64: Re100 99.8 (0.2%), Re300 243.9 (18.7%), Re500 322.9 (35.4%) — non-monotonic in grid at Re300 (15.1%→18.7%, identical deterministic trajectory). Anchors: residual estimator discretization (central-diff time, FFT space) differs from the CN/Heun `NavierStokes2d` that generated the data → recovery is not self-referential; 12 CPU toy-data tests pass (`tests/tta/test_infer_nu.py`), frames time-stepped by the solver, rollout test asserts obj(Rê) < obj(10×-wrong Re).

Interpretation. Residual recovers ν to ≤0.5% at all three Re on both trajectories from 5 frames — the unknown-ν case of (cont. 3) is solved on clean frames; coarse-alone can run at an inferred ν without labels. Residual obj rises with Re (0.005→0.35) while ν̂ stays exact: that mismatch is central-difference truncation growing with small-scale content, and does not contaminate the LS estimate. Rollout is biased low everywhere, worsens with Re, trajectory-sensitive (Re300: 15.1% vs 33.8%). Not grid over-dissipation: `CoarseSolver.solve` zero-pads back to S=128, so its field has zero energy above the coarse Nyquist while observed frames carry high-k energy growing with Re → ν-independent objective floor (Re500 obj bottoms at 0.107), minimizer trades ν down to shrink resolved-band mismatch; the non-monotonic 32→64 at Re300 is inconsistent with a grid-resolution explanation. Rollout as scored conflates wrong-ν with cannot-represent-field. Caveats: clean noise-free data exactly obeying the encoded PDE, GT (not operator-predicted) frames; traj 0/7 lie in train [0:240], not the [240:270] ν-inference validation split of §3; single 5-frame window per Re.

---

# 2026-07-24 (cont. 3)

Network on/off toggle at Re500 — does the OOD (Re100-trained) operator beat coarse-solver-alone? `scripts/coarse_as_prediction_gate.py` on Re500 GT + Re500 coarse_solver16 (network-OFF arm), vs banked cross-Re `mjzrrn6o` @Re500 (network-ON, OOD; same GT + same coarse channel). Test [270:300], sub_t=2.

RAW Re500 COARSE SOLVER (solver16) vs Re500 GT [network OFF]:

| band | rel_l2 | ρ | γ | corr>0.8 |
|---|:--:|:--:|:--:|:--:|
| k0-7 | 0.346 | 0.942 | 1.028 | 63.8/65 (cens 27/30) |
| k8-64 | 1.142 | 0.021 | 0.573 | 0/65 |

W1 all 0.0237 / late 0.0716; covRMSE all 0.0600 / late 0.1365.

| metric (Re500) | OFF coarse-alone | ON Re100-net OOD (banked) |
|---|:--:|:--:|
| W1 all / late | 0.0237 / 0.0716 | 0.0898 / 0.0987 |
| covRMSE all / late | 0.0600 / 0.1365 | 0.131 / 0.179 |

Interpretation. OOD the network is ACTIVELY HARMFUL: coarse-alone beats the Re100-net-on-Re500 by 3.8× on W1 (0.0237 vs 0.0898) and 2.2× on covRMSE (0.060 vs 0.131). Combined with in-dist (cont. 2: network ≈ neutral, slightly worse W1), the operator does not earn its place on top of the coarse solver — neutral in-dist, damaging OOD. Mechanism: the coarse solver is physics (run at Re500 ν → correct Re500 statistics, generalizes across Re for free); the network learned Re100 amplitudes and imports that bias OOD, dragging the physically-correct coarse field toward Re100 stats → Re500's wider fluctuations under-represented → W1/covRMSE blow up. Both toggles use ORACLE ν (correct Re500 coarse). Surviving question: unknown-ν case — does ν-inference (+ optional network) beat a wrong-ν coarse solver? If ν-inference recovers ν and coarse-alone is already this good, the useful machinery may be ν-inference, not the network. Caveat: network k0-7 aggregate corr-horizon not yet matched (report.py run pending); W1/covRMSE are the directly-comparable verdict.

---

# 2026-07-24 (cont. 2)

Coarse solver as predictor (no network) — what does the operator add on top? `scripts/coarse_as_prediction_gate.py` (commit 6269275) feeds the coarse_solver16 channel straight into eval.py field metrics vs GT, no model. Re100 test [270:300], sub_t=2, k-split 8.

RAW COARSE SOLVER (solver16) vs GT, in-dist Re100:

| band | rel_l2 | ρ | γ | corr>0.8 |
|---|:--:|:--:|:--:|:--:|
| k0-7 | 0.329 | 0.948 | 1.031 | 63.8/65 (cens 28/30) |
| k8-64 | 1.160 | 0.103 | 0.700 | 0/65 (cens 0/30) |

W1 all 0.0184, late 0.0505; covRMSE all 0.0865, late 0.1927. vs coarse+network mjzrrn6o (banked in-dist): W1 0.0368, covRMSE 0.087, k8-64 γ 0.641.

Interpretation. In-dist Re100 the raw coarse solver ALONE already resolves k≤7 (95% of energy) near-perfectly (ρ 0.948, corr>0.8 to 63.8/65, 28/30 censored, γ 1.03). The network adds nothing measurable on top: W1 worse (0.0368 vs 0.0184), covRMSE identical (0.087 vs 0.0865), and the network's k8-64 energy (γ 0.641) is LOWER than the raw coarse's (0.700) — the operator slightly damps the high band, it does not add valid high-k. Distributional validity (both W1 < GT-floor 0.0435) is inherited from the solver, not contributed by the network. k8-64 row degenerate for the band-limited coarse. Consequence: the network's value is not in-dist accuracy — it must appear OOD/adaptation or not at all. Caveat: network k0-7 aggregate estimated from per-shell (matched-band report.py run pending).

---

# 2026-07-24 (cont.)

Zero-shot OOD: Re100-trained models on Re500 GT — recurrent U-Net (ry8uot3p) vs one-shot FNO (75prctl5). `msc.tta.report --data-path <Re500 res128> --test-re 500 --op-re 100`, test [270:300], default bands, run from a dev-recurrent-unet git worktree (server main checkout was on another branch). Matched metric/eval/data; both Re100-trained.

corr>0.8 horizon (of 65); ρ,γ = k-band aggr decomposition; W1/covRMSE all-frames:

| metric (Re100→Re500) | one-shot FNO 75prctl5 | recurrent U-Net ry8uot3p |
|---|:--:|:--:|
| k0-7 corr-horizon | 50.7 (cens 8/30) | 54.8 (cens 11/30) |
| k8-64 corr-horizon | 3.9 | 9.9 |
| k0-7 ρ / γ / rel_l2 | 0.867 / 0.807 / 0.501 | 0.893 / 0.929 / 0.451 |
| k8-64 ρ / γ / rel_l2 | 0.272 / 0.355 / 0.966 | 0.409 / 0.657 / 0.945 |
| W1 (value) all / late | 0.1475 / 0.1791 | 0.0555 / 0.0838 |
| covRMSE all / late | 0.328 / 0.439 | 0.179 / 0.317 |

(In-dist Re100 for reference: recurrent k0-7 65.0 cens30/30 ρ0.996 γ1.001 W1 0.0016; FNO k0-7 63.6 ρ0.94 γ0.88 W1 0.083.) Prior banked FNO cross-Re decomposition (2026-07-10, per-shell k1–8) consistent: vanilla 75prctl5 cross ρ falls 0.99(k1)→0.31(k8), γ 0.85→0.43.

Interpretation. The recurrent U-Net generalizes to Re500 BETTER than the one-shot FNO on every measured axis — reverses the naive "autoregressive compounds worse OOD" prior. corr-horizon longer (k0-7 54.8 vs 50.7; k8-64 9.9 vs 3.9), phase higher (ρ 0.893 vs 0.867), amplitude closer to unity (γ 0.929 vs 0.807), value-dist 2.7× closer (W1 0.0555 vs 0.1475), covariance 1.8× closer (0.179 vs 0.328). Both UNDER-evolve OOD (γ<1 everywhere) — the Re100 model's learned step size under-drives the faster Re500 flow; the recurrent's residual/small-step parametrization anchors each step near the current state, keeping the rollout on the KF manifold (W1 near GT) while the one-shot hedges/blurs harder (γ 0.36 k8-64). Note the flip side: the recurrent is "plausibly wrong" OOD (on-attractor value-dist → a distribution/MMD detector would miss its failure), whereas its amplitude deficit + high-band collapse are physics/spectral signatures a residual detector should still catch. Caveats: mild OOD (Re500 = 5×Re100; crossover at higher shift untested), single seed, in-dist corr-horizon saturated so the in-dist row understates separation.

---

# 2026-07-24

Lippe recurrent-UNet (unet2d autoregressive arm) — faithful stabilizer bundle + training recipe, vs the earlier collapsed baseline and the one-shot FNO ceiling. Motivation: the baseline (yhsr7za0) rolled out worse than persistence (corr-horizon 18.6/65, phase ρ=0.48, high-k energy blow-up γ→2.5, off-attractor). Bundle: residual target (u_next=u_prev+output_factor·net_out, output_factor=0.104=std of one-step residual at sub_t=2), time_history=1, circular padding, EMA 0.995; residual-SPACE MSE loss (supervise net_out vs normalized increment — frame-space rel-L2 gave the ~3%-of-field increment no gradient and collapsed to persistence); AdamW wd1e-5, lr1e-4 warmup5→cosine, batch 8, 150 ep. Run ry8uot3p, best val_l2=0.0464 (val split [240:270]). Eval: `msc.tta.report` on test [270:300].

corr>0.8, first frame below (of 65); ρ,γ = k-band aggr decomposition:

| model | k0-7 horizon | k8-64 horizon | k0-7 ρ | k0-7 γ |
|---|:--:|:--:|:--:|:--:|
| baseline yhsr7za0 | 18.6 | 4.9 | 0.48 | 2.5 (late) |
| one-shot FNO 75prctl5 | 63.6 (cens 28/30) | 14.9 | 0.94 | 0.88 |
| bundle ry8uot3p | 65.0 (cens 30/30) | 63.7 (cens 29/30) | 0.9964 | 1.0006 |

k8-64 (ry8uot3p): ρ=0.9383, γ=0.9964. On-attractor: W1(all)=0.0016 (GT-vs-GT floor 0.0435), covRMSE(all)=0.0130 (floor 0.3761) — both below the GT-vs-GT floor. val_l2: bundle 0.0464 vs baseline 1.14, persistence ~0.60, FNO 0.31 (same validation_step / LpLoss d=3 / [240:270]).

Interpretation. Baseline pathologies removed: phase ρ 0.48→0.9964 (k0-7 corr>0.8 for the full 65-frame window, 30/30 test traj censored); high-k energy blow-up γ 2.5→1.0006 (no spurious injection); off-attractor (prior MMD p=0.002) → on-attractor (W1/covRMSE below GT floor). corr-horizon 18.6→65.0 saturates the metric (30/30 censored vs FNO 28/30). val_l2 0.0464 is below the one-shot FNO's 0.31 on the identical metric/split. The two changes that moved the result off persistence: residual-space loss (baseline train_loss was pinned at the 0.031 persistence floor under frame-space loss) and the training recipe (batch-1 + lr1e-3 thrashed at the predict-zero floor; batch-8 + lr1e-4 + warmup converged). Status: single seed, single test read, not yet visually sanity-checked (rollout-vs-GT).

---

# 2026-07-11 (cont. 3)

Solver wall-clock scaling: coarse_res x Re x IC — supervisor question on how expensive the coarse NS solver is and how it scales with grid size. New `scripts/solver_scaling_bench.py` (commit `9f063d2`) times `solve_from_ic` (`src/solver/periodic.py` NavierStokes2d) alone at `coarse_res` for a full T=128, t_interval=1.0 chain, deliberately excluding the resolution-independent zero-pad back to native res that `msc/tta/coarse_solver.py` applies for storage (constant cost, would swamp small-`coarse_res` timings). IC = frame 0 of a real GT chain (spectral-cropped to `coarse_res` via `spectral_resample`), not a synthetic field, so per-Re flow character (speed -> adaptive-dt substep count) is preserved; 5 distinct GT chains (chain_idx 0-4) per Re to check IC representativeness, min-of-3 timed reps per cell after 1 untimed warmup (cuFFT plan cache). Each (Re, coarse_res, ic) cell runs in its own spawned subprocess (`mp.get_context("spawn")`) so a CUDA hang/blowup can't stall the sweep; hard 300s per-cell timeout. `--res 8 12 16 24 36 --re 100 300 500 1000 --n_ics 5 --device cuda`, student06 GPU0.

Mean seconds across 5 ICs (CV% in parens), 100/100 (Re,coarse_res,ic) cells completed — zero blowups, zero timeouts:

| Re | 8 | 12 | 16 | 24 | 36 |
|---|:--:|:--:|:--:|:--:|:--:|
| 100 | 0.167 (0.6%) | 0.168 (0.6%) | 0.175 (5.6%) | 0.291 (1.0%) | 0.355 (9.9%) |
| 300 | 0.168 (0.7%) | 0.168 (0.3%) | 0.291 (0.5%) | 0.292 (0.7%) | 0.444 (15.2%) |
| 500 | 0.168 (1.2%) | 0.190 (8.1%) | 0.289 (0.5%) | 0.346 (11.8%) | 0.470 (8.3%) |
| 1000 | 0.168 (0.4%) | 0.263 (18.4%) | 0.291 (1.3%) | 0.397 (14.3%) | 0.531 (14.5%) |

Anchor: single-IC (chain 0 only) pilot run earlier same session reproduces this table within IC noise (Re100 0.166/0.167/0.169/0.287/0.383 vs 5-IC mean 0.167/0.168/0.175/0.291/0.355 above). GPU utilization sampled mid-run via `nvidia-smi`: 4% on student06 GPU0.

Interpretation. Table is step-quantized, not smooth: within a Re row, timings sit on plateaus (Re100: 0.167/0.168/0.175 flat across S=8/12/16, jump to 0.291 at S=24, 0.355 at S=36) rather than growing continuously with S — consistent with the adaptive-CFL substep count changing only at discrete (S,Re) thresholds. 5-point OLS log-log slope of mean_s vs S is 0.56 at Re100 and 0.74 at Re1000 — both far below the O(S^3)-O(S^4) a compute-bound pseudospectral solver would predict from substep-count x per-step O(S^2 log S) FFT cost; combined with 4% GPU utilization, wall-clock at these sizes (8-36) is overhead-bound (Python-loop + kernel-launch dispatch across 128 steps), not FLOP-bound, so the fitted exponent is not the solver's asymptotic compute scaling. IC comparability (CV% across the 5 chains): 12/20 cells under 2%, but 8/20 cells run 5-18%, concentrated at larger S (24/36) and/or higher Re (500/1000) — Re1000/S=12 highest at 18.4%, Re500/S=24 at 11.8%, Re100/S=36 at 9.9%. Re does move cost independent of S: Re1000/S=12 (0.263s) sits above Re100/S=16 (0.175s) despite the smaller grid.

---

# 2026-07-11 (cont. 2)

MMD grading instrument — in-band 16² MMD as on-attractor guard + Re100/Re500 discrimination gate; port of swirl measure_distances.py (rational-quadratic kernel mixture, biased V-stat) to torch in eval.py (mmd/inband_frames/mmd_bandwidth_median/forward_inband), caller msc/tta/mmd_gate.py; sample = one frame → spectral crop to low 16×16 Fourier block (k≤8, ×(s_out/S)², ifft to 16² real) → R²⁵⁶ point; bags = res128 independent-chain fine GT, test split [270:300) n=30 × T_eff=65 (sub_t=2); median-heuristic bandwidth (root-median off-diag dist × (0.5,1,2)) frozen ONCE per lineage on full 30-traj ref; discrimination + eval via permutation test (n_perm=500, relabel pooled trajectories, one-sided p=P(Δ*≥Δ)); 15v15-traj equal bags (biased-V diagonal cancels); scale-stripped = per-bag (b−mean)/std, bw refrozen on standardized ref.

Frozen kernels (median-heuristic, full 30-traj ref, 16²/k≤8):

| kernel | scales |
|---|:--:|
| bw_raw(Re100) | (32.20, 64.39, 128.79) |
| bw_raw(Re500) | (47.56, 95.12, 190.23) |
| bw_strip(Re100) | (10.77, 21.54, 43.08) |

Discrimination gate (Re100 ref, 15v15, seed1):

| gate | floor | ood | ratio | perm-p | verdict |
|---|:--:|:--:|:--:|:--:|:--:|
| raw | 0.1323 | 0.2211 | 1.67 | 0.002 | discriminates |
| scale-stripped | 0.1310 | 0.1135 | 0.87 | 0.794 | non-discriminative |

Eval — MMD(held-out ref half, pred[:15]) vs ref floor, seed3:

| run | operator | ckpt | data | ref | floor | score | ratio | perm-p | verdict |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | vanilla Re100 | 75prctl5 | Re100 | Re100 | 0.1323 | 0.1422 | 1.07 | 0.377 | on |
| 2 | coarse16 Re100 | mjzrrn6o | Re100 | Re100 | 0.1323 | 0.1362 | 1.03 | 0.435 | on |
| 3 | coarse16 Re500 | b7damqem | Re500 | Re500 | 0.1175 | 0.1225 | 1.04 | 0.433 | on |
| 4 | coarse16 Re100 | mjzrrn6o | Re500 | Re100 | 0.1323 | 0.1784 | 1.35 | 0.026 | OFF |
| 4 | coarse16 Re100 | mjzrrn6o | Re500 | Re500 | 0.1175 | 0.1204 | 1.02 | 0.459 | on |
| 5 | vanilla Re100 | 75prctl5 | Re500 | Re100 | 0.1323 | 0.1482 | 1.12 | 0.265 | on |
| 5 | vanilla Re100 | 75prctl5 | Re500 | Re500 | 0.1175 | 0.1413 | 1.20 | 0.220 | on |

Runs 1&2 reproduce prior-banked (floor 0.1323; 0.1422/p0.377; 0.1362/p0.435) exactly → per-lineage-full-ref bandwidth refactor non-perturbing; frozen bw identical all 5 runs. Gate: raw discriminates Re100/Re500 (p=0.002) by ENERGY only — scale-stripped NS (p=0.794, ood 0.1135 < floor 0.1310) → 256-dim shape non-discriminative at 16²/k≤8 (SD-style degeneracy, confirmed for MMD; ≈ scalar energy ratio for this Re pair). In-regime: all 3 ops on-attractor vs own Re (p 0.377/0.435/0.433, ratio ≤1.07). OOD Re100-op→Re500: run4 coarse OFF-Re100 (ratio1.35, p0.026)/on-Re500 (p0.459) but coarse ch = Re500 coarse_solver16 supplies exactly the graded k≤8 band → confounded, ≠ generalization. run5 vanilla (no coarse) INCONCLUSIVE: NS both refs (Re100 ratio1.12 p0.265; Re500 ratio1.20 p0.220), energy between attractors, marginally closer Re100; energy-only + structure-blind + short window (65 frames from true Re500 ICs) → no power to resolve reversion. Guard confirms in-regime on-attractor; cannot adjudicate vanilla OOD reversion.

---

# 2026-07-11 (cont.)

covRMSE (new metric) — anisotropy companion to the value-distribution W1 banked earlier today. Adds `cov_rmse` to `msc/tta/eval.py` + a covRMSE block in `report.py` reusing the `forward_fields` pass already run for W1 (no second forward), commit `cc98f4a`, motivated by DySLIM (Schiff et al., ICML 2024, Appendix C.4 Eq. 24-25; not in the released swirl-dynamics code, paper formula only). Metric: exploit KF x-translation-invariance — treat each vector along the forced y-axis (field axis 2, from `f=-4cos(4y)` in `src/pde/ns.py`) as one realization, pool samples/x/time into a row set, center on the pooled mean vector, form the S_y×S_y covariance (float64), and report relative Frobenius `‖Cov_pred−Cov_gt‖_F / ‖Cov_gt‖_F`. Reads the directional two-point y-structure the isotropic Chebyshev shell-average erases; second-order, not a coherent-structure detector. GT-vs-GT floor per window from the n/2-vs-n/2 test split (15 vs 15). Same six operators as 2026-07-10 (cont. 2): Re100 own — `75prctl5` (vanilla), `mjzrrn6o` (coarse solver16 m8); Re500 own — `nrow6j9l` (vanilla), `b7damqem` (coarse16 m8); cross-Re (Re100 op @ Re500 data via `--data-path`/`--coarse-path`) — `75prctl5`, `mjzrrn6o`. Eval `report.py`, test [270:300) n=30 (§3), two windows (all frames / late = last 8 of T_eff=65). Companion column, not a Table-1 reproduction (short eval-T, not a long-rollout invariant measure).

covRMSE (all / late), floor (all / late), with companion W1(all) and k8–64 γ (aggr) from the same forward pass:

| operator | covRMSE all | covRMSE late | floor all | floor late | W1 all | k8–64 γ |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| vanilla Re100 (75prctl5) | 0.205 | 0.303 | 0.376 | 0.434 | 0.083 | 0.615 |
| coarse16 Re100 (mjzrrn6o) | 0.087 | 0.129 | 0.376 | 0.434 | 0.037 | 0.641 |
| vanilla Re500 (nrow6j9l) | 0.226 | 0.294 | 0.406 | 0.421 | 0.122 | 0.208 |
| coarse16 Re500 (b7damqem) | 0.092 | 0.119 | 0.406 | 0.421 | 0.080 | 0.291 |
| cross vanilla (75prctl5 @Re500) | 0.328 | 0.439 | 0.406 | 0.421 | 0.148 | 0.355 |
| cross coarse16 (mjzrrn6o @Re500) | 0.131 | 0.179 | 0.406 | 0.421 | 0.090 | 0.371 |

Anchor: W1(all) column reproduces the numbers banked in tta-results.md today exactly (same forward pass). Sanity: `tests/tta/test_cov_rmse.py` 6/6 pass — analytic Cholesky covariance recovery, exact `|c²−1|` scaling for pred=c·gt, one-sided y-permutation sensitivity, x-permutation invariance ~1e-9. Floor split is n/2-vs-n/2 (15 vs 15) while the metric is n-vs-n (30 vs 30).

Interpretation. covRMSE roughly halves under coarse conditioning at every regime: Re100 0.205→0.087, Re500 0.226→0.092, cross 0.328→0.131 (all-frames); late-window 0.303→0.129, 0.294→0.119, 0.439→0.179. Cross-vanilla is the largest value in the table (0.328 all / 0.439 late); late > all in every row. The cross-vanilla vs cross-coarse16 pair has near-equal high-band energy (k8–64 γ 0.355 vs 0.371) yet covRMSE differs 2.5× (0.328 vs 0.131, all frames), a wider gap than the same pair's W1 (0.148 vs 0.090, 1.6×). The GT-vs-GT floor (0.376–0.434) sits above every model covRMSE except cross-vanilla late (0.439 > 0.421); the floor is computed on 15-vs-15 halves whereas the metric compares 30-vs-30, so at n=30 the floor carries more covariance-estimation noise than the pred-vs-gt distance it bounds.

---

# 2026-07-11

W1 value-distribution companion — point-wise Wasserstein-1 on pooled vorticity values (scipy.stats.wasserstein_distance, /std(gt)), new eval.w1_values + eval.forward_fields (full-res field forward, sibling of forward_bands/forward_inband), caller msc/tta/report.py W1 block; per-window GT-vs-GT null floor (n/2-vs-n/2 sample split); same operator set + fixed test split [270:300) n=30 × T_eff=65 (sub_t=2) as 2026-07-06/07; cross-Re via --data-path/--coarse-path override onto Re500 res128 + coarse_solver16; spatial-blind (permute pixels → unchanged); companion column, NOT a DySLIM Table-1 Wass1 reproduction (short eval-T value-PDF distance, not long-rollout invariant measure). GPU student06:0 (A10).

| operator | ckpt | eval data | W1 all | W1 late | floor all | floor late |
|---|---|:--:|:--:|:--:|:--:|:--:|
| vanilla Re100 | 75prctl5 | Re100 own | 0.0834 | 0.1124 | 0.0435 | 0.0361 |
| coarse16 Re100 | mjzrrn6o | Re100 own | 0.0368 | 0.0479 | 0.0435 | 0.0361 |
| vanilla Re500 | nrow6j9l | Re500 own | 0.1222 | 0.1269 | 0.0446 | 0.0420 |
| coarse16 Re500 | b7damqem | Re500 own | 0.0797 | 0.0785 | 0.0446 | 0.0420 |
| vanilla Re100 | 75prctl5 | Re500 cross | 0.1476 | 0.1791 | 0.0446 | 0.0420 |
| coarse16 Re100 | mjzrrn6o | Re500 cross | 0.0898 | 0.0987 | 0.0446 | 0.0420 |

Coarse conditioning ~halves W1 at every Re/regime (Re100 own 0.0834→0.0368; Re500 own 0.1222→0.0797; cross 0.1476→0.0898), same direction as the 2026-07-06/07 rel-L2 gains. Cross-Re vanilla (all 0.1476/late 0.1791) exceeds BOTH own-data baselines — op500 own (0.1222/0.1269) and op100 own (0.0834/0.1124) — reproducing the 2026-07-07 rel-L2 "cross exceeds both own-data" signature on a trajectory-blind metric. OOD concentrates late: cross vanilla late/all 1.21 vs op500 own flat 1.04 — value-PDF degrades with lead time only OOD. Cross coarse16 (all 0.0898) < op500 own vanilla (0.1222): Re100 coarse operator on Re500 coarse trajectories matches Re500 GT value-PDF better than native-Re vanilla training (same direction as coarse16-cross rel-L2 aggr 0.2483 < op500 vanilla 0.3955). coarse16 own late floor-ratio Re100 1.33× (0.0479/0.0361) < Re500 1.87× (0.0785/0.0420) — higher-Re heavier tails intrinsically further from any model. coarse16 Re100 own all-frames 0.0368 < floor 0.0435 → indistinguishable from GT at n=30. All Re500-data rows share GT floor 0.0446/0.0420; Re100 rows 0.0435/0.0361.

---

# 2026-07-10 (cont. 2)

Amplitude/phase decomposition of the one-shot rel-L2 — first application of the new eval lens (`amp_ratio` γ_k = √(E_pred/E_gt) and pooled `corr_pooled` ρ, added to `msc/tta/eval.py` + `report.py`, commit `6009669`), motivated by DySLIM (Schiff et al., ICML 2024, arXiv:2402.04467): pointwise rel-L2 fuses phase decorrelation and energy-spectrum error; γ is the phase-blind amplitude read, and pooled power satisfies rel_l2² = (1−ρ²) + (γ−ρ)² exactly over any band/frame window (ρ = `corr_pooled`, the set-level ratio-of-sums correlation, distinct from the per-sample `corr_curve` used for horizons). Three conditions. Today's three Re100 operators — `75prctl5` (vanilla, no coarse), `mjzrrn6o` (coarse solver16, n_modes=8), `zh4br372` (coarse solver16, n_modes=[32,32,8]) — in (a) in-distribution Re100 (own data) and (b) cross-Re (Re100 operator on Re500 data via `--data-path`/`--coarse-path`, matched Re500 solver16 coarse file; vanilla `--data-path` only). Plus (c) Re500-native: `nrow6j9l` (vanilla) and `b7damqem` (coarse solver16, n_modes=8), no override — their own cfgs already resolve to the identical Re500 res128 GT and solver16 coarse file used by the cross runs, so cross and native are scored on the same trajectories. No Re500-native n_modes=32 checkpoint exists, so m32 has cross but no native. Eval: `msc/tta/report.py`, per-shell `--bands` (65 single Chebyshev shells), `--time-bins 1-8,57-64`, test [270:300) n=30 (§3), op_re=test_re at defaults (inert — the decomposition reads only pred vs GT power).

Decomposition (aggr), cells = rel_l2 / ρ / γ, k1–8 — **in-distribution Re100**:

| k | vanilla 75prctl5 | coarse16 m8 mjzrrn6o | coarse16 m32 zh4br372 |
|---|:--:|:--:|:--:|
| 1 | 0.120 / 0.993 / 0.965 | 0.051 / 0.999 / 0.976 | 0.022 / 1.000 / 1.000 |
| 2 | 0.348 / 0.949 / 0.801 | 0.122 / 0.994 / 0.950 | 0.089 / 0.996 / 0.980 |
| 3 | 0.415 / 0.915 / 0.814 | 0.178 / 0.984 / 0.968 | 0.155 / 0.988 / 0.978 |
| 4 | 0.347 / 0.939 / 0.896 | 0.182 / 0.983 / 0.978 | 0.159 / 0.987 / 0.979 |
| 5 | 0.550 / 0.837 / 0.792 | 0.387 / 0.924 / 0.866 | 0.303 / 0.954 / 0.923 |
| 6 | 0.657 / 0.754 / 0.743 | 0.518 / 0.858 / 0.800 | 0.435 / 0.903 / 0.836 |
| 7 | 0.747 / 0.665 / 0.683 | 0.651 / 0.759 / 0.745 | 0.591 / 0.813 / 0.713 |
| 8 | 0.766 / 0.643 / 0.658 | 0.735 / 0.679 / 0.693 | 0.703 / 0.723 / 0.592 |

Decomposition (aggr), cells = rel_l2 / ρ / γ, k1–8 — **cross-Re (Re100 op @ Re500 data)**:

| k | vanilla 75prctl5 | coarse16 m8 mjzrrn6o | coarse16 m32 zh4br372 |
|---|:--:|:--:|:--:|
| 1 | 0.181 / 0.994 / 0.848 | 0.087 / 0.999 / 0.925 | 0.029 / 1.000 / 0.982 |
| 2 | 0.521 / 0.854 / 0.817 | 0.152 / 0.988 / 0.991 | 0.109 / 0.994 / 1.006 |
| 3 | 0.641 / 0.767 / 0.770 | 0.206 / 0.979 / 0.985 | 0.174 / 0.985 / 0.993 |
| 4 | 0.690 / 0.728 / 0.802 | 0.285 / 0.959 / 0.973 | 0.240 / 0.971 / 0.974 |
| 5 | 0.828 / 0.571 / 0.681 | 0.516 / 0.857 / 0.854 | 0.412 / 0.911 / 0.928 |
| 6 | 0.925 / 0.409 / 0.561 | 0.708 / 0.708 / 0.663 | 0.589 / 0.808 / 0.801 |
| 7 | 0.943 / 0.356 / 0.479 | 0.853 / 0.524 / 0.574 | 0.753 / 0.659 / 0.684 |
| 8 | 0.957 / 0.313 / 0.431 | 0.931 / 0.380 / 0.490 | 0.845 / 0.535 / 0.544 |

Decomposition (aggr), cells = rel_l2 / ρ / γ, k1–8 — **Re500-native** (no m32 native exists):

| k | vanilla nrow6j9l | coarse16 m8 b7damqem |
|---|:--:|:--:|
| 1 | 0.096 / 0.996 / 0.952 | 0.043 / 0.999 / 0.980 |
| 2 | 0.361 / 0.938 / 0.840 | 0.135 / 0.991 / 0.965 |
| 3 | 0.475 / 0.887 / 0.778 | 0.181 / 0.984 / 0.953 |
| 4 | 0.551 / 0.839 / 0.755 | 0.259 / 0.967 / 0.924 |
| 5 | 0.729 / 0.690 / 0.604 | 0.489 / 0.877 / 0.787 |
| 6 | 0.854 / 0.525 / 0.460 | 0.688 / 0.739 / 0.604 |
| 7 | 0.908 / 0.424 / 0.362 | 0.817 / 0.589 / 0.473 |
| 8 | 0.937 / 0.354 / 0.299 | 0.894 / 0.450 / 0.401 |

γ early (t1–8) → late (t57–64), selected shells (Re100 in-dist | Re500 cross):

| k | van @Re100 | m8 @Re100 | m32 @Re100 | van @Re500 | m8 @Re500 | m32 @Re500 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | 0.973→0.956 | 0.985→0.968 | 1.003→0.997 | 0.872→0.824 | 0.934→0.912 | 0.981→0.975 |
| 4 | 0.952→0.858 | 0.974→0.973 | 0.986→0.965 | 0.911→0.686 | 0.967→0.942 | 0.994→0.951 |
| 5 | 0.885→0.736 | 0.930→0.849 | 0.959→0.905 | 0.755→0.577 | 0.861→0.799 | 0.970→0.874 |
| 7 | 0.805→0.621 | 0.873→0.686 | 0.873→0.621 | 0.597→0.436 | 0.692→0.554 | 0.861→0.604 |

Anchor: `75prctl5` @Re100 corr-horizon (same forward pass) reproduces banked 2026-07-10 numbers exactly (corr<0.9: k1 64.2 cens 29/30, k2 61.2 cens 17, k5 33.2 cens 2, k6 16.9, k7 10.9). Decomposition identity rel_l2² = (1−ρ²)+(γ−ρ)² holds to fp by construction. k0/DC shell excluded (γ = ratio of ~1e-10 zero-mean noise floors ≈ 1e6, meaningless; guard finite). Control: cross rel_l2 > in-dist at every shell/operator; cross and native share the identical Re500 GT file.

Interpretation. In-distribution, γ ≤ 1 at essentially every shell k1–8 for all three operators (predicted energy at or below GT; the sole exception is m32 k1 at unity, 1.000), decreasing with k. At fixed k (k1–6) γ rises vanilla < m8 < m32 (m32 nearest 1); the order breaks at k7–8 (k8 γ: m32 0.592 < vanilla 0.658 < m8 0.693). ρ at k1–4 is ≥0.91 (all) and ≥0.98 (coarse arms); the k1–4 rel_l2 there is carried by the phase term, 1−ρ² ≫ (γ−ρ)² (m32 k4: 0.025 vs 7e-5). γ_early > γ_late at every shell (van k5 0.885→0.736; m32 k7 0.873→0.621).

Cross vs in-distribution, same operator: cross rel_l2 is higher at every shell, cross ρ lower, the gap widening with k (mjz k7 ρ 0.759→0.524). The in-dist→cross increase in rel_l2² is carried by the phase term: mjz k7 Δrel_l2² = 0.304 = +0.301 (1−ρ²) + 0.002 (γ−ρ)²; van k5 Δ = 0.384 = +0.373 + 0.010 — amplitude <5% of the increase at these shells. Under cross, early-window γ at k1–4 stays near 1 for the coarse arms (m8 0.93–0.97, m32 0.98–0.99) and lower for vanilla (0.87–0.91); the in-dist→cross drop in k1-early γ is vanilla −0.10, m8 −0.05, m32 −0.02.

Cross vs Re500-native, identical Re500 GT — the arms differ. Coarse (mjz vs b7dam): ρ agrees to ≤0.01 at k1–4, native higher at k5–8 (k7 0.589 > 0.524); γ crosses at k1→k2 — k1 cross < native (0.925 < 0.980), k2–8 cross > native (k7 0.574 > 0.473). Vanilla (75 vs nrow): native ρ exceeds cross already from k2 (k2 0.938 > 0.854; k4 0.839 > 0.728), and native γ exceeds cross at k1–3 (k1 0.952 > 0.848), cross overtaking only from k4 (k7 0.479 > 0.362). The cross−native rel_l2² difference is amplitude at k1 (coarse: +0.0051 (γ−ρ)² vs +0.0006 (1−ρ²)) and phase at k5–8 (coarse k7: +0.072 (1−ρ²) vs −0.011 (γ−ρ)²). At k5–8 cross and native rel_l2 sit closer to each other than either to in-distribution Re100 (coarse k7: 0.853 / 0.817 vs 0.651); at k1–4 that ordering does not hold (coarse k1: native 0.043 nearer in-dist 0.051 than cross 0.087).

---

# 2026-07-10

Per-shell phase-coherence horizon (new metric) — vanilla vs coarse16 vs coarse16+modes32. First integration of the correlation-horizon metric (`corr_curve`/`time_to_threshold`/`bootstrap_ci` in `msc/tta/eval.py`, derived from PDE-Refiner/Lippe's high-correlation rollout time), applied per single Chebyshev shell. Metric: per-sample band correlation ρ_k(t) = (P_pred+P_gt−P_err)/(2·√(P_pred·P_gt)) recovered from the band-power arrays (over all shells this equals the physical-field cosine, and Pearson for the zero-mean vorticity); horizon = first frame ρ_k(t) drops below threshold, right-censored at T_eff=65 when it never does; reported as bootstrap mean [2.5,97.5] CI and censored-sample count over the n=30 test set. Runs: `75prctl5` (vanilla, no coarse), `mjzrrn6o` (coarse solver16, n_modes=8), `zh4br372` (coarse solver16, n_modes=[32,32,8]). Eval: `msc/tta/report.py --run-id <ID> --bands "0-0,1-1,...,64-64"` (65 single-shell groups), test split [270:300) n=30 (§3), op_re=test_re=100 (in-distribution); default `--time-bins` (unused by the horizon).

corr<0.9 horizon, mean (censored/30):

| k | vanilla 75prctl5 | coarse16 m8 mjzrrn6o | coarse16 m32 zh4br372 |
|---|:--:|:--:|:--:|
| 0 | 0.5 (0) | 0.3 (0) | 0.3 (0) |
| 1 | 64.2 (29) | 65.0 (30) | 65.0 (30) |
| 2 | 61.2 (17) | 65.0 (30) | 65.0 (30) |
| 3 | 51.1 (8) | 64.5 (29) | 64.6 (29) |
| 4 | 57.1 (16) | 63.4 (28) | 63.6 (28) |
| 5 | 33.2 (2) | 56.6 (22) | 61.1 (21) |
| 6 | 16.9 (0) | 30.0 (4) | 49.2 (6) |
| 7 | 10.9 (0) | 13.7 (1) | 26.6 (1) |
| 8 | 10.0 (0) | 9.8 (0) | 13.6 (0) |
| 9 | 9.5 (0) | 9.6 (1) | 11.3 (0) |
| 10 | 8.8 (0) | 9.2 (1) | 9.5 (0) |
| 11 | 7.4 (0) | 7.2 (0) | 8.1 (0) |
| 12 | 7.0 (0) | 6.6 (0) | 6.8 (0) |
| 13 | 6.4 (0) | 6.0 (0) | 6.3 (0) |
| 14 | 5.8 (0) | 5.5 (0) | 5.6 (0) |
| 15 | 5.2 (0) | 5.1 (0) | 5.3 (0) |
| 16 | 4.6 (0) | 4.6 (0) | 4.0 (0) |
| 17 | 4.3 (0) | 4.4 (0) | 0.0 (0) |
| 18 | 3.8 (0) | 3.4 (0) | 0.0 (0) |

Tail (k≥19, <0.9): vanilla decays 3.2→0.1 to k25 then 0; m8 decays 3.0→0.2 to k25 then 0; m32 flat 0.0 from k17. Key CIs at the transition shells: k6 m8 [23.0,37.0] vs m32 [44.5,54.2]; k7 m8 [10.5,17.8] vs m32 [21.8,31.8].

corr<0.8 horizon, mean (censored/30):

| k | vanilla 75prctl5 | coarse16 m8 mjzrrn6o | coarse16 m32 zh4br372 |
|---|:--:|:--:|:--:|
| 1 | 65.0 (30) | 65.0 (30) | 65.0 (30) |
| 2 | 64.9 (29) | 65.0 (30) | 65.0 (30) |
| 3 | 61.3 (22) | 64.9 (29) | 64.9 (29) |
| 4 | 63.0 (27) | 64.4 (29) | 64.4 (29) |
| 5 | 52.3 (14) | 63.5 (26) | 63.7 (27) |
| 6 | 36.7 (6) | 57.7 (21) | 60.9 (23) |
| 7 | 22.8 (2) | 35.0 (7) | 50.6 (12) |
| 8 | 23.2 (3) | 20.8 (1) | 29.7 (1) |
| 9 | 17.4 (1) | 18.1 (1) | 20.5 (0) |
| 10 | 15.4 (1) | 15.6 (1) | 18.6 (0) |
| 11 | 14.5 (0) | 15.0 (1) | 14.6 (0) |
| 12 | 13.4 (0) | 14.5 (0) | 12.8 (0) |
| 13 | 11.5 (0) | 12.0 (0) | 11.4 (0) |
| 14 | 10.3 (0) | 11.1 (0) | 11.0 (0) |
| 15 | 9.5 (0) | 9.4 (0) | 10.5 (0) |
| 16 | 9.5 (0) | 8.7 (0) | 9.8 (0) |
| 17 | 8.7 (0) | 8.3 (0) | 4.7 (0) |
| 18 | 7.7 (0) | 7.6 (0) | 1.5 (0) |

Tail (k≥19, <0.8): vanilla decays 6.9→0.1 to ~k33 then 0; m8 decays 6.7→0.1 to k30 then 0; m32 0.0 from k19 (isolated 0.4 at k20). k7 <0.8 CIs: m8 [27.3,42.4] vs m32 [45.1,56.2].

Anchor: rel-L2 aggr columns from the same runs reproduce banked values — vanilla k0-7 0.3469, m8 0.2067, m32 0.1728 (2026-07-06 / 2026-07-09) ✓. Control: DC shell k0 horizon ≈0 for all three arms (vorticity mean ≈0; correlation degenerate there).

k1–k5 are phase-coherent (high censoring) in both coarse arms and less so in vanilla at the same shells (vanilla k2 17/30, k3 8/30, k5 2/30 at 0.9 vs coarse arms 17–30/30). The k6 and k7 horizons rise monotonically vanilla→m8→m32 (k6 <0.9: 16.9→30.0→49.2; k7 <0.9: 10.9→13.7→26.6; k7 <0.8: 22.8→35.0→50.6); the m8-vs-m32 CIs at k6 and k7 do not overlap under either threshold. k8–k16 differ by ≤~4 frames across the three arms with overlapping CIs. m32's horizon is 0.0 from k17 (<0.9) / k19 (<0.8), below both m8 and vanilla, which retain small nonzero horizons through ~k25 / ~k33 — i.e. m32 decorrelates at frame 0 in the k17–32 shells it represents, while modes8/vanilla (blind above k8) do not. Under this per-shell phase metric the k5–k8 shells are not flat across the three runs, unlike the k8-31 aggregate rel-L2 (m8 0.7834 vs m32 0.7822, banked 2026-07-09).

---

# 2026-07-10 (cont.)

Cross-Re phase-coherence horizon — where does the Re100 coarse operator's per-shell phase horizon sit on Re500 relative to the Re500-native operator? Same metric and eval as the 2026-07-10 entry above (per-shell correlation horizon, `msc/tta/report.py`, per-shell `--bands`, test [270:300) n=30). Two new runs: `b7damqem` (coarse solver16, trained on Re500 — in-distribution Re500) on its own data; and `mjzrrn6o` (coarse solver16, trained on Re100) forwarded on Re500 via `--data-path` (`NS_fine_Re500_T128_res128_part0.npy`) + `--coarse-path` (`NS_fine_Re500_T128_res128_coarse_solver16_part0.npy`), the matched-solver Re500 coarse file — the cross-Re override method of 2026-07-07 (cont.). Compared against `mjzrrn6o` on Re100 (in-distribution, banked in the 2026-07-10 entry above). op_re/test_re left at defaults (inert for the correlation horizon; it uses only pred vs GT).

corr<0.9 horizon, mean (censored/30):

| k | Re100 op @Re100 (in-dist) | Re100 op @Re500 (cross) | Re500 op @Re500 (native) |
|---|:--:|:--:|:--:|
| 1 | 65.0 (30) | 65.0 (30) | 65.0 (30) |
| 2 | 65.0 (30) | 65.0 (30) | 65.0 (30) |
| 3 | 64.5 (29) | 63.9 (25) | 64.5 (29) |
| 4 | 63.4 (28) | 60.3 (21) | 62.0 (22) |
| 5 | 56.6 (22) | 21.0 (0) | 21.7 (1) |
| 6 | 30.0 (4) | 6.4 (0) | 1.1 (0) |
| 7 | 13.7 (1) | 3.6 (0) | 0.6 (0) |
| 8 | 9.8 (0) | 2.5 (0) | 0.0 (0) |

corr<0.8 horizon, mean (censored/30):

| k | Re100 op @Re100 (in-dist) | Re100 op @Re500 (cross) | Re500 op @Re500 (native) |
|---|:--:|:--:|:--:|
| 3 | 64.9 (29) | 65.0 (30) | 65.0 (30) |
| 4 | 64.4 (29) | 64.7 (28) | 65.0 (30) |
| 5 | 63.5 (26) | 45.6 (10) | 53.2 (15) |
| 6 | 57.7 (21) | 13.7 (0) | 11.7 (1) |
| 7 | 35.0 (7) | 7.1 (0) | 2.9 (0) |
| 8 | 20.8 (1) | 4.9 (0) | 1.0 (0) |

Transition-shell CIs (cross vs native): k6 <0.9 [5.6,7.2] vs [0.5,1.7]; k7 <0.9 [3.0,4.1] vs [0.2,1.0]. High-k tail: both Re500-facing runs retain small nonzero horizons (~1–2 frames <0.8, ~1 frame <0.9 through ~k42) across k9–k63, whereas the Re100-on-Re100 runs reach 0 above ~k26 (2026-07-10 entry above).

Anchor: rel-L2 aggr (default bands) from the same two runs reproduces banked values — `b7damqem` k0-7 0.2276 (2026-07-06 Re500-own solver16) ✓; `mjzrrn6o` cross-Re k0-7 0.2483 (2026-07-07 cont. solver16 cross) ✓. Control: DC shell k0 horizon ≈0 all three arms.

k1–k2 stay phase-locked (30/30) in all three arms. At k3–k4 the cross run sits slightly below both in-distribution arms (k4 <0.9: 60.3/21-cens vs Re100-own 63.4/28 and Re500-native 62.0/22). At k5 the horizon drops sharply from the Re100-own value (56.6) to both Re500-facing runs, which match each other (cross 21.0, native 21.7). At k6–k8 the cross run's horizon exceeds the Re500-native run (k6 <0.9 6.4 vs 1.1, k7 3.6 vs 0.6; CIs non-overlapping) and both sit far below the Re100-own values. The cross run's k5–k8 phase horizon thus equals (k5) or exceeds (k6–k8) the Re500-native run's, while its banked k0-7 rel-L2 aggr is higher than the Re500-native run's (0.2483 vs 0.2276) — the cross-vs-native rel-L2 difference is not located in the correlation term (rel_l2² = (1−ρ²) + (γ−ρ)²).

---

# 2026-07-09

Mode-cutoff ablation — does raising FNO n_modes above the solver16 baseline (8) close the high-k error gap from the 2026-07-06 sweep, and is that gap actually confined to the band the model can represent? 3 checkpoints from an overnight sweep: `zh4br372` (n_modes=[32,32,8]), `c5ygxrjg` (n_modes=32, n_layers=6), `qbqntka8` (n_modes=24, n_layers=8), all Re100/solver16-coarse, same data/coarse paths as `mjzrrn6o` (n_modes=8 baseline, 2026-07-06). Eval: `msc/tta/report.py`, test split [270:300) n=30 (§3), op_re=test_re=100 (in-distribution). First pass default bands (0-7,8-64); second pass split the high band at the mode cutoff: 0-7,8-31,32-64.

Note: ran with default `--time-bins` (0-8,56-64), not the banked early/late convention (1-8,57-64) used in 2026-07-06/07 — early/late columns below are not directly comparable to those entries; aggr is unaffected (no time restriction) and is comparable.

Default bands (0-7 / 8-64):

| run | n_modes | k0-7 aggr | k8-64 aggr |
|---|---|:--:|:--:|
| zh4br372 | 32 | 0.1728 | 0.7823 |
| c5ygxrjg | 32, layers6 | 0.1818 | 0.7952 |
| qbqntka8 | 24, layers8 | 0.1899 | 0.7978 |

Cutoff-split bands (0-7 / 8-31 / 32-64):

| run | n_modes | k0-7 aggr | k8-31 aggr | k32-64 aggr |
|---|---|:--:|:--:|:--:|
| mjzrrn6o (baseline) | 8 | 0.2067 | 0.7834 | 1.3295 |
| zh4br372 | 32 | 0.1728 | 0.7822 | 1.1899 |
| c5ygxrjg | 32, layers6 | 0.1818 | 0.7952 | 1.1582 |
| qbqntka8 | 24, layers8 | 0.1899 | 0.7978 | 1.0918 |

Anchor: mjzrrn6o k0-7 aggr (0.2067) reproduces banked 2026-07-06 number ✓.

k8-31 is fully inside zh4br372/c5ygxrjg's mode budget (32) and mostly inside qbqntka8's (24, covers k8-23 of the k8-31 band) but entirely outside mjzrrn6o's (8). Despite that capacity difference, k8-31 aggr is flat across all four runs (0.7822-0.7978, range 0.0156) — modes32 (0.7822) is not lower than modes8 (0.7834). k0-7 aggr decreases with raised n_modes (0.2067->0.1728, mjzrrn6o->zh4br372) and k32-64 aggr also decreases (1.3295->1.1899), but the k8-31 band in between shows no such trend. qbqntka8 (fewest raised modes, most layers, of the three raised-mode runs) has the lowest k32-64 (1.0918) and the highest k0-7/k8-31.

---

# 2026-07-09 (cont.)

Capacity-lever ablation (layers/channels, modes unchanged) — do depth or width alone move the error the way raising n_modes did (mode-cutoff ablation, above)? 2 checkpoints, the 2 most recent runs by created_at: `sp0pzlwi` (n_layers=8, solver16 base otherwise unchanged) and `6cu3gcnp` (hidden_channels=256, solver16 base otherwise unchanged), both Re100/solver16-coarse, n_modes=8 (unchanged from baseline `mjzrrn6o`). Eval: `msc/tta/report.py --bands "0-7,8-31,32-64"`, same test split/op_re/test_re as the mode-cutoff entry above.

| run | lever | k0-7 aggr | k8-31 aggr | k32-64 aggr |
|---|---|:--:|:--:|:--:|
| mjzrrn6o (baseline) | n_layers=4, hidden=128 | 0.2067 | 0.7834 | 1.3295 |
| sp0pzlwi | n_layers=8 | 0.2151 | 0.7724 | 1.3302 |
| 6cu3gcnp | hidden=256 | 0.2069 | 0.7839 | 1.3359 |

Anchor: mjzrrn6o row reproduces the value banked in the mode-cutoff entry above and 2026-07-06 ✓.

sp0pzlwi (layers) moves k0-7 by +0.0084 and k8-31 by -0.0110 off mjzrrn6o; 6cu3gcnp (hidden) moves k0-7 by +0.0002, k8-31 by +0.0005, k32-64 by +0.0064 — flat within noise on all three bands. Both levers' shifts are an order of magnitude smaller than the mode-cutoff entry's k0-7 shift (-0.0339, mjzrrn6o->zh4br372) and k32-64 shift (-0.1396) for the same baseline and band set.

---

# 2026-07-07

Cross-Re OOD baseline — does the Re100 backbone degrade further than either in-distribution baseline when forced onto genuine Re500 dynamics? op100 (`75prctl5`, vanilla, no coarse channel) forwarded on Re500 data via a new `--data-path` override in `msc/tta/report.py` (splices into the run's own resolved cfg before `build_dataset`; forward params otherwise unchanged, sourced from `75prctl5`'s own cfg). All rows scored on the same test split [270:300) n=30 (§3), same band/time convention as 2026-07-06 (k0-7/k8-16/k17-32/k33-64; early=frames 1-8, late=last 8, T_eff=65). op_re=test_re=100 throughout (default, inert — physics-residual arrays still unsurfaced).

| run | trained on | eval data | early (k0-7) | late (k0-7) | aggr (k0-7) |
|---|---|---|:--:|:--:|:--:|
| 75prctl5 | Re100 | Re100 (own) | 0.1600 | 0.4618 | 0.3469 |
| nrow6j9l | Re500 | Re500 (own) | 0.2610 | 0.5028 | 0.3955 |
| 75prctl5 | Re100 | Re500 (cross) | 0.2166 | 0.6601 | 0.5014 |

Spectral split (aggr, cross-Re row): k0-7 0.5014, k8-16 0.9615, k17-32 0.9832, k33-64 1.0131.

Anchor: in-distribution rows reproduce banked 2026-07-06 numbers ✓ (0.3469, 0.3955 unchanged).

Cross-Re early (0.2166) sits between the two in-distribution rows, closer to nrow6j9l's own-data early (0.2610) than to 75prctl5's own-data early (0.1600). Cross-Re late (0.6601) and aggr (0.5014) exceed *both* in-distribution rows — higher than nrow6j9l's own-data late (0.5028) and aggr (0.3955), not merely as high. High-k bands (k8-16/17-32/33-64) sit at 0.96-1.01, the same near/above-1 range established in the 2026-07-06 entry.

---

# 2026-07-07 (cont.)

Cross-Re coarse-conditioning — does the coarse channel close the Re100->Re500 gap seen in the vanilla cross-Re baseline (2026-07-07)? Re100 coarse-solver{8,12,14,16} checkpoints (`g32z0ipc`/`qb6m7ux7`/`8140mmty`/`mjzrrn6o`) forwarded on Re500 data via the vanilla run's `--data-path` override plus a new mirrored `--coarse-path` override in `msc/tta/report.py`, paired at matched solver level with Re500's own `coarse_solverN` file. Same test split, band/time convention, and op_re=test_re=100 (inert) as 2026-07-06/07. Confirmed `"coarse"` key present in the dataset for all four runs (override took).

k0-7:

| level | condition | early | late | aggr |
|---|---|:--:|:--:|:--:|
| solver8 | Re100 (own) | 0.1665 | 0.4478 | 0.3366 |
| solver8 | Re500 (own) | 0.2803 | 0.4977 | 0.4002 |
| solver8 | Re500 (cross) | 0.2274 | 0.6169 | 0.4696 |
| solver12 | Re100 (own) | 0.1479 | 0.3909 | 0.2935 |
| solver12 | Re500 (own) | 0.2461 | 0.3689 | 0.3045 |
| solver12 | Re500 (cross) | 0.2087 | 0.4378 | 0.3453 |
| solver14 | Re100 (own) | 0.1315 | 0.3291 | 0.2496 |
| solver14 | Re500 (own) | 0.2163 | 0.3324 | 0.2667 |
| solver14 | Re500 (cross) | 0.1869 | 0.3830 | 0.2957 |
| solver16 | Re100 (own) | 0.1106 | 0.2686 | 0.2067 |
| solver16 | Re500 (own) | 0.1944 | 0.2707 | 0.2276 |
| solver16 | Re500 (cross) | 0.1609 | 0.3153 | 0.2483 |

Spectral split (aggr, cross-Re rows only):

| level | k0-7 | k8-16 | k17-32 | k33-64 |
|---|:--:|:--:|:--:|:--:|
| solver8 | 0.4696 | 0.9581 | 0.9821 | 1.0077 |
| solver12 | 0.3453 | 0.9462 | 0.9804 | 1.0071 |
| solver14 | 0.2957 | 0.9419 | 0.9795 | 1.0087 |
| solver16 | 0.2483 | 0.9457 | 0.9813 | 1.0080 |

Anchor: own-data rows reproduce 2026-07-06 banked numbers ✓.

Early sandwiches between the two own-data rows at every level (same shape as the vanilla cross-Re baseline). Late and aggr exceed *both* own-data rows at every level (solver8 aggr 0.4696 > 0.4002/0.3366; solver16 aggr 0.2483 > 0.2276/0.2067). Coarse channel still lowers cross-Re error monotonically with solver resolution (aggr 0.4696->0.3453->0.2957->0.2483, solver8->16), same monotone direction as both own-data sweeps. High-k bands (k8-16/17-32/33-64) stay in the 0.94-1.01 range already established, flat across solver level.

---

# 2026-07-06

Coarse-backbone sweep, band x time — does coarse-conditioning (solver8/12/14/16) improve the operator, and where (band, lead time)? 10 checkpoints: {vanilla, solver8, solver12, solver14, solver16} x {Re100, Re500}, all pretrained on the current independent-chain res128 data, project msc-base (paths.yaml pretrain_checkpoints/coarse_checkpoints). Eval script: `msc/tta/report.py` (setup.load_model -> setup.build_dataset(cfg,"test") -> eval.forward_bands -> eval.rel_l2), invoked per checkpoint as:

```
python -m msc.tta.report --run-id <ID> --bands "0-7,8-16,17-32,33-64" --time-bins "1-8,57-64"
```

op_re=test_re=each checkpoint's own training Re (no override, in-distribution). Split: test [270:300) n=30 (§3, fixed). Shared physics/data params across all 10 (from each run's own resolved cfg, identical at 128²): sub_t=2, time_scale=1.0, temporal_pad=5, pad_mode=zero, t_interval=1.0, T_eff=65. Time bins reproduce the established early(skip frame 0, nE=T_eff//8=8)/late(last 8) convention; aggr = full-trajectory pooled (report.py default column). k0-7 = FNO-representable band (n_modes=[8,8,8]) and the band coarse-conditioning targets; k8-16/17-32/33-64 = high-k control bands, expected flat under a correctly-localized coarse effect.

Anchor: eval pipeline (forward_bands/rel_l2) validated earlier this session — exact reproduction of a banked historical number (op500/38o0kj3y: err_full 0.4091 vs banked 0.409, err_k7 0.3714 vs 0.371, early 0.2529 vs 0.253, late 0.4724 vs 0.472) and self-consistency on nrow6j9l (aggregate rel_l2 0.4424 vs its own logged val_l2 0.460, close under the known pooled-ratio-of-sums vs mean-of-per-sample-ratios divergence).

k0-7 (decision band), early/late/aggr:

| Re | arm | early | late | aggr |
|---|---|:--:|:--:|:--:|
| 100 | vanilla | 0.1600 | 0.4618 | 0.3469 |
| 100 | solver8 | 0.1665 | 0.4478 | 0.3366 |
| 100 | solver12 | 0.1479 | 0.3909 | 0.2935 |
| 100 | solver14 | 0.1315 | 0.3291 | 0.2496 |
| 100 | solver16 | 0.1106 | 0.2686 | 0.2067 |
| 500 | vanilla | 0.2610 | 0.5028 | 0.3955 |
| 500 | solver8 | 0.2803 | 0.4977 | 0.4002 |
| 500 | solver12 | 0.2461 | 0.3689 | 0.3045 |
| 500 | solver14 | 0.2163 | 0.3324 | 0.2667 |
| 500 | solver16 | 0.1944 | 0.2707 | 0.2276 |

Spectral split (aggr, all bands):

| Re | arm | k0-7 | k8-16 | k17-32 | k33-64 |
|---|---|:--:|:--:|:--:|:--:|
| 100 | vanilla | 0.3469 | 0.8019 | 0.9625 | 1.2630 |
| 100 | solver8 | 0.3366 | 0.7956 | 0.9570 | 1.2635 |
| 100 | solver12 | 0.2935 | 0.7864 | 0.9540 | 1.2834 |
| 100 | solver14 | 0.2496 | 0.7889 | 0.9599 | 1.3205 |
| 100 | solver16 | 0.2067 | 0.7781 | 0.9545 | 1.3863 |
| 500 | vanilla | 0.3955 | 0.9586 | 0.9863 | 0.9928 |
| 500 | solver8 | 0.4002 | 0.9620 | 0.9873 | 0.9929 |
| 500 | solver12 | 0.3045 | 0.9510 | 0.9852 | 0.9939 |
| 500 | solver14 | 0.2667 | 0.9390 | 0.9830 | 0.9932 |
| 500 | solver16 | 0.2276 | 0.9336 | 0.9823 | 0.9936 |

k0-7 late decreases monotonically vanilla->solver16 at both Re (100: 0.4618->0.2686; 500: 0.5028->0.2707). k0-7 aggr likewise monotonic vanilla->solver16 at both Re except solver8 (100: 0.3366<0.3469 vanilla, i.e. still an improvement; 500: 0.4002>0.3955 vanilla, a regression). k0-7 early is non-monotonic at both Re: solver8 early exceeds vanilla (100: 0.1665>0.1600; 500: 0.2803>0.2610), then decreases monotonically solver8->solver16. High-k bands (k8-16/17-32/33-64) are flat across the sweep at both Re — max range 0.024 (Re100 k8-16: 0.7781-0.8019), 0.058 (Re100 k33-64: 1.2630-1.3863) vs the k0-7 aggr range of 0.14-0.19 — an order of magnitude smaller shift outside k0-7. Re100 k33-64 aggr >1 at every arm (1.263-1.386, rising with solver resolution); Re500 k33-64 aggr stays ~0.99 at every arm. Re500 k8-16/17-32/33-64 sit uniformly closer to 1.0 (0.93-0.99) than Re100's (0.78-1.39).

---

# 2026-07-02

Coarse-solver resolution sweep — structural + physics sanity check on `msc/tta/coarse_solver.py` (free-running NS solve at coarse_res, spectrally zero-padded to target grid; replaces `materialize_coarse_solver.py`, confirmed sourced from stale correlated data). coarse_res ∈ {8,12,14,16} bracket the banked cross-resolution oracle-kmax cliff (k=3 no benefit, k=4 minimum useful, k=6 ~2/3 reduction, k=7 native); Re ∈ {100,500}, N=300 each, sourced from independent-chain res128 data. Physics check: per-shell energy fraction above each file's Nyquist (zero-padding validity); early-window (frames 1-8) relL2 vs true GT at k≤min(Nyquist,7), phase-blind (power magnitude only), chain 0 only.

Anchor: blowups=0/300 all 8 cells (sweep log). All 8: shape=(300,129,128,128), dtype=float32, nan=False, inf=False.

| Re | coarse_res | min | max | std |
|---|:--:|:--:|:--:|:--:|
| 100 | 8 | -13.74 | 16.53 | 3.042 |
| 100 | 12 | -15.77 | 17.63 | 3.014 |
| 100 | 14 | -16.93 | 18.43 | 3.043 |
| 100 | 16 | -16.06 | 20.05 | 3.038 |
| 500 | 8 | -21.82 | 19.68 | 4.043 |
| 500 | 12 | -29.85 | 27.74 | 4.063 |
| 500 | 14 | -31.05 | 28.95 | 4.065 |
| 500 | 16 | -31.36 | 29.18 | 4.042 |

Anchor: energy fraction above Nyquist ≈ 2e-14 (fp noise) all 8 cells.

| Re | coarse_res | Nyquist | k≤ (measured) | early relL2 vs GT |
|---|:--:|:--:|:--:|:--:|
| 100 | 8 | 4 | 4 | 0.302 |
| 100 | 12 | 6 | 6 | 0.107 |
| 100 | 14 | 7 | 7 | 0.086 |
| 100 | 16 | 8 | 7 | 0.004 |
| 500 | 8 | 4 | 4 | 0.209 |
| 500 | 12 | 6 | 6 | 0.090 |
| 500 | 14 | 7 | 7 | 0.088 |
| 500 | 16 | 8 | 7 | 0.007 |

All 8 files structurally valid; value range scales with Re (std 3.0 @Re100, 4.0 @Re500). Zero-padding exact at machine precision, all resolutions. Early relL2 vs GT decreases monotonically 8→12→14 both Re (Re100: 0.302→0.107→0.086; Re500: 0.209→0.090→0.088), then drops ~10x from 14→16 despite both resolving k≤7 (Re100: 0.086→0.004; Re500: 0.088→0.007). Nyquist(14)=7=kmax; Nyquist(16)=8>kmax.

---
