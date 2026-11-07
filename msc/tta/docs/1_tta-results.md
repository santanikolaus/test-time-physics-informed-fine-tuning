# ⛔ LEGACY — ENTRY STOP

**Do NOT add new entries to this file.** It is legacy; new results are logged in `5_tta-thesis.md`.

Before writing anything here, **ASK THE USER AGAIN** whether it should actually land in `5_tta-thesis.md` instead.

---

# 2026-07-12

Lippe recurrent-UNet port — corr-horizon vs one-shot ceiling, Re100 test [270:300], `msc.tta.report`. unet2d one-step (yhsr7za0, best val_l2=1.140) vs one-shot FNO/PINO (75prctl5).

corr>0.8, first frame below (of 65); ρ,γ = k0-7 aggr decomposition:

| model | k0-7 horizon | k8-64 | k0-7 ρ | k0-7 γ |
|---|:--:|:--:|:--:|:--:|
| unet2d one-step | 18.6 | 4.9 | 0.48 | 1.02 |
| one-shot FNO | 63.6 (cens 28/30) | 14.9 | 0.94 | 0.88 |

MMD (yhsr7za0): OFF-attractor, score 0.272 vs GT-floor 0.132, ratio 2.06, perm-p 0.002.

Lippe Table 1 (Kolmogorov, corr>0.8 duration): MSE U-Net 9.663s, PDE-Refiner 10.659s.

Interpretation. Not pathological: k0-7 holds corr>0.8 to frame 18.6/65, error is phase (ρ=0.48, γ≈1), not amplitude; k8-64 accumulates spurious energy (γ→2.5 late, banked separately). Headroom exists: the one-shot FNO tracks low-k GT to ~63/65 (28/30 never decorrelate in-window, ρ=0.94), so low-k predictability ≫18 — the 18.6 is not chaos-limited. NO valid cross-paper ratio: ratio-to-ceiling (ours 18.6/63.6=0.29 vs Lippe 9.663/10.659=0.91) is INVALID — incommensurable ceilings. Lippe's ceiling (PDE-Refiner, AR) beats his baseline only 1.1x; ours (one-shot space-time FNO, non-AR, blurs toward low-k mean) is 3.4x the UNet. The gap reflects ceiling TYPE, not baseline quality; cannot separate "faithful" from "underperforming". How far below Lippe is not cleanly measurable (his τ_decorr at Re1000 absent; Table-1 DNS times are solver-accuracy, not decorrelation, horizons). The 63→18 gap bundles the inherent autoregression tax (one-shot can't compound; any AR must) with fixable faithfulness gaps (history 4→1, absolute→residual) — unpartitionable without a matched AR baseline.

---

# 2026-06-30 (cont.)

Pool-neighbor A_k distribution — per-IC heterogeneity of best-neighbor traj A_k (k=7) within 200 training ICs; `phase_alignment_re100_all300.npz` + `neighbor_index_re100_traj.npy`; `sim_traj[:200,:200,6]` diagonal set to −inf, max over pool per IC.

| stat | best-neighbor traj A_k (k=7) |
|---|:--:|
| mean | 0.4639 |
| std | 0.1337 |
| min | 0.1863 |
| p5 | 0.2840 |
| p25 | 0.3662 |
| p50 | 0.4416 |
| p75 | 0.5448 |
| p95 | 0.7265 |
| max | 0.8378 |

| threshold | ICs above | % |
|---|:--:|:--:|
| > 0.80 | 4/200 | 2.0 |
| > 0.70 | 12/200 | 6.0 |
| > 0.60 | 32/200 | 16.0 |
| > 0.55 | 50/200 | 25.0 |
| > 0.50 | 70/200 | 35.0 |

ICs 127–140 cluster: 10 consecutive ICs, best-neighbor A_k = 0.73–0.84 (temporally adjacent in NS simulation). ICs 100–119 desert: A_k = 0.19–0.27. Histogram unimodal, center mass [0.35–0.50], right tail sparse.

Split-resolved (query→training pool [0:200]):

| split | n | mean | p50 | >0.70 | >0.60 | >0.50 | max |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| train [0:200] | 200 | 0.464 | 0.442 | 6.0% | 16.0% | 35.0% | 0.838 |
| val [200:260] | 60 | 0.375 | 0.363 | 0% | 3.3% | 11.7% | 0.629 |
| test [260:300] | 40 | 0.345 | 0.351 | 0% | 0% | 2.5% | 0.526 |

ICs 127–140 cluster is entirely intra-train (temporally adjacent NS sampling). Val and test ICs land in the flat attractor bulk. Train-time conditioning (mean 0.46, peak 0.84) ≠ test-time conditioning (mean 0.35, max 0.53): distribution shift between training and inference. At θ=0.60 threshold zero test ICs qualify for neighbor retrieval; at θ=0.50 only 1/40. Step F (selective conditioning) collapses to solver-only at inference on the test split.

---

# 2026-06-30

Sibling-as-coarse ablation — σ∈{0.3,0.6} × n_sibs∈{1,2,3}: phase-perturbed sibling trajectories (cheb_lowpass k≤7, shape (N,129,S,S) float32, same format as GT coarse) used as FNO channels 5–(4+n_sibs); dropout=0 (ceiling probe); Re100 128², train[0:200]/val[200:260] n=200/60, sub_t=2, h128, batch1, lr1e-3 milestones[25,50,75,100] γ0.5, max_epochs=150; configs kf_sibling_re100_s{30,60}_n{1,2,3}.yaml; wandb pretrain-kol; val_l2 = rel-L2 on val split [200:260] (NOT held-out [260:300]).

| σ | n_sibs | data_ch | best val_l2 | @ epoch |
|---|:--:|:--:|:--:|:--:|
| 0.3 | 3 | 7 | 0.185 | 29 |
| 0.3 | 2 | 6 | 0.193 | 29 |
| 0.3 | 1 | 5 | 0.214 | 29 |
| 0.6 | 3 | 7 | 0.226 | 29 |
| 0.6 | 2 | 6 | 0.242 | 29 |
| 0.6 | 1 | 5 | 0.274 | 29 |

σ=0.3 < σ=0.6 at every n_sibs (Δ0.041 at n=3, Δ0.060 at n=1). n=3 < n=1 within each σ (s30: Δ0.029; s60: Δ0.048). Both effects monotone across the grid. All 6 runs peak at ep29; val_l2 rises 0.007–0.010 by ep149 while train_loss halves — consistent overfit after ep29 for all arms. s30_n3 (0.185) is 0.025 above GT coarse oracle (~0.16, user-stated reference, same val split).

Band-resolved gate — s30_n3 best ckpt (k62n2x26, ep29) via coarse_oracle_gate.py --coarse_paths; held-out [260:300] n=40, sub_t=2, op_re=100; scripts/coarse_oracle_gate.py.

| arm | early_k7 | late_k7 | err_k7 | err_full |
|---|:--:|:--:|:--:|:--:|
| s30_n3 siblings (σ=0.3, n=3) | 0.0904 | 0.1109 | 0.1124 | 0.1874 |
| op100 baseline (banked) | 0.175 | 0.543 | 0.409 | — |

late/early ratio = 1.23 (op100 = 3.10). err_k7 on held-out test (0.112) below val_l2 (0.185) — different metrics (k≤7 band-limited vs full-field).

---

# 2026-06-30 (cont.)

Sibling proximity analysis — three measurements from materialized sibling files + GT, test split [260:300] n=40; `scripts/sibling_proximity_analysis.py`; lowpass validation max|lowpass(GT)−coarse_k7|=5.61e-06 ✓.

Proximity curve — relL2(sib_k7[t], GT_k7[t]) per frame, mean over n=40: s30 t=0: 0.1640, t=128: 0.1212; s60 t=0: 0.3170, t=128: 0.2297. Monotonically decreasing — attractor contraction reduces sibling-GT distance over time. σ_60/σ_30 ratio ≈ 1.93 ≈ 0.6/0.3 preserved.

Dose-response — x = mean proximity (k≤7, test [260:300]), y = val_l2 (full-field, val [200:260]):

| σ | n | mean_prox | val_l2 |
|---|:--:|:--:|:--:|
| 30 | 1 | 0.1389 | 0.1788 |
| 30 | 2 | 0.1364 | 0.1579 |
| 30 | 3 | 0.1355 | 0.1497 |
| 60 | 1 | 0.2682 | 0.2001 |
| 60 | 2 | 0.2646 | 0.1973 |
| 60 | 3 | 0.2635 | 0.1966 |

x-axis flat within σ cluster (2% range), y varies 16% within σ=0.3. Diversity drives model gain, not proxy quality. y-axis is val_l2 not err_k7 — extrapolation to oracle/null anchors not valid until all 6 gate results collected.

n-utility (σ=0.3) — inter-sibling diversity vs marginal val_l2 gain: n=2 diversity=0.1926 Δval_l2=0.021; n=3 diversity=0.1928 Δval_l2=0.008. Diversity flat for n≥2; n=4 predicted negligible. σ is the lever, not n.

---

# 2026-06-30 (cont.)

Pairwise phase alignment — all 300 Re100 ICs, no train/test split; A_k = energy-weighted cos(Δφ) per L∞ shell; k=4 excluded [f=−4cos(4y) forcing inflates base-flow phase]; `scripts/pairwise_phase_alignment.py`; `phase_alignment_re100_all300.npz`. Validations: self-alignment max|A_k(i,i)−1|=0.00e+00; sibling cross-check computed=0.9551 vs exp(−0.09/2)=0.9560 Δ=0.0009.

NN A_k per IC (best neighbor, excl. self):

| metric | mean | p10 | p50 | p90 | max |
|---|:--:|:--:|:--:|:--:|:--:|
| IC-level k=5 (t=0) | 0.6745 | 0.5487 | 0.6662 | 0.8199 | 0.9442 |
| IC-level k=6 (t=0) | 0.6300 | 0.5060 | 0.6234 | 0.7766 | 0.9251 |
| IC-level k=7 (t=0) | 0.5926 | 0.4711 | 0.5876 | 0.7174 | 0.8859 |
| traj-mean k=5 | 0.5739 | 0.4019 | 0.5586 | 0.7566 | 0.9487 |
| traj-mean k=6 | 0.4939 | 0.3435 | 0.4728 | 0.6718 | 0.8910 |
| traj-mean k=7 | 0.4520 | 0.3089 | 0.4285 | 0.6156 | 0.8378 |

A_k decay k=7 — t=0-locked NN vs per-frame-best:

| frame | locked | per-frame-max |
|---|:--:|:--:|
| 0 | 0.5926 | 0.5926 |
| 16 | 0.4592 | 0.5968 |
| 32 | 0.2771 | 0.5997 |
| 64 | 0.1701 | 0.6010 |
| 128 | 0.1144 | 0.5924 |

Fraction above threshold (t=0): k=5 >0.70: 0.34% (302/89700); k=7 >0.70: 0.07% (59/89700). Pool has no dense k=5,6,7 structure. Locked NN A_k decays 0.593→0.114 by t=128; per-frame-max flat ≈0.60 (attractor well covered, best neighbor changes each frame). Pool retrieval BLOCKED at full 300-IC level — a fundamental property of the attractor, not a sample-size artifact.

---

# 2026-06-29

Phase-operator A/B — does coarse-k7 IC amplitude (varB, 5-ch) improve per-shell phase alignment A_k over phase-IC-only (varA, 4-ch) and a persistence floor? A_k = GT-energy-weighted mean cos(Δφ) per Chebyshev shell (1=perfect, 0=decorrelated); A_k_late = last T//8 frames (8 frames). varA: ckpt taxja4zq, 4-ch IC (gridx,gridy,gridt,phase-k7). varB: ckpt tagae16y, 5-ch IC (+coarse-k7 amplitude broadcast). Persistence: IC phase frozen across T. op100: ckpt pvqq97sq, 4-ch raw NS IC. All: held-out [260:300] n=40, sub_t=2; scripts/eval_phase_ak.py; npz saved to /tmp/ak_{varA,varB,persistence,op100}.npz.

| arm | k=5 | k=6 | k=7 | k≤7 agg late |
|---|:--:|:--:|:--:|:--:|
| op100 (raw IC, full ampl) | 0.648 | 0.488 | 0.399 | 0.871 |
| persistence | 0.241 | 0.171 | 0.153 | 0.662 |
| varB (5-ch, coarse ampl IC) | 0.308 | 0.203 | 0.159 | 0.650 |
| varA (4-ch, phase-IC only) | 0.184 | 0.144 | 0.145 | 0.560 |

varB > varA at all three target shells (k5: +0.124, k6: +0.059, k7: +0.014) — amplitude IC channel adds information. varB > persistence at k=5 (8/8 late frames, Δ+0.054–+0.083), k=6 (8/8 frames, Δ+0.019–+0.040), k=7 (6/8 frames, Δ≤+0.016); varA < persistence at all three. varB k≤7 aggregate (0.650) < persistence aggregate (0.662): persistence wins at k=1 (0.979 vs 0.934) and k=4 (0.668 vs 0.621). op100 (full amplitude IC) at k5=0.648 vs varB k5=0.308: varB captures (0.308−0.241)/(0.648−0.241)=16% of op100's above-persistence gain at k=5, 10% at k=6, 2% at k=7.

---

# 2026-06-24

Extreme under-resolution as a capacity-per-data probe — at 16² (256² GT downsampled 16×), with operators of fixed capacity, does shrinking the data (overfit regime, the inverse of growing the model) and stripping high-k change the in-band fit, and FNO vs UNet? Two architectures × two downsample methods, all Re100 16², 150 ep, data5+pde1+ic1, n_train200/n_val60, sub_t2, batch1, lr1e-3 milestones[25,50,75,100] γ0.5. FNO = op100 baseline byte-identical except data_path: n_modes[8,8,8] (=Nyquist 8 ⇒ NO spectral truncation, full-rank) h128 lift0, ~21M. UNet = the spatial-spectral bottleneck parity arch (SpatialSpectralMixer, temporal_mixer=spatial modes8 hidden64) base64 depth1 (16²→8² bottleneck), 2.29M. Downsample: spectral=FFT-crop (clean restriction) vs strided=::16 (aliases high-k into k≤7), via `scripts/materialize_res256.py`. configs `kf_{pino,unet}_re100_res16_{spectral,strided}.yaml`; wandb project `pretrain-kol` (FNO spec vsm4xtg5 / strided seeh9q04; UNet spec 05wrabf5 / strided vfhirgsi). Metric: val_l2 = full-field rel-L2 over all T frames on the n_val=60 split (NOT the held-out [260:300] k≤7 band metric used elsewhere; not comparable to the wall).

| arm | params | val_l2 | train_loss (epoch, =5·data+pde+ic) |
|---|:--:|:--:|:--:|
| FNO  · spectral | 21M   | 0.344 | 0.692 |
| FNO  · strided  | 21M   | 0.378 | 1.120 |
| UNet · spectral | 2.29M | 0.246 | 1.040 |
| UNet · strided  | 2.29M | 0.293 | 1.410 |

UNet < FNO on val_l2 for both downsamples (0.246 vs 0.344 spectral; 0.293 vs 0.378 strided) despite 9× fewer params — at 16² the FNO mode cap is inert (n_modes=8=Nyquist, no truncation), so both operators are full-rank and the UNet's local-conv bias yields lower full-field error. spectral < strided for both archs (FNO 0.344<0.378; UNet 0.246<0.293): the aliased target is harder to fit, Δ≈0.03–0.05. train_loss_epoch is the weighted PINO composite, not rel-L2, so the train>val ordering (e.g. FNO spectral 0.692 vs 0.344) is a metric mismatch, not a generalization gap. All four reached max_epochs=150; FNO ~1.3 GB / UNet ~0.6 GB on 12 GB TITAN V, ~7 s/epoch.

Band+time dissection — same `band_eval` machinery / split / k≤7 definition as the banked wall (held-out [260:300] n=40, early[1..8]/late[last T/8], op_re=test_re=100). `scripts/res16_band_probe.py`. Load guards reproduce training val_l2 (0.3437/0.3759/0.2418/0.2887, all Δ<0.001) ⇒ forward correct.

| arm | val_l2 | err_full | err_k7 | early k≤7 | late k≤7 | late/early |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FNO  · spectral | 0.344 | 0.383 | 0.380 | 0.146 | 0.506 | 3.47 |
| FNO  · strided  | 0.376 | 0.416 | 0.406 | 0.183 | 0.529 | 2.89 |
| UNet · spectral | 0.242 | 0.262 | 0.259 | 0.076 | 0.422 | 5.55 |
| UNet · strided  | 0.289 | 0.313 | 0.301 | 0.125 | 0.450 | 3.60 |

per-shell late k≤7 (k0 undefined: vorticity mean≈0 ⇒ zero denom):
| arm | k1 | k2 | k3 | k4 | k5 | k6 | k7 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| FNO  · spectral | 0.163 | 0.561 | 0.632 | 0.536 | 0.737 | 0.859 | 0.948 |
| FNO  · strided  | 0.172 | 0.596 | 0.637 | 0.543 | 0.773 | 0.880 | 0.977 |
| UNet · spectral | 0.121 | 0.412 | 0.499 | 0.447 | 0.638 | 0.802 | 0.932 |
| UNet · strided  | 0.134 | 0.405 | 0.510 | 0.457 | 0.698 | 0.870 | 0.953 |

The late k≤7 wall SURVIVES 16²: FNO·spectral late 0.506 ≈ the 128² op100 wall 0.543 (err_k7 0.380 vs 0.409, early 0.146 vs 0.175) — collapsing the grid 64×, removing all high-k, and forcing the overfit/high-capacity-per-data regime does NOT lower the in-band wall. The wall is intrinsic (in-band physics/chaos-identifiability), not a capacity or high-k-contamination artifact; this is the inverse-direction confirmation of the banked NO-GO (shrink data instead of grow model). Aliasing corrupts the resolved band, confound-free (spectral vs strided holds 16² DOF fixed): strided worse in-band for both archs, err_k7 +0.026 (FNO) / +0.042 (UNet), late +0.023 / +0.028 — folded high-k energy degrades the k≤7 evolution; the FFT-crop restriction is the easier target. UNet UNDERCUTS FNO in-band at 16² (late 0.422 vs 0.506, early 0.076 vs 0.146, 9× fewer params), reversing the 128²/256² parity where the spatial-spectral UNet only REACHED the FNO wall (0.478/0.577): with no high-k to resolve the conv bias wins in-band, and its high late/early ratio 5.55 = near-perfect early amortization + pure late chaotic drift. Time+spectral signature unchanged from the 128²/256² wall: early≪late (ratio 2.9–5.6), per-shell error rising monotonically to the resolved Nyquist (k7≈0.93–0.98, near-fully wrong), low at k1 (0.12–0.17) — identical failure mode, only the absolute level shifts with architecture.

---

Direct IC→window mapping (Version B) — can a model trained only on a 5- or 10-frame target window (data-only loss, pde_weight=0, ic_weight=0) beat the full-trajectory baseline on that window? From-scratch h64 Re100 128², 150 ep, 5 windows × 2 widths (T=5, T=10); eval held-out [260:300] n=40 via `scripts/latewin_eval.py` (window mode). Baseline = pvqq97sq (h128, full T=65, data+pde+ic), per-window relL2 and k≤7 pooled relL2 over the window slice only. Window indices in 65-frame sub_t=2 trajectory. configs `kf_re100_latewindow.yaml`; wandb project `late-window-kol`.

| window | frames | baseline relL2 | baseline k≤7 | T=5 relL2 | T=5 k≤7 | T=10 relL2 | T=10 k≤7 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| W1 | [6,11)/[6,16)   | 0.329 | 0.335 | 0.310 | 0.308 | 0.273 | 0.263 |
| W2 | [14,19)/[14,24) | 0.344 | 0.351 | 0.492 | 0.487 | 0.373 | 0.369 |
| W3 | [30,35)/[30,40) | 0.403 | 0.422 | 0.647 | 0.653 | 0.534 | 0.541 |
| W4 | [46,51)/[46,56) | 0.486 | 0.513 | 0.757 | 0.767 | 0.648 | 0.661 |
| W5 | [60,65)/[55,65) | 0.573 | 0.600 | 0.819 | 0.831 | 0.675 | 0.691 |

W1 beats baseline on both widths (T=5 k≤7 0.308 vs 0.335; T=10 k≤7 0.263 vs 0.335); baseline is h128 full-trajectory vs h64 specialist so comparison is not capacity-controlled. W2–W5: both T=5 and T=10 are worse than baseline, degrading monotonically with window distance from IC. T=10 consistently better than T=5 across all windows (W5: 0.691 vs 0.831) but does not close the gap to baseline. Baseline note: pvqq97sq per-window numbers differ from banked early/late (0.175/0.543) — different window definition and eval script.

---

# 2026-06-23

Objective-lever reweighting — do an ADDED energy term (band_energy_loss: abs per-shell |E_pred−E_gt|² over k2–7) or phase term (band_phase_loss: GT-energy-weighted (1−cosΔφ) over k2–4) lower the late k≤7 wall? From-scratch op100 Re100 128² (fno_kf h64), 150 ep, data5+pde1+ic1 + the term; eval held-out [260:300] n=40, late=last T/8. Legend: late/early k≤7 = relL2 vs GT; A(k2–4)=GT-energy-weighted phase alignment (1=perfect, manipulation check for angle); R(k≤7)=PRED/GT energy ratio (1=preserved, manipulation check for energy). configs `kf_re100_{energyweight,angleweight}.yaml`; probe `configs/model/fno_kf.yaml` build + `band_eval`/`phase_align_decomp`. Baseline = banked a0 uniform (late 0.543 / early 0.175 / aggr 0.409).

| arm | late k≤7 | early k≤7 | err_k7 | A(k2–4) late | R(k≤7) late |
|---|:--:|:--:|:--:|:--:|:--:|
| a0 baseline (banked) | 0.543 | 0.175 | 0.409 | — | — |
| energy w1 | 0.538 | 0.175 | 0.399 | 0.815 | 0.672 |
| angle w1 | 0.534 | 0.178 | 0.396 | 0.820 | 0.655 |
| angle w5 | 0.522 | 0.171 | 0.384 | 0.822 | 0.703 |
| angle w10 | 0.525 | 0.172 | 0.384 | 0.836 | 0.659 |
| angle w20 | 0.521 | 0.182 | 0.388 | 0.835 | 0.660 |
| angle5+energy5 | 0.522 | 0.165 | 0.383 | 0.840 | 0.702 |

Anchor: err_k7 0.383–0.399 reproduces the ~0.40 op100 Re100 family. Late k≤7 vs baseline 0.543: energy w1 0.538 (−0.005); angle 0.534 (w1) → 0.521–0.525 (w5/w10/w20, −0.018 to −0.022), within 0.004 across w5–w20. err_k7 0.383–0.399 vs baseline 0.409; early k≤7 0.165–0.182 vs 0.175. A(k2–4) 0.815–0.840 across all arms (+0.02 over the angle w1→w20 20× weight increase). combo angle5+energy5 late 0.522 = angle w5 0.522, A 0.840 vs angle w5 0.822. Validity: only energy w1 (rai1ohu9) measured — the w5/w20 multirun arms had no retrievable checkpoint at probe time (sweep still finishing).

---

# 2026-06-22

Collapse-independent per-shell R_k/A_k — is the late-error amplitude collapse a SEPARATE failure from phase, or two faces of one cause? The squared-error split's phase term ∝|Fu| → energy collapse suppresses measured phase (confound). Two |Fu|-independent measures per Chebyshev shell, late window, n=40: R_k = Σ|Fu|²/Σ|Fg|² (energy ratio, magnitude only; 1=preserved) and A_k = Σ|Fg|²cosΔφ/Σ|Fg|² (GT-energy-weighted phase cosine, |Fu|-independent; 1=positions perfect). Test: if ONE cause (hedging=low-pass under positional uncertainty) R_k and A_k co-degrade with k; if independent dissipation, R_k falls while A_k stays flat. Same forward as band_eval. `msc/tta/eval.py energy_phase_band, phase_align_decomp`; `scripts/amp_phase_probe.py --align`. ID op100 Re100 128² (pvqq97sq), OOD op500 Re500 128² (38o0kj3y), OOD spatial-UNet Re500 256² (7sklgx44). k0 omitted (vorticity mean≈0 ⇒ |F0|≈0, R0 undefined).

per-shell LATE  R_k | A_k:
| k | ID R | ID A | OOD128 R | OOD128 A | OOD256 R | OOD256 A |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | 0.74 | 0.99 | 0.83 | 1.00 | 0.84 | 0.99 |
| 2 | 0.40 | 0.84 | 0.73 | 0.81 | 0.59 | 0.80 |
| 3 | 0.41 | 0.80 | 0.54 | 0.84 | 0.42 | 0.72 |
| 4 | 0.65 | 0.85 | 0.46 | 0.76 | 0.47 | 0.73 |
| 5 | 0.40 | 0.65 | 0.33 | 0.56 | 0.28 | 0.40 |
| 6 | 0.33 | 0.49 | 0.24 | 0.47 | 0.20 | 0.27 |
| 7 | 0.31 | 0.40 | 0.16 | 0.26 | 0.11 | 0.13 |
| k≤7 | 0.60 | 0.87 | 0.70 | 0.89 | 0.63 | 0.83 |

R_k and A_k both fall monotonically with k in all three settings (k1: R≈0.74–0.84, A≈0.99–1.00; k7: R≈0.11–0.31, A≈0.13–0.40) — they co-degrade, not R-only. The k≤7 aggregate A (0.83–0.89) is k1-dominated (k1 holds most in-band energy, A≈0.99); per-shell A craters at k5–7 (0.13–0.65). Aggregate R 0.60/0.70/0.63 (≈30–40% late energy loss) — higher than an earlier n=3 estimate (0.37). k1 preserved on both axes everywhere (R≈0.83, A≈0.99). Pattern identical ID vs OOD; Re500/256² steepest at high k (k7 R 0.11 A 0.13). Energy loss (R<1) cannot come from displacement (translation preserves |F|), so it is real magnitude loss; but phase also decorrelates at the same shells (A<1), so amplitude collapse is not an independent defect — both worsen together up the spectrum.

---

# 2026-06-22

Amplitude-vs-phase decomposition of the operator error — is the late k≤7 error magnitude (amplitude) or position (phase), and at which shells? Exact per-mode split |Fp−Fg|² = (ap−ag)² + 2·ap·ag·(1−cosΔφ) [amplitude + phase], Chebyshev-shell × frame, held-out [260:300] n=40, early[1..8]/late[last T/8]. Forward = band_eval (setup TIME_SCALE/PAD, ["x"] IC). `msc/tta/eval.py amp_phase_decomp` (+ `amp_phase_band`), `scripts/amp_phase_probe.py --per-band --window 8`. Three settings: op100 Re100 128² (pvqq97sq), op500 Re500 128² (38o0kj3y), spatial-UNet Re500 256² (7sklgx44).

aggregate k≤7:
| arm | relL2 tot (=err_k7 guard) | relL2 amp | relL2 phase | phase_frac early | late | aggr |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| op100 Re100 128² | 0.401 | 0.302 | 0.264 | 0.326 | 0.447 | 0.434 |
| op500 Re500 128² | 0.371 | 0.281 | 0.243 | 0.288 | 0.488 | 0.428 |
| spatial-UNet Re500 256² | 0.454 | 0.333 | 0.309 | 0.260 | 0.516 | 0.462 |

per-shell LATE phase_frac:
| arm | k0 | k1 | k2 | k3 | k4 | k5 | k6 | k7 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| op100 Re100 128² | 0.00 | 0.20 | 0.40 | 0.43 | 0.50 | 0.48 | 0.51 | 0.53 |
| op500 Re500 128² | 0.00 | 0.31 | 0.59 | 0.51 | 0.49 | 0.47 | 0.43 | 0.41 |
| spatial-UNet Re500 256² | 0.00 | 0.39 | 0.53 | 0.54 | 0.54 | 0.53 | 0.48 | 0.40 |

Load guards reproduce banked err_k7 (0.401 vs op100 baseline 0.409; 0.3714 = 0.371; 0.4543 = 0.454). Aggregate late k≤7 phase fraction 0.447 / 0.488 / 0.516 — near half, rising Re100→Re500 and 128²→256²; only the 256² UNet exceeds 0.5. relL2 amplitude ≥ phase in all three (0.302>0.264, 0.281>0.243, 0.333>0.309) ⇒ energy-weighted aggregate is amplitude-leaning. Per-shell: k0 phase 0 (DC); k1 phase 0.20–0.39 (amplitude-leaning); k2–5 phase 0.40–0.59 (≥0.5 at several shells; peak k2 0.59 @Re500-128²; k2–5 ≈0.53–0.54 @256²); k6–7 taper to 0.40–0.53. Phase concentrated in k2–5. Time: late phase_frac > early at every shell (e.g. k2 @256² 0.35→0.53, k7 @256² 0.12→0.42). The energetic large scales (k0–1, carrying ~half the in-band energy) are amplitude; the displacement (phase) error is in the mid band k2–5 and grows along the trajectory.

---

# 2026-06-22

Loss-emphasis reweighting sweep (time AND scale axes) — does up-weighting late frames or de-biasing per-shell attention lower the late k≤7 wall? op100 @128² FNO n_modes=8 h128, from-scratch 150 ep, data5+pde1+ic1, Re100; GT-supervised DATA term only (pde/ic unchanged). Time: w_t=1+α(t/T)^p, mean-1 norm [α-sweep {0,1,2,4} at p=2; p-sweep {0.25,0.5,1,1.5} at α=4]. Band: per-Chebyshev-shell w_k over k=2-7, num+denom same w_k [ramp w∝k; equalize w∝E_k^-β, β∈{0.5,1.0}, E_k=per-sample GT shell energy]. a0=α=0=uniform baseline. Eval at final.ckpt (ep150, fixed-epoch; band arms also best.ckpt) on held-out Re100 [260:300] n=40. `src/pde/ns.py` {frame_weights,time_weighted_rel,shell_weights,band_weighted_rel}; configs `kf_re100_{timeweight,bandweight}.yaml`; `msc/tta/eval.py band_eval`.

| arm | early k≤7 | late k≤7 | aggr | early k2-7 | late k2-7 | full |
|---|--:|--:|--:|--:|--:|--:|
| a0 baseline | 0.1746 | 0.5432 | 0.4087 | 0.2230 | 0.7075 | 0.4307 |
| tw α1 p2 | 0.1808 | 0.5477 | 0.4142 | 0.2303 | 0.7125 | 0.4361 |
| tw α2 p2 | 0.1779 | 0.5380 | 0.4062 | 0.2270 | 0.7018 | 0.4280 |
| tw α4 p2 | 0.1862 | 0.5481 | 0.4140 | 0.2382 | 0.7129 | 0.4357 |
| tw p0.25 α4 | 0.1748 | 0.5430 | 0.4096 | 0.2257 | 0.7079 | 0.4320 |
| tw p0.5 α4 | 0.1760 | 0.5431 | 0.4076 | 0.2250 | 0.7062 | 0.4294 |
| tw p1 α4 | 0.1910 | 0.5438 | 0.4127 | 0.2426 | 0.7056 | 0.4348 |
| tw p1.5 α4 | 0.1839 | 0.5375 | 0.4075 | 0.2372 | 0.7013 | 0.4294 |
| band ramp | 0.1769 | 0.5456 | 0.4132 | 0.2288 | 0.7107 | 0.4353 |
| band half β0.5 | 0.1737 | 0.5466 | 0.4117 | 0.2196 | 0.7106 | 0.4332 |
| band full β1.0 | 0.1838 | 0.5500 | 0.4180 | 0.2350 | 0.7158 | 0.4398 |
| band ramp BEST | 0.1709 | 0.5432 | 0.4096 | 0.2221 | 0.7090 | 0.4319 |
| band half BEST | 0.1742 | 0.5459 | 0.4114 | 0.2192 | 0.7092 | 0.4329 |
| band full BEST | 0.1806 | 0.5456 | 0.4138 | 0.2295 | 0.7115 | 0.4354 |

Baseline late k≤7 0.5432, late k2-7 0.7075. No arm lowers late k≤7 by more than 0.006. Time: best α2 0.5380 (−0.0052), p1.5 0.5375 (−0.0057); non-monotone — α1 0.5477 and α4 0.5481 both above baseline, p0.25–p1 within ±0.001 of baseline. Band: all three above baseline on both late k≤7 (ramp 0.5456, half 0.5466, full 0.5500) and late k2-7 (0.7107, 0.7106, 0.7158), monotone-rising with β; best.ckpt within 0.005 of final. early k≤7 rises with weighting strength (α4 0.186, p1 0.191, band full 0.184 vs baseline 0.175). σ(late)≈0.001 (prior banked). Net: band-equalize raises late error; time-reweight moves late <0.006 and non-monotone.

---

# 2026-06-22

Cheaper-coupling + second-layer probes. (1) spatial_diag: channel-DIAGONAL spatial spectral bottleneck mixer — same global x,y coupling, one complex gain per (channel,mode), NO cross-channel mixing, ~130K params (vs full block ~1.1M); 128² (4lsdtvpz). (2) second spectral block added at the S/4 encoder level on top of the bottleneck spatial block; 256² (crzo760a). Trained 150 ep, Re500, data5+pde1+ic1, AMP. Same band metric as the prior rungs (band_eval early[1..8]/late[last T/8] k≤7, held-out [260:300] n=40). `scripts/unet_band_probe.py --mixer spatial_diag --modes 8` / `--mixer spatial --levels 1`.

128² [test 260:300]:
| arm | val_l2 | err full | err k≤7 | early k≤7 | late k≤7 |
|---|:--:|:--:|:--:|:--:|:--:|
| FNO op500 n8h128 | ~0.40 | 0.409 | 0.371 | 0.253 | 0.472 |
| UNet none | 0.452 | 0.457 | 0.424 | 0.289 | 0.538 |
| UNet spatial (full) | 0.395 | 0.413 | 0.375 | 0.251 | 0.478 |
| UNet spatial_diag | 0.423 | 0.429 | 0.393 | 0.258 | 0.512 |

256² [test 260:300]:
| arm | val_l2 | err full | err k≤7 | early k≤7 | late k≤7 |
|---|:--:|:--:|:--:|:--:|:--:|
| FNO n16h64 | 0.439 | 0.466 | 0.422 | 0.164 | 0.571 |
| UNet vanilla | 0.646 | 0.666 | 0.644 | 0.428 | 0.763 |
| UNet spatial (1 layer) | 0.466 | 0.495 | 0.454 | 0.282 | 0.577 |
| UNet spatial + 2nd@S/4 | 0.466 | 0.493 | 0.451 | 0.296 | 0.590 |

spatial_diag: early k≤7 0.258 ≈ full-spatial 0.251 ≈ FNO 0.253 and below none 0.289; late k≤7 0.512 lies between none 0.538 and full 0.478, nearer none; err_k7 0.393 and val_l2 0.423 both between none (0.424 / 0.452) and full (0.375 / 0.395). The early in-band equals the full block; the late in-band does not. Second spectral layer @256²: val_l2 0.466 equals the single-layer 0.466; err_full 0.493 vs 0.495 and err_k7 0.451 vs 0.454 (equal within rounding); early k≤7 0.296 > single 0.282 and late k≤7 0.590 > single 0.577 (both higher). The second layer reduced no error; early k≤7 stays ~1.8× FNO 0.164. Validity: load guards reproduce training val_l2 (0.4233 @128², 0.4661 @256²).

---

# 2026-06-21

Spatial FNO bottleneck mixer — does a channel-mixing spectral conv over x,y at the UNet bottleneck close the conv-UNet's in-band k≤7 gap to FNO (P0)? SpatialSpectralMixer (1×1 channel bottleneck 512→64, two-corner low-pass k≤8, per-mode channel mixing, zero-init up-proj → identity at start), replaces the dead temporal mixer. Trained 150 ep, Re500, data5+pde1+ic1, AMP. 256² (7sklgx44, GV100) + 128² (z2o8cbkl, P40). Same band metric as P0. `scripts/unet_band_probe.py --mixer spatial --modes 8 --hidden 64`.

256² [test 260:300]:
| arm | val_l2 | err full | err k≤7 | early k≤7 | late k≤7 | late/early |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FNO n16h64 | 0.439 | 0.466 | 0.422 | 0.164 | 0.571 | 3.47 |
| UNet vanilla | 0.646 | 0.666 | 0.644 | 0.428 | 0.763 | 1.79 |
| UNet spatial-FNO | 0.466 | 0.495 | 0.454 | 0.282 | 0.577 | 2.05 |

128² [test 260:300]:
| arm | val_l2 | err full | err k≤7 | early k≤7 | late k≤7 | late/early |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FNO op500 n8h128 | ~0.40 | 0.409 | 0.371 | 0.253 | 0.472 | 1.87 |
| UNet none | 0.452 | 0.457 | 0.424 | 0.289 | 0.538 | 1.86 |
| UNet spectral(t) | 0.447 | 0.453 | 0.420 | 0.278 | 0.528 | 1.90 |
| UNet spatial-FNO | 0.395 | 0.413 | 0.375 | 0.251 | 0.478 | 1.90 |

128²: spatial-FNO UNet matches FNO op500 on every metric — val_l2 0.395 vs ~0.40, err_full 0.413 vs 0.409, err_k7 0.375 vs 0.371, early k≤7 0.251 vs 0.253, late k≤7 0.478 vs 0.472 — and beats vanilla and both temporal mixers in-band (early 0.251 < spectral(t) 0.278 < none 0.289). 256²: collapse broken — val_l2 0.466 vs vanilla 0.646 (FNO 0.439); err_k7 0.454 vs vanilla 0.644; early k≤7 0.282 vs vanilla 0.428; late k≤7 0.577 ≈ FNO 0.571. Late in-band reaches the FNO wall at both resolutions; early in-band equals FNO at 128² (0.251 vs 0.253) but at 256² is 1.7× FNO (0.282 vs 0.164) — the residual gap to FNO is 256² early in-band only. Validity: load guard reproduces training val_l2 (0.4661 @256², 0.3949 @128²). One channel-mixing spectral bottleneck layer (~1.1M params, +5%) moves the conv-UNet from collapse to FNO-equal at 128² and within 6% val_l2 at 256²; the temporal mixers did not.

---

# 2026-06-19

Solver-closure gate (operator-free, physics) — does high-k IC content causally drive low-k (k≤7) evolution, or was the operator closure NO-GO a model limitation? Perturb ONLY k>kc of a true IC, roll the TRUE solver (`NavierStokes2d`+`kf_forcing`, 256² faithful), measure low-k divergence from GT vs the operator late k≤7 wall ≈0.47. Variants: zero (drop high-k), swap (donor's high-k), scramble (keep per-mode |û| in high-k, randomize phase = correct energy/wrong phase). Held-out [200:300] n=16, kc=7, both Re. `scripts/solver_closure_gate.py`.

late k≤7 divergence (low_late) + high-band divergence (high_late), pooled relL2 vs GT:
| Re | ctrl | zero | swap | scramble | high_late (zero/swap/scr) |
|---|:--:|:--:|:--:|:--:|:--:|
| 500 | 0.000 | 0.085 | 0.112 | 0.115 | 0.55 / 0.75 / 0.78 |
| 100 | 0.000 | 0.060 | 0.079 | 0.090 | 0.25 / 0.37 / 0.39 |

ctrl≈0 ⇒ solver faithful both Re. high_late 0.25–0.78 confirms the high band is genuinely corrupted. low_late 0.06–0.12 ≪ wall 0.47 (~5× below) BOTH Re ⇒ a 100%-wrong/absent high-k band barely perturbs the energetic low-k evolution ⇒ high-k carries little low-k-relevant info in the TRUE dynamics ⇒ the operator closure NO-GO is PHYSICS, not operator inability to carry high-k. scramble≈swap ⇒ the small residual coupling is PHASE-driven not energy ⇒ a spectral-ENERGY loss cannot capture even that. Re100 weaker coupling (0.06–0.09) + high-k decays faster (high_late 0.25–0.39 vs Re500 0.55–0.78).

---

# 2026-06-19

UNet operator-arm band probe — is the conv-UNet's val_l2 gap to FNO in-band (k≤7) or high-k? Forward trained UNet3D ckpts through the SAME load guard + band metric as the FNO arms (tta_eval.probe val_l2 + band_eval early[frames 1..8]/late[last T/8] k≤7 relL2 vs GT on held-out Re500 [260:300] n=40, op500-recipe data5+pde1+ic1). 256² vanilla (g367bsi5) vs FNO n16h64@256²; 128² none(2xfg6hw4)/spectral(1u585p0f) vs op500 n8h128@128². `scripts/unet_band_probe.py`.

256² [test 260:300]:
| arm | val_l2 | err full | err k≤7 | early k≤7 | late k≤7 | late/early |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FNO n16h64 | 0.439 | 0.466 | 0.422 | 0.164 | 0.571 | 3.47 |
| UNet vanilla | 0.646 | 0.666 | 0.644 | 0.428 | 0.763 | 1.79 |

128² [test 260:300]:
| arm | val_l2 | err full | err k≤7 | early k≤7 | late k≤7 | late/early |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FNO op500 n8h128 | ~0.40 | 0.409 | 0.371 | 0.253 | 0.472 | 1.87 |
| UNet none | 0.452 | 0.457 | 0.424 | 0.289 | 0.538 | 1.86 |
| UNet spectral | 0.447 | 0.453 | 0.420 | 0.278 | 0.528 | 1.90 |

256²: UNet err_k7 0.644 ≈ err_full 0.666 ⇒ the error is in k≤7, not high-k. UNet k≤7 worse than FNO at every lead time (early 0.428 = 2.6× FNO 0.164; late 0.763 > 0.571). 128²: UNet none in-band worse than op500 at every stage (early 0.289 > 0.253, late 0.538 > 0.472, err_k7 0.424 > 0.371) while the full-field val_l2 gap is smaller (0.452 vs ~0.40); spectral ≈ none (early 0.278 vs 0.289, late 0.528 vs 0.538, Δ ≤0.011). UNet late/early 1.79–1.90 vs FNO 3.47 @256²: UNet early k≤7 already high, FNO early k≤7 low. Validity: load-guard val_l2 reproduces logged training val (0.6464 @256², 0.4522/0.4473 @128² none/spectral). The conv-UNet's deficit vs FNO is in the k≤7 band at both resolutions, larger at 256² (early 2.6×) than 128² (early 1.14×); the temporal bottleneck mixer does not change the in-band error.

---

# 2026-06-19

Closure-gate kc sweep — where in k does the recoverable late-error signal live? Same band-split restart chaining (frozen fno16, held-out Re500 256² [260:300] n=40, stride 16), split kc swept {1,3,7,16}. hi_oracle = own k≤kc + GT k>kc; lo_oracle = GT k≤kc + own k>kc. `scripts/chain_gate_256.py --kc 1 3 7 16`.

anchors: oneshot 0.571, oracle 0.230, model 0.525 (late k≤7).
| kc | hi_oracle late | lo_oracle late |
|:--:|:--:|:--:|
| 1 | 0.293 | 0.489 |
| 3 | 0.407 | 0.357 |
| 7 | 0.521 | 0.230 |
| 16 | 0.525 | 0.230 |

Recoverable signal lives entirely in k≤7, spread across k≈2–7 (mid-low modes dominant, not the single largest scale): lo_oracle fills in k≤1→0.489, k≤3→0.357, k≤7→0.230(=oracle); hi_oracle decays GT-k>1→0.293, GT-k>3→0.407, GT-k>7→0.521(=model). Everything above k=7 is a bystander (hi_oracle kc7/kc16 = model), incl. the Fig-1 Re-divergent band (k>16). k≤1 alone minor (~0.06–0.08); the bulk of the recoverable signal is k=2–7.

---

# 2026-06-19

Closure gate — does correct high-k feed the k≤7 wall? (decides the per-band spectral-loss idea, no train). Band-split restart chaining: frozen fno16 on held-out Re500 256² [260:300] n=40, restart every stride from a constructed field, late k≤7 relL2 vs GT, paired vs one-shot. Split kc=7. sources: oracle (full GT restart), model (own field), hi_oracle (own k≤7 + GT k>7), lo_oracle (GT k≤7 + own k>7). `scripts/chain_gate_256.py` + `chain_gate.py` (`_band_mix`, sources hi_oracle/lo_oracle).

| source | stride | late | aggr | late win% | Δ̄ late |
|---|:--:|:--:|:--:|:--:|:--:|
| oneshot | — | 0.571 | 0.422 | — | — |
| oracle | 16/24 | 0.230 | 0.197 | 100% | −0.328 |
| model | 16 | 0.525 | 0.405 | 80% | −0.048 |
| model | 24 | 0.503 | 0.392 | 90% | −0.068 |
| hi_oracle | 16 | 0.521 | 0.403 | 80% | −0.052 |
| hi_oracle | 24 | 0.501 | 0.391 | 95% | −0.070 |
| lo_oracle | 16 | 0.230 | 0.197 | 100% | −0.327 |
| lo_oracle | 24 | 0.231 | 0.232 | 100% | −0.327 |

hi_oracle ≈ model (Δ̄ −0.052/−0.070 vs −0.048/−0.068) ⇒ correct high-k buys ~nothing. lo_oracle ≈ oracle (0.230, 100% win, Δ̄ −0.327) ⇒ correct k≤7 alone recovers the FULL oracle win even with the operator's own (wrong) high-k. Cascade closure into k≤7 ≈ 0 within the restart window ⇒ the high-k band (Re100→Re500 divergent band, 6–1000× offset) is a BYSTANDER for the wall; the full recoverable win is in k≤7. Validity: oneshot/oracle/model reproduce banked chain_gate (0.571/0.230/0.525) ✓.

---

# 2026-06-18

Predictability-horizon — per-band prediction error vs lead-time e(t,band) on held-out Re500 256² [260:300] n=40; e(t)=√(Σ_band err_pt / Σ_band gt_pt) per frame, Chebyshev L∞ bands; horizon = t/T where e≥0.95. FNO@16/@32 + UNO@32 (anchor op500@8 OOM'd as 4th model on P40, sanity-only). `scripts/time_band_resolved.py` (now saves gt_pt) + `scripts/horizon_bands.py`, fig `msc/tta/outputs/figs/horizon_bands.png`.

horizon (t/T at e≥0.95):
| band | FNO@16 | FNO@32 | UNO@32 | e(end) |
|---|:--:|:--:|:--:|:--:|
| k0-7 | never | never | never | ~0.58 |
| k8-16 | 0.33 | 0.33 | 0.31 | ~1.03 |
| k17-32 | 0.06 | 0.05 | 0.06 | ~1.00 |
| k33-64 | 0.05 | 0.00 | 0.05 | ~1.00 |
| k65-128 | 0.02 | 0.00 | 0.02 | ~1.00 |

k0-7 e(0)→e(end): 0.13→0.58 (FNO@16), 0.11→0.59 (FNO@32), 0.14→0.57 (UNO@32). Load guard: FNO@16/@32 reproduce banked err_full 0.4657/0.4643, late k≤7 0.571/0.575 ✓.

Horizon falls monotonically with k: k≤7 never decorrelates within T (concave drift, ratio ~3.5), k>16 decorrelated by t/T≈0.05, k8-16 by ~0.33. Horizons identical across capacity (16↔32) AND architecture (FNO↔UNO) ⇒ dynamical, not model. e(end)>1 for k>7 = decorrelated AND energy-mismatched (not merely low-energy). FNO@16 k>16 pinned at 1 from t=0 is ceiling+chaos confounded; FNO@32 k17-32 representable yet still saturates ≈immediately ⇒ unpredictability, not representation. k≤7 is predictable within T (drift); k>7 decorrelated.

---

# 2026-06-18

Gate 2 — phase-spread test (CAPSTONE of the migrate-fork). Is the late k≤7 PHASE intrinsically IC-sensitive (chaos ⇒ irreducible) or learnable? Phase analog of Gate 1 (which is power-based ⇒ phase-blind), and phase is 54–66% of the late wall. Per held-out Re500 256² IC [260:300] n=20: M=8 perturbed copies w0+δ·(k≤7 band-limited noise), δ∈{0.1,0.2,0.3} (brackets operator early error); roll the TRUE solver; energy-weighted ensemble circular phase dispersion D(k,t)=1−|⟨e^{iφ}⟩_M| over k≤7 modes, early(frames 1..8)/late(last T/8). 256² (faithful solver, not under-resolved 128²). `scripts/gate2_phase_spread.py`.

control = k≤7 relL2(unperturbed solve − GT) = **4e-6** ⇒ solver FAITHFUL @256² (solver chaos == data chaos) ✓.

| δ | D_early | D_late | late/early |
|---|:--:|:--:|:--:|
| 0.10 | 0.057 | 0.073 | 1.28 |
| 0.20 | 0.136 | 0.192 | 1.41 |
| 0.30 | 0.209 | 0.296 | 1.42 |

D_late LOW (0.07–0.30) and grows only ~1.3–1.4× (does NOT saturate→1), even at δ=0.30 (> operator early error 0.16–0.28). Late k≤7 phase does NOT intrinsically decorrelate ⇒ phase is LEARNABLE (grows like the power spread, Gate 1 ~1.3×, not the ≫1 chaos signature). n=20 confirms n=2 smoke (δ0.3 D_late 0.296 vs 0.308). READ WITH chain_gate (Gate 2 bounds the one-shot-from-IC map only): D_late low ⇒ one-shot SHOULD predict late-from-IC ⇒ the 0.55 wall is an OPTIMIZATION/OBJECTIVE failure, fixable IN PLACE — NOT chaos, NOT a re-anchor necessity.

Migrate-fork synthesis (elimination of chaos/capacity/representation → objective-lever hypothesis), self-consistency-bystander cross-ref, late-weighted candidate + chaos-amplification caveat, and user sequencing → **tta-roadmap.md** (forward-looking, kept out of the results bank).

---

# 2026-06-18

chain gate @256² — does the rollout lever (banked 128²: oracle≫one-shot, model compounds) transfer? Frozen op forwarded over held-out Re500 256² [260:300] n=40; k≤7 pooled early/late relL2 vs GT. oracle = restart every `stride` frames from the TRUE mid-traj field (best-case rollout); model = restart from the op's OWN predicted field; paired late win% vs one-shot. `scripts/chain_gate_256.py`. op500 (n8@128², res-flexible) + fno16 (256²-native, adb4tfh0).

| op | source | stride | late | aggr | late win% | Δ̄ late |
|---|---|:--:|:--:|:--:|:--:|:--:|
| op500 | one-shot | — | 0.528 | 0.416 | — | — |
| op500 | oracle | 16/24 | 0.313 | 0.291 | 100% | −0.200 |
| op500 | model | 16 | 0.531 | 0.433 | 32% | +0.013 |
| op500 | model | 24 | 0.504 | 0.413 | 57% | −0.014 |
| fno16 | one-shot | — | 0.571 | 0.422 | — | — |
| fno16 | oracle | 16/24 | 0.230 | 0.197 | 100% | −0.328 |
| fno16 | model | 16 | 0.525 | 0.405 | 80% | −0.048 |
| fno16 | model | 24 | 0.503 | 0.392 | 90% | −0.068 |

Rollout lever REAL @256² (confirms 128²): oracle re-anchor beats one-shot late on 100% of instances, both ops — prize op500 0.528→0.313, fno16 0.571→**0.230**. Model (own-field) chaining: op500 ≈ one-shot (32–57% win, weak fields, like 128²); fno16 modestly net-positive (80–90% win, 0.571→0.503) — its lower early error (0.164 vs op500 0.279) makes its OWN restart fields cleaner, but the gap to oracle (~0.27) is still large. Given a recent TRUE field the operator predicts the next window far below its one-shot wall ⇒ the wall is error accumulation from a stale IC under the one-shot objective, NOT capacity/arch. [Objective-lever synthesis & next steps → tta-roadmap.md.]

---

# 2026-06-18

Spectrum diagnostic — time-averaged radial E(k), Re100 (ID) vs Re500 (target) on 256² GT, 1000 snapshots/Re sampled evenly over (300,129); offset = log10(E_Re500/E_Re100); k_d bracketed by log-slope knee and 99.9%-palinstrophy. `scripts/spectrum_diag.py`, fig `msc/tta/outputs/figs/spectrum_diag.png`.

| Re | k_d knee | k_d pal99.9 |
|---|:--:|:--:|
| 100 | 22 | 31 |
| 500 | 25 | 78 |

| band | k | offset log10(E500/E100) | ratio |
|---|:--:|:--:|:--:|
| forcing | 4 | +0.172 | 1.5× |
| resolved | 7 | +0.113 | 1.3× |
| n_modes16 | 16 | +0.779 | 6× |
| n_modes32 | 32 | +2.068 | 117× |
| residual | 42 | +2.984 | ~1000× |

crossover k_c = **nan** (no sign flip; E_Re500 ≥ E_Re100 at every k).

Offset is sign-definite and monotone-rising ⇒ no crossover exists. In-band (k≤7) the two Re are near energy-matched (+0.11 dex = 1.3×); the gap is 6× at FNO cutoff n_modes=16, 117× at n_modes=32, ~1000× by k=42 — doubling modes 16→32 does not reach a regime where the two Re match. Right-panel peak ≈9 dex near k≈80 = Re100 hitting its dissipation floor (~1e-7) while Re500 still has signal, not physical at that k. k_d brackets: Re100 [22,31], Re500 [25,78] — Re500 knee fires early (eyeball: roll-off extends to ~78). Re500 resolved within 256² Nyquist (128).

---

# 2026-06-18

Two 256² re-confirmations of the horizon-vs-capacity decider below. (1) Paired op-quality — is late k≤7 systematic (learnable) or chaos? op300 vs op500 (both n8@128², FNO res-flexible) forwarded on the SAME Re500 256² ICs [200:300] n=100; paired per-traj late/aggr k≤7 (`eval.per_instance_k7`), Wilcoxon + sign test. `scripts/paired_op_quality_256.py`. (2) Temporal-mode point — truncate FNO@16@256² one-shot to K_t time-modes, late k≤7 vs full + gt_late(K_t) floor; inference proxy for a 16-time-mode train (256² train needs ≥24 GB). `scripts/temporal_payoff_256.py`.

| op | early k≤7 | late k≤7 | aggr k≤7 |
|---|:--:|:--:|:--:|
| op300 (off-Re) | 0.2305 | 0.5241 | 0.4031 |
| op500 (on-Re) | 0.2686 | 0.5045 | 0.3965 |

paired late: op500 > op300 **61/100** (Δ̄ +0.0195, wilcoxon 1.2e-2, sign 3.5e-2). paired aggr: 52/100 (Δ̄ +0.0067, wilcoxon 0.24, sign **0.76 = chance**).

| K_t | model_late | Δ vs full | gt_late floor | model_early |
|---|:--:|:--:|:--:|:--:|
| full | 0.550 | — | — | 0.186 |
| 3 | 0.593 | +0.043 | 0.258 | 0.289 |
| 8 | 0.571 | +0.022 | 0.135 | 0.186 |
| 16 | 0.556 | +0.006 | 0.065 | 0.161 |
| 24 | 0.553 | +0.003 | 0.057 | 0.165 |

Temporal-mode lever DEAD @256² (confirms 128²): model_late(K_t) ≥ full at EVERY K_t (Δ +0.003…+0.043 — truncation recovers nothing ⇒ model late error is in the smooth ≤8-mode band); gt_late(8)=0.135 ≪ 0.55 wall ⇒ 8 t-modes over-represent the GT late field. No 16-t-mode train needed. Chaos-vs-systematic @256²: the migrate-relevant metric is LATE, and late is op500>op300 61/100 p=0.035 = WEAKLY SIGNIFICANT SYSTEMATIC (chaos would be 50/50; it is not). aggr 52/100 p=0.76 is NOT a chaos signal — it is early/late CANCELLATION: op300 wins early (0.231<0.269, the banked Re-invariant early floor) while op500 wins late, so pooling cancels — CONSISTENT with the 128² story, not against it. The late effect is small (Δ̄ 0.02 on a 0.55 wall), plausibly attenuated by the OOD-resolution confound (both ops n8@128²). Read: a small reducible late component is demonstrated; whether a LARGE reducible gap exists is the open question — and that is answered directly by chaining, not by op-pair size. Capacity dead on ALL axes (spatial + temporal + arch) = solid; a small reducible late component is demonstrated. [Direction / next-steps → tta-roadmap.md; superseded by the Gate 2 capstone + chain_gate @256² above.]

---

# 2026-06-18

Horizon-vs-capacity decider — on CONVERGED best ckpts, is the ~0.44 val_l2 wall late-time drift or spatial capacity? Forward each operator over LOCKED test [260:300] n=40 at 256²; k≤7 energy-pooled relL2 vs GT per lead-time + spectral split of error power, via `eval.band_eval`. Arms: FNO n16h64 / n32h64 (flat), UNO n32 (U-shaped, param-matched 83M ≈ FNO@32), op500 n8h128@128² anchor forwarded on 256². val_l2 = full-field LpLoss d3p2 rel on val [200:260] (load guard; n32/uno/anchor on n=16 subset). `scripts/time_band_resolved.py`.

| arm | best ep | val_l2 | err full | err k≤7 | early k≤7 | late k≤7 | late/early |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| op500 n8h128 anchor @256² | — | 0.4403 | 0.4614 | 0.4163 | 0.2789 | 0.5281 | 1.89 |
| FNO n16h64 | 89 | 0.4393 | 0.4657 | 0.4218 | 0.1643 | 0.5707 | 3.47 |
| FNO n32h64 | 64 | 0.4260 | 0.4643 | 0.4220 | 0.1476 | 0.5745 | 3.89 |
| UNO n32 | 39 | 0.4154 | 0.4596 | 0.4167 | 0.1681 | 0.5576 | 3.32 |

k≤7 relL2 e(t) by lead time (frac of horizon T=65):

| arm | t=0 | 1/8 | 1/4 | 1/2 | 3/4 | end |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| op500 n8 anchor | 0.249 | 0.297 | 0.336 | 0.413 | 0.481 | 0.542 |
| FNO n16h64 | 0.133 | 0.195 | 0.280 | 0.427 | 0.524 | 0.583 |
| FNO n32h64 | 0.112 | 0.183 | 0.274 | 0.429 | 0.527 | 0.586 |
| UNO n32 | 0.137 | 0.201 | 0.286 | 0.425 | 0.513 | 0.570 |

error rel-L2 per disjoint k-band + frac of total error power in k≤7:

| arm | k0-7 | k8-16 | k17-32 | k33-64 | k65-128 | errfrac k≤7 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| op500 n8 anchor | 0.416 | 0.968 | 0.989 | 0.993 | 0.999 | 0.772 |
| FNO n16h64 | 0.422 | 0.963 | 0.993 | 0.995 | 0.999 | 0.778 |
| FNO n32h64 | 0.422 | 0.944 | 0.993 | 0.999 | 1.002 | 0.784 |
| UNO n32 | 0.417 | 0.944 | 0.992 | 0.995 | 1.002 | 0.780 |

Late-time drift in resolved scales; spatial capacity + arch saturated. Validity: load guards reproduce logged val_l2 (n16h64 0.4393 = best_score; anchor reproduces banked 256² 0.5281/0.2789/0.4163/0.4614 ✓). CAVEAT val_l2 col = load guard, NOT a ranking — n32/uno/anchor on n=16 (≈0.01 low vs logged 60-sample); not cross-row comparable. early k≤7 0.15–0.17 (256²-native) < n8 anchor 0.279 < banked 0.248 ⇒ the "0.248 Re-invariant intrinsic floor" is 128²-specific, beaten by 256²-native train (contradicts findings "intrinsic"). late 0.558–0.586 across modes 16→32 + flat→U (Δlate ≤0.017, saturated; val_l2 0.439→0.415 marginal). e(t) climbs 0.11–0.14→0.57–0.59, ratio 3.3–3.9; ALL arms converge to one late wall from different early — alone this is ALSO the chaos/predictability-horizon signature, so e(t) slope does NOT by itself prove fixable. 78% err power in resolved k≤7; k>7 relerr≈1 but low GT energy → not the gap. n32 best ep64 / uno ep39 (overfits past ~40). EARNED here: 256² late wall = drift in resolved scales; more spatial modes (16→32) + flat→U both dead. CARRIED from 128² (NOT re-run at 256²): late k≤7 SYSTEMATIC not chaos (op500>op300 35/40 p=1e-6, findings) + temporal-mode sweep DEAD (temporal_budget: 8 t-modes hold late 0.19 ≪ wall) + residual non-identifying ⇒ prior "next = rollout/self-consistency". ⇒ rollout / windowed-per-step (self-consistency) loss is the CONVERGENT hypothesis (this + 3 banked), NOT more spatial modes — but OPEN at 256²: paired op-quality test (systematic?) + one temporal-mode point, before committing to an autoregressive rewrite. [SUPERSEDED by the Gate 2 capstone + chain_gate @256² above: both closed — late phase LEARNABLE (Gate 2), temporal-mode DEAD; lever = the training OBJECTIVE on the existing FNO, not a new architecture.]

---

# 2026-06-17

Capacity gate — does widening the FNO representable band (n_modes) or width move the late k≤7 wall? Forward each operator over the LOCKED held-out Re500 test split [260:300] n=40; energy-pooled early(post-IC frames 1..8)/late(last T/8) k≤7 relL2 vs GT, ‖GT‖-normalised, via `eval.band_eval`. All arms scored on the SAME data so cross-arm delta = capacity effect (FNO res-flexible → n8@128 anchor forwards on 256² ICs). op500 throughout. `scripts/k7_modes_gate.py`.

| arm | train | eval data | late k≤7 | early k≤7 | err k≤7 | err full |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| anchor n8 h128 | 128² | 128² | 0.4723 | 0.2527 | 0.3714 | 0.4091 |
| anchor n8 h128 | 128² | 256² | 0.5281 | 0.2789 | 0.4163 | 0.4614 |
| n16 h32 | 256² | 256² | 0.5890 | 0.1794 | 0.4303 | 0.4743 |
| n16 h64 | 256² | 256² | 0.5799 | 0.2043 | 0.4355 | 0.4781 |

Anchor: n8@128² reproduces banked late floor (0.4723 ≈ 0.473) ✓ harness. On faithful 256² GT the n8 floor is late=0.528; n16 late=0.580–0.589 (> 0.528, no gain) at h32≈h64 (Δlate=0.009, width-invariant on the gate, not just val_l2). early: n16 0.179–0.204 < n8 0.279 (modes improve early; gain decays by late). Caveat: n8 trained @128², n16 @256² → n16 has more modes AND cleaner data, still no late gain (strictly stronger negative). n16h64 ckpt @epoch~24 val_l2 0.454 (h32 plateau 0.448, converged); n16h32 fully converged.

---

# 2026-06-17

Gate 1 — chaos-spread vs operator dispersion. Perturb true IC by band-limited (k≤7) noise of rel-size δ; roll M=8 ensembles from the SAME perturbed ICs via the true solver (σ_phys) and the operator (σ_op); late & early(post-IC, frames 1..8) k≤7 pooled relL2, ‖GT‖-normalised. growth=σ_phys_late/σ_phys_early; σ_op=operator intra-ensemble spread; op_mean=op-ensemble-mean vs GT; pred-Em=relL2(pred,phys-ens-mean); cov_op=relL2(GT,op-mean)/σ_op; op_single=deterministic operator wall (=`wall`); control δ=0 = solver from true IC vs GT. adapt_op100 & op300 @ Re500 (OOD), op300 @ Re300 (ID); held-out [200:300] n=20. `scripts/chaos_spread_gate.py`.

| op @ Re | wall | δ | σph_e | σph_l | growth | σ_op | op_mean | pred-Em | cov_op |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| adapt_op100 @ Re500 | 0.4937 | 0.05 | 0.0459 | 0.0646 | 1.41 | 0.0200 | 0.4939 | 0.4920 | 24.64 |
| adapt_op100 @ Re500 | 0.4937 | 0.10 | 0.0918 | 0.1278 | 1.39 | 0.0410 | 0.4953 | 0.4845 | 12.08 |
| adapt_op100 @ Re500 | 0.4937 | 0.20 | 0.1837 | 0.2432 | 1.32 | 0.0807 | 0.4990 | 0.4589 | 6.19 |
| op300 @ Re500 | 0.4537 | 0.05 | 0.0459 | 0.0646 | 1.41 | 0.0244 | 0.4536 | 0.4525 | 18.63 |
| op300 @ Re500 | 0.4537 | 0.10 | 0.0918 | 0.1278 | 1.39 | 0.0473 | 0.4544 | 0.4442 | 9.61 |
| op300 @ Re500 | 0.4537 | 0.20 | 0.1837 | 0.2432 | 1.32 | 0.0935 | 0.4595 | 0.4227 | 4.92 |
| op300 @ Re300 | 0.4282 | 0.05 | 0.0464 | 0.0559 | 1.20 | 0.0240 | 0.4285 | 0.4264 | 17.86 |
| op300 @ Re300 | 0.4282 | 0.10 | 0.0927 | 0.1086 | 1.17 | 0.0484 | 0.4288 | 0.4210 | 8.86 |
| op300 @ Re300 | 0.4282 | 0.20 | 0.1854 | 0.2094 | 1.13 | 0.0933 | 0.4307 | 0.4062 | 4.61 |

Anchor: control δ=0 = 0.0000 all three ✓. σ_phys ≈ linear in δ (σph_l ≈ 1.3δ), growth = 1.32–1.41 (Re500) / 1.13–1.20 (Re300), no saturation; growth higher at Re500 than Re300 at every δ. σ_op ≪ σ_phys (σ_op 0.020–0.094 vs σph_l 0.056–0.243); cov_op = 4.6–24.6. σ_op(op300) ID≈OOD: Re300 0.024/0.048/0.093 ≈ Re500 0.024/0.047/0.094. Walls 0.494/0.454/0.428 (cf. op500 frozen floor 0.473, banked earlier). pred-Em ≈ wall, falls slightly with δ.

---

# 2026-06-14

Resolution gate (operator-free) — distributional k≤7 target validity, native 512² vs native 128² Re500 GT; ICs independent between sets (frame-0 cross-dataset relL2≈1.27). Ensemble + window-mean Chebyshev E(k), power /S⁴ (physical density, resolution-comparable); ratio = 512²/128², 95% CI = 4000× bootstrap resampling the two ensembles separately. Re500 all-300 (offset 0), sub_t=2. Confirmatory: GT NS-residual floor rel-L2(Du,f) per band, n=40 (reuse `floor_ablation`, cd2 aliased). `scripts/res512_gate.py`.

late k≤7 E(k), 512²/128² per shell:
| k | E128_late | E512_late | ratio | CI95 | early ratio |
|---|:--:|:--:|:--:|:--:|:--:|
| 1 | 11.37 | 8.637 | 0.759 | [0.749,0.770] | 0.757 |
| 2 | 2.520 | 2.291 | 0.909 | [0.852,0.971] | 0.927 |
| 3 | 3.282 | 2.849 | 0.868 | [0.798,0.944] | 0.869 |
| 4 | 2.544 | 2.261 | 0.889 | [0.814,0.971] | 0.899 |
| 5 | 0.996 | 0.915 | 0.918 | [0.847,0.997] | 0.983 |
| 6 | 0.421 | 0.447 | 1.060 | [0.991,1.131] | 1.016 |
| 7 | 0.246 | 0.272 | 1.104 | [1.041,1.172] | 1.065 |

k≤7 total (late): 128=21.38, 512=17.67, ratio 0.826 CI95=[0.799,0.855]. 512² roll-off: k64=9.6e-6, k128=1.5e-9, k200=5.7e-14. residual floor rel-L2(Du,f) [128²/512²/256²]: k≤7 = 0.021/0.018/0.018; k≤42 = 0.210/0.160/0.160.

256² working-res validation (spectral-downsample 512²→256², FFT-crop |k|≤128): energy retention k≤128 vs 512²-native = 1.000000; 256/128 k≤7 ratio 0.826 [0.799,0.855] (= 512/128, in-band exact); residual floor = 512² to 4 digits (above).

Anchor: 512² E(k) tail monotone to 5.7e-14 (roll-off present), 128² absent; 256² retention 1.000000. Numbers: ratio CI excludes 1.0 for k=1-5 (0.76-0.92) and k=7 (1.10); includes 1.0 for k=6 (1.06 [0.991,1.131]); k≤7 total 0.826 [0.799,0.855]. residual k≤7 ≈ equal all res (0.021/0.018/0.018); k≤42 512²=256²<128² (0.160<0.210).

---

Low-k positioning ceiling — late k≤7 relL2 under a third GT oracle full=replace shells ≤b with GT complex modes (magnitude AND phase); amp_lev=base−modemag (phase-blind ceiling), POS_lev=modemag−full (value of positioning). Adapted op100→Re500 (λ=0) + frozen op100, held-out Re500[200:300] n=100. Cross-check: late phase-energy share from banked `amp_phase.npz` (Tb−Ab over shells ≤b, /late k≤7 wall). `scripts/ic_identifiability.py`.

| op | band | base | modemag | full | amp_lev | POS_lev |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| adapt_op100 | k≤1 | 0.5625 | 0.5455 | 0.5351 | 0.0170 | 0.0104 |
| adapt_op100 | k≤2 | 0.5625 | 0.5425 | 0.4835 | 0.0200 | 0.0591 |
| adapt_op100 | k≤3 | 0.5625 | 0.5428 | 0.3814 | 0.0197 | 0.1614 |
| adapt_op100 | k≤7 | 0.5625 | 0.5683 | 0.0000 | -0.0058 | 0.5683 |
| op100 | k≤1 | 0.6657 | 0.6330 | 0.6198 | 0.0326 | 0.0132 |
| op100 | k≤2 | 0.6657 | 0.6393 | 0.5414 | 0.0263 | 0.0979 |
| op100 | k≤3 | 0.6657 | 0.6608 | 0.4155 | 0.0048 | 0.2453 |
| op100 | k≤7 | 0.6657 | 0.6957 | 0.0000 | -0.0300 | 0.6957 |

npz cross-check, late phase-energy / wall: k≤1 = 0.025 (adapt_op100) / 0.018 (op100); k≤2 = 0.118 / 0.135; k≤3 = 0.309 / 0.304.

Anchor: full at k≤7 = 0.0000 ✓. POS_lev(k≤1) = 0.0104 (adapt) / 0.0132 (op100); grows with band to 0.57/0.70 at k≤7. Two methods agree at k≤1 (oracle POS_lev 0.010 ↔ npz phase-share 0.025, both ≪ wall). amp_lev ≤ 0.033 every band; negative at k≤7.

---

IC-moment identifiability + band-resolved spectral ceiling — (1) per-instance vorticity-moment violation |M_n(pred_late)−M_n(IC)|, M_n=mean_x ω^n, correlated across instances vs late k≤7 relL2 (r,sig=|t|>1.984), with GT drift |M_n(gt_late)−M_n(IC)|; (2) late k≤7 relL2 under GT oracles per band: shellE=per-shell GT energy match (phase kept), modemag=per-mode |û|=|ĝ| (phase kept). Frozen op{100,500} + adapted op100→Re500 (λ=0), held-out Re500[200:300] n=100. `scripts/ic_identifiability.py`.

moments (mean over n=100):
| op | n | \|viol\| | \|gtdrift\| | r | sig |
|---|:--:|:--:|:--:|:--:|:--:|
| op100 | 1 | 0.0338 | 3.9e-8 | 0.118 | no |
| op100 | 2 | 13.48 | 1.244 | 0.617 | yes |
| op100 | 3 | 92.06 | 34.81 | 0.444 | yes |
| op100 | 4 | 4881 | 1031 | 0.532 | yes |
| op500 | 1 | 0.0474 | 3.9e-8 | 0.352 | yes |
| op500 | 2 | 8.407 | 1.244 | 0.774 | yes |
| op500 | 3 | 104.2 | 34.81 | 0.576 | yes |
| op500 | 4 | 4076 | 1031 | 0.641 | yes |
| adapt_op100 | 1 | 0.0668 | 3.9e-8 | -0.140 | no |
| adapt_op100 | 2 | 9.376 | 1.244 | 0.698 | yes |
| adapt_op100 | 3 | 68.84 | 34.81 | 0.442 | yes |
| adapt_op100 | 4 | 3450 | 1031 | 0.570 | yes |

band spectral ceiling (late k≤7 relL2):
| op | baseline | k≤1 shellE | k≤1 modemag | k≤7 shellE | k≤7 modemag |
|---|:--:|:--:|:--:|:--:|:--:|
| op100 | 0.6657 | 0.6359 | 0.6330 | 0.7335 | 0.6957 |
| op500 | 0.4585 | 0.4543 | 0.4509 | 0.4908 | 0.4441 |
| adapt_op100 | 0.5625 | 0.5501 | 0.5455 | 0.6060 | 0.5683 |

Anchor: n=1 |gtdrift|≈4e-8 (circulation conserved) ✓. Numbers: k≤7 shellE > baseline for all 3 ops (+0.068/+0.032/+0.044); k≤1 shellE−baseline = −0.030/−0.004/−0.012; k≤7 modemag ≥ baseline except op500 (−0.014). |gtdrift|/|viol| = 0/0.09/0.38/0.21 (op100), 0/0.15/0.33/0.25 (op500), 0/0.13/0.51/0.30 (adapt) for n=1/2/3/4; pearson r sig for n≥2 (op100, adapt) and all n (op500).

---

Phase-blindness oracle — keep the operator's phase, swap its k≤7 Fourier magnitudes toward GT, re-measure pooled relL2 per window (= ceiling of any phase-blind energy-matching loss): raw=û; shell=pooled per-L∞-shell energy match α_s=√(ΣC_s/ΣA_s); mode=per-mode |û|→|g|; gt=full swap. Frozen op{100,500} + adapted op100→Re500 λ={0,10}, held-out Re500[200:300] n=100. `scripts/phase_oracle.py`.

| ckpt | window | raw | shell | mode | gt |
|---|---|:---:|:---:|:---:|:---:|
| adapt_lam0  | early | 0.1852 | 0.1671 | 0.1306 | 0.0 |
| adapt_lam0  | late  | 0.5813 | 0.6308 | 0.6025 | 0.0 |
| adapt_lam0  | aggr  | 0.4239 | 0.4408 | 0.4118 | 0.0 |
| adapt_lam10 | late  | 0.6120 | 0.6514 | 0.6167 | 0.0 |
| op100_froz  | late  | 0.6782 | 0.7558 | 0.7243 | 0.0 |
| op500_froz  | late  | 0.4727 | 0.5136 | 0.4683 | 0.0 |

raw late reproduces banked 0.678/0.473/0.581 (pipeline valid). At late, both magnitude-only oracles exceed raw (shell 0.631, mode 0.603 > 0.581): no phase-blind magnitude correction lowers the late error, it raises it. At early, both fall below raw (0.185→0.167→0.131): it lowers the early error.

---

# Amp/phase ladder — frozen vs adapted, 5 source ops → Re500 (2026-06-13)
Frozen + lr3e-4/N160-adapted op{100,200,300,500,1000} → Re500, held-out [200:300] n=100, k≤7. spec = systematic shell-spectrum (pooled-E(k)-steerable part). `scripts/amp_phase_split.py --ckpt`. Anchors ✓: frozen op100/300/500 aggr = banked 0.531/0.404/0.370; each adapt aggr = its run aggr_final.

| source | LATE relL2 froz→adapt | LATE phase% | LATE spec% | aggr froz→adapt |
|---|:---:|:---:|:---:|:---:|
| op100  | 0.678→0.580 | 56→61 | 24.9→15.5 | 0.531→0.424 |
| op200  | 0.567→0.538 | 52→60 | 22.6→14.7 | 0.428→0.394 |
| op300  | 0.527→0.527 | 53→60 | 21.6→15.0 | 0.404→0.388 |
| op500  | 0.473→0.489 | 46→46 | 24.8→27.1 | 0.370→0.382 |
| op1000 | 0.480→0.509 | 60→62 | 10.9→10.5 | 0.373→0.380 |

per-shell (late): systematic spectrum sits at k0-1; k2-7 phase-dominated. Read: residual+IC adaptation already eats ~40% of the OOD systematic-spectrum (op100/200/300 late spec 22-25%→15%) and late error becomes MORE phase (chaos ~60%); op500 (ID) flat/up, op1000 (rough) phase-heavy + low spec throughout. Late relL2 only improves for far-OOD op100/op200; op300/500/1000 late flat-or-worse (late wall ~adaptation-invariant). Steerable spectral lever left on the deployment op (adapt_op100) ≈ 15.5% → ~0.047 late upper bound, marginal over the 13% GO line.

---

# Amplitude/phase split of the k≤7 error — Exp 1 (2026-06-13)
Frozen op{100,300,500} one-shot, held-out Re500[200:300] n=100, k≤7. Per mode |û−g|²=(|û|−|g|)²+2|û||g|(1−cosΔφ): amp=(|û|−|g|)², phase=T−amp; spec=Σ_k(√ΣU_k−√ΣG_k)² (systematic shell-spectrum). Pooled error-energy fractions per early/late/aggr. `scripts/amp_phase_split.py`.

| op | window | relL2 | phase% | amp% | spec% |
|---|---|:---:|:---:|:---:|:---:|
| op100 | early | 0.2726 | 20.2 | 79.8 | 60.8 |
| op100 | late  | 0.6782 | 56.1 | 43.9 | 24.9 |
| op100 | aggr  | 0.5307 | 51.3 | 48.7 | 23.9 |
| op300 | early | 0.2344 | 28.2 | 71.8 | 45.5 |
| op300 | late  | 0.5269 | 53.1 | 46.9 | 21.6 |
| op300 | aggr  | 0.4042 | 47.8 | 52.2 | 23.0 |
| op500 | early | 0.2480 | 31.2 | 68.8 | 41.6 |
| op500 | late  | 0.4727 | 46.4 | 53.6 | 24.8 |
| op500 | aggr  | 0.3700 | 41.6 | 58.4 | 27.4 |

anchor (k≤7 aggr vs banked): op100 0.5307/0.531 · op300 0.4042/0.404 · op500 0.3700/0.370 ✓.

late per radial shell k=0..7 (ph=phase% / sp=spec%):
| op | k0 | k1 | k2 | k3 | k4 | k5 | k6 | k7 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| op100 ph | 0 | 15 | 56 | 63 | 66 | 65 | 62 | 54 |
| op100 sp | 100 | 73 | 24 | 19 | 11 | 13 | 19 | 27 |
| op300 ph | 0 | 18 | 53 | 58 | 59 | 55 | 50 | 46 |
| op300 sp | 100 | 63 | 19 | 17 | 14 | 21 | 26 | 32 |
| op500 ph | 0 | 30 | 51 | 46 | 49 | 46 | 44 | 39 |
| op500 sp | 100 | 35 | 14 | 25 | 23 | 27 | 32 | 39 |

---

# GT-floor term ablation — ∂ₜ stencil × advection dealias (2026-06-13)
GT Re500 [260:300] n=40, sub_t=2, ν=1/500. NS residual rel-L2(Du,f) on ground truth (no model). ∂ₜ ∈ {cd2,cd4,cd6} × advection ∈ {aliased, 2/3-dealias}, scored k≤7 / k≤42 / full (Nyquist 64); common interior window frames 3..T-4. `scripts/floor_ablation.py`. anchor (cd2,aliased,full)=0.6635 vs banked 0.666 ✓.

| stencil | advection | k≤7 | k≤42 | full |
|---|---|:---:|:---:|:---:|
| cd2 | aliased     | 0.0347±0.019 | 0.3570±0.286 | 0.6635±0.464 |
| cd2 | 2/3-dealias | 0.0347±0.019 | 0.5576±0.308 | 0.8019±0.486* |
| cd4 | aliased     | 0.0013±0.001 | 0.1204±0.139 | 0.5250±0.346 |
| cd4 | 2/3-dealias | 0.0013±0.001 | 0.4332±0.197 | 0.8073±0.510* |
| cd6 | aliased     | 0.0001±0.000 | 0.0720±0.101 | 0.5022±0.316 |
| cd6 | 2/3-dealias | 0.0001±0.000 | 0.4191±0.179 | 0.8327±0.540* |

term mags ‖·‖/‖f‖: |wt| = 5.997 (cd2) / 6.154 (cd4) / 6.178 (cd6); |adv| = 6.309 (aliased) / 6.252 (dealias); |diff| = 0.274. *full-band not comparable for the dealiased arm (modes >42 zeroed in adv only, not ∂ₜ/diff).

Read (eyeball): full floor 0.666 = ~0.50 spatial under-resolution (dealias→worse ⇒ not removable aliasing) + ~0.16 CD2 time-error; CD4 removes the time-error (k≤7 0.035→0.001). Cross-checked: dt-sweep native-dt mean 0.666 + CD6 0.502 ≈ CD2@half-dt 0.493; magnitude matches temporal_budget floor 0.031 (the banked "0.0025" is an energy fraction, not a magnitude).

---

# Dose-response gate (transductive, consist-isolated, collapse-controlled) (2026-06-13)
Per-instance TTIA (adapt+measure SAME held-out instance), opX→Re500[200:300] n=20, m=32, 20 steps, lr1e-3. Loss = consist(restart F(u[32]) as teacher) + 5·ic, **NO physics** (isolates consist; ic forbids collapse). headline = of-si-down instances, frac with gt_late also down. `scripts/self_consistency_dose.py`.

| op | si_late 0→F (frac↓) | gt_late 0→F (Δ̄) | gt_early 0→F | dose ρ(Δsi,Δgt) | si↓→gt↓ |
|---|:---:|:---:|:---:|:---:|:---:|
| op100 | 0.302→0.204 (1.00) | 0.615→0.899 (+0.284) | 0.258→0.287 | +0.217 (p.36) | 0% |
| op500 | 0.228→0.253 (0.40) | 0.409→0.957 (+0.548) | 0.224→0.242 | +0.507 (p.02) | 0% |

gt_early stable (ic anchor prevents collapse, unlike no-anchor). op100 (clean cell): si↓ on 100%, gt_late↑ on 100% — lowering si moves truth the wrong way. op500 objective barely engaged (40%↓). Confirms self-inconsistency = **bystander, not steerable**, collapse-controlled & consist-isolated (corroborates the pooled dose-response below).

---

# Dose-response gate — is self-inconsistency a STEERABLE lever for late GT error? (2026-06-12)
Label-free TTA opX→Re500, pool[0:160], held-out[200:300] n=100, m=32, 3000 steps, lr 3e-4. Loss = consist + 5·ic + 1·pde(k≤7); one-shot tail pulled toward its own frozen restart F(u[32]). `scripts/dose_response_gate.py`. dose = held-out self-inconsistency relL2(F(u[m]),u[m:]); response = held-out k≤7 GT late.

| op | dose start→final | GT late start→final (min) | early start→final |
|---|:---:|:---:|:---:|
| op100 | 0.246 → 0.126 | 0.678 → 0.659 (min 0.610) | 0.271 → 0.162 |
| op500 | 0.190 → 0.145 | 0.473 → 0.503 | 0.248 → 0.185 |

Dose halves (op100) / drops (op500); GT late does NOT follow — op100 noisy-flat, stays ABOVE the k≤7-physics floor 0.556; op500 gets worse. ⇒ **self-inconsistency is a BYSTANDER, not a steerable lever** (truth-gate correlation was non-causal). Plots: outputs/dose_response/dose_response_op{100,500}.png.

---

# Chaos-vs-artifact split — is the late wall fixable physics or unfixable chaos? (2026-06-12)
Frozen opX one-shot field at frame t_r → fed to the TRUE NS solver (data generator `NavierStokes2d`, Re500, f=−4cos4y, 2/3-dealias) → integrated to t=1. Held-out Re500[200:300] n=100, late k≤7, normalised by ‖GT‖. A=relL2(pred,GT) total; B=relL2(pred,solver) fixable artifact; C=relL2(solver,GT) chaos. Control = solver from TRUE field@t_r → C_ctrl. `scripts/chaos_artifact_split.py`.

C_ctrl = 0.0000 at every t_r (solver reproduces GT exactly → setup validated).
| op | t_r | A total | B fixable | C chaos |
|---|:---:|:---:|:---:|:---:|
| op100 | 16 | 0.678 | 0.432 | 0.562 |
| op100 | 32 | 0.678 | 0.337 | 0.627 |
| op100 | 48 | 0.678 | 0.182 | 0.659 |
| op500 | 16 | 0.473 | 0.287 | 0.345 |
| op500 | 32 | 0.473 | 0.211 | 0.395 |
| op500 | 48 | 0.473 | 0.112 | 0.443 |

C_ctrl=0 ⇒ frame 0 fully determines the future; ALL late error = operator mid-field error × chaos. C (chaos) dominates and grows with t_r; B (fixable artifact) shrinks. Even perfect late dynamics (B→0) from the operator's own mid-field lands at C (~0.56 op100). ⇒ **late wall is mostly chaos-amplified mid-field error, not fixable physics.** A/B/C are three pairwise distances (triangle-related, not additive); min-C is at the grid edge t_r=16. Plots: outputs/chaos_split/chaos_split_op{100,500}.png.

---

# IC + data + consistency — does data anchoring stabilise the distillation? (2026-06-12)
TTA opX→Re500, pool[0:160] stride-1, held-out[200:300] n=100, 5000 steps, lr 3e-4. Loss = 1·distill + 5·ic + 1·data (data = full-traj GT rel-L2; distill teacher = F(GT@32); m=32). `scripts/oracle_distill_probe.py --data_weight 1.0`. Extends the dw=0 oracle-distill probe.

| op | late start | late final | early final | aggr final |
|---|:---:|:---:|:---:|:---:|
| op100 | 0.678 | 0.522 | 0.156 | 0.376 |
| op500 | 0.473 | 0.488 | 0.177 | 0.372 |

Data term removes the dw=0 reversion (op100 holds 0.522, no bounce-back to 0.678); lands stable at the supervised frame-0 ceiling (~0.48–0.52 ≈ reach gate / op500-level). Consistency adds nothing below it; early fully protected.

---

# Temporal-payoff gate — does low-passing the model trajectory in time lower late error? (2026-06-12)
Time-axis mirror of E_bandK (ceiling for any ∂ₜ-term regularizer). Frozen opX, held-out Re500[200:300] n=100, k≤7, late=last 8 frames. model_late=late-k7(trunc_t(F(ic),K_t),GT); gt_late=late-k7(trunc_t(GT,K_t),GT) (truncation floor). FNO time n_modes=8. `scripts/temporal_payoff_gate.py`.

| op | full | K_t=4 | K_t=8 | K_t=16 | K_t=24 | gt_late @8 / @16 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| op100 | 0.666 | 0.704 | 0.681 | 0.671 | 0.668 | 0.136 / 0.064 |
| op500 | 0.458 | 0.523 | 0.487 | 0.468 | 0.463 | 0.136 / 0.064 |

model_late ≥ full at every K_t, both ops (Δ +0.002…+0.071, monotone, min at largest K_t→full). gt_late floor grows as K_t falls. Late error sits in the temporally-smooth band; time-truncation removes real content, not recoverable model error.

---

# Self-consistency truth gate — does late self-inconsistency rank-track late GT error? (2026-06-12)
Frozen opX, held-out Re500[200:300] n=100, k≤7, late=last 8 frames. self_incons(m)=relL2_{k≤7,late}(F(u[m]),u[m:]); gt_late=relL2_{k≤7,late}(u,GT); u=F(ic). `scripts/self_consistency_gate.py`.

| op | m | Spearman(late) | p | Pearson(late) | Spearman(aggr) |
|---|:---:|:---:|:---:|:---:|:---:|
| op100 | 16 | +0.399 | 4.0e-5 | +0.430 | +0.500 |
| op100 | 32 | +0.460 | 1.4e-6 | +0.512 | +0.550 |
| op500 | 16 | +0.637 | 1.1e-12 | +0.649 | +0.576 |
| op500 | 32 | +0.650 | 2.4e-13 | +0.670 | +0.593 |

Positive, significant every cell; Pearson≈Spearman (monotone). m32>m16 both ops; op500 (.65) > op100 (.46). Self-inconsistency tracks GT late error, weaker on op100.

---

# Oracle-distillation probe — can a perfect restart teacher move the one-shot tail? (2026-06-12)
Setup: TTA opX→Re500, pool[0:160] stride-1, held-out[200:300] n=100, 5000 steps, lr 3e-4, Adam. Loss = 1.0·distill + 5.0·ic (teacher uses GT frame@32 — throwaway upper bound). distill = relL2(F(ic)[...,32:], stopgrad F(GT@32)[...,:33]); ic = relL2(F(ic)[...,0], IC). `scripts/oracle_distill_probe.py`. Metric: held-out k≤7 early/late vs GT.

| op | late start | late min (@step) | late final | early start→final |
|---|:---:|:---:|:---:|:---:|
| op100 | 0.678 | 0.542 (1000) | 0.684 | 0.271→0.315 |
| op500 | 0.473 | 0.473 (0) | 0.566 | 0.248→0.505 |

---

# Self-inconsistency + physics diagnostic (frozen, forward-only) (2026-06-12)
Setup: frozen opX, held-out Re500[200:300] n=100, full field. u=F(ic); v_m=F(u[m]). `scripts/self_consistency_diag.py`. self-inconsistency = relL2(v_m[j], u[m+j]); one-shot NS residual relL2(Du,f) + terms (wt=∂ₜ, adv, diff); residual of disagreement (v_m−u[m:]).

self-inconsistency aggr by restart m:
| op | m4 | m8 | m16 | m32 | m48 |
|---|:---:|:---:|:---:|:---:|:---:|
| op100 | 0.188 | 0.204 | 0.230 | 0.248 | 0.202 |
| op500 | 0.108 | 0.132 | 0.175 | 0.190 | 0.160 |

one-shot residual (rel forcing, per-frame range): op100 total 3.6→2.0 (adv 3.5→2.5, wt 2.7→1.4, diff 0.17→0.05); op500 total 1.0–1.2 (adv 2.8→2.2, wt 2.5→1.9, diff ~0.03–0.05). Plots: outputs/selfcons/selfcons_op{100,500}.png.

---

# Gate A — chaining stride × source sweep (2026-06-12)
Frozen-operator chaining, held-out Re500 [200:300] n=100, k≤7, pooled. `scripts/chain_gate.py`. one-shot: op100 early 0.271 / late 0.678 / aggr 0.531; op500 early 0.248 / late 0.473 / aggr 0.370. win% / Δ̄ = paired per-instance late vs one-shot.

**op100**
| source | stride | early | late | aggr | late win% | Δ̄ late |
|---|---:|:---:|:---:|:---:|:---:|:---:|
| oracle | 4  | 0.236 | 0.243 | 0.238 | 100% | −0.424 |
| oracle | 8  | 0.260 | 0.272 | 0.263 | 100% | −0.396 |
| oracle | 16 | 0.271 | 0.361 | 0.311 | 100% | −0.310 |
| oracle | 32 | 0.271 | 0.522 | 0.405 | 100% | −0.155 |
| model  | 4  | 0.343 | 0.912 | 0.728 | 0%   | +0.243 |
| model  | 8  | 0.282 | 0.841 | 0.655 | 1%   | +0.170 |
| model  | 16 | 0.271 | 0.771 | 0.591 | 3%   | +0.097 |
| model  | 32 | 0.271 | 0.711 | 0.549 | 11%  | +0.035 |

**op500**
| source | stride | early | late | aggr | late win% | Δ̄ late |
|---|---:|:---:|:---:|:---:|:---:|:---:|
| oracle | 4  | 0.234 | 0.237 | 0.235 | 100% | −0.227 |
| oracle | 8  | 0.243 | 0.249 | 0.244 | 100% | −0.217 |
| oracle | 16 | 0.248 | 0.283 | 0.262 | 100% | −0.185 |
| oracle | 32 | 0.248 | 0.353 | 0.299 | 100% | −0.118 |
| model  | 4  | 0.291 | 0.874 | 0.649 | 0%   | +0.397 |
| model  | 8  | 0.255 | 0.625 | 0.478 | 3%   | +0.156 |
| model  | 16 | 0.248 | 0.488 | 0.393 | 33%  | +0.021 |
| model  | 32 | 0.248 | 0.449 | 0.366 | 71%  | −0.020 |

**Reads:**
- **oracle ↓↓** — beats one-shot on 100% of instances, both ops, down to ~0.24. lever is real.
- **model ↑** — worse almost everywhere. own restart field injects error → compounds.
- **stride flips sign:** oracle shorter → better (more true resets); model shorter → worse (more injection). model knob = *fewer* restarts.
- **op500 > op100:** better operator = better own fields = less injection. op500 crosses to net-+ only @ s32 (0.449<0.473, 71% win); op100 never crosses.
- **gap (oracle−model) ≈ 0.19 op100 / ≈0.10 op500** = the cost of bad restart fields = the prize.
- ⇒ lever real, own fields too weak ⇒ next = self-consistency **training** (make own restart fields oracle-like).

---

# Gate A — frozen-operator oracle chaining (2026-06-11)
GT-restart every 16 frames (stride=16), held-out Re500 [200:300] n=100, k≤7. `scripts/chain_gate.py`. one-shot self-check reproduces refs (op100 late 0.678, op500 late 0.473).

| op | late one-shot | late chained | aggr one-shot | aggr chained | Δ̄ late (paired) | improved | wilcoxon p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| op100 | 0.678 | **0.361** | 0.531 | 0.311 | −0.310 | 100% | 3.9e-18 |
| op500 | 0.473 | **0.283** | 0.370 | 0.262 | −0.185 | 100% | 3.9e-18 |

early unchanged by construction (op100 0.271, op500 0.248). **Oracle short-horizon chaining breaks the late wall on every instance ⇒ late wall is not a short-horizon representability limit.**

---

# Reachability gate — is a good late field reachable from op100? (2026-06-11)
Supervised adapt op100→Re500[0:160], eval held-out [200:300] late k≤7. LR sweep, 6000 steps, no early-stop. `reach_gate{,_pino}.yaml`. Best cell:

| objective | late_min (oracle, sub25) | late_final (no-stop, full100) | aggr_final |
|---|:---:|:---:|:---:|
| data-only (d1) | 0.498 | 0.536 | 0.385 |
| data+pde 5:1 k≤7 (d5/p1) | **0.481** | **0.497** | 0.374 |

refs: wall ~0.55 · op300 0.527 · op500 0.473 · 8-mode floor 0.19. late_min = best-over-curve (label-selected, sub25); late_final = full100 last step. Pool autocorrelated (N_eff≈35/160).

**Findings (1 seed):** supervised adapt-from-op100 reaches late ~0.48–0.50 ≈ op500, far above the 0.19 representable floor → shared-operator generalization saturates near op500-level (this budget; sub-0.48 untested, not a proven floor). data+pde stays below the wall even without oracle stopping (0.497); data-only overfits back to ~0.54. Physics gain (Δlate ~0.02–0.04) single-seed, not established.

---

# Temporal-mode budget — is 8 time-modes the late wall? (2026-06-11)
GT Re500 held-out [200:300], n=100, k≤7 band, pad=5. `scripts/temporal_budget.py`.

**A — GT projected onto 8 time-modes**
| E_t<8 | dE_t<8 | field err early | field err late | resid floor | resid_t@8 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.992 | 0.183 | 0.107 | **0.189** | 0.031 | 11.0 |

**B — op300 vs op500 (both 8 time-modes), late k≤7**
| op | late k≤7 | err frac>8 |
|:---:|:---:|:---:|
| 300 | 0.513 | 0.011 |
| 500 | 0.458 | 0.011 |
| *bracket ref* | *0.527 / 0.473* | — |

**C — resid_t@8 source** (Δterm = trunc−full, k≤7, rel forcing; early≈late)
| Δwt (∂ₜ) | Δadv | Δdiff | floor | total |
|:---:|:---:|:---:|:---:|:---:|
| **11.07** | 0.87 | 0.005 | 0.035 | 11.07 |

**Conclusion:** 8 time-modes hold the late field at 0.19 ≪ wall 0.46–0.53 (op err frac>8≈0.01) ⇒ capacity NOT the wall — **temporal-mode sweep DEAD** (mirrors nm16), **not super-res**. resid_t@8=11 = ~all Δ∂ₜ: a metric artifact of *truncating GT* (not the model), diagnostic-only, not an optimization target. Real problem = **residual objective is non-identifying** (residual↓, late err flat; best op500 ≠ lowest-residual). ⇒ next = **rollout/self-consistency**. Open: held-out reachability (op500 0.47 ≫ 0.19 floor), 256² target.

---

# Spectral budget per Re (128² data, 2026-06-10)

**(1) 8 modes carries the field but not the physics.** `n_modes=8` keeps 93–96% energy at every Re, but the residual (∝ derivatives, ~k⁴) of a band-8 field is wrecked even at Re100. Mode budget set by the TTA *target*, not source.
**(2) 128² has a physics ceiling ≈ Re300.** Full-field GT residual (`floor`) blows up with Re — non-dealiased advection. The Re500 ν-oracle TTA objective is ~½ aliasing, not physics.

| Re | E<8 | Z<8 | resid@8 | floor(128²) | clean? |
|---:|:---:|:---:|:---:|:---:|---|
| 100 | 95.7 | 66.9 | 0.84 | 0.013 | physics-clean |
| 300 | 94.3 | 46.1 | 1.47 | 0.125 | borderline |
| 500 | 93.2 | 22.8 | 2.02 | **0.666** | residual ½ aliasing |
| 1000| 95.6 | 38.7* | 2.15 | **1.460** | GT fails own PDE |

E<M,Z<M = % energy/enstrophy below radial mode M. resid@M / floor = rel-L2(NS-residual, forcing) of band-M / full GT field (1.0 = forcing-sized error). Field err @8 only 0.14–0.22 → field looks fine, physics doesn't. *Re1000 `_indep` lineage, tail Nyquist-capped. resid@M = strict band-M projection (FNO leaks some high-k via GELU) → pessimistic bound. **op100→Re500 physics-TTA sits in worst quadrant: high-k objective + 8 modes + 128².** Plots: `outputs/figs/{energy_spectrum_allRe,truncation_budget}.png`.

---

# TTA Results — range-N × LR matrix (op100 → Re500, physics-only)

**Status:** banked 2026-06-10. 29/33 cells (seed-0 grid complete bar 1e-3 N100; +3 of 6 seed reps).
Run `E_rangeN` (12k steps, stride-1 adapt, oracle ν=1/500). Source: `msc/tta/runs/E_rangeN/2026*/summary.json`.

## Setup
- **Operator:** op100 (FNO `n_modes=[8,8,8]`, pretrained Re100, KF 128²), frozen → adapted.
- **Target:** Re500 (OOD), 128². Adapt pool = Re500`[0:N]` **stride-1** (genuinely decorrelated range, not dense copies); buffer `[N:200]`; held-out = Re500`[200:300]` (disjoint, n=100, FIXED).
- **Objective (label-free):** `L = pde_residual(ν=1/500, oracle) + 5·ic_anchor`, `data_weight=0`. 12k step cap.
- **Metric:** held-out band-resolved **k≤7 rel-L2 vs GT**, time-split **early** (first ~8 frames) / **late** (last ~8) / **aggr** (all). Lower = better. k>7 = under-resolved-GT, excluded.

## Reference points (zero-shot, same split/metric)
| operator | early | late | aggr |
|---|:---:|:---:|:---:|
| op100 (source, must-beat) | 0.271 | 0.678 | 0.531 |
| op300 (best free ID op)   | 0.234 | 0.527 | 0.404 |
| op500 (supervised ceiling)| 0.248 | 0.473 | 0.370 |
| per-instance floor (transductive) | — | — | ~0.037 |

## Held-out k≤7 — final per cell (seed 0; lower = better)

**EARLY** — solved below op500's 0.248 from N≈40 up:
| N | lr=1e-4 | lr=3e-4 | lr=1e-3 |
|---:|:---:|:---:|:---:|
| 20  | 0.241 | 0.299 | 0.404 |
| 40  | 0.210 | 0.234 | 0.309 |
| 60  | 0.205 | 0.226 | 0.304 |
| 80  | 0.199 | 0.226 | 0.273 |
| 100 | 0.192 | 0.213 |  —    |
| 120 | 0.188 | 0.211 | 0.238 |
| 140 | 0.182 | 0.195 | 0.215 |
| 160 | 0.179 | 0.182 | 0.216 |
| 180 | **0.176** | **0.175** | 0.216 |

**LATE** — the wall; plateaus ~0.59 for N≥120, never reaches op300's 0.527:
| N | lr=1e-4 | lr=3e-4 | lr=1e-3 |
|---:|:---:|:---:|:---:|
| 20  | 0.650 | 0.650 | 0.685 |
| 40  | 0.622 | 0.615 | 0.656 |
| 60  | 0.612 | 0.610 | 0.646 |
| 80  | 0.609 | 0.613 | 0.648 |
| 100 | 0.608 | 0.601 |  —    |
| 120 | 0.596 | 0.592 | 0.619 |
| 140 | 0.596 | 0.587 | 0.608 |
| 160 | 0.592 | **0.581** | 0.597 |
| 180 | 0.595 | 0.588 | 0.607 |

**AGGR** — best ≈ 0.424 (3e-4, N160), short of op300's 0.404:
| N | lr=1e-4 | lr=3e-4 | lr=1e-3 |
|---:|:---:|:---:|:---:|
| 20  | 0.493 | 0.513 | 0.584 |
| 40  | 0.464 | 0.472 | 0.530 |
| 60  | 0.456 | 0.463 | 0.520 |
| 80  | 0.450 | 0.462 | 0.511 |
| 100 | 0.443 | 0.447 |  —    |
| 120 | 0.436 | 0.440 | 0.473 |
| 140 | 0.435 | 0.432 | 0.458 |
| 160 | 0.431 | **0.424** | 0.446 |
| 180 | 0.431 | 0.425 | 0.453 |

**Pool val_l2** (1e-4): rises with N (0.227→0.404) — shared weights never fit the pool; the inductive limit is in the pool itself.

## Seed band — N=180, lr=1e-4 (seeds 0/1/2)
| seed | early | late | aggr |
|---:|:---:|:---:|:---:|
| 0 | 0.176 | 0.595 | 0.431 |
| 1 | 0.175 | 0.597 | 0.432 |
| 2 | 0.175 | 0.594 | 0.430 |

σ(late) ≈ 0.001 → the late plateau is **real saturation, not seed noise**.

## Conclusion — early solved, late hard-walled
- **EARLY is solved.** Best 0.175 — below op500's supervised 0.248 (best-in-class), monotone in N. In-band early physics is learnable from the instantaneous residual.
- **LATE is a wall at ~0.58–0.59**, plateaued for N≥120; best anywhere **0.581** (3e-4, N160) — still **+0.054 above op300's 0.527** and +0.108 above op500's 0.473.
- **Wall is lever-invariant.** Invariant across N (5× more decorrelated data → only ~0.06, then flat), LR (1e-4 ≈ 3e-4; 1e-3 worse = forgetting), seed (σ≈0.001), and capacity (prior nm16: early 0.238→0.145, late 0.531→0.547). Four independent knobs, same wall.
- **LR sets the path, not the wall.** 1e-3 is much worse at low N (forgetting) but converges to the same ~0.60 plateau — the limit is the objective, not optimization.

⇒ **Late k≤7 is a model/objective limit of the *instantaneous* PDE residual, not data, LR, seed, or capacity.** Points at the unrun fork: a band-limited (k≤7) and/or temporal/multi-step residual objective.

## Caveats
- ν = oracle 1/500 only.
- op300/op500's late edge is measured vs the same 128² GT — a 256² reference still gates whether the late target is physically real (carry on any "approached the target" claim).
- 4 gap-fill cells (1e-3 N100; seeds at N20-s2, N100-s1/s2) still running 2026-06-10; will not move the story. All landed cells sit exactly on trend.

---

## Exp #1 — band-limited (k≤7) residual objective (E_bandK, banked 2026-06-11)
Setup: `KFLoss(pde_band_kmax=7)` (penalise only k≤7 Chebyshev band of `Du`; `cheb_lowpass` in `src/pde/ns.py`). Cell = E_rangeN plateau (lr=3e-4, N=160, 12k steps, oracle ν=1/500). Control = same cell, `pde_band_kmax=null`, same seed. Metric = held-out k≤7 band_eval (unchanged). Control late 0.578 reproduces banked E_rangeN wall (0.581).

| held-out (n=100, final) | control (full) | kmax=7 | Δ |
|---|:---:|:---:|:---:|
| early | 0.187 | 0.144 | −0.043 |
| late  | 0.578 | 0.556 | −0.022 |
| aggr (err_k7) | 0.423 | 0.400 | −0.023 |
| resu_f7 (û k≤7 residual) | 0.693 | 0.469 | −0.224 |

resgt_f7 (GT k≤7 residual) = 0.0025. seed σ(late) ≈ 0.001. Plots: `outputs/figs/bandk{7_treatment,_control}_curve.png`.

---

## Prior matrix — E_lrN (small-N, 3000-step, superseded)
N∈{1..20}×LR{2.5e-3,1e-3,3e-4,1e-4}, dense stride-5 pool `[0:100]`. Best held-out aggr ≈ **0.486** (N=18–20, lr≤3e-4); high-LR (2.5e-3) forgets (>0.531 baseline). Superseded by E_rangeN, which extends the decorrelated range to N=180 and exposes the early/late split. Plots: `runs/E_lrN/slope_{full,k7_aggr,k7_early,k7_late}.png`.
