# Sim2Sic Journal

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
4. **Interpretation** — raw interpretation of what the numbers mean, No outlook, no next steps, no mechanism speculation, pivots, or broader cross-experiment synthesis.
5. Writing style: each entry should be the extreme minimal. This document is only the gist of the findings within a few lines. Every interesting result will land here as a few line summary, the full outcomes will be stored elsewehere this is just the journey documenting what we did and how - so we can reconstruct it quickly or find where we stored it. ---

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
- `msc/configs/paths.yaml` — checkpoint and data paths.
- `proof_sketch.md` — derivation, assumptions and predictions behind the 2026-07-29 entry.

---

# 2026-08-01 (cont. 2)

blur-horizon (`gamma`) against corr-horizon (`rho`), per shell, all three rows, defaults
(`0e1423c`). First `blur` run; both tables share thresholds so the pair is comparable per shell.
Anchor — row A rho reproduces 2026-07-31 (cont. 2) exactly. CIs and cens dropped for space, in
`msc/tta/results/`; row A rho cens 29/30 at k1, so those rows read >64.

```
shell    rho_A  gam_A   rho_C  gam_C   rho_B  gam_B     (first frame < 0.9)
k1        64.2   57.6    65.0   59.0    65.0    1.0
k2        61.2   26.0    52.8   19.3    31.9    4.4
k3        51.1   31.1    36.2   12.5    23.1    8.2
k4        57.1   36.8    22.2    3.3    18.1   18.5
k5        33.2   24.8     3.8    0.0     8.8    0.0
k6        16.9    9.8     0.0    0.0     4.5    0.0
k7        10.9    6.3     0.1    0.0     3.7    0.0
k8        10.0    5.8     0.2    0.0     3.5    0.0
k12        7.0    2.6     0.9    0.0     2.8    0.0
k16        4.6    5.5     1.5    0.0     2.3    0.0
k20        2.5   10.6     1.5    0.0     2.1    0.0
k24        0.3   22.0     1.3    0.0     1.9    0.0
k32        0.0   46.0     1.1    0.0     1.5    0.0
k48        0.0   62.7     0.4    0.0     0.9    0.0
```

- Amplitude fails before phase in every represented band, all three rows. (b) is not a companion
  to (a); it is the earlier and tighter constraint.
- Gap widens with shift, k1-4: A 47/62 = 0.75, C 33/59 = 0.56, B 2.9/38.8 = **0.07**. Cross-Re
  holds position 39 frames while losing amplitude at frame 3.
- B k1 is a sag, not a collapse: gamma<0.9 at 1.0 but <0.8 only at 62.8 (cens 27/30). A ~10%
  deficit appears at the largest scale immediately and then sits there.
- A's gamma-horizon is U-shaped, min k12 (2.6), rising to no zero tail at all: gamma > 1 above
  ~k16, so the model is under-energetic mid-k and over-energetic high-k. A gamma gate must be
  two-sided, and the sign flips inside the adaptation band.
- A's weak shell differs by axis: amplitude k2 (26.0), phase k3 (51.1).
- C is monotone with no excess anywhere; B has excess at k64 (16.3). C k5+ and B k5+ have no
  amplitude window at all (gamma<0.9 at frame 0).

---

# 2026-08-01 (cont.)

`w1` rebuilt paired: `w1_curve` scores each prediction against its OWN gt, then means over the 30
(`6fea414`). The pooled form cannot distinguish a correct pairing from a permutation — verified,
both score 0.0 on trajectories with different distributions. W1 is convex, so pooled <= these
means: NOT comparable to the entry below. Last row is `w1_lag_drift`, the same trajectory 32
frames away.

```
                    t0      t1      t2      t4      t8     t12     t16     t24     t48     t64    aggr
A  W1           0.0350  0.0405  0.0408  0.0451  0.0512  0.0571  0.0630  0.0752  0.1042  0.1131  0.0811
A  width-corr   0.0151  0.0168  0.0176  0.0197  0.0241  0.0279  0.0309  0.0348  0.0448  0.0541  0.0307
A  drift 32f    0.0396  0.0397  0.0399  0.0404  0.0419  0.0433  0.0442  0.0447  0.0450  0.0443     nan

C  W1           0.1091  0.1143  0.1144  0.1175  0.1205  0.1220  0.1234  0.1261  0.1325  0.1316  0.1256
C  width-corr   0.0639  0.0663  0.0664  0.0672  0.0680  0.0681  0.0683  0.0693  0.0760  0.0785  0.0644

B  W1           0.0874  0.0995  0.1040  0.1152  0.1252  0.1295  0.1340  0.1435  0.1695  0.1852  0.1501
B  width-corr   0.0252  0.0283  0.0301  0.0342  0.0404  0.0445  0.0475  0.0508  0.0568  0.0746  0.0431

B/C drift 32f   0.0328  0.0328  0.0328  0.0326  0.0320  0.0312  0.0302  0.0293  0.0302  0.0296     nan
```

- Matching the width removes 62/49/71% (A/C/B), down from the pooled 80/74/83. Still the dominant
  term, no longer overwhelming.
- C's residual is 2x the flow's own 32-frame drift at every frame; B crosses from 0.77x to 2.5x;
  A stays under until t48. The Re500-native model gets the value distribution's SHAPE wrong from
  frame 0 — the first second axis found, and pooling was hiding it.
- The drift row is NOT a floor, and the entry below is wrong to call it one. Paired W1 compares
  two fully observed fields, so a perfect prediction scores exactly 0 and there is no sampling
  noise to clear. It is a yardstick only: "2x the flow's own drift".
- It cannot be calibrated to independence. Swept on GT it climbs monotonically (Re100 t8: 0.0037
  at lag 2 -> 0.0569 at lag 48) and never plateaus inside 65 frames, so lag=32 is a stated
  convention (half the window). Two DIFFERENT chains at one frame read 0.21 — 4x the lag-48 value,
  the realisation spread the pairing exists to remove. A within-frame spatial split reads 0.52
  and is unusable.

---

# 2026-08-01

SUPERSEDED by (cont.) above: pooled, and its floor is not a null. Kept as the route taken.

New `w1` row: `w1_width_corrected` = W1 after rescaling pred to GT's width, so what is left is
the distribution mismatch gamma cannot express (`e308297`; replaced a W1/W1(gamma*gt,gt) ratio
whose |1-gamma| denominator inflated a fixed defect 1.08 -> 4.79 -> nan as gamma -> 1).
All three rows, defaults. Anchor — W1 aggr 0.0833/0.1222/0.1475 reproduces A/C/B as banked.

```
                    t0      t1      t2      t4      t8     t12     t16     t24     t48     t64    aggr
A  W1           0.0363  0.0425  0.0421  0.0465  0.0522  0.0575  0.0631  0.0756  0.1047  0.1122  0.0833
A  width-corr   0.0075  0.0074  0.0079  0.0087  0.0094  0.0100  0.0109  0.0144  0.0241  0.0261  0.0164
A  floor        0.0235  0.0234  0.0233  0.0232  0.0230  0.0234  0.0236  0.0231  0.0202  0.0221  0.0190

C  W1           0.1060  0.1111  0.1109  0.1137  0.1163  0.1174  0.1184  0.1209  0.1271  0.1251  0.1222
C  width-corr   0.0442  0.0453  0.0446  0.0440  0.0422  0.0403  0.0379  0.0338  0.0331  0.0337  0.0318

B  W1           0.0862  0.0980  0.1024  0.1131  0.1225  0.1265  0.1307  0.1401  0.1662  0.1809  0.1475
B  width-corr   0.0151  0.0187  0.0206  0.0235  0.0264  0.0281  0.0294  0.0293  0.0322  0.0447  0.0245

B/C floor       0.0609  0.0608  0.0606  0.0604  0.0598  0.0591  0.0583  0.0574  0.0587  0.0578  0.0585
```

- Matching the width removes 80/74/83% of W1 (A/C/B). W1 is the amplitude ratio in every regime
  tested, across a 5x Re change and across ID vs OOD. Direct measurement, no Gaussian assumption,
  so it supersedes the 2026-07-31 (cont.) corr-0.99981 route to the same conclusion.
- The residual clears its floor only in A, only at t48/t64 (1.19x/1.18x) — 5x past the t10 blur
  cutoff. Never in C or B, no column.
- Negative result on the OOD rows is not clean: the floor is unpaired 15-vs-15 while the signal is
  paired 30-vs-30, and reads 0.0585 on Re500 vs 0.0190 on Re100 (realisation spread, not noise —
  same defect as the covRMSE TODO). Too loose to support "no shape error".
- Reportable only as a negative, with that caveat: the field's preferred distributional metric is
  measuring variance here. It folds location, skew and tail weight together, so it never isolates
  intermittency even where it does resolve.

---

# 2026-07-31 (cont. 4)

Conclusion on the Re100 blur horizon, over the four entries below (`75prctl5`, own data).

- The model stays useful for about 10 frames. Below k=5 it never loses the flow inside the
  65-frame window; above k=8 it is gone within 5. The shells k5-7 set the limit, so ≈t10 is
  the honest cutoff for the band the model can represent (k1-7).

- Amplitude and distribution measure the same thing. W1 and covRMSE both turn out to be gamma
  rewritten (corr 0.999 and 0.997), and L2 forces gamma to follow rho. Of the field metrics
  only rho — how well the prediction sits in the right place — carries information on its own.

- Two unrelated metrics name the same frame. W1 crosses its noise floor at t8, rho drops below
  0.9 at ≈t10. One works on spectra with paired trajectories, the other on value histograms
  with unpaired ones; nothing is shared, so the agreement is evidence for the cutoff.

- The physics residual points the other way in both directions: better with lead time and
  better at high k, exactly where L2 gets worse. Its minimum is a smooth laminar field and the
  prediction smooths as it degrades, so it rewards the blur we are trying to detect. Usable
  only alongside another target, never alone.

---

# 2026-07-31 (cont. 3)

Physics residual on the L2 bands and frame ladder, for direct contrast. `75prctl5` own data.
Anchor — k8-16/k17-32/k33-64 aggr (0.4633/0.1423/0.05216, ratio 23.25/6.919/9.789) reproduce
2026-07-30 exactly. t0/t64 are "-": the centred stencil has no residual at the endpoints.

```
res_rms/|f|     t1      t2      t4      t8     t12     t16     t24     t48    aggr
k1-4        0.7600  0.6931  0.6531  0.5684  0.5415  0.5522  0.5568  0.5956  0.5927
k5-7        1.0040  0.8850  0.8556  0.7990  0.7210  0.6540  0.5417  0.5226  0.6149
k8-16       0.8047  0.7435  0.7149  0.6448  0.5633  0.5007  0.4068  0.3599  0.4633
k17-32      0.2553  0.2417  0.2253  0.2011  0.1826  0.1654  0.1282  0.1009  0.1423
k33-64      0.1044  0.0976  0.0886  0.0823  0.0734  0.0639  0.0453  0.0297  0.0522

res_pred/res_gt (1 = below detection)
k1-4         447.7   406.8   380.3   323.6   298.0   298.2   293.9   297.6   301.2
k5-7         244.4   212.5   200.0   177.9   155.2   136.5   96.59   74.21   100.6
k8-16        86.54   79.06   73.87   61.37   47.60   36.79   22.47   14.04   23.25
k17-32       20.11   18.88   17.41   15.38   13.47   10.92   6.189   4.447   6.919
k33-64       22.15   19.85   16.76   14.51   12.71   10.70   7.056   5.188   9.789
```

Interpretation.

- The residual improves with lead time while every L2-family metric degrades. k5-7 1.004 ->
  0.523, k8-16 0.805 -> 0.360, k33-64 0.104 -> 0.030, monotone; only k1-4 is flat (dips 0.54
  at t12). At t48 the residual is at its best in each band while rho in k5-7 is 0.71 and
  rel_l2 0.70. The two metrics disagree about which frames are good.

- It also falls with k (0.59/0.61/0.46/0.14/0.05) where rel_l2 rises: physics violation is
  concentrated at low k, exactly where the L2 prediction is strongest.

- Consistent with the laminar attractor (2026-07-30): the residual's global minimum is a
  smooth field, and the prediction smooths with lead time (k5-7 gamma 0.90 -> 0.71). Blurring
  lowers the residual without lowering the L2 error.

- k5-7 t1 = 1.004 is the only cell reaching the w=0 null, in the band that sets the L2 cutoff.
  Below t8 it is the worst band on both metrics — the one region where they agree.

- Detectability is not the limit: the ratio is 4.4-448 everywhere, no cell at the stencil floor.

- Therefore the physics residual can only be a CO-TARGET at Re100, never a stand-alone TTA
  objective: minimising it rewards the smoothing that L2 already over-produces. Any use needs
  a gamma gate.

---

# 2026-07-31 (cont. 2)

corr-horizon, per shell and on the banked L2 bands. `75prctl5` own data, defaults otherwise.
Anchor — k5/k6/k7 corr<0.9 (33.2/16.9/10.9) reproduce 2026-07-24 exactly.
cens = samples still above the threshold at t64, so their crossing was never observed and
enters the mean as 65: a high cens makes the row a lower bound, not a measurement.

```
band       corr<0.9                      corr<0.8
k1-4       62.5 [59.3,64.9] cens 26/30   63.9 [61.7,65.0] cens 29/30
k5-7       22.4 [16.7,28.2] cens  1/30   43.0 [35.7,50.1] cens 11/30
k8-16       8.3 [ 6.7,10.8] cens  0/30   15.1 [11.9,19.5] cens  1/30
k17-32      2.7 [ 2.0, 3.4] cens  0/30    6.3 [ 5.2, 7.7] cens  0/30
k33-64      0.0 [ 0.0, 0.0] cens  0/30    0.0 [ 0.0, 0.0] cens  0/30

shell      corr<0.9                      corr<0.8
k0          0.5 [ 0.3, 0.8] cens  0/30    0.5 [ 0.3, 0.8] cens  0/30
k1         64.2 [62.5,65.0] cens 29/30   65.0 [65.0,65.0] cens 30/30
k2         61.2 [58.8,63.3] cens 17/30   64.9 [64.6,65.0] cens 29/30
k3         51.1 [46.1,55.5] cens  8/30   61.3 [57.7,64.1] cens 22/30
k4         57.1 [52.7,61.2] cens 16/30   63.0 [59.9,65.0] cens 27/30
k5         33.2 [26.1,39.8] cens  2/30   52.3 [46.4,57.7] cens 14/30
k6         16.9 [12.7,21.2] cens  0/30   36.7 [29.1,44.2] cens  6/30
k7         10.9 [ 8.2,14.6] cens  0/30   22.8 [16.9,29.0] cens  2/30
k8         10.0 [ 7.7,13.4] cens  0/30   23.2 [17.1,29.9] cens  3/30
k12         7.0 [ 5.6, 8.7] cens  0/30   13.4 [10.3,17.4] cens  0/30
k16         4.6 [ 3.9, 5.4] cens  0/30    9.5 [ 7.0,13.2] cens  0/30
k20         2.5 [ 1.9, 3.3] cens  0/30    5.9 [ 4.8, 7.2] cens  0/30
k24         0.3 [ 0.1, 0.6] cens  0/30    2.7 [ 1.9, 3.4] cens  0/30
k26+        0.0                           0.0 from k35
```

New here, against the pooled-rho tables banked above:

- k3 is the weak shell (51.1), not k5 — k1/k2/k4 are 64.2/61.2/57.1. The forced shell k4 is
  not the most coherent one. Invisible to any banded table.
- k1-4 vs k5-7 CIs do not overlap (62.5 [59.3,64.9] vs 22.4 [16.7,28.2]): the band split,
  chosen on energy-weighting grounds, is statistically real.
- k5 is the break, 57.1 -> 33.2 -> 16.9 -> 10.9 across k4-k7.
- Per-sample horizons run ~2x the pooled-rho crossings (k5-7 22.4 vs ≈t10 from `decomp`).
  Cauchy-Schwarz: a ratio-of-sums is <= the weighted mean of per-sample ratios, so pooled rho
  crosses any threshold first, always. Both correct, never interchangeable in a quote.
- cens 26/30 at k1-4 and 29/30 at k1: those rows are the window length in disguise, read >64.

---

# 2026-07-31 (cont.)

W1 resolved over the same frame ladder (`df57f55`, `--late` retired). `75prctl5` own data.
Anchor — aggr 0.0834 / floor 0.0435 reproduces the banked 2026-07-30 all-frames pair.

```
                    t0      t1      t2      t4      t8     t12     t16     t24     t48     t64    aggr
W1              0.0363  0.0425  0.0421  0.0465  0.0523  0.0575  0.0631  0.0757  0.1047  0.1122  0.0834
GT-GT floor     0.0519  0.0518  0.0517  0.0515  0.0510  0.0504  0.0494  0.0470  0.0386  0.0357  0.0435
```

Interpretation.

- W1 tracks gamma, carrying no shape signal. W1 = 0.85 x `0.798|1-gamma_full|`, the closed
  form for a Gaussian scale change, at corr 0.99981 over all 65 frames. Same ordering across
  rows: A/C/B W1 0.0834/0.1222/0.1476 vs gamma 0.877/0.833/0.791. The predicted value
  distribution is GT's, narrowed; the flat 0.85 is the only non-Gaussian residue. Retires the
  premise that W1 reads intermittency/tails gamma cannot — true for a band-split field (CLT),
  false for the full field of a smoothed prediction.

- rho is then the only independent field metric: W1 is a function of gamma, L2 drives gamma to
  rho, and rel_l2 fuses both. A distribution- or variance-matching objective is one axis with
  W1, not a second.

Also: W1 crosses its floor at t8, matching the rho<0.9 crossing at k5-7 (≈t10) from paired
spectral data. Below t8 W1 sits under its floor — the floor pairs GT halves from different
trajectories, so it is not a null at lead times where the IC still dominates.

---

# 2026-07-31

New error/amp/decomp defaults (`6d686cb`): bands split the old k0-7 at the forcing scale,
tbins a single-frame ladder. `75prctl5` own data, test [270:300] n=30, T_eff=65.
Anchor — k8-16/k17-32/k33-64 aggr reproduce 2026-07-29 (cont.) exactly, all three metrics.

```
rel_l2         t0      t1      t2      t4      t8     t12     t16     t24     t48     t64    aggr
k1-4       0.0932  0.1017  0.1062  0.1195  0.1437  0.1657  0.1888  0.2377  0.3571  0.4198  0.2900
k5-7       0.1819  0.2174  0.2385  0.2979  0.4075  0.4890  0.5418  0.5957  0.7013  0.7462  0.6132
k8-16      0.2506  0.2985  0.3384  0.4397  0.6138  0.7241  0.7852  0.8330  0.8753  0.9233  0.8019
k17-32     0.4007  0.4436  0.5063  0.6494  0.8481  0.9525  0.9937  1.0073  1.0031  1.0087  0.9625
k33-64     1.5122  1.4502  1.4818  1.5134  1.5529  1.4600  1.3624  1.3136  1.0990  1.1223  1.2636

rho
k1-4       0.9965  0.9958  0.9953  0.9940  0.9909  0.9875  0.9834  0.9732  0.9366  0.9091  0.9589
k5-7       0.9871  0.9818  0.9761  0.9596  0.9166  0.8742  0.8416  0.8037  0.7129  0.6669  0.7904
k8-16      0.9785  0.9687  0.9546  0.9104  0.7944  0.6903  0.6194  0.5550  0.4895  0.4100  0.5978
k17-32     0.9184  0.9005  0.8647  0.7604  0.5473  0.3798  0.2805  0.2008  0.1737  0.1494  0.3181
k33-64     0.4346  0.4179  0.3921  0.3456  0.2538  0.1769  0.1400  0.0415  0.0299  0.0323  0.1232

gamma
k1-4       0.9560  0.9505  0.9510  0.9463  0.9414  0.9368  0.9304  0.9124  0.8682  0.8572  0.8993
k5-7       0.9010  0.8764  0.8774  0.8614  0.8378  0.8161  0.7994  0.7743  0.7140  0.7062  0.7632
k8-16      0.8363  0.8029  0.7940  0.7615  0.7063  0.6614  0.6306  0.5989  0.5650  0.5540  0.6178
k17-32     0.8545  0.8125  0.8010  0.7616  0.6847  0.6067  0.5378  0.4354  0.3644  0.3488  0.4842
k33-64     1.6494  1.5482  1.5537  1.5329  1.4686  1.2553  1.0757  0.8943  0.4867  0.5428  0.9054
```

Cutoff, first frame rho below 0.9:

| band  | k1-4 | k5-7 | k8-16 | k17-32 | k33-64 |
|:------|:----:|:----:|:-----:|:------:|:------:|
| frame | >64  | ≈10  |  ≈5   |   ≈2   |   0    |

Interpretation. The rho<0.9 crossing moves none → ≈10 → ≈5 → ≈2 → 0 across the five bands.
Inside the represented band k≤7 the split separates a lower half that never crosses in 65
frames from an upper half crossing at ≈t10; the retired k0-7 row reported 27 for that pair.

---

# 2026-07-30 (cont. 2)

Re100 vs Re500 native, controlled contrast — architecture identical (fno
n_modes=[8,8,8], 4 layers, hidden 128, weights data 5.0/pde 1.0/ic 1.0, sub_t=2);
only Re and data differ.
`msc/tta/results/compare_row-A-Re100_vs_row-C-Re500_NATIVE_d315925.txt`
External — 75prctl5 logged train_pde_loss_epoch 0.244 vs test res_rms/|f| 0.9841.

|                          | Re100  | Re500  |
|:-------------------------|:------:|:------:|
| res_rms/\|f\| k0-64      | 0.9841 | 0.9812 |
| GT floor k0-64           | 0.0298 | 0.4121 |
| k32 gamma                | 0.659  | 0.060  |
| k32 rho                  | 0.145  | 0.165  |
| k32 gamma<0.9 horizon    | 46.0   | 0.0    |

Interpretation. res_rms/|f| is invariant to 0.3% across 5x Re at fixed architecture and
loss weights, while the GT floor moves 14x. H = -0.066/-0.147/-0.085 at Re500 under the
same n_modes=8 refutes 2026-07-29 (cont.) "the sign change falls at k=8, this repo's
mode-cutoff split" — Re100's positive high-k H is near-GT-amplitude noise, not the cutoff.
Re100 shell k32 gamma-horizon 46.0 frames at rho 0.145: gamma is phase-blind, so a blur
horizon is only readable where rho is jointly high (Re100 k<=5, Re500 k<=4).

---

# 2026-07-30 (cont.)

Re500 in-dist baseline — native Re500 ceiling, first read. `nrow6j9l` own data,
`report.py --run-id nrow6j9l --reports all` (all defaults), test [270:300] n=30, T_eff=65.
`msc/tta/results/row-C_nrow6j9l_op500_test500_NATIVE_all-reports_defaults_d315925.{txt,npz}`
Anchor — rel_l2 k0-64 0.4421 vs logged val_l2 0.4350.

| band | rel_l2 | gamma | rho | res_rms/\|f\| | GT floor | ratio | share |
|---|---|---|---|---|---|---|---|
| k0-7   | 0.3955 | 0.8539 | 0.9209 | 0.9101 | 0.0267 | 34.09 | 86.0% |
| k8-16  | 0.9586 | 0.2268 | 0.2923 | 0.3334 | 0.1009 |  3.31 | 11.5% |
| k17-32 | 0.9863 | 0.0739 | 0.2205 | 0.1397 | 0.2341 |  0.60 |  2.0% |
| k33-64 | 0.9928 | 0.0615 | 0.1469 | 0.0615 | 0.3228 |  0.19 |  0.4% |
| k0-64  | 0.4421 | 0.8334 | 0.8994 | 0.9812 | 0.4121 |  2.38 |  100% |

share = fraction of total residual mean-square. Per-shell ratio <1 from k=19, <3 from
k=12. gamma<0.9 horizon 0.0 frames at every shell >=5, 3.3 at k4; corr<0.9 0.0 at k6.

Interpretation. Stencil-limited above k=19: res_gt exceeds res_pred, so the ratio column
is uninterpretable there and only res_rms/|f| carries signal. 86% of the residual
mean-square sits in k0-7, inside the 8 retained modes. Above k4 there is no horizon to
measure — gamma is already below 0.9 at the first frame.

---

# 2026-07-30

Re100 in-dist physics residual — first read in physical units (`resid_rms`, 8983ee9;
regime-scoped nu, 5c74541). `75prctl5` own data, `report.py --run-id 75prctl5 --reports
all` (all defaults), test [270:300] n=30, T_eff=65.
`msc/tta/results/row-A_75prctl5_op100_test100_NATIVE_all-reports_defaults_d315925.{txt,npz}`
Anchors — field tables byte-identical to 2026-07-29 (cont.); disjoint band squares sum to
the total exactly; NATIVE invariant res(nu_test)==res(nu_op) maxdiff 0.0.

| band | rel_l2 | gamma | rho | res_rms/\|f\| | GT floor | ratio | share |
|---|---|---|---|---|---|---|---|
| k0-7   | 0.3469 | 0.8836 | 0.9396 | 0.8549 | 0.0064 | 133.13 | 75.5% |
| k8-16  | 0.8019 | 0.6178 | 0.5978 | 0.4633 | 0.0199 |  23.25 | 22.2% |
| k17-32 | 0.9625 | 0.4842 | 0.3181 | 0.1423 | 0.0206 |   6.92 |  2.1% |
| k33-64 | 1.2636 | 0.9054 | 0.1232 | 0.0522 | 0.0053 |   9.79 |  0.3% |
| k0-64  | 0.3690 | 0.8768 | 0.9310 | 0.9841 | 0.0298 |  32.99 |  100% |

share = fraction of total residual mean-square. No shell below detection: min per-shell
ratio 5.58 at k=27.

Interpretation. The predicted field violates the vorticity equation at 0.98x the forcing
amplitude, 33x the stencil floor. 75.5% of the residual mean-square sits in k0-7 — the
band fully inside n_modes=8, where rel_l2 is 0.347 and rho 0.94. No stencil limit at Re100
at any wavenumber; retires the banked k~22 (Re100) figure, which came from the
residual-vs-field comparison removed in 8983ee9.

---

# 2026-07-29 (cont.)

Blur-horizon baseline, Re100 in-dist — first per-sample gamma horizon (`amp_curve`/
`print_blur`, e0c3beb) against the corr horizon at matched thresholds; tests gamma*=rho.
`75prctl5` own data, `report.py --reports error,amp,horizon,blur --bands
"0-7,8-16,17-32,33-64,0-64"`, per-frame time-bins, thresholds 0.9/0.8/0.5/0.3, test
[270:300] n=30, T_eff=65. rho from the identity rho=(1+g^2-r^2)/2g.
Anchors — reproduces banked 2026-07-06 vanilla Re100 (k0-7 early/late/aggr, k8-16 aggr,
k33-64 aggr) and 2026-07-24 k0-7 corr-horizon/gamma/rho exactly.

| band | rel_l2 | gamma | rho | H = gamma-rho |
|---|---|---|---|---|
| k0-7   | 0.3469 | 0.8836 | 0.9396 | -0.056 |
| k8-16  | 0.8019 | 0.6178 | 0.5978 | +0.020 |
| k17-32 | 0.9625 | 0.4842 | 0.3181 | +0.166 |
| k33-64 | 1.2636 | 0.9054 | 0.1232 | +0.782 |

k0-7 H = -0.046..-0.062 across all 65 frames. k8-16 H crosses zero near t16.
k33-64 gamma(t0)=1.65. Horizons at 0.9 (corr vs gamma), non-overlapping CIs:
k0-7 59.5 vs 43.2, k17-32 2.7 vs 7.8 — order reverses.

Interpretation. H is frame-invariant at -0.056 in k0-7 and positive and growing outside
(+0.02/+0.17/+0.78); the sign change falls at k=8, this repo's mode-cutoff split for
n_modes=[8,8,8] (2026-07-09), with k8-16 straddling and transitioning. Above the cutoff
amplitude is not a hedge — near-GT energy (gamma 0.91) at unrelated positions (rho
0.12), 65% over-energetic at t0 — matching the 2026-07-09 ablation where n_modes
[8,8,8]->[32,32,8] moved k32-64 further than k0-7 (-0.1396 vs -0.0339). gamma and corr
horizons separate at every band with reversing order: two independent measurements.
k0-7 has no knee, gamma monotone 0.950->0.840: no frame-localised cut in the
represented band.

---

# 2026-07-29

Blur = L2 optimum under phase uncertainty.

Claim — amplitude deficit γ<1 (one-shot FNO, all bands/regimes) is the L2 minimiser, not a
fitting failure. Motivation — Anna Option-2 constraint (reconstruction loss at TTA re-creates
the causal averaging); Lippe PDE-Refiner arXiv:2308.05732 Table 4.
Sketch — polar expand `|F_p−F_g|²`; expectation over unpredictable Δφ at lead t; minimise over a_p.

```
γ* = ρ                       ρ = E[cos Δφ], γ = a_p/a_g
rel_l2² = (1−ρ²) + (γ−ρ)²    exact, any band/frame window
```

Consequence — optimal amplitude = GT amplitude × achievable phase correlation ⇒ γ<1 irremovable
by L2 fitting. `1−ρ²` predictability floor; `(γ−ρ)²` calibration, 0 at optimum.
Assumption-free form (2026-07-31) — minimising `E|F_p−F_g|²` gives `F_p = E[F_g]` outright, then
`E|F_g|² = |E F_g|² + Var` gives `γ*² = 1 − Var/E|F_g|² ≤ 1`: the damping IS the conditional mean,
no polar step, while the `Sketch` line above is its amplitude reading and additionally assumes a_g
deterministic and Δφ symmetric — quote the conditional-mean form where the claim must hold without
those. Var is over futures conditional on one input, which the split does not sample (one future
per input); the client's pooled ρ² is not that fraction — row A k2 t0 0.989 vs 0.314.
Diagnostic — `H := γ−ρ`. H<0 over-damped (fitting-recoverable); H>0 pointwise traded for spectral.
Detail — `proof_sketch.md`.
