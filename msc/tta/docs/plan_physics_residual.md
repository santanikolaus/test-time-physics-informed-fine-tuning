# Plan — physics-residual reporting + array storage

Transient implementation plan; delete once landed. Converged 2026-07-29.

## What the metric is

GT is a spectral-solver solution, so it satisfies the PDE exactly in the continuum: its
measured residual under our coarse centred time stencil is *entirely our measurement
error*, not physics. Hence

```
ratio_gt = (model violation + measurement noise) / (measurement noise)
```

— a signal-to-noise ratio, structurally the same as the GT-vs-GT floors already used for
W1/MMD. `~1` = violation below detection; `>>1` = detectable violation.

## Measured constraint (GT floor check, 30 test traj, both Re, no model)

sqrt(res_gt / field) pooled over samples and all 63 residual frames:

| band | Re100 | Re500 |
|---|:--:|:--:|
| k0-7 | 0.0061 | 0.0179 |
| k8-16 | 0.109 | 0.323 |
| k17-32 | 0.673 | 1.664 |
| k33-64 | 2.712 | 8.400 |

Every shell sits >=7e8x above the float32 power floor -> not a precision problem. The
noise floor reaches the field amplitude (ratio 1.0) at **k~17 (Re500)** and **k~22
(Re100)**; above that any violation is buried, so the metric loses *sensitivity*, not
validity. Cause: the time stencil `(w[t+1]-w[t-1])/2dt` at dt=1/64 cannot resolve modes
whose turnover time ~1/(u*k) approaches dt. Nyquist shell k64 spikes (176/198), DC shell
k0 is degenerate (field power ~1e-5, ratio 40).

Property of the shared 65-frame time grid, therefore **identical for FNO and U-Net** —
which is why it belongs in a default. Spatial resolution is not binding (128^2 resolves
k<=64); `sub_t=2` discarding half the available frames is what sets it. Not exploitable
here: the model's t-grid is fixed at 65 frames.

Frames are stable: k0-7 varies 0.0170-0.0190 over j=0..62 at Re500 and 0.0043-0.0081 at
Re100, no edge pathology. A flat floor is what licenses reading a per-frame ratio — drift
in the ratio is then attributable to the numerator.

**=> bands restricted, frames fully resolved per frame (not aggregated).**

## Steps

1. **`eval.py`** — one new function, one deletion.
   - `resid_ratio(res_pt, den_pt, bands=slice(None), frames=slice(None), per_frame=False)`
     -> `sqrt(sum(res) / sum(den))`, house pooling (sum both, then one ratio). Serves
     every reading by choice of `den_pt`.
   - Delete the `pde_residual` stub: its declared shape omits the sample axis
     `forward_bands` produces, and an "RMS magnitude" off unnormalised-FFT power is
     `S**2`-scaled and in no physical unit. Removes 3 dead-parameter warnings.

2. **`report.py`** — `print_physics`, three band x time sub-tables in the existing
   `_band_time_header` layout:
   | table | denominator | GT? | use |
   |---|---|:--:|---|
   | `ratio_gt` | `pde_res_gt_pt` | yes | offline: violation vs detection floor |
   | `ratio_self` | `pred_pt[:, :, 1:64]` | no | test-time TTA trigger |
   | `raw` | none | no | diagnostic only; `S**2`-scaled, never cross-resolution |

3. **Frame mapping.** Residual index `j` = field frame `j+1`, spanning field frames
   1..63. `--time-bins` stays in field-frame coordinates everywhere; map by -1, clip to
   the valid range, print the effective window. A window with no residual frames (e.g.
   `0-0`) prints as empty, never as zero.

4. **Regime header**, printed once per run in `main()` regardless of selected reports:
   `physics regime: op_re=100 test_re=500 -> CROSS (operator physics Re100 vs data Re500)`
   `physics regime: op_re=100 test_re=100 -> NATIVE (same physics both sides)`
   Surfaces the case where both silently fell back to the training Re.

5. **Default bands** — new `PHYS` sentinel in `_resolve_bands`:
   per-shell k1..24, then coarse `1-8, 9-16, 17-24`, then `25-64` as a single flagged
   dead-zone row. 28 rows. Rationale: step 1 because band groups are ratios-of-sums and
   **do not re-aggregate**, so pooling not done is irreversible while per-shell is free;
   to k24 because that brackets the crossing at *both* Re, making the breakdown empirical
   rather than asserted; the coarse triple is the model-neutral headline comparable across
   FNO and U-Net; the dead-zone row shows the collapse in one line instead of 40.
   k0 excluded (DC degenerate), k64 folded into the dead zone.
   gamma/rho reports keep `0-7,8-16,17-32,33-64` unchanged for journal comparability.

6. **Tests** (CPU-only, toy arrays, nothing true by construction): self-ratio == 1;
   pooling is sum-then-ratio not mean-of-ratios; band selection; zero-denominator guard;
   field-frame <-> residual-index mapping pinned on a field whose residual is nonzero at
   one known frame; `--time-bins` clipping incl. the empty-window case; regime header
   NATIVE vs CROSS; `PHYS` sentinel expansion; npz round-trip.

7. **`--save-npz PATH`** — `np.savez_compressed` of `cache["bands"]` plus metadata:
   `run_id`, resolved `data_path`, `op_re`, `test_re`, `sub_t`, `T_eff`, split offsets,
   git SHA (best-effort, non-fatal). ~3.0 MB band arrays + ~2.0 MB residual arrays per
   row. Fields excluded (255 MB/row, only `w1`/`cov` need them).

8. **Storage run**, `--op-re`/`--test-re` explicit on every row, never defaulted:
   A `75prctl5` @ Re100, op=test=100 (NATIVE) ·
   B `75prctl5` @ Re500, op=test=500 (CROSS data; nu identifiable from a few frames, so
   the residual is evaluated at the data's true physics) ·
   C `nrow6j9l` @ Re500, op=test=500 (NATIVE).

## Notes carried from the discussion

- `ratio_gt < 1` is not "better than GT": a blurred field has small derivatives and
  satisfies the equation too easily. With gamma=0.88 in k0-7 the operator is already
  under-energetic, so a low residual there is partly the blur, not physics. Residual and
  gamma are not independent readings.
- `ratio_self` partly cancels that (smoother field -> smaller residual and smaller
  energy), which helps a detector but means it is not a clean violation measure either.
- Floor is Re-dependent (k0-7: 0.0061 at Re100 vs 0.0179 at Re500), so cross-Re ratio
  comparisons carry whichever GT's floor sits in the denominator.
- `print_error`/`print_amp`/`print_physics` now share a near-identical band x time loop;
  collapsing them into one helper is a later cleanup, deliberately not done here to avoid
  touching the printers that produce the banked anchor numbers.
