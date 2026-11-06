# E04 — self-consistency restart + checkpoint drift (banked weights, GPU inference)

Goal: test two label-free, phase-sensitive stop signals that the spectral shape (amplitude-free but phase-blind) cannot provide.

## Hypotheses
- D3 self-consistency: restart the model from its own predicted frame t0 (r = G(û[t0])); correlation horizon between û[t0:] and r[:65−t0] (ρ*=0.9) tracks the GT horizon across steps: a model that has learned the target dynamics agrees with itself under restart.
- D4 drift: ‖û_s − û_{s−20}‖² / ‖û_s‖² on unlabeled ICs decays to a plateau; plateau = stop.
- Both must separate physics+ from IC-only / PDE-only / source-pool, and must not reward the k2/k3 dip (s 5–65).

## Method
- Runs (val, weights every 5 steps, `msc/tta/outputs/adapt/{unet_nu100,unet_srcpool}/weights/`): physics+ `camc8fxl`, IC-only `au9wj7tk`, PDE-only `czu1ocfu`, source-pool `ei38s8ax`.
- Steps 0, 20, …, 500 (26 ckpts; step 0 = pretrained `3z5bxjzp`). ICs: held-out val [240:270] (30) + the run's own pool (5).
- Per sample, shells 0..32, `band_power_t`: GT pred/gt/err (anchor vs E01 H), restart t0 ∈ {16, 32}: power of û-segment, r-segment, difference; drift power of û_s − û_prev.
- `selfcons.py <weights_dir> <out.npz>`; one process per run, GPUs 0–3. Analysis `analyze.py` (local).
- Anchor: GT whole-field H from this script must match E01 `gt_heldout_H_all` at matching steps.

## Status
4/4 runs done 2026-11-06. Source-pool run is a no-shift control (target_re=100): every signal flat, GT H saturated. Analysis runs on the server (local .venv scipy is broken): `PYTHONPATH=. python analyze.py <E04 dir>`.

## Result (held-out val n=30, pool n=5; SC = self-consistency restart, ρ̄ = mean correlation over overlap)

Anchor: physics+ GT H 34.30→39.23 at s=500 = E01 `gt_heldout_H_all` (39.2) ✓.

| run | s | GT H | SC16 H held | SC32 H held | SC16 ρ̄ held | SC16 ρ̄ pool | drift late |
|---|---|---|---|---|---|---|---|
| physics+ `camc8fxl` | 0 | 34.30 | 49.0 | 33.0 | 0.9902 | 0.9883 | — |
| | 100 | 35.37 | 49.0 | 33.0 | 0.9887 | 0.9948 | 6e-4 |
| | 300 | 38.03 | 49.0 | 33.0 | 0.9844 | 0.9936 | 4e-4 |
| | 500 | 39.23 | 49.0 | 32.67 | 0.9817 | 0.9925 | <1e-4 |
| IC-only `au9wj7tk` | 100 | 33.23 | 49.0 | 33.0 | 0.9886 | 0.9866 | 1e-4 |
| | 500 | 30.93 | 49.0 | 33.0 | 0.9833 | 0.9814 | <1e-4 |
| PDE-only `czu1ocfu` | 100 | 27.30 | 49.0 | 33.0 | 0.9799 | 0.9755 | 1.6e-3 |
| | 500 | 18.90 | 41.17 | 21.33 | 0.9395 | 0.9339 | 1e-4 |
| source-pool `ei38s8ax` (target = Re100, no shift) | 0 | 63.60 | 48.87 | 32.63 | 0.9914 | 0.9958 | — |
| | 500 | 63.43 | 49.0 | 32.70 | 0.9922 | 0.9963 | 1e-4 |

(full per-20-step tables: rerun `analyze.py`)

## Interpretation
- SC horizon saturated (49/49, 33/33 = full overlap) except under PDE-only collapse → restarting from the model's own blurred frame is too easy; no resolution for the stop step.
- SC ρ̄ held-out falls for physics+ (0.990→0.982) while GT H rises → anti-tracks within run; unusable as a stop signal.
- SC ρ̄ pool: physics+ ↑ (0.988→0.995, peak s≈100–160), IC-only ↓, PDE-only ↓↓ → separates helpful from harmful runs (abort-type information) but peaks long before the GT optimum → would stop early (same failure as the residual).
- Drift decays to ~1e-4 by s≈100–300 for all three runs, harmful included → plateau carries no information on quality.
- D3/D4 done−: no phase-sensitive label-free stop signal from restart or drift in this form. Spectral shape (E02b) remains the only working stop signal.

## Provenance
outputs `student05:/system/user/studentwork/wehofer/stop_research/E04/<run>.npz`, logs beside.
