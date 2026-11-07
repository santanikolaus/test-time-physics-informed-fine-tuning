# E02 — label-free stop/start rules scored on banked runs (zero GPU)

Goal: which label-free per-step series picks a checkpoint with GT horizon ≥ declared S, and refuses harmful runs.

## Method
- Input: E01 features (local copy: scratchpad `feat/`; canonical: `student05:.../stop_research/E01/features`).
- `score.py <feat_dir> [val|test]`; protocol per `../../PLAN.md` §3. Positive run: oracle gain > 1 frame (whole-field H, ρ*=0.9); negative otherwise.
- Signals:
  - M(s) = mean_{k=1..16} |log E_pred(k, window) / E_IC(k)| (invariant-measure mismatch; E_IC = IC-ensemble energy, label-free).
  - shape(s) = std_{k=1..16} log(E_pred/E_IC) (amplitude-free spectral shape).
  - residual power (pool / held-out ICs, loss-ν).
- Rules: argmin; gate05 = argmin only if min < 0.95·value(s=0), else s=0 (do not adapt).

## Result (2026-11-06, val split, 146 runs: 22 positive, 124 negative; negatives dominated by lr ≥ 1e-3 stress sweeps)

| rule | pos ΔH vs S0 | pos regret | neg ΔH vs S0 | neg worst | k2 dip violations |
|---|---|---|---|---|---|
| declared_S | 3.80 | 0.14 | −5.00 | −24.1 | 63 |
| min_loss | 3.74 | 0.20 | −4.56 | −24.8 | 63 |
| min_res_held | 0.27 | 3.67 | −3.71 | −30.2 | 74 |
| min_M_late_pool | 3.47 | 0.47 | −0.75 | −23.0 | 18 |
| gate05_M_late_pool | 3.47 | 0.47 | −0.72 | −23.0 | 17 |
| min_shape_late_pool | 3.54 | 0.40 | −1.31 | −23.0 | 18 |

Key trajectories (pool ICs, late window; s = 0 → 500):
- physics ν_src (positive): H 34.3→39.1; M 1.75→1.86 (dip, s≈60)→1.29; shape 0.90→0.93→0.77. Dip in H_k2 mirrored by M/shape rise.
- IC-only (negative): H 34.3→30.9; M 1.75→1.62 (false "go"); shape 0.90→0.96 (correct "stop").
- PDE-only (negative): H →18.9; M, shape both rise (correct); residual ↓ to 0.14·R0 (wrong).
- ν_r=1/50 (positive H, over-diffusive): H →40.8; M 1.75→2.04, shape 0.90→1.27 (both say "no").
- lr 1e-2 sweeps: one step destroys H; M/shape sometimes dip at s=1 → argmin picks it.

## Interpretation
- log E_pred/E_IC < 0 at every shell, every run: M = energy-deficit (amplitude) signal; it rewards any energy gain → fails IC-only.
- shape removes amplitude: separates physics vs IC-only vs PDE-only correctly on the canonical runs; residual does not.
- ν_r=1/50: higher horizon but spectrum moves off the IC measure → criterion rejects a smoother-but-more-correlated model; consistent with reviewer R1 (viscosity as smoothness knob). Decision needed: does the method target horizon or attractor fidelity.
- Remaining failure: single-step blow-ups at lr ≥ 1e-3 (argmin catches a transient). Fix = sustained-improvement rule (min over smoothed series, require k consecutive snapshots below threshold).

## γ check (held-out, window all, E_pred/E_gt, s=0 → 500)
| run | k1-4 | k5-8 | k9-16 |
|---|---|---|---|
| physics ν_src `0m` | 0.672→0.782 | 0.486→0.517 | 0.175→0.188 |
| physics ν_r=1/50 | 0.672→0.747 | 0.486→**0.419** | 0.175→**0.130** |
| IC-only `au9` | 0.672→0.709 | 0.486→0.492 | 0.175→0.161 |
| physics ν_r=1/500 | 1.347→0.953 | 1.147→0.871 | 1.027→0.786 |

- ν_r=1/50 damps k≥5 below the unadapted model → violates the thesis validity condition (γ ≥ unadapted); shape rejecting it = feature, not miss.
- ANOMALY: ν_r=1/500 run has γ(s=0)=1.35 ≠ 0.67 of the others although H0 identical → check its ckpt/meta before using it.

## Caveats (advisor review, 2026-11-06) — headline table NOT yet valid
1. argmin over a declared S is not a stopping rule (sees future, needs S). Positives carry no discriminative evidence (H rises to the end; min_loss also ≈3.7). Need causal rule: stop when shape improved < tol over last w snapshots; abort if shape > shape(0) after probe P (P must clear the ~65-step dip). Budget test runs: `s1000_ie46pt7g` (best@935), `s800_pm1135ol` (best@660).
2. Repeats (GPU nondeterminism) duplicate rows: `0m/ca/eu`, `au9/o5s`, `5hd/czu`, `k8/x5`, `0g/dv` → collapse by config. Headline on protocol family (UNet 3z5bxjzp, lr ≤ 3e-4, S ≥ 200); lr-stress = separate robustness row.
3. Shells 1–16 × late window chosen after the smoke run → grid k∈{1–8,1–16,4–16,1–32} × window∈{mid,late,all}, freeze one, report grid.
4. Start gate: no no-shift negative yet. Zero-GPU test: shape(0) of native Re100@Re100 vs Re100@Re500 (`outputs/report/*.npz`).
5. Pool (transductive, 3.54) vs held-out (inductive, 2.99): report both.

Command: `python score.py <features_dir> val` (features canonical: `student05:/system/user/studentwork/wehofer/stop_research/E01/features`).

## Next
- E02b: sustained/smoothed rules; stratify negatives (protocol lr ≤ 3e-4 vs lr stress); report γ/rel_l2 at s_R; per-band shape (k1-4 vs k5-16).
- test split read only once, after rule frozen.
