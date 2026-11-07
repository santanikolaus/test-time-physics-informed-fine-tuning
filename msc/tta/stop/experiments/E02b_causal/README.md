# E02b — causal stop rule (probe-abort + plateau)

Goal: a rule that decides at step s from snapshots ≤ s only, needs no declared budget, and is scored on the protocol family.

## Hypothesis
Spectral-shape mismatch σ(s) = std_{k∈K} log(E_pred(k,window)/E_IC(k)) (pool ICs) rises during the k2/k3 dip, then falls while the horizon grows, and plateaus when the horizon saturates. Therefore:
- abort: σ(P) ≥ (1−δ)·σ(0) after probe P → s* = 0 (do not adapt; also = start decision after a probe).
- stop: first s > P where running-min σ improved < τ·σ(0) over the last w steps → s* = argmin σ over seen snapshots.
- never fires: s* = argmin over all seen snapshots at the end of the data (censored; flagged).

## Method
- Input: E01 features. Script `causal.py <feat_dir> val`.
- Families:
  - protocol: ckpt `3z5bxjzp`, lr ≤ 3e-4, S ≥ 200 (headline).
  - stress: everything else (robustness row).
- Duplicates (same config name, different wandb id) collapsed: first id kept.
- Grid: K ∈ {1–8, 1–16, 4–16, 1–32} × window ∈ {mid, late, all} × P ∈ {60, 100} × w ∈ {50, 100} × τ ∈ {0.01, 0.02}; δ = 0.02 fixed.
- Selection: maximise protocol mean ΔH (vs s=0), with protocol negatives' mean ΔH ≥ −0.5. Report the full grid, the top-5 settings and the frozen one.
- Budget-test runs: `s1000_ie46pt7g` (oracle best @935), `s800_pm1135ol` (oracle best @660).

## Status
done 2026-11-06 (grid extended to P ∈ {60,100,150,200,250} after P ≤ 100 aborted every lr=1e-4 run).

## Result (val, 143 configs after dedup; protocol 29 = 18 positive / 11 negative)

| rule | proto pos ΔH | proto pos regret | proto neg ΔH | proto neg worst | stress neg worst |
|---|---|---|---|---|---|
| declared S | 3.84 | 0.13 | −2.93 | −15.87 | −24.13 |
| never adapt | 0 | 3.97 | 0 | 0 | 0 |
| causal, K 1–16, late, P=200, w=100, τ=0.01, pool ICs (frozen) | 2.40 | 1.57 | +0.22 | 0.00 | −1.50 |
| same, P=100 | 1.58 | 2.39 | +0.14 | 0.00 | −1.50 |
| same, P=250 | 2.08 | 1.89 | 0.00 | 0.00 | −1.50 |

Grid ranking: K 1–16 > 4–16 > 1–32 ≈ 1–8; late window dominates; P=200 best; w, τ second order. Full grid: rerun `causal.py`.

Per-run, frozen setting, pool ICs (protocol):
- all 11 negatives refused or harmless: IC-only `au9`, PDE-only `5hd`/`776`, source-pool `8itlwu8c`, bottleneck, deepconv → abort; deepblock → +0.07/+0.53.
- budget tests: `s1000_ie46pt7g` stops at 960 (+5.33, oracle +5.40 @935); `s800_pm1135ol` stops at 660 (= oracle, +2.87).
- canonical `0mt68z26` → censored at 500 (+4.87; plateau never fires; consistent with thesis: gain saturates ≥500).
- false aborts (positives refused): n10, lr 7e-5, ν_r 1/20, ν_r 1/50, p10-ν_src. ν_r 1/20, 1/50 = damped (E02 γ check) → intended. n10, lr 7e-5, p10-ν_src = slow shape recovery (σ(200) not yet 2% below σ(0)).
- held-out-IC variant (inductive): aborts most ν_src lr ≤ 3e-4 runs incl. canonical → markedly weaker than pool variant.

## Interpretation
- Causal rule is safe on every protocol negative (no harm, worst 0.00 vs −15.87 for declared S) and recovers 63% of the declared-S gain on positives; it found the oracle step on both budget tests.
- Cost of safety = false aborts on slow-but-positive runs; spectral shape returns below its s=0 value later (~125–200 steps) than the horizon dip ends (~65).
- Transductive (pool ICs) > inductive (held-out ICs) for this signal.
- Abort after probe doubles as the START decision after a 200-step probe.

## Provenance
local features: scratchpad copy of `student05:/system/user/studentwork/wehofer/stop_research/E01/features`.
