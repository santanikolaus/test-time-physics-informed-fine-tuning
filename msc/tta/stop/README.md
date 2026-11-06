# STOP — lead file

Goal: label-free START (adapt or not) + STOP (which step) criterion for test-time physics-informed fine-tuning (TT-PIFT); then generality on a 2nd PDE (Kuramoto–Sivashinsky). Together = the method.

Read order for a new agent: this file → `CONTROLS.md` (done, do not rerun) → `PLAN.md` (protocol, data, scoring) → newest `experiments/E*/README.md` → `lit/LIT.md`, `ks/KS_SPEC.md`.

## Directions (status: idea | queued | running | done+ | done− | parked)

| id | direction | ends | status | pointer |
|---|---|---|---|---|
| D0 | scoring harness: rule → step → GT regret (horizon ρ*=0.9) + safety on negative runs | both | done+ (207 runs) | E01 |
| D1 | invariant-measure consistency: spectral SHAPE of predictions vs IC-ensemble spectrum | both | forward-only: works in E02b family, fails reverse + τ100 (E05) | E02, E02b, E05 |
| D2 | held-out-IC residual (inductive, deployment-legal) as rule input | stop | done− (regret 3.67/3.94) | E02 |
| D3 | trajectory self-consistency: restart from predicted frame, compare overlap | stop | done− (saturated / anti-tracks) | E04 |
| D4 | prediction drift between checkpoints (plateau = stop) | stop | done− (decays for harmful runs too) | E04 |
| D5 | start gate: phase-2 residual OOD score + D1 mismatch at step 0 | start | idea | — |
| D6 | Kuramoto–Sivashinsky: shift ν (u_xxxx) 1.0→0.5, L=64; ν term = sole energy sink → misspecification not favorable | generality | spec done, build after KF criterion survives | ks/KS_SPEC.md |
| L | literature: label-free selection/stopping, TTA gating | both | done (several [R] cites unverified) | lit/LIT.md |

Prior controls (summary + pointers: `CONTROLS.md`; source `~/Downloads/thesis/controls/ladder.md`, A1–A8, test split): R1 regularizer, R2 IC exposure, R3 amplitude closed; reverse Re500→Re100 adapts strongly (k_all 38.2→55.4, rec 50%, A2); gain needs stationary target-attractor pool states (A5); held-out residual min at s≈30 (A8). Any criterion must keep A2 reverse and abort A5 τ≤2.
Scope decision (user, 2026-11-06): KF only until a criterion survives; KS = later control, not built before.

Infra: studentwork 84% (34 GB free after deleting `adapt/fno/weights`, 21 GB, 2026-11-06); student05 missing from `msc/configs/gpus.yaml`.
Weights every 5 steps exist for 4 val runs (`msc/tta/outputs/adapt/{unet_nu100,unet_srcpool}/weights/`): physics+ `camc8fxl`, IC-only `au9wj7tk`, PDE-only `czu1ocfu`, source-pool `ei38s8ax` → D3/D4 need GPU inference only, no new training. DO NOT delete.

## Experiments

| id | goal (one line) | GPU | status |
|---|---|---|---|
| E00 | landscape survey, within-run Spearman | no | superseded |
| E01 | compact per-run features (GT readouts + label-free series), 207 runs | no | done |
| E02 | argmin rules over declared S: shape vs energy vs residual | no | done, caveats |
| E02b | causal stop rule (probe-abort + plateau) on protocol family, k/window grid | no | done+: safe on all negatives, 63% of declared-S gain |
| E03 | start gate: shape(0) in-dist vs shifted, pool-size-5 bootstrap AUROC | inference | done+: two-sided \|log σ(0)/σ_src\| AUROC 1.0 both directions; one-sided σ inverts in reverse |
| E04 | self-consistency restart + checkpoint drift on 4 banked weight runs | yes (inference) | done−: SC horizon saturated, SC ρ̄ anti-tracks held-out, drift uninformative; pool SC ρ̄ = abort-only info |
| E06 | few-frame setting (user 2026-11-06: drop misspecification, observe m frames/IC): ν̂ start + short-window-error stop | no (CPU) | done+: ν̂ ±1% from 3 frames; e_8 causal stop pos +4.69 (oracle 5.14), neg mean 0.00, keeps reverse +18.2 |
| E05 | stop rule + gate on reverse, A5 wrong-IC pools, no-shift (10 new val runs) | yes (training) | partial 5/9: frozen σ-rule falsely aborts A5 τ100 (−6.2) and reverse oracle-ν (−18.3); aborts harmful τ0/τ2 ✓ |

## Current best criterion (few-frame setting, E06)
Start: ν̂ least squares from m≥3 observed frames → adapt iff ν̂ ≠ ν_src. Stop: pool-IC relative error over observed frames 1–8, causal rule P200/w100/τ0.01. Pending: locked test read; data-loss-on-m-frames baseline.

Deployment (E06 H2b): 5 adaptation ICs (no frames, label-free L_pde(ν̂)+L_ic) + 3 monitoring windows (IC + 8 frames) for ν̂ and stop.
Novelty warning: physics-residual fine-tuning at correct ν incl. KF Re100→Re500 = PINO instance-wise fine-tuning (Li et al. 2111.03794), fixed budget, no stop. Contribution must be: start/stop protocol, pool-level surrogate adaptation, recurrent arch, controls.

## Previous criterion (label-free, superseded)
Causal rule (E02b, frozen): σ(s) = std_{k=1..16} log(E_pred(k, frames 55–62)/E_IC(k)) on the 5 pool ICs; probe 200 steps → abort (s*=0) unless σ(200) < 0.98·σ(0); then stop when running-min σ improves < 1%·σ(0) over 100 steps; return argmin σ seen.
Protocol val: positives +2.40 frames (declared S +3.84, never 0); negatives +0.22, worst 0.00 (declared S worst −15.87); oracle step hit on both budget tests (s1000→960, s800→660).
Weak spots: false aborts on slow-but-positive runs (n10, lr 7e-5); inductive (held-out IC) variant much weaker; phase-blind.
Known negative: training/held-out residual — minimum at S≈30, inside k2/k3 dip (thesis §5 "Model selection").

## Next action
E05 finishing (remaining runs feed E06 scoring too). Then: E07 = data-loss-on-m-frames baseline (new objective, GPU) → locked test read of E06 rule. Recurrent U-Net (Lippe recipe) ckpts `ry8uot3p` (Re100), `sl3fp15e` (Re500) = 2nd architecture / PDE-Refiner idea — DO NOT delete.
