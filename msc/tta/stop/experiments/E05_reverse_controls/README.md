# E05 — stop/start rules on reverse shift, wrong-IC pools, no-shift (new val runs)

Goal: stress-test the frozen E02b stop rule and E03 two-sided gate outside the forward protocol family, on val only.

## Hypotheses
- H1 reverse (Re500 model `qr6rs0jb` → Re100): unadapted σ(0)=0.22 is already low (E03) → "minimise σ" has no target. Rule is safe iff it aborts or stops at ≈0 regret when adaptation does not help; fails if it chases σ below the attractor level into a smoothed solution.
- H2 A5 wrong-IC pools (Re100 states evolved τ ∈ {0, 2, 16, 100} under Re500): E_IC of the pool is the wrong spectrum for small τ → σ points at the wrong target. Expect: τ0/τ2 harmful or abort; τ100 ≈ correct pool (positive).
- H3 no-shift (model adapted on own Re): gate must say "do not adapt"; stop rule must abort or keep ΔH≈0.

- Known GT outcome (thesis controls, test split): A2 reverse is strongly positive (k_all 38.2→55.4 at S500, all ν_r); A5 IC+physics is harmful for τ ≤ 2, ≈0 at τ16, +47% at τ100. Pass = rule keeps reverse gain and aborts τ0/τ2. Risk: reverse unadapted model is over-energetic (γ_Z 1.17) but shape-matched (σ 0.22) → σ may not fall → false abort of a +17-frame gain.

## Method
- `launch.sh` (server copy `stop_research/E05/launch.sh`): `msc.tta.adapt.adapt`, steps 500, lr 1e-4, probe_every 5, pool_n 5, physics (pde + 5·ic), heldout val [240:270], wandb `unet-final`. One sequential queue per GPU (student05 0–3).
- Runs (10):

| exp | ckpt | target | pde ν | objective | pool | role |
|---|---|---|---|---|---|---|
| e05_rev | qr6rs0jb | 100 | 1/500 (ν_src) | physics | p0 | reverse protocol |
| e05_rev | qr6rs0jb | 100 | 1/500 | physics | p10 | pool replicate |
| e05_rev | qr6rs0jb | 100 | 1/100 (oracle) | physics | p0 | reverse oracle-ν |
| e05_rev | qr6rs0jb | 100 | — | ic | p0 | reverse negative |
| e05_a5 | 3z5bxjzp | 500 | 1/100 | physics | τ ∈ {0,2,16,100} | wrong-IC pools |
| e05_noshift | qr6rs0jb | 500 | 1/500 | physics | p0 | reverse no-shift |
| e05_noshift | 3z5bxjzp | 100 | 1/100 | physics | p0 | forward no-shift (= `unet_srcpool` config, 3rd seed) |

- Analysis: E01 `extract.py` on new npz → E02b `causal.py` frozen setting (no retuning) → per-run ΔH at s*, regret, abort/stop; gate σ(0) vs σ_src (E03 two-sided).

## Status
launched 2026-11-06 on student05 GPUs 0–3, ~3 waves. Launch-time bug: log names truncated → the 4 A5 runs share one log file (npz unaffected; fixed in local `launch.sh` afterwards). Throughput: probe_every 5 on Titan X Pascal → 1.7–2.8 h/run (not 36 min). Forward no-shift run dropped (duplicate of `unet_srcpool` ×2) via server watcher → 9 runs, ETA ~6 h from 13:05.

## Result (partial, 5/9 runs, 2026-11-06 16:10; `code/score.py` on E01 features, frozen setting, no retuning)

| run | GT ΔH s500 | GT best (s) | σ(0) | log σ(0)/σ_src | σ(200)/σ(0) | rule s* | outcome | ΔH at s* | regret |
|---|---|---|---|---|---|---|---|---|---|
| A5 τ0 (harmful) | −1.57 | 0 | 0.126 | −0.66 | 1.098 | 0 | abort ✓ | 0 | 0 |
| A5 τ2 (harmful) | −4.73 | 0 | 0.781 | +1.17 | 1.137 | 0 | abort ✓ | 0 | 0 |
| A5 τ100 (≈ target pool) | +6.20 | +6.20 (500) | 0.946 | +1.36 | 1.162 | 0 | abort ✗ | 0 | 6.20 |
| reverse ν_r 1/100 (oracle) | +18.27 | +18.30 (465) | 0.238 | −1.14 | 1.595 | 0 | abort ✗ | 0 | 18.30 |
| reverse ν_r 1/500 (ν_src) | +17.77 | +17.87 (260) | 0.238 | −1.14 | 0.755 | 205 | stop ✓ | +17.77 | 0.10 |

Trajectories (σ/σ(0) at s 0/50/100/200/300/500; GT ΔH):
- τ100: σ 1.0/1.10/1.14/1.16/1.15/1.10 while ΔH 0/1.9/2.1/3.9/5.2/6.2 → σ rises during a monotone gain.
- reverse ν 1/100: σ 1.0/1.28/1.60/1.60/1.40/1.13; ΔH 0/13.0/17.1/18.0/18.2/18.3 → gain saturates by s≈100, σ anti-tracks.
- reverse ν 1/500: σ 0.99/0.94/0.76/0.83/1.17; same ΔH curve → its "stop" is coincidental (σ U-shape unrelated to H).
- energy level M (mean log ratio) reverse: +0.33 → −0.7 (s100) → −0.35: reverse gain = removal of over-energy, overshoots then partly returns.

Interpretation (partial):
- Frozen σ-rule fails outside the forward protocol family: 2/3 positives falsely aborted (−6.2, −18.3 frames); harmful A5 pools aborted, but "never adapt" does the same.
- σ tracks forward-protocol horizon only; the E02b mechanism story (blur vs IC spectrum) does not transfer to reverse (over-energetic, not blurred) nor to τ100 pool.
- Reverse gain is fast (≈95% by s100) vs forward (needs ≥200) → any fixed probe length P is direction-specific.
- D1 status → done− for generality unless a two-sided / direction-aware variant is pre-registered and passes on remaining runs + E02b family.

## Provenance
npz `msc/tta/outputs/adapt/e05_{rev,a5,noshift}/` (server), logs `stop_research/E05/logs/`.
