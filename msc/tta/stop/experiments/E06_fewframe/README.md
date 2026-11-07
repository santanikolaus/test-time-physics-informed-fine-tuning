# E06 — few-frame setting: observed frames give start (ν̂) and stop (short-window error)

Setting change (user, 2026-11-06): drop the misspecification arc. Deployment observes m consecutive frames per pool IC (m ≤ 8 of 64, dataset step); viscosity is estimated from them and the residual uses ν̂. Adaptation objective L_pde(ν̂) + L_ic; observed frames 1..m are NOT in the gradient (validation signal only). Mandatory baseline later: data loss on the same m frames.

## Hypotheses
- H1 start: ν̂ from m frames by least squares (PDE linear in ν: ω_t + u·∇ω − f = ν ∇²ω) is accurate enough to decide shift / no shift.
- H2 stop: pool-IC error over observed frames 1..m, fed to the E02b causal rule (probe-abort + plateau + argmin), keeps the gain on forward, reverse and A5 τ100 positives and is safe on negatives. Risk: short-window error tracks the S≈30 residual minimum (A8) instead of the long horizon.

## Method
- `nu_hat.py`: GT val chains [240:270], frames ::2 (dt 1/64), m ∈ {3, 5, 9, 17, 65}; ν̂ = Σ⟨a,l⟩/Σ⟨l,l⟩ pooled over 5-IC pools (2000 draws); `src.pde.ns.NSVorticity` parts.
- `stop_short.py`: all val runs with E01 features (dedup by config) joined with raw npz `{pool,heldout}_err_pt`; e_m(s) = Σ err power frames 1..m / Σ GT power; causal rule grid side × m × P {25,50,100,200} × w {50,100} × τ {0.01,0.02}; families tagged (fwd ν_src, fwd oracle-ν, rev, A5 τ, IC-only, PDE-only, no-shift).
- Server: `stop_research/E06/`, `PYTHONPATH=<repo>:.`.

## Status
2026-11-06: nu_hat done; stop_short running.

## Result
### H1 ν̂ (val, 5-IC pools, p5–p95)
| Re | m=3 | m=5 | m=9 | m=17 | m=65 |
|---|---|---|---|---|---|
| 100 | 0.01000 [.00999, .01001] | same | same | same | same |
| 500 | 0.00203 [.00201, .00206] | 0.00202 | 0.00203 | 0.00202 | 0.00203 |
→ ν̂ within 1% from 3 frames at both Re; start gate = |log ν̂/ν_src| > tol is trivially perfect for a 5× shift. Caveat: noise-free solver data, same discretization as the residual (dt-stencil bias +1% at Re500, consistent with A8 floor).

### H2 stop (148 val runs, 23 pos / 125 neg; output `stop_research/E06/stop_short.out`)
Selected (pool side, neg mean ≥ −0.5, max pos ΔH; 128-setting grid on val → mild selection, needs locked test read): e_8 (frames 1–8), P 200, w 100, τ 0.01.

| rule | pos ΔH | pos regret | neg mean ΔH | neg worst |
|---|---|---|---|---|
| short-window e_8 causal | +4.69 | 0.46 | 0.00 | −1.87 |
| declared S | +5.01 | 0.14 | −4.86 | −24.13 |
| oracle | +5.14 | 0 | +0.11 | 0 |
| σ-rule (E02b, forward family only) | +2.40 | 1.57 | +0.22 | 0.00 |

Per-run highlights: reverse oracle-ν +18.20 (s* 310, regret 0.10), reverse ν_src +17.80; A5 τ100 +6.20 (regret 0); A5 τ0/τ2 abort ✓; IC-only, PDE-only, no-shift abort ✓; s1000 budget run +5.33 (regret 0.07); s800 run stop 360 (regret 0.50); ν_r 1/20 false abort (−5.8, irrelevant in ν̂ setting); n1 pool +0.50 (regret 1.33). Worst negative −1.87 = FNO n1 lr3e-4 s300. Robust over grid: top 13 pool settings 4.0–4.7 pos ΔH, neg mean ≈ 0. m=8 > m=4 > m≤2 (m=2 heldout worst −4.77). Heldout-IC variant (observed frames on other ICs) equal/better: +4.86, worst −1.17.
Caveats: family tags mislabel FNO forward runs as "rev" (cosmetic, ckpt-based tag); most positives "censored" → rule ≈ adapt while observed error falls, plateau rarely fires before budget; k2 dip not rewarded (ΔH_k2 > 0 for all positives).

### H2b stop from k validation ICs only (`stop_fewval.py`, rule P200/w100/τ0.01, 20 random draws of k from the 30 held-out ICs; GT H on all 30 → mild overlap)
| k | m | pos ΔH | regret | neg mean | neg worst (mean / any draw) |
|---|---|---|---|---|---|
| 2 | 3 | 3.65 | 1.50 | −0.01 | −1.75 / −4.77 |
| 3 | 3 | 3.87 | 1.27 | −0.03 | −3.35 / −4.77 |
| 3 | 4 | 4.47 | 0.68 | −0.01 | −2.29 / −4.77 |
| 3 | 8 | 4.84 | 0.31 | +0.01 | −1.19 / −1.23 |
| 5 | 4 | 4.53 | 0.61 | 0.00 | −1.40 / −4.77 |
| 5 | 8 | 4.82 | 0.32 | +0.01 | −1.21 / −1.23 |
→ 3 validation ICs × 8 frames ≈ 30 ICs; m ≤ 4 has a −4.8 tail. Deployment recommendation: 5 adaptation ICs (IC only, label-free physics) + 3 monitoring windows of IC + 8 frames (ν̂ + stop).

## Interpretation
- Few observed frames solve both ends on val: ν̂ exact → start; short-window (8 frames of 64) observed error → stop keeps 91% of oracle gain incl. reverse and τ100, harmful runs abort. Short-window error does NOT inherit the S≈30 residual minimum.
- Next: locked test read of the frozen setting; mandatory baseline = data loss on the same m frames (does physics add anything over fitting the observed frames?).
