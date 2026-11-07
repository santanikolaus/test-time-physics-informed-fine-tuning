# Prior controls (done before stop research; do not rerun)

Source: `~/Downloads/thesis/controls/` — overview `ladder.md`, per-control `A*.md` (results + Provenance: redo command, scripts, server paths), raw tables `a*_out.md`. Server outputs `msc/tta/outputs/adapt/controls_a{2,3,5}/` (student05/04), A5 pool files `studentwork/wehofer/controls/a5_data/`.
Protocol: UNet `3z5bxjzp` Re100→Re500, physics (L_pde + 5 L_ic), full, n=5, lr 1e-4, S=500; replicates = pool_offset (no seed); locked test [270:300] unless noted.

| id | question | reviewer | result (test) | file |
|---|---|---|---|---|
| A1 | amplitude γ per band (R3: ρ amplitude-blind) | om7y, x6ay | ν_src raises γ in every band (dγ_Z +.05, CI excl. 0) → R3 rejected at ν_src; ν_r 1/50, 1/20 gain ρ by damping k≥9 (partly R1); L_pde only collapses (γ_E .44); early dip γ_E .84→.78 at S15 | A1.md |
| A2 | reverse Re500→Re100, ν_r ∈ {1/500, 0, 1/100} × 3 pools | x6ay, rWec | strong gain all ν_r: rec 50.5 / 49.8 / 52.1 %, k_all 38.2→55.4; under-diffusive misspec ≈97% of oracle; unadapted over-energetic (γ_Z 1.17) | A2.md |
| A3 | 6 pool draws × ν_r ∈ {1/100, 1/500, 0} | all | rec 49.9 (sd 4.8) at ν_src, paper's 43 = lowest; src−tgt +6.5 rec 6/6 pools, tgt more k5-8 amplitude 6/6; ν=0 keeps 85% of gain → R1 not main effect | A3.md |
| A4 | explicit Laplacian regularizer | om7y | not run (A2/A3 closed R1) | ladder.md |
| A5 | IC-state dial τ ∈ {0, .5, 2, 16, 100} × {IC only, IC+physics} × 3 pools (R2: IC exposure) | om7y, x6ay | IC only negative at every τ (−33…−63%); physics share +24→+94 (15/15); net gain only at τ100 (+47%), harmful τ≤2 → precondition: pool states from stationary target attractor | A5.md |
| A6 | fine-shell inversion (native < unadapted at k≥12) | — | not persistence; specific to ρ*=0.9 (frame-0 offset + early decorrelation of native); vanishes at ρ* 0.8 | A6.md |
| A7 | thesis tables: capped/uncapped rec, per-shell H fwd + rev, spectra | om7y (band choice) | rec 43.3 uncapped / 40.6 capped; spectra `a7_spectrum_re500_test.npz` | A7.md |
| A8 | residual convergence + GT floor | rWec (stop) | training L_pde falls to S1000; held-out residual min at S≈30 then rises past native (fwd) → residual ≠ stop signal; reverse min S≈200; floor set by dt=1/64 | A8.md |

## Still open (reviewer points not covered by A1–A8)
- stopping / start criterion (rWec) → this folder, E02b/E03/E05.
- second PDE (all) → `ks/KS_SPEC.md`.
- external TTA baseline (rWec).
- code-level seed (rWec; A3 covers pool variance only).
- deployment scenario + lr chosen on target labels (x6ay) → stop rule reduces lr sensitivity; writing.

## Constraints for stop/start criteria derived from controls
- must keep A2 reverse gain (no abort there).
- should abort A5 τ0/τ2 pools (known harmful) — tested on val in E05.
- must not stop inside S5–65 dip (A1, A8).
