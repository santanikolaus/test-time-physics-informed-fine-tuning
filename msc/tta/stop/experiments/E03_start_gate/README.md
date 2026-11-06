# E03 — start gate: does the unadapted model's spectral shape separate shifted from unshifted data?

Goal: a label-free decision "adapt / do not adapt" at step 0, from 5 unlabeled ICs + the unadapted model's predictions.

## Hypothesis
σ(0) = std_{k=1..16} log(E_pred(k, late)/E_IC(k)) (frozen E02b setting) is larger on shifted data than on the model's training distribution. Comparison: phase-2 OOD score = residual power at ν_op, shells 9–16 (B3), all frames.

## Method
- `gate.py` (GPU inference, val split [240:270], n=30): forward_bands for UNet `3z5bxjzp` (Re100) on Re100 and Re500, and `qr6rs0jb` (Re500) on Re500 and Re100.
- Deployment unit = pool of 5 ICs: 2000 random 5-subsets per (model, data) → distribution of σ(0), M(0) (energy level), B3 residual (normalised by forcing-free scale: per-sample residual / IC power in B3).
- Metrics per direction: AUROC(ID vs OOD); threshold = 95th percentile of ID → OOD detection rate.

## Status
done 2026-11-06. `analyze.py <gate.npz>`; two-sided variant computed inline (reference σ_src = median over independent 5-IC draws of the model on its own training-Re val data).

## Result (val, 2000 draws of 5 ICs per cell)

| direction | statistic | ID median | OOD median | AUROC | OOD detected @ ID p95 |
|---|---|---|---|---|---|
| forward Re100→Re500 | shape σ | 0.243 | 1.107 | 1.000 | 1.00 |
| forward | energy M | 0.339 | 1.792 | 1.000 | 1.00 |
| forward | B3 residual (raw) | 1.26e10 | 4.87e10 | 0.936 | 0.46 |
| forward | B3 residual / IC B3 power | 288 | 280 | 0.451 | 0.00 |
| reverse Re500→Re100 | shape σ | 0.742 | 0.220 | 0.000 | 0.00 |
| reverse | energy M | 1.156 | 0.249 | 0.000 | 0.00 |
| reverse | B3 residual (raw) | 5.37e10 | 7.98e10 | 0.828 | 0.64 |
| reverse | B3 residual / IC B3 power | 314 | 1841 | 1.000 | 1.00 |
| forward | two-sided \|log σ/σ_src\| | — | — | 1.000 | 1.00 |
| reverse | two-sided \|log σ/σ_src\| | — | — | 1.000 | 0.999 |

## Interpretation
- σ is not one-sided: the Re500 model is already far off-shape on its own data (0.74; fine scales decorrelate → late-frame blur) and closer on smoother Re100 data (0.22). σ measures late-frame blur vs IC spectrum, which scales with target chaoticity.
- START gate that works both directions: deviation of σ(0) from the model's own source-data reference σ_src (source data is owned → legal). AUROC 1.0 both ways with 5 ICs.
- Phase-2 residual scores: each normalisation works in one direction only.
- Consequence for STOP: "minimise σ" (E02b) is validated only forward. Reverse: unadapted σ (0.22) already ≈ native Re100 model ID σ (0.24, forward table) → minimising σ has no defined target. Stop rule must be retested on reverse runs; A2 reverse controls exist only on the test split → val reverse runs needed (E05 candidate).
- Caveat: ID test and reference draws come from the same 30 val chains; a cleaner reference = model on source train/val, target pools from target train.

## Provenance
outputs `student05:/system/user/studentwork/wehofer/stop_research/E03/gate.npz`.
