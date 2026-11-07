# E01 — feature extraction + scoring harness

Goal: one compact per-run npz (GT readouts + label-free series) for every banked TTA run, so stop/start rules are scored locally without re-reading 110 GB.

## Method
- `extract.py <adapt_root> <out_dir> [shard n_shards]`; run from server repo root with `PYTHONPATH=.`.
- Per run keys (S = snapshots):
  - labels (prefix `gt_`): `gt_{heldout,pool}_H_all` (S,), `_H_k` (S,32; shells 1..32), `_rel_all`, `_rel_k7` (S,), `_E_w` (33 shells × 4 windows, GT energy).
  - label-free: `{side}_E_ic` (33,) IC-ensemble energy, `_E_ic_n` (N,33), `_E_pred_w` (S,33,4), `_E_pred_tot_t` (S,T), `_res_w` / `_res_op_w` (S,33,4) residual power at loss-ν / op-ν, `_w1wc_t` (S,T) [W1 uses GT → treat as label].
  - `losses_*` (per optimizer step), `meta_*` (split, objective, pde_re, lr, pool, …).
- Windows: early 1–8, mid 28–36, late 55–62, all 1–62. Horizon ρ*=0.9 first_cross (thesis convention).

## Status
running (8 shards, started 2026-11-06). Smoke: `unet_nu100/...0mt68z26` (val): H_all 34.27 → 39.13, 422 KB, ~90 s/run (NFS read bound).

## Provenance
- outputs: `student05:/system/user/studentwork/wehofer/stop_research/E01/features/*.npz`, logs `E01/extract_*.{log,err}`.
- server repo commit cc10a19 (eval.py functions).
