# STOP — plan

## 1. Problem

Method = TT-PIFT: pretrained surrogate (UNet3D `3z5bxjzp`, Re100) → Re500; objective L_pde(ν_src) + 5·L_ic; pool n=5 unlabeled target ICs; Adam lr 1e-4; S=500 declared. Thesis: `~/Downloads/thesis` (Ch5 "Model selection", Ch6 "Limitations"). Workshop: Sim2Sci 2026 #180, rejected; reviewer rWec: "no observable stopping criterion … which checkpoint has the best predictive skill". Reviews: `~/Downloads/thesis/presentation/Reviews.pdf`.

- STOP: choose s* ∈ {0..S} from label-free data only.
- START: decide adapt vs. not (shift present AND adaptation expected to help).
- Generality: same criteria on a 2nd PDE (KS), shift on a coefficient that dominates the residual (answers "favorable misspecification", rWec/om7y; thesis: no-diffusion keeps 85% of gain).

Known facts (thesis, run `0s25y123`, test split):
- whole-field horizon H: 33.1 (S=0) → 37.3 (S=500); native Re500 model 48.8.
- k2/k3 regress S=5–65 (k2 −1.2 frames at S=15, back at S=65); whole-field gain significant from S=200; saturates ≥500.
- residual (even ν_tgt) min at S=30 → early stopping on residual = wrong; across models residual ranking inverts.
- loss decreases monotonically also on harmful runs (E00 first rows: `tau16` pool, best step 0, loss ρ=0.88).

## 2. Label-free information at deployment (legal inputs)

| symbol | quantity | legal because |
|---|---|---|
| u0_pool, u0_new | target ICs (attractor snapshots, stationary measure) | inputs |
| E_IC(k) | per-shell energy of IC ensemble | inputs only; stationarity ⇒ = E_true(k,t) ∀t |
| ŷ_s | predictions of checkpoint s on pool and on new ICs | model output |
| R_ν(ŷ_s) | PDE residual at any ν | equation + output |
| θ_s, θ_0 | weights per step, source weights | model |
| source train data | Re100 trajectories | owned |

Not legal: GT target trajectories (y), ν_tgt (unless estimated), held-out GT spectra beyond frame 0.

## 3. Scoring protocol (fixed; every experiment uses it)

- GT readout: correlation horizon H_k, ρ*=0.9, `first_cross`, per shell k and whole field (`msc/tta/eval/eval.py` `corr_curve` + `time_to_threshold`); secondary rel_l2 and γ (amplitude) to address x6ay.
- Rule R maps label-free per-step series → s_R.
- Per run: regret_oracle = H(s_best) − H(s_R); gain_vs_declared = H(s_R) − H(S); gain_vs_0 = H(s_R) − H(0).
- Safety: on negative runs (GT best at s≈0) require H(s_R) ≥ H(0) − ε; dip check: s_R ∉ k2/k3 dip unless H_k2(s_R) ≥ H_k2(0).
- Ranking metric: mean gain_vs_declared over positive runs, subject to safety on all negatives.
- Splits: design/tune on runs with `meta_heldout_split` = val [240:270]. Runs scored on test [270:300] (`*-test*` files, thesis controls A2/A3/A5) = one confirmatory read at the end only.
- Caveat: Spearman over steps within a run is NOT evidence (shared trend with step).

## 4. Data inventory

- Banked runs: `student05:/system/user/studentwork/wehofer/pino-lightning/msc/tta/outputs/adapt/{fno,unet,unet150,unet_nu100,unet_srcpool,controls_a2,controls_a3,controls_a5}/*.npz` (~200 runs). Per snapshot step: `{pool,heldout}_{pred,gt,err}_pt` (S,N,65 shells,65 frames), `*_pde_res_{pred,pred_op,gt}_pt` (S,N,65,63), `w1_t`, `w1wc_t`; `losses_*` (per step); `meta_*`.
- Weights: only `unet_srcpool/weights` (+ runs with `save_every`) → D3 needs new runs.
- Thesis control meanings: `~/Downloads/thesis/controls/{ladder.md,A*.md}` (A2 reverse Re500→100, A3 pool offsets × ν, A5 τ-evolved Re100 ICs = start-gate negatives).
- Research outputs: `student05:/system/user/studentwork/wehofer/stop_research/E??/`.

## 5. Directions — hypotheses

- D1 invariant measure: stationarity ⇒ E_ŷ(k,t) should equal E_IC(k) for all t. M(s) = Σ_k w_k |log E_ŷs(k, late) − log E_IC(k)|. Hyp: M decreases while adaptation helps, plateaus at convergence; harmful runs raise M or leave it. START: M(0) vs ID-reference level (source model on source ICs). Addresses x6ay energetics.
- D2 held-out residual: R on unlabeled new ICs. Known to anti-track within run; test as cross-run discriminator and in combination with D1.
- D3 self-consistency: G(u0)[t0+τ] vs G(G(u0)[t0])[τ]; horizon proxy without labels. Needs weights per step.
- D4 drift: ‖ŷ_s − ŷ_{s−Δ}‖ / ‖ŷ_s‖ on new ICs; stop on plateau. Spectral proxy available from npz; field version needs weights.
- D5 start: phase-2 score (residual at ν_src, band B3=[9,16], `documentation/ood/`) at s=0 + D1 M(0).
- D6 KS: see `ks/KS_SPEC.md`.

## 6. Sequencing

1. E00 landscape (done/running) → 2. E01 harness + inventory (val runs, positives/negatives labelled by GT) → 3. E02 score D1/D2/D4-spectral from banked npz (zero GPU) → 4. if a rule survives: E03 GPU runs with `save_every` for D3/D4-field + new negatives (wrong ICs on val) → 5. confirmatory test read → 6. KS.
GPU runs require user sign-off before launch.

## 7. Conventions for every experiment folder `experiments/E##_name/`

`README.md` sections: Goal (1 line) · Hypothesis · Method (inputs, script, command) · Status · Result (raw numbers) · Interpretation (≤5 lines) · Provenance (server path of outputs, commit, date). Scripts live beside it; outputs on server under `stop_research/E##/`. Update the lead table in `README.md` after every status change.
