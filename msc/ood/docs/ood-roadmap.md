# OOD Roadmap — residual decomposition trust signals

Core claim: **don't aggregate the PDE residual — decompose it (terms × bands), the structure is free and diagnostic.**
General principle (architecture-free): decompose > aggregate. FNO-specific instance: B3=[n_modes+1, 2·n_modes] onset at truncation cutoff.
Calibration axis (NOT "ground truth"): online running-stats | offline ID baseline | conformal (coverage guarantee).
Physics fix (2D Kolmogorov): inverse energy cascade, forward **enstrophy** cascade E(k)∝k⁻³ → cite Kraichnan 1967, Boffetta&Ecke 2012. Do not say "energy to high k".

---

## Map

```
[1] DECOMPOSE: per PDE term        ── core method, novel axis      ── THESIS+WS
      advection / diffusion / forcing / divergence
      compare: aggregate vs term-split   metric: AUC, Cohen's d, FPR@95
      Q: are terms separable + differentially excited by Re? (PROVEN — diff dominant, adv secondary; signal anti-symmetric across Re diagonal)

[2] DECOMPOSE: term × spectral band ── which term dominates per Re shift ── THESIS+WS
      builds anna-ood B3 bands × [1]    output: dominating-term map across 5×5 op/Re
      this powers the "what broke" diagnosis story

[3] REGIME: online self-calibration ── running stats, no ID set       ── THESIS
      build running μ,σ → outlier alert; band/term localizes fault
      lit: streaming anomaly det.; frame vs DiverseNO (ensemble) — ours single-model, residual-native
      → research refs THEN build

[4] REGIME: offline ID baseline     ── calibrated detector, beat CP-PRE ── THESIS+WS(core)
      z-score vs train-Re ID baseline (= current anna-ood); + conformal wrapper = coverage
      BASELINE TO BEAT: Gopakumar CP-PRE (aggregate L1, unsigned, never framed OOD)
      ground: Certificate of Trust (Raonic) — p(x) fails NS; ours task-aware for free
      survival exp: CP-PRE vs our score on 5×5, per-cell AUC+d
      → research refs THEN build

[5] GENERALITY: FNO → UNO/U-Net     ── tests GENERAL claim, not band   ── THESIS (WS if lands)
      U-Net has no spectral cutoff → band mechanism may not transfer; per-term should
      separates universal principle from FNO instantiation; industry-relevant (CNN surrogates std)

[6] APPLICATION: trust fingerprint  ── infra [1-4] on fine-tuned instances ── THESIS
      avg per-term performance = fingerprint; same detector re-pointed (honest: application not new result)
   [6.1] opt: loss-weight adaption during training ── Phase 3, NOT detection chapter

[7] STRETCH: PRISMA-style warm-start diffusion ── residual decomp as spectral guidance ── if time
      generalize toward other architectures
```

## Scope

- **Workshop paper** = [1]+[2]+[4] only. One claim: term/band-resolved residual beats aggregate (CP-PRE) as post-hoc OOD detector w/ direction under Re shift. Tight, named baseline, data mostly exists.
- **Thesis** = [1]–[6]. [4] calibrated core, [3] online regime, [5] generality, [2] diagnosis layer throughout.
- **Cut from detection chapter**: [6.1] loss-weight adaption → Phase 3 (act on trigger), different contribution.
- **Industry card**: trust signals on physical models; [5]+[6] are the demo.

## Open / verify before claiming

- [x] ~~[1] per-term separability — run before building framing~~
- [x] ~~[2] base tensors computed — raw |FFT2|² (N,S,S,T) per term, 5×5 grid~~
- [x] ~~[2] shell_power → Cohen's d per k~~ — located: NOT B3, broadband + k=1
- [x] ~~CP-PRE head-to-head~~ — banded residual 0.835 > aggregate 0.603 (held-out); but MOOT, see 02.06 (input-trivial)
- [x] ~~random/adjacent-band control~~ — banding is workhorse (+0.18), term-split secondary (+0.05)
- [x] input-triviality control — raw IC high-k separates Re (AUC 0.95) > residual: Re-sweep too easy
- [ ] Li&Yang dual-branch PDF (S0955799724005551) — grep spectrum/FFT/residual before "first NO OOD" language
- [ ] enstrophy vs energy: check anna-ood Fig.1 label (E(k) vs k²E(k))

refs detail → ood-literature.md, metric-literature.md

thoughts from self: 
- generalist opeator vs fine tuning operator needs reserach: who needs tta'd operators? use case predates research
- diffusion prisma like warm start with pde loss split; no warm start if thats easier

---

## 31.05.2026

[1] confirmed. 5×5 Cohen's d heatmap (op Re × test Re, per term).

**Findings:** diff dominant OOD axis (|d|up to 4.7), adv secondary. wt weakest — FNO temporal output stays smooth even OOD. Signal anti-symmetric: low→high Re positive (terms rise), high→low Re negative (terms collapse). op500↔op1000 nearly blind (adjacent Re, d≈0).

**Interpretation:** diff blows up because ν mismatch amplifies fine-scale Laplacian errors; adv rises because turbulent IC structure propagates into nonlinear term. wt flat because architecture smoothly interpolates temporal derivative regardless of Re. Original bet (wt/spin dominant) wrong — spatial terms carry the signal.

**Next:** [2] band-resolve diff and adv to locate where in frequency space the signal lives, then [4] AUC vs CP-PRE.

---

## 02.06.2026 — Re-sweep with snapshot-ICs is input-trivial (IC leaks Re)

Control chain (each peeled a layer):
1. [2] per-k signal located: NOT in B3[9-16]; broadband + k=1 spike.
2. k=1 spike = ENERGY CONFOUND not model error: per-traj k1-diff power vs rollout-L2 Pearson=0.20 (weakest shell). Re500 scare ruled out (input spectra smooth across Re; cross-pattern = Re500 large-scale-energy max, file co-gen Apr7). Re1000=_indep file, off energy trend → IC-distribution confound flag.
3. CP-PRE head-to-head (held-out ID split, AUC): aggregate(Du-scalar) 0.603 | Du-shells banded 0.781 (+0.18 BANDING) | term×band 0.835 (+0.05 term) | diff high[17-32] 0.865 best. CP-PRE one-sided below random in 8/10 collapse cells (unsigned aggregate flips direction). Cohen's-d parity ≠ AUC parity (multivariate separability). → contribution = spectral BANDING; term-split = secondary/attribution.
4. KILLER — model-free baseline: raw IC high-k[17-32] AUC=0.954 (far 0.991, adj 0.897) > residual 0.865. Deployable (KFDataset NO normalization; x=frame0). Flat over t → ICs are developed-turbulence snapshots, so high-k(IC) encodes Re (input leaks label).
   [protocol note: input baseline = raw two-sample + max(s,1-s) direction; residual = held-out z-score. Matched comparison → fold into AUC-heatmap stress test.]

VERDICT: Re-sweep with snapshot-ICs is input-trivial — high-k(IC) leaks Re, so residual STRICTLY DOMINATED by free high-pass filter. NOT a flaw of Re-shift per se: an IC distribution that does not leak Re would make Re-mismatch detection non-trivial again. A hand-crafted input summary (high-k energy) suffices here, so this benchmark never exercises the regime where input-density methods fail — too easy for input-based detection. Earlier "banded residual beats CP-PRE" true but MOOT (task trivially input-solved).

L2/trust framing weak (~0.5): full-traj L2 poor target in chaos (Lyapunov phase decorrelation). Cutoff notch k=8 (d≈0 at n_modes) = real FNO fingerprint, curiosity not contribution here.

INFRA BUILT (reusable any benchmark): ResidualAnalysis (shell_power, cohens_d), shell_features cache (op,test,traj,term,shell), AUC harness, held-out protocol.

---

## Next up

- [x] clean memory: remove stale local files, verify outputs/ gitignored on server
- [x] compute base tensors (server, one-time):
      saved raw |FFT2|² (N=40, S=128, S=128, T=63) float32 per term — no pre-aggregation.
      all downstream banding/shell/Cohen's d runs locally from .npz.
      5×5 grid done. nm16-noskip ablation: per_mode_residual done; base tensor TODO.
- [x] ResidualAnalysis class: load(), time_mean(), shell_power() → (N, k_max+1), cohens_d() → (k_max+1,)

---

## Next steps — pivot to input-in-distribution OOD

Requirement: OOD where deployable input (IC) does NOT separate → input baselines fail, residual must carry signal (the regime input-density methods are for).

1. FORCING/BC PERTURBATION at fixed Re (most principled). Matched Re → matched dissipation spectrum → input genuinely cannot see the shift → clean "p(IC) in-distribution, model wrong". Needs new data gen, but it is the test that actually exercises the residual.
2. TEMPORAL EXTRAPOLATION (cheap but AT RISK). Same IC dist, predict past training horizon; p(IC) in-dist by construction. RISK: wt (temporal residual) was FLAT under Re-shift — FNO interpolates time smoothly; may not track extrapolation error. Also operator is one-shot space-time (kf_forward: IC→fixed-T block), not autoregressive → verify it even produces detectable degradation. VERIFY wt-not-flat before committing.
3. STRUCTURAL OOD: different flow class, matched input stats. Most work.

Stress tests (carry to chosen benchmark): ROC + FPR@95; per-cell AUC heatmap with MATCHED protocols (residual vs input-baseline vs CP-PRE); conformal wrapper [4]; UNO/U-Net [5] generality — only meaningful once benchmark is input-nontrivial (else U-Net inherits the same input-triviality).
