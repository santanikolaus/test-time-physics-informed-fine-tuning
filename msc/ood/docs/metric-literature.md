# OOD Detection Metrics

Goal: establish the right reporting bundle for a thresholded z-score OOD detector,
continuous parametric shift, n=40 per class.

---

## Recommended Bundle

| Metric | Purpose | Status |
|--------|---------|--------|
| AUROC | threshold-free global ranking | done |
| FPR@95TPR | OOD slip-through at 95% ID recall — operational low-FPR point | add |
| AUPR | useful if OOD is rare positive class | add |
| Spearman ρ(z-score, \|error\|) | does score track PDE prediction degradation? | add |
| AUROC grouped by \|ΔRe\| | turns 5×5 matrix into a severity curve — IS-OOD style | add |
| Sign accuracy of z-score | directional variant evaluation: correct sign vs train_Re | add |
| n-MeRCI | regression-specific error-uncertainty correlation (Mouli 2024) | optional |
| Bootstrap 95% CI on all above | n=40 requires it | done |

**Small-sample protocol:** paired bootstrap or permutation test when comparing two detectors on the same test set; TOST equivalence test if claiming adjacent-Re pairs are at physical detection limit (not detector failure).

**5×5 matrix summary:** report three numbers — mean off-diagonal AUROC, adjacent-gap AUROC (\|ΔRe\|=1), severity slope (AUROC vs \|ΔRe\| bin).

---

## TODO

- [ ] Moukari et al. 2019 (IROS) — original n-MeRCI source; Mouli 2024 borrows it; bibliography must cite origin not just Mouli
- [ ] OpenOOD v1.5 benchmark paper — source for AUROC/AUPR/FPR@95 as standard trio; cite for metric definitions
- [ ] "Recent Advances in OOD Detection" survey — find exact reference; defines standard evaluation trio
- [ ] TOST / equivalence testing reference — needed to formally support "at physical detection limit" claim for weak adjacent-Re pairs
- [ ] "Rethinking the Evaluation of OOD Detection: A Sorites Paradox" — IS-OOD: evaluates detectors across ordered shift severities; closest match to train_Re × test_Re matrix evaluation style
- [ ] "An Information Theoretical View for Out-Of-Distribution Detection" — information-theoretic framing of detection limits; language for "physical limit vs detector failure" on weak adjacent-Re pairs
- [ ] "Layer-Wise Distance Awareness for OOD Detection" — distance/direction-aware scoring; conceptual grounding for signed B3 z-score as directional regime probe

---

## Cross-references

- [88_Mouli] — n-MeRCI (Eq. 7): correlation between uncertainty estimate and prediction error; regression-specific
- [89_certificate] — ACC, FPR, FDR, AUROC (Table 1): standard binary OOD classification suite for scientific ML
