# Proof sketch — blur as the L2 optimum under phase uncertainty

Reference doc for the `6_sim2sic.md` entry of 2026-07-29. Not a journal: no dated entries,
updated in place when the argument changes.

## Claim

The amplitude deficit γ<1 the one-shot FNO shows in every band and regime is the exact
minimiser of its training objective, not a defect of the trained operator — hence not
removable by better L2 fitting.

Motivation: Anna's Option-2 constraint (a pure reconstruction loss at TTA reproduces the same
averaging that caused the deficit) and Lippe (PDE-Refiner, arXiv:2308.05732) Table 4's
loss-vs-metric tradeoff.

## Derivation

Per Fourier mode, GT and prediction in polar form, `F_g = a_g·e^{iφ_g}`, `F_p = a_p·e^{iφ_p}`,
with `Δφ = φ_p − φ_g`:

```
|F_p − F_g|² = a_p² + a_g² − 2·a_p·a_g·cos Δφ
```

At lead time t a deterministic one-shot operator faces a distribution over the true phase
(chaotic divergence from the IC). Taking the expectation over that distribution at fixed `a_g`
gives the L2 risk as a function of the amplitude the operator chooses:

```
R(a_p) = E|F_p − F_g|² = a_p² + a_g² − 2·a_p·a_g·E[cos Δφ]

dR/da_p   = 2a_p − 2a_g·E[cos Δφ] = 0   ⟹   a_p* = a_g·E[cos Δφ]
d²R/da_p² = 2 > 0                        ⟹   minimum
```

With `ρ := E[cos Δφ]` the phase coherence at that lead time and `γ := a_p/a_g` the amplitude
ratio:

```
γ* = ρ
```

The L2-optimal amplitude is the GT amplitude shrunk by exactly the achievable phase
correlation. Pooling over a Chebyshev band with energy weights leaves the algebra intact and
reproduces the identity already printed by `msc/tta/report.py::print_decomp`:

```
rel_l2² = (1 − ρ²) + (γ − ρ)²
          └ irreducible ┘   └ excess over the optimal hedge ┘
```

Substituting `γ = ρ` kills the second term and leaves `rel_l2² = 1 − ρ²`: the floor imposed by
predictability alone.

## Hedging excess

```
H := γ − ρ
```

| H | reading |
|---|---|
| ≈ 0 | operator is L2-optimal for its own phase skill; the deficit is the objective's answer, not a fitting failure |
| < 0 | over-damped past the optimum; L2 left on the table, recoverable by better fitting |
| > 0 | over-energetic; already trading pointwise accuracy for spectral fidelity |

## Cost of restoring variance

An objective that matches spectra drives `γ → 1` at fixed ρ, so `(γ−ρ)² → (1−ρ)²` and

```
rel_l2²(γ=1) / rel_l2²(γ=ρ) = [(1−ρ²) + (1−ρ)²] / (1−ρ²) = 1 + (1−ρ)/(1+ρ) = 2/(1+ρ)
```

rel-L2 inflates by exactly `√(2/(1+ρ))`: +2.6% at ρ=0.9, +15.5% at ρ=0.5, +29% at ρ=0.2,
bounded by +41.4% as ρ→0. Full variance restoration is cheap in the dominant band and expensive
only where phase is already lost.

## Assumptions

Stated so falsification stays honest.

- **A1** `a_g` deterministic per band — exact at band level, since GT band energy is measured,
  not modelled.
- **A2** `a_p` freely choosable per mode — a finite-capacity network with shared parameters can
  only approach the per-mode optimum, so `H ≈ 0` is an upper bound on achievable calibration,
  not an identity the network must obey.
- **A3** `Δφ` independent of `a_p`, i.e. no amplitude–phase error correlation.
- **A4** the measured ρ (`eval.py::corr_pooled`) is the energy-weighted `E[cos Δφ]`, so the
  identity is exact rather than approximate over any band/frame window.

## Predictions

To be checked against the blur-horizon sweep (`report.py --reports decomp,horizon`, bands
`0-7,8-16,17-32,33-64`, 65 single-frame `--time-bins`, T_eff=65, test [270:300] n=30) over three
rows: `75prctl5` on Re100 (in-dist reference), `75prctl5` on Re500 data (the OOD regime TTA
targets), `nrow6j9l` on Re500 (native ceiling).

| # | prediction | falsified if | outcome |
|---|---|---|:--:|
| P1 | γ_k(t) tracks ρ_k(t): H small and invariant, varying no more than ±0.05 across bands, frames, both Re | H varies systematically with band, frame or regime beyond that band | **confirmed inside k≤7, falsified outside** |
| P2 | at a common threshold θ, `T_blur(k,θ) = T_corr(k,θ)` within the bootstrap CI — no blur horizon separate from the decorrelation horizon | the two horizons separate by more than the CI at any band | **falsified** |
| P3 | γ_k is a function of ρ_k alone — band index and Re enter only through ρ, all rows collapsing onto γ=ρ | γ_k depends on band index at fixed ρ_k | **falsified** |
| P4 | a pure L2/reconstruction TTA objective cannot push γ above ρ; variance is restorable only by first raising ρ | γ rises with ρ flat under an L2-only TTA objective | |
| P5 | a spectrum/variance-matching objective raises γ→1 at ~fixed ρ, inflating rel_l2 by `√(2/(1+ρ))` | rel-L2 inflation exceeds `√(2/(1+ρ))` at the measured ρ (would mean ρ degraded too, not a pure amplitude trade) | |

P1–P3 need the sweep only, no new training. P1 and P2 become *CI*-testable only once a
per-sample amplitude curve exists — `amp_ratio` pools the sample axis, so pooled γ carries no
bootstrap interval while `corr_curve`-derived ρ horizons do. P4–P5 need the TTA arms and gate
Anna's a/b split: P4 states her reconstruction-loss concern quantitatively, P5 bounds what the
alternative loss can cost.

## Outcome of P1–P3 — 2026-07-29, Re100 in-dist row (numbers in `6_sim2sic.md`)

**P1 — confirmed inside k≤7, falsified outside.** H = −0.056 in k0-7, frame-invariant to
±0.008 over all 65 frames, and −0.06 at the same band in the banked cross-Re run: within its
represented band the operator sits at a constant, Re-invariant, near-zero offset from the L2
optimum, exactly as derived. Outside: +0.020 (k8-16), +0.166 (k17-32), +0.782 (k33-64).
**A2 is what breaks, and it breaks at the spectral cutoff** — above `n_modes=8` the network has
no mode-mixing machinery, so amplitude there is not a chosen hedge.

**P2 — falsified.** T_blur ≠ T_corr at matched θ, non-overlapping CIs at every band, and the
order reverses: at θ=0.9 blur precedes corr in k0-7 (43.2 vs 59.5) and follows it in k17-32
(7.8 vs 2.7). Amplitude and phase horizons are independent measurements.

**P3 — falsified.** k33-64 has ρ=0.12 with γ=0.91 — near-GT energy at unrelated positions. γ is
not a function of ρ; band index carries information ρ does not.

Consequences, kept out of the journal entry because §2 bars outlook there:

- **Restoring γ→1 in k0-7 is nearly free.** At the measured ρ=0.9396, `√(2/(1+ρ))` = 1.0155 —
  +1.5% rel-L2 for full variance restoration in the represented band. The same figure is
  +11.9% at k8-16, +23.2% at k17-32, +33.4% at k33-64.
- **No TTA loss can reach above the cutoff.** The high-k deficit is not a hedging failure a
  variance-matching objective can correct; it is the absence of a representation. Raising
  `n_modes` is an architecture decision, not a loss decision — which bounds what Option 2 can
  deliver before any adaptation arm runs.

## Resolved tension

The prior open question — `H = +0.12` at the k8 shell, γ above ρ, which the optimum forbids —
resolves to cause (i): A2 binds. H > 0 appears only above the mode cutoff, grows monotonically
with band index, and is largest in the band the spectral convolution cannot touch at all. The
alternative explanation (pooled ρ diverging from per-mode `E[cos Δφ]` at a low-mode-count
shell) would not produce a monotone band ordering. Not yet checked against the cross-Re and
Re500-native rows.
