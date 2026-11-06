# One-Shot Latent Prediction: Architecture Routes
## Research note — June 2026

Sources: 2505.09063 (Flexi-VAE/DCP), 2604.06881 (MENO), 2402.17853 (LNS), 2410.20100 (LNOP), 2605.25413 (AFNO); internal results: tta-results.md 2026-06-22/23/24

---

## Does Route A gate Route B? (added 2026-06-24)

### Route A is already run — not a future experiment

The 16² experiment (FNO·spectral + UNet·spectral, Re100, FFT-crop GT) was completed on 2026-06-24. Results from tta-results.md:

| arm | late k≤7 | early k≤7 |
|---|:--:|:--:|
| FNO · spectral 16² | 0.506 | 0.146 |
| UNet · spectral 16² | 0.422 | 0.076 |
| FNO 128² baseline | 0.543 | 0.175 |

Wall survives 16²: 0.506 ≈ 0.543. The config confirms what was removed — n_modes=8 = Nyquist at 16², so the FNO ran full-rank on k≤7 with zero high-k contamination. Collapsing the grid 64×, removing all high-k modes, and running in the overfit/high-capacity-per-data regime did not lower the in-band late error. The per-shell pattern is identical: late error rises monotonically to k7≈0.93–0.98 (near-fully wrong), early ≪ late (ratio 3.5–5.6).

### Is 16² comparable to a learned encoder/decoder?

No — structurally different. The 16² model operates entirely in FFT-truncated pixel space with no encoding. The training loss, the objective, and the output space are identical to the 128² model; only the data resolution changes. A Route B encoder/decoder sits in a nonlinear learned representation that is not constrained to Fourier modes. These are different things.

**However**, the question is whether that structural difference can escape the wall. Two pieces of evidence say no, if the objective stays L2-in-pixel-space:

1. **Coordinate-change argument (geometric):** a smooth invertible encoder is a smooth change of coordinates on the same k≤7 dynamics. Lyapunov exponents — the rate at which initially close trajectories diverge — are invariant under smooth coordinate changes. If the wall is chaos-driven, a learned latent cannot escape it.

2. **Gate 2 / D_late rebuttal (from file, 2026-06-22 capstone):** D_late = 0.07–0.30, growing only 1.3–1.4× from δ=0 to δ=0.30. This is NOT the chaos signature (D_late → 1 at large δ). Late phase does not intrinsically decorrelate. One-shot SHOULD predict late-from-IC. The conclusion in the file is explicit: the 0.55 wall is an **optimization/objective failure**, not chaos. Oracle restart (chain gate: late 0.230 from GT mid-traj field) and lo_oracle (GT k≤7 + own k>7 → late 0.230, matching oracle) both confirm the dynamics are fine — given the true field, the operator predicts the next window well.

These two points do not contradict. The wall is not "chaos makes prediction physically impossible." The wall is "the one-shot IC→late map, trained with L2 over all T frames, is poorly optimized for late k≤7" — L2 forces hedging under positional uncertainty. This is an objective failure, not a representation failure. A learned encoder with the same L2 objective inherits the same failure mode.

### Why the user's intuition is correct — and grounded

The user's point: "maybe kolmogorov is not solvable with l2 or late is not solvable in pixel space."

This is supported by the band-energy experiment (tta-results.md 2026-06-23): adding the energy term raised R(k≤7) from ≈0.60 toward ≈0.70 — restoring ~15% of collapsed in-band energy, making the prediction physically more correct — but late L2 moved only 0.543→0.538. Restoring physical energy barely registers in L2. The angle term plateaus at late ≈ 0.52 even at 20× weight. **L2 at late time is dominated by phase/position error that the model hedges against; it does not measure whether the field is physically reasonable.**

### What this means for Route B

Route B (learned encoder + one-shot propagator with L2 loss decoded to pixel space) is gated by the 16² result if and only if the loss is L2 trajectory comparison in pixel space. The 16² experiment shows that representation changes (Fourier vs arbitrary grid, high-k included vs excluded) do not move the wall — so a richer learned representation with the same L2 objective will not either.

Route B is NOT gated if the objective changes — e.g., if the latent architecture is trained to match energy spectra, phase statistics, or distributions rather than point trajectories. But that objective change does not require a new architecture: the same change can be applied to the existing FNO, which is simpler.

### Conclusion

The latent architecture route and the objective rethink are orthogonal. The evidence — 16² wall survival + D_late ≠ chaos + L2-insensitivity to physical energy restoration — all point at the **objective** as the lever, not the representation. Build Route B only if targeting a non-L2 objective that specifically benefits from the latent structure (e.g., distribution matching in latent space). Otherwise the wall follows you into the encoder.

The prior conclusion from tta-roadmap: lever = training objective on existing FNO. This is consistent with all three evidence threads above.

---

## What "latent one-shot" means precisely

The full chain is:

```
IC (B,S,S)  →  Encoder  →  z₀ (B,C,h,h)  →  Propagator  →  Z (B,C,h,h,T or T×calls)  →  Decoder  →  traj (B,S,S,T)
```

The word "one-shot" means the propagator never feeds a predicted frame back as input. Recurrency lives in the **propagator**, not the decoder.

---

## Where recurrency is / is not

**Recurrent (avoid):**
```
z_t = f(z_{t-1})       ← each latent depends on the previous one
```
This is what LNS, DIANO, LNOP, and MENO's coarse arm all do. It is autoregression in latent space — same error-accumulation mechanism, just cheaper per step.

**Non-recurrent — two realizations:**

(i) All-channels: operator emits all T latent frames as output channels in one call.
```
z₀ → F(z₀) → Z[…, 1..T]         ← one call, T frames simultaneously
```
This is exactly what the current one-shot FNO already does in physical space — T output channels. The only change is that F acts in a compressed latent rather than at full resolution.

(ii) Time-query (DCP, Flexi-VAE 2505.09063): operator takes z₀ and a time offset τ, returns z_τ for that τ only.
```
z_τ = P([z₀ || τ])               ← T independent calls, each using z₀ only
```
Each call is independent — no frame's output is another frame's input. This is what the Direct Concatenation Propagator does (fully connected network on concatenated [z₀, ζ, τ]). Called T times but non-recurrent because the input is always z₀, never z_{t-1}. The "flexible" property is that τ can be any horizon not seen at training.

Per-frame **decode** in either case is also non-recurrent: each z_t is decoded independently. Not a loop.

---

## Route A — Fixed spectral codec (k≤7 FFT crop to 16²)

```
IC (B,S,S)  →  FFT crop k≤7  →  latent (B,16,16)   [no learned params]
            →  one-shot FNO at 16² (type i above)   [existing architecture]
            →  iFFT zero-pad  →  trajectory (B,S,S,T) [no learned params; high-k = 0]
```

**The encode step is a fixed FFT crop — zero learned parameters in codec.** The 16² grid retains exactly k≤7 modes (Nyquist at 16² = mode 8). The "latent" is just the low-pass band.

**What "already exists":** the 16² FFT-cropped data does exist (`ns_re100_res16_spectral`, `ns_re500_res16_spectral`). The one-shot FNO architecture runs at any resolution. No two-stage training hazard because the codec is not learned.

**What does NOT yet exist:** a 16² config, training run, or number. This needs a new config (n_modes ≤ 8, grid 16²) and an actual training run. "Near-free" means no new code — not that it is already trained.

**The go/no-go test:** does dedicated 16² (k≤7-only) training break the late relL2 wall of 0.54? The 16² model's relL2 IS the k≤7 band error — it has no other modes. Compare against 0.54, not the oracle-restart 0.230 (different setup).

Run Re100 first (wall is 0.543 for op100/Re100 at 128²); Re500 is a second run once the in-distribution question is answered. Note: 16² data is FFT-cropped from 256², baseline is 128² — k≤7 content is near resolution-independent so the comparison holds to first order.

**What Route A tests mechanistically:** the closure gate showed high-k modes are bystanders — lo_oracle (GT k≤7 + own k>7) fully recovers oracle error. The hypothesis is that including high-k in the loss geometry dilutes gradient signal for the failing k≤7 band. Route A removes high-k entirely. If the wall holds in the 16² model, the mechanism is not capacity allocation — it is intrinsic to k≤7 prediction, and the full latent direction is a dead end.

---

## Route B — Learned 2D encoder + one-shot propagator (Flexi-VAE style, no VAE)

```
IC (B,1,S,S)  →  2D encoder (CNN or FNO, no T-axis)  →  z₀ (B,C,h,h)
              →  propagator (type i or ii, non-recurrent)  →  Z (B,C,h,h,T)
              →  2D decoder per frame  →  traj (B,S,S,T)
[trained end-to-end; encoder never frozen]
```

**Propagator options:**

(i) All-T channels: natural extension of the current one-shot FNO; latent just adds an encode/decode wrapping layer.

(ii) Time-query (DCP): z_τ = P([z₀ || ζ || τ]) via FC network. T independent forward passes but fully parallel; no recurrency. More flexible (arbitrary τ at inference), at the cost of T decoder calls vs one.

**Critical constraint from LNOP (2410.20100):** freezing the encoder then training the propagator gives ~8× worse error (0.22 vs 0.027). Reconstruction-optimal representations are not propagation-optimal. Must train end-to-end from scratch.

**Blocker from current UNet:** `PeriodicConv3d` pads the T-axis, so the existing encoder cannot take a single frame as input. Restructuring requires replacing 3D periodic convolutions with 2D in the encoder path. This is the main implementation cost — estimated 2-3 days.

**What Route B adds over A:** the encoder learns a representation optimally shaped for the propagator's task, not forced into Fourier basis. It may find latent structure that better captures phase relationships. Unknown whether this helps vs A for KF at Re=500.

---

## Route C — Flexi-VAE (full)

As Route B but encoder is a VAE (KL regularization, reparameterization trick). Paper also uses Jacobian spectral analysis of the decoder to diagnose latent geometry and explain prediction quality. Only tested on 1D Burgers and 2D advection — no NS, no turbulence results exist.

**Overkill for PoC.** Adds VAE loss tuning and Jacobian diagnostics. 1-2 weeks. Thesis-grade novelty if applied to KF turbulence.

---

## Recommended sequence

| Step | Route | Cost | Decision gate |
|---|---|---|---|
| 1 | A (16², Re100) | new config + 1 training run | late k≤7 < 0.54? |
| 2 | A (16², Re500) | 1 training run | OOD consistent? |
| 3 | B (if A breaks wall) | 2-3 days, encoder restructuring | beats A? |
| 4 | C (if B validates) | 1-2 weeks | thesis contribution |

Route A is cheap enough that the go/no-go test costs less than designing Route B up front. If A stalls at 0.54, the wall follows into the coarse band — objective-allocation is not the lever — and the entire latent direction is dropped before spending a day on it.
