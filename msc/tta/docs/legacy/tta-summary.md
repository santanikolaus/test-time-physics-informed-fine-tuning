---
title: "TTA Results: op100 $\\to$ Re500 Kolmogorov Flow"
date: "2026-06-23"
geometry: margin=2.5cm
fontsize: 10pt
papersize: a4
mainfont: "STIX Two Text"
mathfont: "STIX Two Math"
monofont: "Menlo"
header-includes:
  - \usepackage{unicode-math}
  - \usepackage{booktabs}
  - \setlength{\parskip}{4pt}
---

**Setting.** FNO pretrained at Re100 (`op100`, $n_\text{modes}=8$, 128$^2$), adapted label-free to Re500 at inference via physics-residual TTA ($\mathcal{L} = \text{pde} + 5 \cdot \text{ic}$, oracle $\nu$). Metric: held-out $k \leq 7$ relL2 vs GT, time-split early / late. Reference ceiling: op500 (supervised, same arch) late 0.473.

---

## 1. The main finding

Physics-residual TTA **solves early, hard-walls late** — and the wall is real:

| | early $k \leq 7$ | late $k \leq 7$ |
|---|:---:|:---:|
| op100 source (baseline) | 0.271 | 0.678 |
| op300 best free ID op | 0.234 | 0.527 |
| **op500 supervised ceiling** | **0.248** | **0.473** |
| **TTA best (N=160, lr=3e-4)** | **0.175** | **0.581** |

Early is below op500 supervised and monotone in data size. Late plateaus at 0.58–0.59 for $N \geq 120$, $\sigma(\text{seeds}) \approx 0.001$. A band-limited $k \leq 7$ residual objective (Exp#1) moves late by $-0.022$ only ($0.578 \to 0.556$).

**The wall is invariant to every lever tried on the instantaneous objective**: data size, learning rate, seed, and residual band.

---

## 2. What the wall is not

Five axes ruled out:

**Capacity (spatial, temporal, architectural).** Modes $n_8 \to n_{32}$, width $h_{32} \to h_{128}$: late $\Delta \leq 0.017$. GT projected onto $K_t = 8$ time-modes holds at 0.135 $\ll$ wall 0.55 — the GT late field lives in 8 modes; the model error does not. UNet with a single spatial-FNO bottleneck reaches the FNO late wall at both 128$^2$ (0.478 vs 0.472) and 256$^2$ (0.577 vs 0.571): the wall is architecture-invariant.

**Chaos / IC-unpredictability.** Gate 2 (phase-spread capstone): ensemble phase dispersion $D_\text{late} = 0.07$–$0.30$, growing $\approx 1.3\times$, does not saturate toward 1 even at perturbation $\delta = 0.30$ exceeding the operator's early error. Late phase is learnable, not intrinsically chaotic. Paired op-quality at 256$^2$ ($n=100$): op500 $>$ op300 late 61/100 ($p = 0.035$), not 50/50 — the wall has a systematic reducible component.

**Self-consistency as a trainable surrogate.** Self-inconsistency rank-tracks GT late (Spearman 0.40–0.65) but is non-causal: halving the inconsistency leaves GT late flat or worse. Transductive dose-response: inconsistency $\downarrow$ on 100% of instances, GT late $\uparrow$ on 100%. Oracle distillation with data anchoring saturates at the supervised ceiling $\approx 0.48$–$0.52$, not below it.

---

## 3. What the error looks like

Per-mode decomposition: $|\hat{u} - \hat{g}|^2 = (a_u - a_g)^2 + 2 a_u a_g (1 - \cos\Delta\phi)$.
Collapse-independent per-shell: $R_k = \Sigma|\hat{u}|^2 / \Sigma|\hat{g}|^2$ (energy ratio), $A_k$ = GT-energy-weighted phase cosine.

Both $R_k$ and $A_k$ fall monotonically with $k$ — they co-degrade, not independently:

| $k$ | $R_k$ (OOD 128$^2$) | $A_k$ (OOD 128$^2$) |
|:---:|:---:|:---:|
| 1 | 0.83 | 1.00 |
| 3 | 0.54 | 0.84 |
| 5 | 0.33 | 0.56 |
| 7 | 0.16 | 0.26 |

Aggregate $k \leq 7$: amplitude-leaning (relL2 amp 0.281 $>$ phase 0.243, op500). Phase fraction grows with wavenumber and with time (early $\to$ late at every shell). Phase concentrated $k = 2$–$5$ (fraction 0.40–0.59); $k=0$–$1$ purely amplitude.

**One mechanism: low-pass hedging under positional uncertainty.** Phase-blind magnitude correction raises late error ($0.581 \to 0.631$) — energy cannot be fixed while phase is wrong. Loss reweighting (time axis, band equalization) moves late $< 0.006$ and non-monotone.

---

## 4. The one lever that works

Oracle re-anchoring — restart from a true GT mid-trajectory field every $s$ frames — breaks the wall **on 100% of instances** across all operators:

| operator | oneshot late | oracle late | $\Delta$ |
|---|:---:|:---:|:---:|
| op100, 128$^2$, stride 16 | 0.678 | **0.361** | $-0.310$ |
| op500, 128$^2$, stride 16 | 0.473 | **0.283** | $-0.185$ |
| fno16, 256$^2$, stride 16 | 0.571 | **0.230** | $-0.328$ |

The recoverable signal lives entirely in $k \leq 7$, concentrated $k = 2$–$7$ (closure gate kc-sweep). Correct $k > 7$ buys nothing: hi_oracle $\approx$ model at $k_c = 7$. Solver closure confirms this is physics: true-solver low-$k$ divergence from 100%-corrupted high-$k$ = $0.06$–$0.12$ $\ll$ wall $0.47$.

Own-field restart compounds error for far-OOD ops (op100 model: $0.912$–$0.771$, all worse than one-shot $0.678$). fno16 own-field is modestly positive (80–90% win) due to lower early error (0.164 vs 0.279), but oracle gap $\approx 0.27$ remains.

**The gap between oracle (0.230) and own-field (0.525) restart is 0.295 — the remaining addressable loss.**

---

## 5. Proposed direction

Self-consistency cannot close the oracle gap (non-causal). The missing piece is a training objective that lets the model see correct mid-field phase states.

**Multi-step rollout training with segmented GT anchoring.** Train the 256$^2$ FNO from mid-trajectory GT snapshots at stride $s \in \{8, 16, 32\}$, not only from IC. Each segment starts from a true field and is supervised $T'$ steps forward (data + pde loss). Training-objective change only — no architecture change, no label requirement at inference.

*Mechanistic basis:* given a recent true field, the operator already predicts far below the one-shot wall (fno16: $0.571 \to 0.230$). The wall is stale-IC accumulation that the one-shot objective never penalises. Segmented training exposes the model to correct phase-state anchors at multiple trajectory points, directly targeting the low-pass hedging mechanism.

*Discriminator:* late $k \leq 7 < 0.47$ without oracle access at inference confirms trajectory accumulation as the mechanism. If the wall persists, the failure mode is the Re-shift domain gap rather than trajectory length — testable with a segmented-start arm on Re500 pool directly.
