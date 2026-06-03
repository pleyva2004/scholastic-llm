# Pre-Registration — Content-test (causal: starvation vs intrinsic)

**APPEND-ONLY.** Logged before the content adapter is trained or probed; committed
before training. Design in [`README.md`](README.md).

- **Logged:** 2026-06-03, before `adapters/content-cf` and `results/` exist.
- **Setup:** base `qwen2.5-7b-mlx-q8`; voice `sft-v2-iter400` (reference); **new**
  `content-cf` = SFT LoRA (rank 8, 16 layers, lr 1e-5, ~400 iters — voice parity)
  on plain-language counterfactual data asserting the FALSE member of 10 cloze
  pairs (10 held out). Probe = same exact cloze identity `ΔM=⟨δh,Δu⟩`, frozen U,
  null std 0.017, bar 0.033. `Δu = u_true − u_false`.

## Hypotheses
- **M_grad / M_capacity** (favored): content gradient → δh acquires content
  component. Trained `cos(δh,Δu)` strongly **negative**, margins **flip**; held-out
  `cos≈0`.
- **M_subspace** (rival): LoRA can't write content → trained margins don't flip,
  `cos≈0` even on trained.

## Decision rule (pre-committed)
M_grad confirmed iff: trained `|cos(δh,Δu)| > bar` AND its CI excludes 0 AND trained
|cos| ≫ held-out |cos| (CIs separate) AND ≥ 6/10 trained margins flip sign. If
margins flip but cos stays ≈0 → the "non-content channel" cell (investigate). If
nothing flips → M_subspace. Ambiguous (flips on some, cos marginal) → report
discriminator with CI, suggestive only.

---

## Pre-registered prediction (signed, with confidence)

**I predict M_grad / M_capacity (~75%).**
- **Trained pairs:** `cos(δh,Δu)` strongly **negative — predict < −0.10** (clears
  the −0.033 bar with non-overlapping CI); **≥ 8/10 margins flip** (ΔM strongly
  negative, the taught false token wins). Confidence: high on the flip (counterfactual
  SFT with guaranteed gradient), medium-high on cos < −0.10.
- **Held-out pairs:** `cos(δh,Δu) ≈ 0` (within null); margins essentially unchanged.
  Rank-8 on 10 facts **memorizes, does not generalize**. Confidence: medium-high.
- **Fingerprints:** content δh **pairwise cos low** (predict < 0.15, vs voice 0.36);
  **marker-boost z ≈ 0** (vs voice +1.0). Mirror image of the voice adapter.
- **Net claim if confirmed:** voice training left understanding intact **because the
  content was already there (no gradient), not because LoRA cannot touch content.**
  The orthogonality is **gradient-routed (starvation)**, not intrinsic.

**Falsifier → M_subspace (~20%):** trained margins **fail to flip** and `cos≈0`
even on taught pairs → low-rank adaptation on these modules structurally cannot write
the content-readout direction. My whole gradient-routing mechanism would be wrong and
I'd owe an account of how voice changed outputs so much without touching this subspace.

**Most-surprising (~5%):** margins flip on trained pairs but `cos(δh,Δu)≈0` (content
written through a non-Δu channel — e.g. suppressing the true token via context rather
than boosting the false-token readout direction). Would force a rethink of "the cloze
margin reads the content axis."

---

## Outcomes (appended post-hoc only)

### 2026-06-03 — Run (results commit follows this block)

Content adapter trained to voice parity (rank 8, scale 10, 16 layers, lr 1e-5,
400 iters; train loss 0.08). null std 0.017, bar 0.033.

| condition | split | cos(δh,Δu) [CI] | ΔM | flipped | cos_style |
|---|---|---|---|---|---|
| voice | trained | +0.001 [−.008,+.011] | +0.02 | 0/10 | +0.023 |
| voice | held-out | −0.019 [−.031,−.007] | −1.22 | 0/10 | +0.020 |
| **content-cf** | **trained** | **−0.107 [−.149,−.067]** | **−11.5** | **9/10** | −0.008 |
| content-cf | held-out | −0.012 [−.025,+.001] | −0.70 | 0/10 | +0.000 |

fingerprints — voice: pairwise cos 0.363, marker-z **+1.03**; content-cf: pairwise
cos **0.380**, marker-z **−0.32**.

**VERDICT — prediction CONFIRMED → M_grad / M_capacity.** With a content gradient,
the same rank-8 LoRA rotates δh onto the content axis (trained cos −0.107, clears the
bar, CI excludes 0; ~9× the held-out/voice values) and **flips 9/10 trained margins**
(ΔM −11.5). Held-out cos ≈ 0, 0/10 flipped → **memorized, did not generalize**
(M_capacity). Marker-z ≈ 0 (content write, not style — mirror of voice's +1.03).
**Conclusion: voice's orthogonality was GRADIENT-ROUTED (starvation) — the content
was already correct, so there was no content gradient — NOT a structural inability of
LoRA to write content.**

**Calibration:**
- ✅ M_grad headline (~75% → confirmed); trained margins flip (predicted ≥8/10, got
  9/10); held-out ≈ 0 (memorized, not generalized); marker-z ≈ 0; trained cos < −0.10
  (got −0.107, just clears).
- ❌ **Fingerprint disconfirmed:** I predicted content δh would have **low** pairwise
  cos (< 0.15, "fact-specific"). It was **0.38** — *higher* than voice. Diagnosis: the
  content adapter ALSO carries a large shared component — a generic "answer
  plainly about Catholic theology" topic/format shift applied to every prompt — on top
  of the small fact-specific content rotation. So **pairwise cos does NOT discriminate
  content-write from style-write** (any topically-coherent SFT induces a shared
  direction). The clean discriminators are the **targeted** axes — `cos(δh,Δu)` for
  content and marker-z for register — which separated perfectly.
- **Honest magnitude note:** even the content adapter's δh is only `cos −0.107` from
  orthogonal — i.e. still ~99% of its norm is OFF the content axis. The content axis is
  a tiny slice of δh; it flips facts only because ‖δh‖‖Δu‖ are large (small-cos ×
  large-norm again). The voice-vs-content difference (0 vs −0.107, signed) is decisive,
  but in absolute terms content is a thin sliver of either adapter's write.
