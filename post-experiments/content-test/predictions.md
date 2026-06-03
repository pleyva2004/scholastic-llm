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

<!-- dated block: trained vs held-out cos(δh,Δu) ± CI, margin-flip count, pairwise
     cos, marker-z; content vs voice vs base; verdict M_grad / M_subspace -->
