# Findings — the orthogonality was gradient starvation, not a wall

**Causal question:** voice-SFT's δh was orthogonal to content. Was that **forced**
by there being no content gradient (the base already knew the doctrine), or
**intrinsic** to low-rank adaptation? Answer: **starvation.** Given a content
gradient, the same rank-8 LoRA writes content.

## The decisive contrast

Same base, same LoRA config (rank 8 / 16 layers / lr 1e-5), same cloze probe — only
the training data differs (plain counterfactual facts vs scholastic register):

| adapter | trained-pair `cos(δh,Δu)` | trained margins flipped | marker-z (register) |
|---|---|---|---|
| **voice** (register data) | +0.001 (≈0) | 0 / 10 | **+1.03** |
| **content-cf** (counterfactual facts) | **−0.107** (clears bar) | **9 / 10** | −0.32 (≈0) |

- **Content is reachable.** Trained to assert the false member, the content adapter
  rotated δh onto the content axis (`cos −0.107`, CI [−0.149,−0.067], ~9× voice/held-out)
  and **flipped 9/10 trained margins** (ΔM −11.5). A rank-8 LoRA can absolutely write
  the content-readout direction — when the gradient points there.
- **So voice's orthogonality was starvation.** Voice training left truth-margins
  untouched **because the content was already correct** (no content gradient), not
  because LoRA can't touch content. The earlier H0 is now *causally* explained.
- **Clean dissociation.** Content adapter: content-aligned, register-neutral
  (marker-z ≈ 0). Voice adapter: register-aligned (marker-z +1.0), content-neutral.
  The two write to **different axes**, selected by where their gradient had signal.
- **Memorized, not generalized (M_capacity).** Held-out cos ≈ 0, 0/10 flipped — the
  rank-8 adapter wrote the 10 taught facts and did **not** spill onto untrained content.

## Two honest refinements (diagnostic, not buried)

1. **My fingerprint prediction was wrong.** I expected content δh to be *fact-specific*
   → low pairwise cos. It was **0.38** — as high as voice's 0.36. Why: the content
   adapter also learned a large **shared** "answer-plainly-about-Catholic-theology"
   topic/format component on top of the small fact rotations. ⇒ **pairwise cos does not
   discriminate content-writes from style-writes** — any topically-coherent SFT induces
   a shared direction. The *targeted* axes (`cos(δh,Δu)`, marker-z) are the clean
   discriminators, and they separated perfectly.
2. **Content is a thin sliver even when you train for it.** Even the content adapter's
   δh is only `cos −0.107` off orthogonal — ~99% of its norm lies off the content axis.
   It flips facts purely because `‖δh‖‖Δu‖` are large (the same small-cos × large-norm
   leverage seen throughout). The voice-vs-content gap (0 vs −0.107, *signed*) is
   decisive for the verdict, but in absolute terms the content-readout direction is a
   tiny fraction of any adapter's write.

## The arc, closed

1. **Phase 3** (behavioral): voice ↑6×, truth-discrimination flat → **H0 (orthogonal)**.
2. **math-test** (geometric): `δh ⟂ Δu` at the readout (`cos≈0`, large `‖δh‖`).
3. **layer-frame** (robustness): orthogonal at **every** layer; the style vector is
   **frame-independent**.
4. **content-test** (causal): the orthogonality is **gradient-routed** — supply a
   content gradient and the same LoRA rotates onto content and flips facts. Voice
   didn't, because the content was already there.

**One-sentence mechanism:** a LoRA writes wherever its cross-entropy gradient has
signal; scholastic-register targets put all the signal on output-form (the base
already had the doctrine), so voice-SFT wrote a style vector orthogonal to content —
and when you instead put the signal on content, the very same architecture writes
content.

## Boundaries / next
- Counterfactual injection maximizes content gradient; a true-direction, base-weak
  fact set would test the same claim without teaching falsehoods (expected: same
  geometry, smaller ΔM).
- M_capacity (no generalization at rank 8) is a *capacity* statement for this rank/data
  size; higher rank or more facts might begin to generalize — untested.
- Final-layer, teacher-forced, single-token cloze regime throughout.
