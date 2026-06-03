# Findings — orthogonality is depth-wide, and the style vector is frame-independent

Two robustness tests on the math-test mechanism. Both pre-registered predictions
**confirmed**. No training; pure forward-pass geometry on base vs sft-v2-iter400
over 20 single-token cloze pairs.

## Part A — content-orthogonality holds at *every* layer (H_throughout)

The adapter sits on the last 16 of 28 blocks. At every one of them:

| | raw `cos(δh^(ℓ),Δu)` | post-norm `cos(dn^(ℓ),Δu)` | bar |
|---|---|---|---|
| max over LoRA layers 12–27 | **0.0097** | **0.0088** | 0.033 |

- `‖δh^(ℓ)‖/‖h^(ℓ)‖`: exactly **0** on frozen blocks 0–11 (built-in negative
  control — a nonzero cos there would be a bug), then **0.04 → 0.39** across 12–27.
- The displacement grows large through the stack yet stays orthogonal to the
  content axis at **every depth**, in both raw and readout (post-norm) space.
- **Style write is late:** `cos(δh^(ℓ), r_style)` ≈ 0 through blk18, rising to a
  **peak +0.077 at blk26** — register/output-form is shaped in the last few blocks,
  exactly where you'd expect it.

**The subtlety, diagnosed (not buried).** My pre-registered *co-metric* — the
logit-lens partial margin `ΔM^(ℓ)` — went **positive mid-stack** (+0.23 at blk26)
before crashing to −0.60 at the readout, which naively reads as "content moves then
cancels" (H_cancel). It does **not**: the *primary* discriminator (raw `cos`) was ≈0
at those layers, and the tiebreaker I added (post-norm displacement cosine) is ≤0.004
there too. So the ΔM bump is a **norm-magnitude wobble** — RMSNorm rescales the
base's large pre-existing content projection — not a directional content write. It is
the same `small-cos × large-norm` leakage the math-test found at the readout
(`ΔM=−0.60` with `cos≈0`), now resolved layer-by-layer: the *direction* is orthogonal
throughout; the *margin* fluctuates only because the norm couples magnitudes.

> Methods lesson: at intermediate layers the logit-lens margin conflates scale and
> direction (RMSNorm). The cosine is the clean construct; report both, trust the
> direction.

## Part B — the shared style direction is frame-independent (H_style)

The math-test's "one content-independent shift" (pairwise cos 0.36) could have been a
response to the shared frame *"Here is a statement about Catholic theology:"*. Render
the same 20 statements under 5 different frames (incl. bare, no-frame):

- **Within-frame** pairwise cos is ~0.34 for **all five** frames (0.36/0.36/0.34/0.34/0.33)
  — the effect was never special to the theology frame.
- **Per-frame mean-δh** vectors are nearly the same vector across frames:
  cos **0.78–0.96** (min 0.78 ≫ 0.4).
- **Cross-frame × cross-statement** pairwise cos = **+0.31** — when *both* the frame
  and the statement differ, the displacement still points the same way (~19× null,
  barely below the within-frame 0.34).

→ The adapter applies essentially **the same displacement regardless of frame or
content** — a genuine global register vector. Confound killed; it barely moved the
number (0.36 → 0.31), so it was real, not the frame.

## Bottom line

The math-test's two caveats are closed. Voice-SFT writes a **large, frame-independent
register vector that is orthogonal to the content-truth axis at every layer it
touches** — it never rotates content at any depth; the only trace it leaves on truth
margins is an untargeted, norm-coupled magnitude wobble that nets to a small
compression and flips no decisions. This is the clean geometric form of Phase-3's H0.

**Still open (the causal question):** orthogonality here is *observed*; whether it is
*forced* by gradient starvation (base already correct) or *intrinsic* to low-rank
adaptation is the next test — train a plain-language content adapter on base-wrong
facts and check whether *its* δh becomes content-aligned (staged in
`../` discussion; needs a training run).
