# Pre-Registration — Layer-resolved orthogonality + frame-control

**APPEND-ONLY.** Logged before `layer_frame_probe.py` produces hypothesis
outcomes; committed in its own commit before the results commit. Design in
[`README.md`](README.md). Setup facts (28 blocks, LoRA on blocks 12–27, δh≡0 on
0–11) came from a tooling spike, not from the metrics below.

- **Logged:** 2026-06-03, before `results/` exists.
- **Setup:** base `qwen2.5-7b-mlx-q8` vs `sft-v2-iter400`; 20 single-token cloze
  pairs (from `../math-test/cloze_pairs.jsonl`); 5 frames for Part B. Final-layer
  `U` dequantized & frozen; per-layer residual via embed→blocks→norm replay;
  intermediate content readout via logit-lens. Null std (random) = 1/√3584 ≈ 0.017,
  bar = 2·null = 0.0334. Uncertainty via across-item bootstrap (10k).

---

## Part A — Layer-resolved
- **H_throughout** (favored): `cos(δh^(ℓ),Δu)≈0` within ±bar at **every** LoRA
  layer (12–27) and the norm; `ΔM^(ℓ)` flat/monotone, **no positive bump**.
- **H_cancel** (rival): `cos(δh^(ℓ),Δu)>bar` at some intermediate ℓ with a
  `ΔM^(ℓ)` bump that cancels by readout.

**Prediction (signed):** **H_throughout.** `cos(δh^(ℓ),Δu)` stays within ±0.033
at all layers 12–27 and norm; `ΔM^(ℓ)` drifts **slightly negative, monotonically**
(consistent with the −0.6 final leakage), never positive. `‖δh^(ℓ)‖/‖h^(ℓ)‖` rises
0 (≤blk11) → ~0.39 (blk27). **`cos(δh^(ℓ),r_style)` rises with depth**, largest in
the last ~4 blocks (output-form is shaped late). Confidence: **medium-high** on
content-orthogonality at every layer; **medium** on the late-layer style
concentration.
**Falsifier → H_cancel:** any layer with `cos(δh^(ℓ),Δu)` clearly >bar and a
`ΔM^(ℓ)` rise-then-fall. That would mean the readout-level orthogonality hid
mid-stack content movement — I'd retract "orthogonal throughout."

## Part B — Frame-control
- **H_style** (favored): shared shift is frame-independent.
- **H_frame** (rival): the 0.36 was the frame.

**Prediction (signed):** **H_style, with partial frame contribution.** `cos`
between per-frame mean-`δh` vectors is **high (> 0.5)** → the register shift is
largely frame-independent. Within-frame pairwise cos ≈ **0.3** per frame
(reproduces the math-test). The strictest **cross-frame × cross-statement**
pairwise cos **drops from 0.36 but stays clearly above null (predict ~0.15–0.25)**.
Net: the shared direction is **mostly genuine register-generality, partly frame.**
Confidence: **medium.**
**Falsifier → H_frame:** per-frame mean-`δh` cos near 0 and cross-everything cos
collapsing to ≈null → the shared direction was an artifact of the shared frame, and
the math-test's "content-independent style vector" claim weakens to "frame
response."

## Decision rule (pre-committed)
- Layer orthogonality "holds at ℓ" iff `|mean cos(δh^(ℓ),Δu)| < bar` AND bootstrap
  CI ∋ 0. "Throughout" iff holds for all ℓ ∈ {12…27, norm}.
- Frame shift "frame-independent" iff **min** pairwise cos among per-frame mean-`δh`
  > 0.4 AND cross-everything pairwise cos > bar.
- **Ambiguous** (e.g. content cos clears bar at one isolated layer but `ΔM^(ℓ)`
  shows no cancel-bump): report the trajectory, call it suggestive, no verdict flip.

---

## Outcomes (appended post-hoc only)

<!-- dated block: per-layer cos_content/cos_style/ratio/ΔM trajectory; frame
     mean-δh cos matrix, within/cross pairwise cos; verdict vs H_throughout/H_cancel
     and H_style/H_frame -->
