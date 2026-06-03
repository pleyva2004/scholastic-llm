# Layer-resolved orthogonality + frame-control

*Robustness follow-up to `../math-test/`. No training — pure forward-pass
geometry. Hardens the two soft spots in the math-test:*
1. *orthogonality was measured only at the **final** normed layer (where the
   cloze identity is exact);*
2. *the shared-direction cosine 0.36 may be inflated by the shared frame
   "Here is a statement about Catholic theology:".*

Design to the `CONCEPTUAL/HYPOTHESIS/EXPERIMENT` bar; pre-registration in
[`predictions.md`](predictions.md).

## Setup facts (from the extraction spike — tooling, not outcome)
- Qwen2.5-7B = **28 transformer blocks**; the adapter (rank 8) is on the **last
  16 (blocks 12–27)**. Verified: `‖δh^(ℓ)‖` is **exactly 0** for blocks 0–11
  (frozen) then rises monotonically to ~0.39 of `‖h‖` by block 27.
- Per-layer residual at the final position is reconstructed by replaying
  `embed → blocks → norm` (final matches `model.model(...)` to <1e-3).
- `Δu = u_true − u_false` lives in the (frozen, dequantized) unembedding; at
  intermediate layers we read it via the **logit-lens** (project the residual
  through the final norm + `U`). Caveat: logit-lens skips the remaining blocks'
  processing — it is the standard diagnostic, used as such, not as exact.

---

## Part A — Layer-resolved orthogonality

**Question (mechanism):** does the adapter write a content-aligned component at
*any* depth that merely **cancels** by the readout, or is it orthogonal to the
content axis **throughout**?

- **H_throughout (favored):** `cos(δh^(ℓ), Δu) ≈ 0` (within the random null) at
  every LoRA layer and the final norm; the per-layer logit-lens margin
  contribution `ΔM^(ℓ)` never bumps up. The style write, by contrast, concentrates
  in **later** layers.
- **H_cancel (rival):** `cos(δh^(ℓ), Δu)` is clearly **>0** at some intermediate
  layer and `ΔM^(ℓ)` rises then returns to ≈0 — "orthogonal at readout" hiding
  content movement that cancels downstream.

**Metrics:** primary discriminator `max_ℓ |cos(δh^(ℓ),Δu)|` and the `ΔM^(ℓ)`
trajectory (H_throughout: flat≈0/monotone; H_cancel: a bump). Companions:
`‖δh^(ℓ)‖/‖h^(ℓ)‖` (injection profile — known to rise 0→0.39) and
`cos(δh^(ℓ), r_style)` (where the register write lands).

## Part B — Frame-control

**Question:** is the content-independent shared direction (pairwise cos 0.36)
genuine **register-generality**, or an artifact of the shared frame?

Render each of the 20 cloze statements under **5 frames** (incl. a bare
no-frame). Compute final-layer `δh` for every (statement × frame).

- **H_style (favored):** the shared shift is **frame-independent** — per-frame
  mean-`δh` vectors are well aligned across frames (`cos > 0.4`), and the
  strictest **cross-frame × cross-statement** pairwise cos stays clearly above
  null. The 0.36 is mostly real register-generality.
- **H_frame (rival):** per-frame mean-`δh` are weakly aligned and the
  cross-frame×cross-statement cos collapses toward null → the shared direction was
  the frame.

**Metrics:** primary discriminator = `cos` between per-frame mean-`δh` vectors
(frame-independence). Companions: within-frame pairwise cos (reproduce 0.36 per
frame), cross-everything pairwise cos, and the variance share along the grand-mean
direction.

---

## Controls / cleanliness (both parts)
- **Random-direction null:** `1/√d ≈ 0.017`; "orthogonal" = within ±2·null.
- **Permutation null** for content cos (carried from math-test).
- **Frozen-layer sanity:** blocks 0–11 give `δh≡0` (built-in negative control —
  any nonzero cos there would mean a bug).
- **Identity check** at the final layer still holds (`⟨δh,Δu⟩ = ΔM_direct`).
- **Minimality:** one model pair (base vs sft-v2-iter400); the only varied factors
  are *depth* (Part A) and *frame* (Part B). No new training, same probe set.

Reproduce: `python layer_frame_probe.py` (~2 min).
