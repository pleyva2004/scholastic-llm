# Findings — the math checks out: the adapter moves ⊥ to content

**Claim tested.** Phase 3 found voice ↑↑ with doctrinal margin flat (H0). The
proposed mechanism: the LoRA update displaces the residual stream a lot, but in a
direction nearly *orthogonal* to the content-truth axis that minimal-pair margins
read. We measured the vector directly via the exact cloze identity
`ΔM = ⟨δh*, Δu⟩`. **Confirmed (pre-registered, H_mech).**

## The numbers (sft-v2-iter400 vs base, 20 single-token cloze pairs)

| quantity | value | meaning |
|---|---|---|
| `‖δh*‖ / ‖h*‖` | **0.26** | the adapter moves the final hidden state by ¼ of its norm — a **large** move |
| `cos(δh*, Δu)` | **−0.009** (CI [−.018,−.0004]) | within the random null (std 0.017; bar 0.033) → **orthogonal** to the content axis |
| `\|cos(δh*,Δu)\|` vs permutation null | 0.017 vs 0.012 | no content-*specific* alignment |
| mean pairwise `cos(δh_i, δh_j)` | **+0.36** | the move is **content-independent** — one shared shift across 20 different prefixes |
| marker logit-boost (z vs random tokens) | **+1.0** | that shared shift is **register-ward** (`CCC` = #1 logit-lens token) |
| identity error | 2.6e-2 | `⟨δh*,Δu⟩` = directly-measured `ΔM` → code valid |

Dose-response: every style signal is larger for sft-v2 than sft-v1 (pairwise cos
0.36 vs 0.31; marker-z 1.0 vs 0.1), while content cos ≈ 0 for both.

## What this proves, mechanistically

1. **The displacement is real and big** — not a null adapter. ¼–⅓ of `‖h*‖`.
2. **Its direction avoids the content axis.** `cos(δh*, Δu) ≈ 0`, indistinguishable
   from a random direction and from an unrelated pair's content axis. The adapter
   performs **no targeted rotation** of the true-vs-false representation.
3. **It is one shared, content-independent shift.** Across 20 unrelated theological
   prefixes the displacements point the *same way* (`cos ≈ 0.36`). The adapter does
   roughly the same thing to the residual stream regardless of *what* is being said
   — the operational signature of a **style/register** transform, not a content one.
4. **That shared shift leans register-ward** — it boosts scholastic discourse /
   Catechism-citation tokens (`CCC` top of the logit-lens; marker-boost z ≈ 1).

Together: voice-SFT learned a **style vector**, applied content-independently, that
lives in a subspace ≈ orthogonal to the directions encoding doctrinal truth. That
is the geometric reason voice rose 6× while truth-discrimination did not.

## The subtle part (diagnostic understanding)

`cos ≈ 0` does **not** mean `ΔM = 0`. On these high-confidence cloze items the mean
margin change is **−0.60**, and it is **negative-biased** (14 of 20) and
**concentrated on the most confident items** (`ΔM ≈ −0.4·base_margin`; e.g.
*Thomas/Augustine* base **+6.77 → ΔM −3.47**). Reason: a tiny cosine times two large
norms (`‖δh*‖`, `‖Δu‖`) still leaks a real number. So the adapter **mildly
compresses extreme content margins** as a side effect — but it is *untargeted*
(direction orthogonal), and because base margins (+6 … +11) dwarf the leakage,
**no decision flips**. That is exactly Phase-3's signature: accuracy at ceiling,
margin drifting slightly down (−0.19 there, larger here only because cloze facts are
higher-confidence). H0 and its margin drift are the **same** phenomenon —
orthogonal leakage, not content erosion (H_override is *not* supported: the cosine
never clears the bar).

## Honest limits

- **Logit-lens was the wrong tool**, as I flagged in advance: raw `U·mean δh*` is
  dominated by rare-token artifacts; only `CCC` read cleanly. The robust style
  evidence is content-independence (pairwise cos) + targeted marker-boost, not the
  lens. A single marker-unembedding axis (`r_style`) was also too crude (cos 0.02).
- **Shared-frame confound:** all prefixes begin "Here is a statement about Catholic
  theology:"; part of the high pairwise cos may be a shared response to that frame
  rather than pure register-generality. Varying the frame separates the two; it does
  **not** affect the content-orthogonality result.
- **Scope:** final-layer, teacher-forced, single-token cloze. The exact identity
  only holds there. A layer-resolved version (where does δh enter the residual
  stream?) and a free-generation logit-attribution are the natural next steps.

**Bottom line.** The hand-wavy inner-product argument was right where it counts:
`δh* ⟂ Δu`. Voice training writes a large, content-independent, register-ward vector
into a subspace orthogonal to the model's truth axis — so it changes how the model
*sounds* without rotating *what it judges true*.
