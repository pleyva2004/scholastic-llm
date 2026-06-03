# Content-test: is the orthogonality forced (starvation) or intrinsic (subspace)?

*The causal experiment. Phase 3 + math-test + layer-frame showed voice-SFT's δh is
orthogonal to the content axis at every layer. But the base already **knew** the
content (94% acc, margins +6…+11), so there was no content gradient. Was the
orthogonality **caused** by that starvation, or is low-rank adaptation **structurally
unable** to write content? This decides it.*

## The manipulation

Train a LoRA — **same config as voice (rank 8, 16 layers, lr 1e-5)**, the only
deliberate difference being the **data** — to assert the **false** member of half the
cloze pairs, in **plain language with zero scholastic markers**. Teaching the model
a relation it is confidently against is a **maximal, guaranteed content gradient**.
Then measure whether *this* adapter's δh rotates onto the content axis.

- **Trained pairs (10):** the adapter is taught the false completion.
- **Held-out pairs (10):** never trained — tests spillover / generalization.

## Hypotheses

- **M_grad / M_capacity (favored):** displacement direction is set by where the CE
  gradient has signal. With a content gradient, δh acquires a content component.
  → on **trained** pairs, `cos(δh,Δu)` goes **strongly negative** (δh points toward
  the taught false token, i.e. along −Δu) and the **margins flip**; on **held-out**
  pairs `cos≈0` (rank-8 memorizes, doesn't generalize). ⇒ content IS reachable;
  voice's orthogonality was **starvation**.
- **M_subspace (rival):** low-rank adaptation on these modules is structurally
  confined to an output-form subspace and **cannot** write content. → even on trained
  pairs the margins **don't flip** and `cos(δh,Δu)≈0`. ⇒ a deeper, more surprising
  result; my gradient-routing account would be wrong.

## Metrics (discriminator + companion + fingerprints)

- **PRIMARY (discriminator):** `cos(δh,Δu)` on **trained** pairs (content adapter).
  M_grad: clears the −bar; M_subspace: within null. Contrast with voice (≈0) and
  with the **held-out** pairs (≈0 under both).
- **COMPANION / sentinel:** `ΔM` on trained pairs — did the margins actually flip
  (sign change)? If training didn't move the behavior, don't interpret the geometry.
- **Fingerprints (mechanism-implied):** content δh **pairwise cos** should be **low**
  (fact-specific writes), unlike voice's 0.36 (one shared style vector); content
  **marker-boost z ≈ 0** (no register write — mirror image of voice).

## Controls / cleanliness

- **Baseline parity (stated):** content adapter = identical LoRA rank/layers/lr as
  voice; comparable iters (≈400). Only the data differs.
- **Held-out pairs:** the within-experiment control isolating "wrote the taught
  axis" from "rotated content in general."
- **Style isolation:** training text rubric ≈ 0 (verified) → any δh is content, not
  style.
- **Same nulls / identity check** as math-test (random null 0.017, bar 0.033).
- **Sentinel:** content adapter must flip ≥ majority of trained-pair margins, else
  training failed.

## Pre-committed joint interpretation table

| trained `cos(δh,Δu)` | trained margins flip | held-out `cos` | reading |
|---|---|---|---|
| strongly < −bar | yes | ≈ 0 | **M_grad / M_capacity** — content reachable, gradient-routed, not generalized |
| strongly < −bar | yes | also < −bar | **M_grad (generalizing)** — unlikely at rank 8 |
| ≈ 0 | **no** | ≈ 0 | **M_subspace** — LoRA structurally cannot write content |
| ≈ 0 | yes | ≈ 0 | margins flipped via a non-content channel ⇒ investigate (would surprise me) |

Reproduce: `build_content_data.py` → train (≈7 min) → `content_probe.py` (seconds).
