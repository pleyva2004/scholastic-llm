# Pre-Registration — Math-test: δh* decomposition

**APPEND-ONLY.** Logged before `math_probe.py` produces any result; committed in
its own commit *before* the results commit (git SHA = integrity proof). Design in
[`README.md`](README.md).

- **Logged:** 2026-06-03, before `results/` exists.
- **Setup:** base `qwen2.5-7b-mlx-q8`; adapters `scholastic-v1`,
  `scholastic-v2-iter400` (headline). Probe = 20 single-token cloze pairs.
  Final-layer normed hidden `h*` at last prefix token; `U` dequantized from 8-bit,
  frozen (verified `base U == sft U`). Deterministic; uncertainty via across-item
  bootstrap (10k) and a random-direction + permutation null.

## Hypotheses
- **H_mech** (orthogonal displacement): `‖δh*‖/‖h*‖` large; `cos(δh*,Δu)≈0`
  (within ±2·null-std, CI∋0); `cos(δh*,r_style)` ≫ null. Margin barely moves
  because a big vector projects ≈0 onto `Δu`.
- **H_entangle** (rival): `mean cos(δh*,Δu) > 0` beyond bar; real pairing >
  permutation null. (What H1 needed.)
- **H_override** (rival): `mean cos(δh*,Δu) < 0` beyond bar.

## Metric stack
- **Primary (discriminator):** `mean cos(δh*,Δu)` vs random-null (std≈0.017) and
  permutation null.
- **Companion:** `‖δh*‖/‖h*‖`; `cos(δh*,r_style)`; style/content variance share.
- **Diagnostic:** logit-lens of `mean δh*` → top-boosted tokens.

## Decision rule (pre-committed)
Content alignment is **non-orthogonal** only if `|mean cos(δh*,Δu)| > 2×null-std`
**AND** bootstrap CI excludes 0 **AND** real-pairing |cos| beats permutation-null
|cos| (CIs separate). Else **H_mech confirmed**. One-of criteria → suggestive only,
report the discriminator with CI, no verdict promotion.

---

## Pre-registered prediction (signed, with confidence)

**I predict H_mech.**

1. **Large move:** `‖δh*‖/‖h*‖` ≈ 0.2–0.4 across items (spike: 0.32). Confidence: high.
2. **Content-orthogonal (headline):** `mean cos(δh*,Δu)` lands **within ±0.03**
   (≤ ~2×null-std), bootstrap CI **includes 0**, and real ≈ permutation null →
   **does NOT clear the bar**. If anything **slightly negative** (consistent with
   the Phase-3 −0.19 aggregate margin drift and the `grace_nature` flip), but a
   small-cos×large-norm leakage with ~zero mean, **not** a systematic push.
   Confidence: medium-high (~70%) on "within bar"; low on the exact sign of the tiny residual.
3. **Style-aligned (companion):** `cos(δh*, r_style)` **positive and ≫ null**
   (predict mean ≳ 0.05, several× the 0.017 null). The **logit-lens of mean δh***
   decodes to **scholastic discourse / register tokens**, not content words.
   Confidence: medium on the cosine magnitude, medium-high on the logit-lens being
   register-dominated.
4. **Dose:** sft-v2 shows **larger** `‖δh*‖` and style cos than sft-v1; content cos
   ≈ 0 for **both**. Confidence: medium.

**Falsifier → H_entangle:** `mean cos(δh*,Δu)` clearly **> 0** beyond bar with real
> permutation null. That would mean voice-SFT *did* rotate content toward truth and
my Phase-3 mechanism story is wrong (I'd have to explain why the margin didn't then rise).

**Other informative surprise → H_override:** content cos clearly **< 0** beyond bar
→ the margin drift is a *systematic* truth-erosion, not leakage noise — a real cost
of register mimicry, upgrading the Phase-3 sub-threshold hint to a confirmed effect.

**What would surprise me most:** style cos *also* ≈ null (clause 3 fails) while
‖δh*‖ is large — meaning the adapter's big move is along *neither* my style axis nor
the content axis, i.e. my decomposition names the wrong basis. Then the logit-lens
becomes the diagnostic: whatever tokens `mean δh*` boosts is the real axis, and I
update my mechanism account to match.

---

## Outcomes (appended post-hoc only)

<!-- dated block after the run: norms, content cos ± CI vs nulls, style cos,
     logit-lens top tokens, identity check, verdict vs H_mech/H_entangle/H_override -->
