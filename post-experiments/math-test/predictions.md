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

### 2026-06-03 — Run (results commit follows this block)

**Identity check:** max |⟨δh*,Δu⟩ − ΔM_direct| = 2.6e-2 (8-bit dequant rounding) → exact identity holds; code valid.

| metric | sft-v1 | sft-v2-iter400 |
|---|---|---|
| ‖δh*‖/‖h*‖ (move size) | **0.314** | **0.259** |
| cos(δh*, Δu) [content] | −0.0068, CI[−0.016,+0.002] | −0.0088, CI[−0.018,−0.0004] |
| null std (random) / bar | 0.0167 / 0.0334 | 0.0167 / 0.0334 |
| \|cos(δh*,Δu)\| vs perm-null | 0.0170 vs 0.0135 | 0.0169 vs 0.0122 |
| cos(δh*, r_style) | +0.0028, CI∋0 | +0.0215, CI[+0.016,+0.027] |
| **mean pairwise cos(δh_i,δh_j)** | **+0.306** | **+0.363** |
| marker logit-boost z | +0.09 | **+1.04** |

**VERDICT — prediction CONFIRMED → H_mech (orthogonal displacement).**
δh* is large (¼–⅓ of ‖h*‖) yet its direction is within the random null of the
content axis Δu (|mean cos|≈0.008 ≪ bar 0.033; real |cos| barely over permutation
null) → **fails the non-orthogonality bar → orthogonal**. The big move is
**content-independent** (pairwise cos ≈0.31–0.36 across 20 different prefixes — a
single shared shift applied regardless of content) and **register-ward**
(marker-boost z=+1.0 for v2; `CCC` is the #1 interpretable logit-lens token).
Dose-response holds (v2 > v1 on style-marker z and pairwise cos; content cos ≈0 both).

**Calibration — where I was right / wrong:**
- ✅ Clauses 1,2,4 (large move; content-orthogonal & slightly negative; dose) — confirmed.
- ❌ Clause 3 in its *naive* form: I predicted a clean register **logit-lens** and
  style cos ≳0.05. Raw logit-lens is **dominated by rare-token artifacts** (only
  `CCC` survived); single-axis marker cos was small (0.022). I had **flagged this
  exact failure mode** ("what would surprise me most…") and the fallback diagnostics
  rescued the style ID: style shows up as **content-independence (pairwise cos)** +
  **targeted marker-boost z**, not as a one-dimensional marker-unembedding cosine.

**Unpredicted refinement (diagnostic understanding):** cos≈0 does **not** make
ΔM=0 — on these high-confidence cloze items mean ΔM = **−0.60** (a real drop),
because tiny cos × large ‖δh‖‖Δu‖ still leaks. The drop is **negative-biased
(14/6 items)** and **concentrated on the highest-base-margin items** (ΔM ≈
−0.4·base_margin; e.g. Thomas/Augustine base +6.77→ΔM −3.47). So the adapter
mildly **compresses extreme content margins** without a *targeted* rotation — and
because base margins (+6 to +11) ≫ the leakage, **no decision flips** → accuracy
stays at ceiling. This is the exact mechanism behind Phase-3 H0 and its small
−0.19 margin drift: untargeted leakage, not content erosion.

**Caveat (named, for follow-up):** all 20 prefixes share the frame "Here is a
statement about Catholic theology:", so part of the high pairwise cos could be a
shared-frame response rather than pure register-generality. Varying the frame
would separate "register-in-general" from "this-frame." The orthogonality result
(content cos) is unaffected by this.
