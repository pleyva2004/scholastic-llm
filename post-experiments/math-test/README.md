# Math-test: does the adapter's displacement avoid the content axis?

*Testing the mechanism proposed for the Phase-3 H0 result. Built to the
`CONCEPTUAL.md` / `HYPOTHESIS.md` / `EXPERIMENT.md` bar; pre-registration in
[`predictions.md`](predictions.md).*

## 1. Restatement (mechanism, not surface)

Phase 3 found voice ↑↑ while doctrinal margin stayed flat (H0, orthogonality).
The proposed *why*: the LoRA update perturbs the residual stream a lot, but along
a **style direction** nearly **orthogonal** to the **content-contrast direction**
that minimal-pair margins read. That is a geometric claim about a vector. This
experiment measures the vector.

## 2. The exact identity this rides on

For a **single-token cloze pair** (shared prefix `P`, true/false next tokens
`x_+, x_-`), the margin is — with **no approximation**, because the log-partition
is shared and cancels:

```
M = log p(x+|P) − log p(x−|P) = <h*, u_x+> − <h*, u_x−> = <h*, Δu>
```

`h*` = final normed hidden state at the last prefix position; `u_v` = row `v` of
the (frozen, 8-bit→dequantized) unembedding `U`; `Δu = u_x+ − u_x−`. The LoRA
touches only attention/MLP on 16 layers (rank 8) — **not** `lm_head` (verified:
`base U == sft U`). So `U` and `Δu` are identical across conditions, and the
adapter's entire effect on the margin is the **exact** projection

```
ΔM = <δh*, Δu>,   δh* = h*_sft − h*_base.
```

*(Validated numerically in the spike: ΔM_direct = −0.6861 vs <δh*,Δu> = −0.6873.)*

## 3. Hypotheses (mechanism + rivals; full form in predictions.md)

- **H_mech (orthogonal displacement).** `‖δh*‖` is large, but `δh*` lies in a
  style subspace ≈ orthogonal to `Δu`: `cos(δh*, Δu) ≈ 0` (within the
  random-direction null), while `cos(δh*, r_style)` is well above null. The
  margin barely moves because a large vector has near-zero projection on `Δu`.
- **H_entangle (rival).** `δh*` has a systematic truth-ward component:
  `mean cos(δh*, Δu) > 0` beyond the null, and real (item,Δu) pairing aligns more
  than a permuted pairing. This is what H1 would have needed; we expect it refuted.
- **H_override (rival).** `δh*` is systematically *anti*-aligned with truth:
  `mean cos(δh*, Δu) < 0` beyond the null — the truth-erosion reading of the
  small −0.19 margin drift seen in Phase 3.

## 4. Metrics (discriminator + stable companion)

- **PRIMARY (discriminator):** `mean_i cos(δh*_i, Δu_i)` over the 20 pairs —
  signed, separates H_mech (≈0) / H_entangle (>0) / H_override (<0).
- **COMPANION (stable):** `‖δh*‖/‖h*‖` (is the move large?) and
  `cos(δh*, r_style)` + variance share along style vs content (is the move
  style-shaped?).
- **Diagnostic (logit-lens):** decode `U · mean_i δh*_i` — the across-prompt
  *common* component of the adapter's shift — to its top-boosted tokens. H_mech
  predicts register/discourse tokens, not content words.
- **Reference direction `r_style`** = mean unembedding of scholastic discourse
  markers (" therefore", " hence", " thus", " Objection", " Whether", …) minus
  the global mean `ū`. Built independently of the content pairs (not circular).

## 5. Controls / cleanliness

- **Random-direction null:** `cos(δh*_i, random)` for many randoms → null band
  `≈ N(0, 1/d)`, std `≈ 1/√3584 ≈ 0.017`. Content cos is judged against this, not
  against literal zero.
- **Permutation null:** `cos(δh*_i, Δu_j)`, `i≠j` → "does δh align with its *own*
  pair's truth axis more than with an unrelated one?" Isolates content-specific
  alignment from generic content-direction geometry.
- **Norm sanity:** if `‖δh*‖≈0`, "orthogonal" is vacuous — the companion guards this.
- **Exact-identity check:** assert `<δh*,Δu> == ΔM_direct` per item (code is correct).
- **Minimality:** one manipulated variable (adapter on/off); same prefixes, same
  `U`, same code. Dose row (sft-v1) is the only extra condition.

## 6. Pre-committed joint interpretation table

| `mean cos(δh*,Δu)` | `cos(δh*, r_style)` | reading |
|---|---|---|
| within ±2·null-std, CI∋0 | ≫ null | **H_mech** — orthogonal displacement (math confirmed) |
| > bar, real > perm-null | (any) | **H_entangle** — adapter pushes content toward truth |
| < −bar, real > perm-null | (any) | **H_override** — adapter pushes content away from truth |
| ≈ 0 | ≈ 0 | δh inert at final layer / measurement broken — debug, don't interpret |

Threshold: content alignment is "non-orthogonal" only if `|mean cos| > 2×null-std`
**and** its across-item bootstrap CI excludes 0 **and** real-pairing alignment beats
the permutation null (CIs separate). Else → orthogonality confirmed. Ambiguous
(one criterion met): report the discriminator value with its CI and call it
suggestive, not a verdict.

Reproduce: `python build_cloze.py && python math_probe.py` (seconds).
