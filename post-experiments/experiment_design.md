# Experiment design

*Assessed against `EXPERIMENT.md`: discrimination, construct validity,
counterfactual cleanliness, falsifiability, a declared primary discriminator with
a stable companion, and minimality. Aim: highest information per compute-hour
that can actually finish.*

## The one comparison that carries the weight

Everything reduces to a single, minimal, well-controlled contrast:

> **Same** base model, **same** probe items, **same** scoring code — vary
> **only** the adapter. Read the within-pair log-probability margin.

That is the counterfactual-clean manipulation: the *only* thing that differs
between base and SFT is `ΔW`, so any change in the margin is caused by `ΔW`.

## Conditions (minimality — only the adapter varies)

| Cond | Model | Adapter | Role |
|---|---|---|---|
| **C0** | qwen2.5-7b-mlx-q8 | — | base / counterfactual reference |
| **C1** | "" | scholastic-v1 | dose point (Phase-1 SFT) |
| **C2** | "" | scholastic-v2-iter400 | **headline SFT** (best checkpoint) |
| **C3** | "" | scholastic-v3-dpo | DPO extension (voice negative-result) |

Headline test: **C0 vs C2**. Dose–response: C0→C1→C2. Mechanism specificity:
C2 vs C3.

## Construct: what we measure and why it bears on the claim

For each minimal pair we present a fixed, **neutral** context
("Here is a statement about Catholic theology:") and score the two members as
assistant continuations. The neutral (non-evaluative) frame matters: we measure
the model's *intrinsic* probability of the statement, not its reaction to a
"which is correct?" cue that the SFT model might answer differently for stylistic
reasons.

- **Scoring rule:** sum the teacher-forced token log-probs of the statement,
  divide by token count → **per-token logprob** (length-normalized).
- **Why per-token, not raw sum:** the de-risking spike showed matched-vocabulary
  pairs can tokenize to *different lengths* ("Grace…" 5 tok vs "Merit…" 6 tok).
  Raw sum favors the shorter side. Per-token normalizes most of it; and the
  *key inferential quantity is the paired SFT−base delta*, in which any
  residual per-item length bias is a constant that cancels.

This bears on "understanding" because the within-pair difference is, by
construction, blind to register (both members are equally scholastic) and
isolates sensitivity to the *relation*.

## Metrics — declared discriminator + stable companion

Per `EXPERIMENT.md`, a single metric says *whether* something moved; a pair says
*what kind*. Both reported with uncertainty regardless of which moves.

- **PRIMARY (discriminator):** **within-pair discrimination accuracy** on the
  doctrinal set = fraction of pairs with `pertok_lp(correct) > pertok_lp(incorrect)`.
  Binary per item, higher variance, sharply separates H1 (> base) from H0
  (= base). This is the headline; declared **before** data.
- **COMPANION (stable):** **mean per-token margin**
  `Δ = pertok_lp(correct) − pertok_lp(incorrect)`, averaged over the doctrinal
  set. Continuous, lower variance, *signed* — its sign is what distinguishes H0
  (Δ_SFT−Δ_base ≈ 0) from M_override (< 0).
- **Positive control (separate axis):** the **voice rubric** (`src/scholastic/
  rubric.py`: scholastic register, Augustinian voice, CCC grounding, structure)
  scored on free-text generations from the *same* models on a small open-question
  set. Confirms the intervention is potent.
- **Diagnostics (instrumentation):** per-category margins, per-item table,
  anchor-set accuracy, and a **label-shuffle control**.

## Counterfactual cleanliness / controls

- **Baseline parity:** the base receives the *identical* probe items and the
  *identical* scoring code — there are no hyperparameters to tune on either side
  (scoring is deterministic), so parity is exact by construction.
- **Trivial control (floor):** the 5 **factual-anchor** pairs (authorship,
  epithet) — base must already score these ≈ ceiling. If it doesn't, the metric
  is broken, not the model.
- **Chance control:** a **label-shuffle** run (randomly relabel which member is
  "correct," fixed seed) must collapse accuracy to ≈ 50% and margin to ≈ 0. If a
  "signal" survives shuffling, the scoring pipeline leaks.
- **Determinism:** teacher-forced scoring has no sampling → no seed variance on
  the logprob numbers. "Noise" is **across-item variance**, handled by
  bootstrap CIs and a **paired** test (base and SFT see identical items, so we
  compare per-item, the most powerful and cleanest contrast).

## Falsifiability & pre-committed analysis (the joint interpretation table)

Written **before** any data exists (mirrored into `predictions.md`):

| Doctrinal margin (SFT−base) | Voice rubric | Reading |
|---|---|---|
| **↑** beyond noise bar | ↑ | **H1 — entanglement**: voice training also taught content |
| **≈ 0** (within bar) | ↑ | **H0 — orthogonality**: voice is a surface skin |
| **↓** beyond noise bar | ↑ | **M_override**: register learned at the cost of truth-tracking |
| ≈ 0 | ≈ 0 | null/bug — intervention inert (positive control failed) → debug, do not interpret |

**Pre-committed success threshold (decision rule).** The doctrinal SFT−base
effect is **confirmed** only if it exceeds **2× the across-item std** of the
paired per-item difference **and** the base vs. SFT 95% bootstrap CIs (on the
companion margin) do **not** overlap. Marginal → reported as **suggestive only**,
not a verdict. **Ambiguous-case pre-commitment:** if accuracy moves but the
companion margin's CIs overlap (or vice versa), we report the discriminator's
result as the verdict and flag the other as an unexplained secondary needing
follow-up — we do **not** retro-promote whichever looks better.

## Compute budget (why this finishes)

Teacher-forced scoring is a *single forward pass per statement* — no
autoregressive generation. 39 pairs × 2 members × 4 conditions ≈ 312 short
forward passes + ~4 model loads. The voice control adds ~8 prompts × 4 models of
real generation. Total: minutes, not hours. High information per compute-hour.
