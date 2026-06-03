# Voice ≠ Understanding — a one-pager

**Question.** We fine-tuned Qwen 2.5 7B to *speak* like Aquinas and Augustine.
Did that also change what the model *understands* about their ideas?

**Answer (pre-registered prediction confirmed).** No. Voice training is a
**surface skin**: it changed *how the model sounds* by ~6× while leaving its
representational grip on *which doctrine is true* essentially unchanged. Voice
and understanding are **orthogonal** under this recipe.

> Pre-registration: [`predictions.md`](predictions.md) (committed at `dacc230`,
> **before** any results existed). Framing: [`conceptual.md`](conceptual.md);
> hypotheses: [`hypothesis.md`](hypothesis.md); design:
> [`experiment_design.md`](experiment_design.md). Raw: [`results/`](results/).

---

## How "understanding" was made measurable

Free-text rubrics conflate the two things we want to separate — a model can
*sound* scholastic and be *wrong*. So understanding was operationalized as a
**necessary condition**: does the model's probability mass track the *truth of a
relation* rather than the *surface form of the sentence*?

We built **39 minimal pairs** (34 doctrinal + 5 factual-anchor) — two sentences
with **identical vocabulary**, differing only in the relation (a swapped term, a
reversed dependency, a negation):

- *"Grace is prior to merit."* vs *"Merit is prior to grace."*
- *"Evil is the privation of a due good."* vs *"…a positive substance created by God."*
- *"Christ is one divine person in two natures."* vs *"…two persons in one nature."*

Both members are equally scholastic, so any preference for scholastic-sounding
text **cancels within the pair**. What remains is sensitivity to the relation.
Score = mean per-token teacher-forced log-prob; **understanding = the model
assigns higher likelihood to the true member.**

## The design in one line

Same base, same items, same code — **vary only the adapter** (base → sft-v1 →
sft-v2-iter400 → dpo-v3). Primary discriminator: within-pair **accuracy**;
stable signed companion: **margin** Δ = logp(true) − logp(false). Positive
control: the project **voice rubric** on free-text, same models.

## Results

| Condition | Voice rubric (/12) | Doctrinal acc | Doctrinal margin [95% CI] |
|---|---:|---:|---|
| **base** | **1.33** | 0.941 | **+0.929** [+0.69, +1.17] |
| sft-v1 | 6.67 | 0.971 | +1.024 [+0.80, +1.26] |
| **sft-v2-iter400** | **7.83** | 0.912 | **+0.738** [+0.57, +0.90] |
| dpo-v3 | 7.33 | 0.941 | +0.836 [+0.64, +1.04] |

**Controls passed:** label-shuffle accuracy = **0.500** (no scoring leakage);
factual anchors **4/5 for every condition** (the one miss is a base
tokenization artifact, identical across conditions → no forgetting).

**The joint table — the actual verdict:**

| Doctrinal margin (SFT−base) | Voice | Reading |
|---|---|---|
| **≈ flat** (Δ small, fails confirmation bar) | **↑↑ (1.33→7.83)** | ✅ **H0 — orthogonality** |
| ↑ beyond bar | ↑ | H1 — entanglement *(not observed)* |
| ↓ beyond bar | ↑ | M_override *(only a sub-threshold local hint)* |

- **Voice moved enormously** (+488%), with a clean dose-response (base < v1 <
  v2) — the intervention is unquestionably potent.
- **Understanding did not.** The base already discriminates doctrine at **94%
  accuracy** with large positive margins — it is a competent theologian out of
  pretraining. SFT left accuracy at ceiling and did **not raise** the margin
  (refuting H1). The most-trained checkpoint *lowered* margin by −0.19 (paired CI
  excludes 0) but **did not clear the pre-committed confirmation bar** (base/SFT
  CIs overlap), so per the pre-registered ambiguous-case rule the verdict is the
  discriminator: **flat → H0**.

## Why this is the right answer mechanistically

A rank-8–16 LoRA on ~377 register-heavy targets has a cheap way to cut loss:
shift **output-form** directions (register tokens, citation cadence). It has no
pressure to reorganize **content** that the frozen base already represents well.
The minimal-pair probe strips exactly the surface channel the adapter moved, so
the content channel shows through unchanged. Voice and understanding live in
different parts of the model, and this recipe only touched one.

## Honest nuances (calibration)

- **Ceiling limits the discriminator.** Base accuracy 0.94 leaves almost no room
  for accuracy to *rise*; the **margin** carries the inferential weight, and it
  did not support H1 — it drifted slightly *down*.
- **A localized M_override hint.** sft-v2 *flipped* `grace_nature` (+1.43→−0.02:
  "grace perfects nature" → it now slightly prefers "grace destroys nature") and
  weakened `faculties`/`metaphysics`, partly offset by *gains* on `trinity`
  (acc 0.67→1.00), `justification`, `grace_faith`. Net = scattered reshuffle, not
  a coherent effect — H0-with-noise plus a faint truth-erosion tail worth a
  closer look, not a claim.
- **Boundary of the claim.** This tests *representational discrimination of
  stated relations*, a **necessary not sufficient** condition for understanding.
  It does **not** test multi-step reasoning, counterfactual propagation (negate a
  premise → does the conclusion flip in generation?), or novel application. Those
  are the next experiment.
- **Authorship circularity.** Claude authored the training data, the probe pairs,
  and the labels; minimal pairs control the style confound but not a shared
  Claude/Qwen prior. Human-rated pairs would close this.

## Reproduce

```bash
source .venv/bin/activate
python post-experiments/data/build_pairs.py        # -> minimal_pairs.jsonl
python post-experiments/run_understanding.py        # -> results/understanding.json  (~1 min)
python post-experiments/run_voice.py                # -> results/voice.json           (~4 min)
```

**Bottom line.** Teaching a small LLM the *voice* of theologians did not teach it
their *ideas* — and it did not need to: the ideas were already there from
pretraining, and the voice rode in on top, orthogonally.
