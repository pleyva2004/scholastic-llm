# Conceptual understanding — voice vs. understanding

*From-first-principles pass before any experiment is designed. Restating the
question as a claim about mechanism, naming the phenomenon, surfacing competing
mechanisms, and scoping the claim.*

## 1. The surface task, and its mechanistic restatement

**Surface question.** "We fine-tuned a 7B model to *speak* like Aquinas and
Augustine. Did that also change what it *understands* about their ideas?"

**Restatement (mechanism, not surface).** Voice-SFT is a LoRA update: a low-rank
`ΔW` added to a frozen base. The question is *where `ΔW` lives*:

> Does `ΔW` act almost entirely in a **surface / output-form subspace** — the
> directions that govern register tokens ("I answer that", "§"), citation
> formatting, and Latinate cadence — leaving the model's **internal
> representation of theological relations** (what depends on what, what is true
> of what) essentially fixed? Or does `ΔW` also **rotate the conceptual
> representation**, so that the relative probability the model assigns to a true
> vs. a false doctrinal proposition changes?

This is the *same* shape as the question we already worked through in the
subliminal-learning study (`predictions.md`): there, "does distillation transfer
*representation* (CKA convergence) or only *projection through a shared head*?"
Here: "does voice-SFT transfer *form* only, or also *content*?" Same dichotomy —
a surface channel vs. a representational one — pointed at theology.

## 2. What "understanding" means here — operationalized end-to-end

A definition is useless unless every term maps to a number we can compute. We
adopt a **necessary-condition** definition, deliberately narrow so it is
testable on the hardware in front of us:

> **Understanding (operational).** A model "understands" a theological relation
> to the extent that its probability mass tracks the *truth of the relation*
> rather than the *surface form of the sentence*. Concretely: given a
> **minimal pair** — two sentences with identical vocabulary and near-identical
> length, differing only in the relation (a swapped term, a reversed
> dependency, a negation) — the model assigns higher likelihood to the
> doctrinally correct member.

Why minimal pairs are the load-bearing design choice: they **subtract out
voice**. Both members are equally scholastic in vocabulary, so any preference a
model has for "scholastic-sounding text" applies equally to both and *cancels*
in the within-pair comparison. What survives is exactly the thing we want —
sensitivity to the *relation*. This is the LLM analogue of the
"flatten/permute the non-target logits while preserving the target" intervention
in `HYPOTHESIS.md`: hold the surface fixed, vary only the structural content,
see if the model notices.

This definition is a **necessary, not sufficient** condition for understanding
in the rich sense (generalization, counterfactual propagation, novel
application). It tests *representational discrimination* — does the model's
distribution encode the doctrinal relation — not *reasoning*. We state that
boundary up front (§4) and do not over-claim past it.

## 3. The phenomenon, and ≥2 competing mechanisms

The training pairs are not style-only: the teacher (Claude) wrote answers that
are *semantically* theological — they encode correct doctrinal relations *and*
scholastic form together. So the gradient sees both signals. Two a-priori
plausible mechanisms predict different outcomes under the *same* intervention
(apply adapter vs. not), measured on the *same* quantity (within-pair Δlogprob):

- **M_entangle ("voice carries content").** Because the SFT targets are
  semantically correct theology, fitting them sharpens the model's
  representation of the underlying relations, not just the surface. The adapter
  cannot cleanly factor style from content because the training signal didn't.
  → minimal-pair discrimination **rises above base**.

- **M_skin ("voice is a surface skin").** `ΔW` is low-rank (rank ~8–16) and
  concentrates on output-form directions; the heavy lifting of *which
  proposition is true* already lives in the frozen base from pretraining, and
  the adapter leaves it untouched. → minimal-pair discrimination is
  **indistinguishable from base**; only the voice rubric moves.

- **M_override ("style at the expense of truth").** A real third possibility,
  not a strawman: the adapter learns to push probability toward *scholastic
  surface features* so strongly that it *degrades* the base's truth-tracking
  (a register-flavored catastrophic forgetting). → discrimination **falls below
  base**; margin goes *negative* relative to base on some categories.

The signed metric (Δmargin, base vs. SFT) can land in all three regions, so the
experiment can be *surprised*, and the surprise points somewhere specific.

## 4. Boundaries of the claim (what generalizes, what's an artifact)

- **Holds for:** representational *discrimination* of stated relations, read
  through teacher-forced log-probability. If SFT moves this, voice training
  measurably touched content representation; if not, it didn't (within our
  detection power).
- **Does not cover:** multi-step *reasoning*, counterfactual *propagation*
  (negate a premise → does the conclusion flip in generation?), or *novel
  application* to cases absent from training. A model could pass the minimal-pair
  probe by representational pattern-matching and still fail to reason. That is
  the natural next experiment, named not run.
- **Setup-specific risk:** the base (Qwen 2.5 7B) already carries substantial
  theology from pretraining, so headroom may be small — a near-null SFT−base
  delta is *expected under M_skin* and must not be mistaken for "no signal in the
  data." The positive control (voice rubric on the same models) guards this: it
  confirms the intervention is potent, so a flat understanding delta means
  *orthogonal*, not *inert*.
- **Authorship circularity (flagged honestly):** the same model family (Claude)
  authored the training data, the probe pairs, and the ground-truth labels.
  Minimal pairs control the main style confound, but a residual "Claude and Qwen
  share priors" effect cannot be fully excluded by this design. Human-rated pairs
  would be the fix.

→ Coverage check against `CONCEPTUAL.md`: restatement ✓, phenomenon named ✓,
≥2 mechanisms ✓ (three), discriminating measurable identified (within-pair
Δlogprob; full pre-registration in `predictions.md`) ✓, confounds named (length,
base-rate/fluency, intervention-potency, authorship) ✓. Ready to design.
