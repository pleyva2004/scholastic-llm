# Hypothesis

*Assessed against `HYPOTHESIS.md`: a strong hypothesis names a mechanism, is
falsifiable, diverges from a specific plausible rival on a computable measurable,
and is signed.*

## The starting (weak) hypothesis, and why it fails the bar

The implicit hypothesis we began the brainstorm with was:

> *"Voice training and understanding are different things."*

This is **too weak to test** — it names no mechanism, forbids no observation, and
is compatible with almost any result. Per `HYPOTHESIS.md`, it is a phenomenon,
not a hypothesis. We sharpen it into a mechanism claim with a named rival.

## Improved hypothesis

**H1 — entanglement (voice carries content).**
Because the SFT targets are semantically correct theology (not style tokens
attached to random content), the LoRA update sharpens the model's representation
of doctrinal *relations*, not only its output form. The update does not — and
given the training signal, cannot — cleanly factor "how a scholastic answer
*sounds*" from "what it *asserts*."

**Intervention.** Apply the SFT adapter vs. the bare base; measure preference on
**vocabulary-matched minimal pairs** (style held fixed, only the relation
varies). Style preference cancels within the pair; any change in within-pair
margin is attributable to representational change in *content*.

**Signed prediction.** SFT raises within-pair discrimination accuracy and mean
per-token logprob margin on the **doctrinal** set, above base, by more than the
pre-committed noise bar.

## The specific rival it must beat

**H0 — orthogonality (voice is a surface skin).**
The rank-8–16 update lives in an output-form subspace; the knowledge of which
proposition is true already resides in the frozen base from pretraining and is
left untouched. → within-pair margin on the doctrinal set is **statistically
indistinguishable from base**, even as the voice rubric rises sharply.

This rival is **not a strawman**: low-rank adapters demonstrably can change
surface style with minimal effect on factual/representational probes, and the
base is a strong pretrained theologian already (its summed-logprob on
"Grace is prior to merit" already beats the reverse — see the de-risking spike).
H0 is the *default* expectation a skeptic would hold.

**Third outcome the design must be able to see — M_override.** SFT *lowers* the
doctrinal margin below base (register-flavored forgetting). The metric is signed,
so this is observable, not folded into H0.

## Why this passes the `HYPOTHESIS.md` checklist

| Property | How H1 satisfies it |
|---|---|
| Names a mechanism | "semantically-correct targets sharpen relational representation," not "SFT helps." |
| Falsifiable / risky | Forbids the observation "voice ↑ while doctrinal margin flat." If we see that, H1 is dead. |
| Diverges from a plausible rival | H0 predicts a *different number* (Δmargin ≈ 0) under the *same* intervention → Platt strong inference. |
| Operationalizable end-to-end | Every term → teacher-forced log-probability, computed locally in `mlx_lm`. |
| Signed / directional | H1: Δmargin(SFT−base) **> 0**; H0: **≈ 0**; M_override: **< 0**. |

## Secondary (mechanism-implied) predictions — cheap, same data

If H1 is the true mechanism, more downstream consequences should hold; confirming
the headline *and* these is much stronger than the headline alone:

1. **Dose–response.** Margin gain should *track training amount*:
   base < sft-v1 < sft-v2-iter400. Under H0 the doctrinal margin is flat across
   all checkpoints while the voice rubric climbs.
2. **DPO is inert on content.** dpo-v3 (a documented voice/preference negative
   result) should not move the doctrinal margin beyond its SFT-v2 parent — if it
   does, the effect isn't the SFT content signal.
3. **Anchors don't regress.** Factual-anchor accuracy stays ≈ ceiling across all
   adapters; a drop there would signal forgetting (evidence *for* M_override) and
   reframe any doctrinal-margin change.
4. **Voice rises regardless.** The rubric (register/structure/CCC) rises sharply
   for all SFT checkpoints — the positive control proving the intervention is
   potent, so a null on understanding means *orthogonal*, not *inert*.
