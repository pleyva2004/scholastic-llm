# Pre-Registration — Voice vs. Understanding (scholastic-llm, Phase 3)

**APPEND-ONLY.** The prediction below is logged *before any model is run on the
probe set*. Never edit it after the fact — outcomes are appended in a dated block
at the bottom. This file's commit SHA + timestamp are the integrity proof that
the prediction preceded the data (it is committed in a separate commit *before*
`run_understanding.py` produces any results).

- **Logged:** 2026-06-03, before `post-experiments/results/` exists.
- **Setup:** base `models/qwen2.5-7b-mlx-q8`; adapters `scholastic-v1`,
  `scholastic-v2-iter400`, `scholastic-v3-dpo`. Probe = 39 vocabulary-matched
  minimal pairs (34 doctrinal + 5 factual-anchor), neutral framing, per-token
  teacher-forced logprob. Deterministic scoring; uncertainty via paired
  across-item bootstrap (10k resamples, fixed seed).

---

## Hypotheses (full statements in `hypothesis.md`)

- **H1 (entanglement):** semantically-correct SFT targets sharpen the model's
  *relational* representation, so within-pair doctrinal margin **rises** above
  base.
- **H0 (orthogonality, the rival):** the low-rank update is a surface skin;
  doctrinal margin is **unchanged** from base while the voice rubric rises.
- **M_override:** register is learned at the expense of truth-tracking;
  doctrinal margin **falls** below base.

### Metric stack
- **Primary (discriminator):** within-pair discrimination accuracy, doctrinal set.
- **Companion (stable, signed):** mean per-token margin Δ, doctrinal set.
- **Positive control:** voice rubric on free-text generations (same models).
- **Controls:** factual-anchor accuracy (floor), label-shuffle (chance check).

### Decision rule (pre-committed)
Doctrinal SFT−base effect **confirmed** only if (a) the paired per-item mean
difference exceeds **2× its across-item std**, AND (b) base vs. SFT 95% bootstrap
CIs on Δ do not overlap. Else **suggestive only**. Ambiguous (one criterion
met, not the other): report the discriminator (accuracy) as the verdict; flag
the other as an unexplained secondary. Do not retro-promote.

---

## Pre-registered prediction (signed, with confidence)

**Headline — I predict H0 (orthogonality) on the doctrinal set.**
Δmargin(C2 sft-v2-iter400 − C0 base) is **small and within the noise bar**;
discrimination accuracy moves by **< ~5 points** and the CIs overlap. **Voice
rubric rises sharply** for all SFT checkpoints (≈ prior Phase-1/2 results).
**Confidence: medium-high (~70%).**

*Mechanistic reason for the prior:* (1) the base already encodes these
relations from pretraining — the de-risking spike showed it prefers "Grace is
prior to merit" unprompted; there is little headroom on well-known doctrine.
(2) A rank-8–16 LoRA on ~377 pairs for 400 iters has small capacity, and the
gradient's *cheapest* way to lower loss on register-heavy targets is to shift
output-form directions, not to reorganize content already present in the base.
The minimal-pair design strips exactly the surface channel the adapter most
plausibly moved.

**The falsifier that would most surprise me → H1:** doctrinal margin rises
beyond the 2×-std bar with non-overlapping CIs, *and* it shows a monotone
dose–response (base < v1 < v2). That would mean fitting semantically-correct
targets reorganized content representation — voice training is not content-neutral.

**The other informative surprise → M_override:** doctrinal margin goes
*negative* (SFT < base) while anchors hold — register mimicry actively eroded
truth-tracking. Lower prior (~10%) but the signed metric is built to catch it.

**Secondary predictions (if H0 holds, as I expect):**
- Dose–response on doctrinal margin: **flat** across v1/v2 (rises only if H1).
- DPO (C3) doctrinal margin ≈ its SFT-v2 parent (no content movement).
- Factual anchors: ≈ ceiling for **all** conditions including base (≈ no
  forgetting). A drop under SFT would be evidence for M_override.
- Voice rubric: base ≪ all SFT checkpoints (the potency check).

**Worth noting against my own prior:** the per-category cut is where H1 could
still win locally even if the aggregate is H0 — e.g. Augustinian/Confessions
relations (`aug*`), which Phase-2 specifically oversampled to close the
Augustinian-voice gap, are the most likely place for a *content* gain to ride
in with the *voice* gain. I pre-commit to reporting per-category so a localized
H1 effect isn't averaged away or, conversely, cherry-picked post hoc.

---

## Outcomes (appended post-hoc only — do not edit predictions above)

<!-- Append a dated block here after the run: per-condition accuracy & margin
     ± CI, anchor floor, shuffle control, voice rubric, and confirmed /
     disconfirmed / ambiguous vs. the prediction above. -->
