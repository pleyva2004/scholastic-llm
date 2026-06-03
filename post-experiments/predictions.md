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

### 2026-06-03 — Run on probe set (commit of results follows this block)

**Controls (pipeline valid):** label-shuffle = **0.500** (no leakage); base
factual-anchor accuracy = **0.800 (4/5)** — the single miss (`anc02`,
"Augustine wrote the Confessions", a name-swap) is identical across *all*
conditions, a base tokenization/frequency artifact, not adapter forgetting.

**Understanding — doctrinal set (n=34), per-token margin Δ:**

| condition | disc. acc | mean margin | 95% CI |
|---|---|---|---|
| base (C0) | 0.941 | **+0.929** | [+0.691, +1.166] |
| sft-v1 (C1) | 0.971 | +1.024 | [+0.797, +1.257] |
| sft-v2-iter400 (C2) | 0.912 | **+0.738** | [+0.575, +0.904] |
| dpo-v3 (C3) | 0.941 | +0.836 | [+0.640, +1.035] |

**Paired vs base (doctrinal):** C1 Δmargin +0.095 (CI [−0.068,+0.257], null);
**C2 Δmargin −0.191 (CI [−0.350,−0.032], paired CI excludes 0, >2·SEM, but base/SFT
margin CIs OVERLAP)**; C3 Δmargin −0.093 (null). Δaccuracy ≈ 0 everywhere
(+0.029 / −0.029 / 0.000), at a base ceiling of 0.94.

**Voice positive control — rubric /12 (n=6):** base **1.33** → sft-v1 **6.67**
→ sft-v2-iter400 **7.83** → dpo-v3 7.33. Sharp, dose-responsive. Intervention is
potent.

**VERDICT — prediction CONFIRMED (H0, orthogonality).** Voice rose ~6× with a
clean dose-response; the doctrinal *discriminator* (accuracy) did not move
(ceiling, base already 0.941); the *companion* margin did not rise (refutes H1)
and its small C2 decrease did **not** clear the pre-committed confirmation
conjunction (unpaired CIs overlap). Per the **pre-committed ambiguous-case rule**,
the discriminator is the verdict → **flat → H0**, and the margin decrease is
logged as an **unexplained sub-threshold secondary** (a localized *M_override*
hint), not a confirmed effect.

**Where I was right / wrong (calibration):**
- Right: H0 headline at ~70% prior — confirmed. Right: voice ↑ sharply; anchors
  no forgetting; base already a strong theologian.
- Wrong/surprising: I expected dead-flat margin; instead the most-trained SFT
  checkpoint *nudged margin down* (−0.19), localized to core axioms
  (`grace_nature` flipped +1.43→−0.02; `faculties`, `metaphysics` down) and
  partly offset by gains (`trinity` acc 0.67→1.00, `justification`,
  `grace_faith`). Net: scattered reshuffle, not coherent H1 — consistent with
  H0-plus-noise with a faint M_override tail. The per-category pre-commitment
  caught a *localized* H1 (trinity) that the aggregate washed out, and a
  localized M_override (grace_nature) — neither survives correction to a global
  claim.
