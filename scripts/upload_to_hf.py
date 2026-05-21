"""Publish scholastic-llm LoRA adapters to the Hugging Face Hub.

One Hub repo per adapter, all cross-linked. Adapter files are MLX-format
LoRA safetensors (~44 MB each). The base model (Qwen/Qwen2.5-7B-Instruct)
is referenced by metadata, not re-uploaded.

Usage:
    python scripts/upload_to_hf.py sft-v1
    python scripts/upload_to_hf.py --all
    python scripts/upload_to_hf.py --all --dry-run    # generate model cards, don't push
    python scripts/upload_to_hf.py sft-v1 --dry-run   # show what would happen

Requires HF_TOKEN with `write` scope (set in env or .env), OR a prior
`huggingface-cli login` (cached token).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTERS_ROOT = PROJECT_ROOT / "adapters"
HF_USER = "pleyva2004"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class Variant:
    key: str                     # short name used as CLI arg
    local_dir: str               # subdir of adapters/
    repo_suffix: str             # appended to HF_USER/scholastic-llm-
    title: str                   # human title for model card
    role: str                    # one-paragraph role description
    iters: int                   # training iterations
    pairs: int                   # # training pairs (or preference pairs for DPO)
    pair_type: str               # "Q/A pairs" or "preference pairs"
    recommended: bool
    # rubric totals (max-30 per dimension, max-120 strict, max-90 balanced)
    reg: int
    aug: int
    ccc: int
    structure: int
    strict_total: int
    balanced_total: int
    extra_recipe: str = ""       # appended to Training section


VARIANTS: list[Variant] = [
    Variant(
        key="sft-v1",
        local_dir="scholastic-v1",
        repo_suffix="sft-v1",
        title="scholastic-llm SFT-v1 (Phase 1 paper headline)",
        role=(
            "Phase 1 of the project. 200 LoRA-SFT iterations on 83 "
            "teacher-distilled (question, scholastic-answer) pairs. This is "
            "the model whose +258% rubric gain (19/120 → 68/120) is the "
            "headline result in the paper."
        ),
        iters=200,
        pairs=83,
        pair_type="Q/A pairs",
        recommended=False,
        reg=20, aug=3, ccc=19, structure=26,
        strict_total=68, balanced_total=66,
    ),
    Variant(
        key="sft-v2-iter400",
        local_dir="scholastic-v2-iter400",
        repo_suffix="sft-v2-iter400",
        title="scholastic-llm SFT-v2 @ iter 400 (best Phase 2 checkpoint)",
        role=(
            "Best checkpoint from Phase 2. 400 LoRA-SFT iterations on 377 "
            "teacher-distilled pairs with per-source register hints (Aquinas "
            "form for Summa/CCC chunks, Augustinian for Confessions/City of "
            "God). Val loss bottoms at iter 400 before mild overfitting. "
            "Matches Phase 1 on the strict rubric (68/120) AND beats it on "
            "the balanced rubric (68/90 vs 66/90); closes the Augustinian "
            "voice gap. **Recommended variant for general use.**"
        ),
        iters=400,
        pairs=377,
        pair_type="Q/A pairs",
        recommended=True,
        reg=21, aug=7, ccc=18, structure=22,
        strict_total=68, balanced_total=68,
    ),
    Variant(
        key="sft-v2",
        local_dir="scholastic-v2",
        repo_suffix="sft-v2",
        title="scholastic-llm SFT-v2 @ iter 800 (Phase 2 final)",
        role=(
            "Phase 2 final weights at iter 800 (twice the iter-400 best "
            "checkpoint). Mildly overfit: leans more strongly Augustinian "
            "but loses some scholastic-register and structure scores "
            "compared to iter 400. Useful as a 'maximum drift' reference."
        ),
        iters=800,
        pairs=377,
        pair_type="Q/A pairs",
        recommended=False,
        reg=15, aug=13, ccc=18, structure=18,
        strict_total=64, balanced_total=64,
    ),
    Variant(
        key="dpo-v3",
        local_dir="scholastic-v3-dpo",
        repo_suffix="dpo-v3",
        title="scholastic-llm DPO-v3 (negative result: saturation)",
        role=(
            "Phase 2 DPO refinement chain on top of SFT-v2 (iter 800). 300 "
            "DPO iterations on ~50 preference pairs (chosen = SFT-v2 output, "
            "rejected = base output). **Documented negative result:** val loss "
            "= 0.000 from iter 1, val accuracy = 1.000, chosen/rejected "
            "margin 35.4 nats. Within-model preference data saturated the "
            "policy at initialization; DPO had no gradient signal. The "
            "resulting adapter is functionally identical to its SFT-v2 "
            "starting point. Published as a teaching artifact for the "
            "setup pitfall, not as a recommended model."
        ),
        iters=300,
        pairs=50,
        pair_type="preference pairs (DPO)",
        recommended=False,
        reg=16, aug=12, ccc=20, structure=16,
        strict_total=64, balanced_total=63,
        extra_recipe=(
            "DPO step on top of SFT-v2: β=0.05, sigmoid loss, "
            "reference = base, 300 DPO iterations."
        ),
    ),
]


def variants_table(highlight_key: str) -> str:
    """Markdown table of all 4 variants for cross-linking; bolds the current one."""
    rows = []
    for v in VARIANTS:
        repo = f"{HF_USER}/scholastic-llm-{v.repo_suffix}"
        link = f"[`{v.repo_suffix}`](https://huggingface.co/{repo})"
        if v.repo_suffix == highlight_key:
            link = f"**{link} (this card)**"
        star = " ⭐" if v.recommended else ""
        rows.append(
            f"| {link}{star} | {v.iters} | {v.pairs} {v.pair_type} | "
            f"{v.strict_total}/120 | {v.balanced_total}/90 |"
        )
    return (
        "| Variant | Iters | Training data | Strict total | Balanced total |\n"
        "|---|---:|---|---:|---:|\n"
        + "\n".join(rows)
    )


MODEL_CARD_TEMPLATE = """---
license: mit
language:
- en
library_name: mlx
tags:
- lora
- mlx
- apple-silicon
- philosophy
- catechism
- scholastic
- qwen
- fine-tuning
base_model: {base_model}
pipeline_tag: text-generation
---

# {title}

> **⚠ NOTICE — research experiment, not theological authority**
>
> This is a personal portfolio / research project exploring how small
> open-weights LLMs can be fine-tuned to adopt a specific historical
> register and citation style. The trained model is **not** a reliable
> source of Catholic doctrine, biblical interpretation, or philosophical
> truth. It can hallucinate citations, misrepresent the Catechism, and
> confidently err. Outputs must not be cited as catechetical
> instruction, theological argument, or spiritual direction. For
> doctrinal questions, consult the actual Catechism of the Catholic
> Church, a qualified priest, or a trained theologian.

## What this is

LoRA adapter for [`{base_model}`](https://huggingface.co/{base_model}),
trained to respond to philosophical and theological questions in a
**scholastic / Latin-inflected register** grounded in the Catechism of
the Catholic Church (CCC, 1992) and modeled after Aquinas's *Summa
Theologica* and Augustine's *Confessions* / *City of God*.

{role}

For full background, recipe, and Phase 1+2 results, see:
- **Paper:** https://pleyva2004.github.io/scholastic-llm/main.pdf
- **GitHub:** https://github.com/pleyva2004/scholastic-llm

## Variants (all four published)

{variants_table}

(Strict total $= \\textsc{{reg}} + \\textsc{{aug}} + \\textsc{{ccc}} + \\textsc{{str}}$, max 120.
Balanced total $= \\max(\\textsc{{reg}},\\textsc{{aug}}) + \\textsc{{ccc}} + \\textsc{{str}}$, max 90;
introduced in Phase 2 because the strict total penalizes appropriate
register switching.)

## How to load (MLX)

```python
from mlx_lm import generate, load

model, tokenizer = load(
    "{base_model}",
    adapter_path="{repo_id}",
)

prompt = "How do you reconcile divine foreknowledge with free will?"
messages = [{{"role": "user", "content": prompt}}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=text, max_tokens=300))
```

Requires `mlx-lm` (≥ 0.27) on Apple Silicon. For other inference engines
you will need to convert the adapter manually; the repo contains MLX-format
weights only.

## Training

| | |
|---|---|
| Base | `{base_model}` |
| Quantization | MLX 8-bit weight quantization (Q8) |
| Method | LoRA on top 16 of 28 transformer layers |
| Optimizer | AdamW |
| Learning rate | $10^{{-5}}$ |
| Batch size | 1 |
| Max sequence length | 2048 |
| Iterations | {iters} |
| Training data | {pairs} {pair_type} |
| Hardware | Apple M4 Pro, 48 GB unified memory |
| Trainable parameters | 2.6M / 7.6B (0.034%) |
| Peak resident memory | 12.7 GB |

{extra_recipe_block}

Training data was generated by **Claude Sonnet 4.6 as teacher**, per a
strict system prompt requesting scholastic register and CCC citations,
applied to ~50–150 cleaned source chunks scraped from the Catechism,
the Summa, and Augustine's works. See the GitHub repo's
`scripts/generate_training_pairs.py` for the exact prompt.

## Evaluation

Rubric-based evaluation on 10 held-out philosophical prompts (not seen
during training). Four dimensions, each scored 0–3 per prompt, summed
across 10 prompts (per-dimension max 30).

| Dimension | BASE | **This variant** | Δ vs BASE |
|---|---:|---:|---:|
| Scholastic register (Summa markers) | 3 | **{reg}** | {reg_delta:+d} |
| Augustinian voice (autobiographical) | 0 | **{aug}** | {aug_delta:+d} |
| CCC grounding (paragraph citations) | 0 | **{ccc}** | {ccc_delta:+d} |
| Structure (multi-para, obj/reply) | 16 | **{structure}** | {structure_delta:+d} |
| **Strict total** | **19** | **{strict_total}** | **{strict_delta:+d}** |
| **Balanced total** ($\\max(\\textsc{{reg}},\\textsc{{aug}}) + \\textsc{{ccc}} + \\textsc{{str}}$) | **19** | **{balanced_total}** | **{balanced_delta:+d}** |

Full per-prompt scores and qualitative samples are in the paper.

## Data licensing

Training data sources:

| Source | Status |
|---|---|
| Catechism of the Catholic Church (1992) | © USCCB / Libreria Editrice Vaticana; used under fair-use research posture |
| Summa Theologica (Shapcote 1920) | Public domain (US) |
| Augustine — *Confessions* (Pusey trans.) | Public domain |
| Augustine — *City of God* (Dods trans.) | Public domain |

The training-data JSONL itself is **not** redistributed with this
adapter; only the LoRA weights and this card. See
[`DATA_LICENSING.md`](https://github.com/pleyva2004/scholastic-llm/blob/main/DATA_LICENSING.md)
for the full posture.

## License

- **This adapter (LoRA weights):** MIT — see
  [LICENSE](https://github.com/pleyva2004/scholastic-llm/blob/main/LICENSE) in the repo.
- **Base model (`{base_model}`):** Apache 2.0 (governed by the
  base-model card on Hugging Face).
- **Source corpus:** terms above.

## Limitations & ethics

- **Hallucinated citations.** The fine-tuned model confidently emits
  CCC paragraph numbers with the surface form of ground truth. Many
  citations do **not** correspond to the actual content of the cited
  paragraph. Always verify against the actual Catechism.
- **No human evaluation.** Reported numbers come from a regex/keyword
  rubric. The rubric measures lexical and structural surface form,
  not theological correctness.
- **Small held-out set (N=10).** Confidence intervals are wide; the
  +49-point delta is large relative to noise but not bootstrapped.
- **No doctrinal authority.** The model speaks in a voice culturally
  associated with magisterial authority. It has none. It can confidently
  err and should not be relied upon for spiritual direction.

## Citation

```bibtex
@misc{{leyva2026scholastic,
  title  = {{Teaching a Small LLM Scholastic Voice: Fine-Tuning Qwen 2.5 on the Catechism, Summa, and Augustine via Local MLX}},
  author = {{Pablo Leyva}},
  year   = {{2026}},
  url    = {{https://github.com/pleyva2004/scholastic-llm}},
  note   = {{Independent Research}}
}}
```
"""


def make_model_card(v: Variant) -> str:
    extra_block = (
        f"### Phase 2 extras\n\n{v.extra_recipe}\n" if v.extra_recipe else ""
    )
    return MODEL_CARD_TEMPLATE.format(
        base_model=BASE_MODEL,
        title=v.title,
        role=v.role,
        variants_table=variants_table(v.repo_suffix),
        repo_id=f"{HF_USER}/scholastic-llm-{v.repo_suffix}",
        iters=v.iters,
        pairs=v.pairs,
        pair_type=v.pair_type,
        extra_recipe_block=extra_block,
        reg=v.reg, aug=v.aug, ccc=v.ccc, structure=v.structure,
        strict_total=v.strict_total, balanced_total=v.balanced_total,
        reg_delta=v.reg - 3,
        aug_delta=v.aug - 0,
        ccc_delta=v.ccc - 0,
        structure_delta=v.structure - 16,
        strict_delta=v.strict_total - 19,
        balanced_delta=v.balanced_total - 19,
    )


def find_variant(key: str) -> Variant:
    for v in VARIANTS:
        if v.key == key or v.repo_suffix == key:
            return v
    raise SystemExit(
        f"Unknown variant '{key}'. Pick one of: "
        + ", ".join(x.key for x in VARIANTS)
    )


def upload_variant(v: Variant, *, dry_run: bool) -> str:
    """Build model card + upload to HF. Returns the repo URL."""
    from huggingface_hub import HfApi, login

    src = ADAPTERS_ROOT / v.local_dir
    if not src.exists():
        raise SystemExit(f"Adapter dir missing: {src}")

    repo_id = f"{HF_USER}/scholastic-llm-{v.repo_suffix}"

    card = make_model_card(v)
    if dry_run:
        print(f"--- {repo_id} ---")
        print(f"Local source: {src}")
        print(f"Files that WILL be uploaded:")
        for name in ("adapters.safetensors", "adapter_config.json"):
            fp = src / name
            if fp.exists():
                print(f"  {fp.name}  ({fp.stat().st_size / 1e6:.1f} MB)")
            else:
                print(f"  {name}  (MISSING — would fail)")
        print(f"  README.md  ({len(card)/1024:.1f} KB generated, model card)")
        print(f"\nModel card preview (first 800 chars of {len(card)} total):")
        print(card[:800] + ("\n...[truncated]" if len(card) > 800 else ""))
        return f"https://huggingface.co/{repo_id} (DRY RUN — not pushed)"

    token = os.environ.get("HF_TOKEN")
    if token:
        # Explicit token in env — use it
        login(token=token, add_to_git_credential=False)
        api = HfApi(token=token)
    else:
        # Fall back to cached login (created by `hf auth login`)
        api = HfApi()
        try:
            who = api.whoami()
            print(f"  using cached HF login as {who['name']}")
        except Exception as e:
            raise SystemExit(
                "No HF_TOKEN in env and no cached login found. Either run "
                "`hf auth login` interactively, or set HF_TOKEN in .env "
                "with write scope. Token page: "
                f"https://huggingface.co/settings/tokens (error: {e})"
            )
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Stage to a temp dir so we don't pollute the local adapter folder with README.md.
    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        # Copy ONLY the adapter + config (skip intermediate checkpoints).
        for name in ("adapters.safetensors", "adapter_config.json"):
            (staging / name).write_bytes((src / name).read_bytes())
        (staging / "README.md").write_text(card)
        print(f"Uploading {repo_id}…")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Publish {v.repo_suffix} adapter + model card",
        )
    return f"https://huggingface.co/{repo_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", help="variant key (sft-v1, sft-v2-iter400, sft-v2, dpo-v3)")
    parser.add_argument("--all", action="store_true", help="Upload all four")
    parser.add_argument("--dry-run", action="store_true", help="Build cards locally; no HF push")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    if not args.variant and not args.all:
        parser.print_help()
        return 2

    targets = VARIANTS if args.all else [find_variant(args.variant)]
    for v in targets:
        url = upload_variant(v, dry_run=args.dry_run)
        print(f"  ✓ {url}\n")

    if args.dry_run:
        print("Dry run complete; nothing pushed. Re-run without --dry-run to publish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
