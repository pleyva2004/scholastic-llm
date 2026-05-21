# scholastic-llm

Fine-tuning Qwen 2.5 7B-Instruct on Apple Silicon (MLX) to debate philosophy in
a **scholastic, Latin-inflected register**, grounded in the **Catechism of the
Catholic Church**, in the structural voices of **Aquinas (Summa Theologica)**
and **Augustine (Confessions, City of God)**.

> **⚠ NOTICE — research experiment, not theological authority**
>
> This is a personal portfolio / research project exploring how small
> open-weights LLMs can be fine-tuned to adopt a specific historical register
> and citation style. The trained model is **not** a reliable source of
> Catholic doctrine, biblical interpretation, or philosophical truth. It can
> hallucinate citations, misrepresent the Catechism, and confidently err.
> Outputs must not be cited as catechetical instruction, theological argument,
> or spiritual direction. For doctrinal questions, consult the actual
> Catechism of the Catholic Church, a qualified priest, or a trained
> theologian.

**[Read the paper](https://pleyva2004.github.io/scholastic-llm/main.pdf)** — single-column arXiv-style preprint, auto-rendered on every push.

**[View the poster](https://pleyva2004.github.io/scholastic-llm/poster.pdf)** — A0 portrait conference poster, auto-rendered on every push.

## Try it live

[**🤗 Hugging Face Space (free CPU demo)**](https://huggingface.co/spaces/pleyva2004/scholastic-llm-demo) — Gradio chat interface using the PEFT-converted adapter on top of Qwen 2.5 7B-Instruct. Slow on free CPU (~30-60 s/response); embedded directly into the [landing page](https://pleyva2004.github.io/scholastic-llm/) for convenience.

## Models on Hugging Face

Four MLX LoRA adapters for `Qwen/Qwen2.5-7B-Instruct`, each in its own Hub repo with a model card cross-linking the rest:

| Variant | Role | Strict (max 120) | Balanced (max 90) | Hub |
|---|---|---:|---:|---|
| `sft-v1` | Phase 1 paper headline | 68 | 66 | [`pleyva2004/scholastic-llm-sft-v1`](https://huggingface.co/pleyva2004/scholastic-llm-sft-v1) |
| `sft-v2-iter400` ⭐ | Best Phase 2 checkpoint | 68 | **68** | [`pleyva2004/scholastic-llm-sft-v2-iter400`](https://huggingface.co/pleyva2004/scholastic-llm-sft-v2-iter400) |
| `sft-v2` | Phase 2 final (iter 800, mild overfit) | 64 | 64 | [`pleyva2004/scholastic-llm-sft-v2`](https://huggingface.co/pleyva2004/scholastic-llm-sft-v2) |
| `dpo-v3` | Negative result (DPO saturation) | 64 | 63 | [`pleyva2004/scholastic-llm-dpo-v3`](https://huggingface.co/pleyva2004/scholastic-llm-dpo-v3) |

Load any of them with `mlx_lm`:

```python
from mlx_lm import generate, load
model, tok = load(
    "Qwen/Qwen2.5-7B-Instruct",
    adapter_path="pleyva2004/scholastic-llm-sft-v2-iter400",
)
```

This is an experimental personal project. See `DATA_LICENSING.md` for the
status of source texts used during training.

## Status

🚧 In active development. Phase 1 (end-to-end smoke test) in progress.

## Approach

1. **Scrape** primary sources (CCC, Summa, Confessions, City of God) into a
   structured corpus.
2. **Reformat** corpus chunks into instruction pairs using the **Claude API**
   as a teacher model — questions paired with scholastic answers that cite
   CCC paragraphs and argue in Summa- or Augustinian-style.
3. **SFT** (LoRA fine-tune) Qwen 2.5 7B-Instruct (MLX Q8) on the pairs.
4. **Evaluate** with a rubric scoring scholastic register, CCC grounding,
   Augustinian rhetorical moves, and argument structure.
5. **(Optional) DPO** refinement against bland base-model outputs.

## Requirements

- macOS with Apple Silicon (tested on M4 Pro, 48GB unified memory)
- Python 3.12
- [`uv`](https://github.com/astral-sh/uv) for environment management
- Anthropic API key (for training-data generation step)
- ~30 GB free disk (7B Q8 weights + training data + adapters)

## Setup

```bash
git clone <repo-url> scholastic-llm
cd scholastic-llm

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .

cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

## Usage

```bash
# 1. Scrape primary sources (rate-limited, cached)
python scripts/scrape_sources.py --phase 1

# 2. Clean and structure scraped HTML
python scripts/clean_corpus.py

# 3. Generate training pairs via Claude API
python scripts/generate_training_pairs.py --max-calls 200

# 4. Convert Qwen 2.5 7B-Instruct to MLX Q8
mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct \
    --mlx-path models/qwen2.5-7b-mlx-q8 -q --q-bits 8

# 5. Run SFT (LoRA)
mlx_lm_lora.train --model models/qwen2.5-7b-mlx-q8 \
    --train --train-mode sft --data data/ \
    --batch-size 1 --num-layers 16 --iters 200 \
    --learning-rate 1e-5 \
    --adapter-path adapters/scholastic-v1

# 6. Evaluate
python notebooks/02_eval.py
```

Each notebook in `notebooks/` is a `.py` file with `# %%` cell markers that
opens as an interactive notebook in VSCode/Cursor.

## Layout

```
notebooks/   end-to-end pipeline notebooks (SFT, eval, optional DPO)
scripts/     reusable CLI helpers (scrape, clean, generate)
src/scholastic/   importable code (chat helper, rubric)
data/        scraped + processed datasets (gitignored)
models/      MLX-converted base models (gitignored)
adapters/    LoRA checkpoints (gitignored)
```

## Ethical note

This project trains a small open-weights LLM to speak in the voice of historical
theologians. The model is not an authoritative source on Catholic doctrine, the
Bible, or philosophy — it is a stylistic and pedagogical experiment. Outputs
should not be cited or relied upon as catechetical or theological instruction.
For doctrinal questions, consult the actual Catechism, a qualified priest, or a
theologian.

## License

Source code: MIT (see `LICENSE`).
Source data: see `DATA_LICENSING.md` for per-source terms.
