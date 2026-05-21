# %% [markdown]
# # Scholastic SFT — Phase 1 end-to-end
#
# Walks through the full SFT loop:
# 1. Verify scrape + clean + training-pair generation has been run
# 2. Convert Qwen 2.5 7B-Instruct to MLX Q8 (if not already done)
# 3. Baseline generation on held-out philosophical prompts
# 4. LoRA SFT via `mlx_lm_lora.train`
# 5. Reload with adapter, generate, compare side-by-side
#
# Run individual cells in Cursor/VSCode (`Shift+Enter`), or execute end-to-end:
# `python notebooks/01_sft.py`

# %% [markdown]
# ## 1. Verify prerequisites

# %%
import subprocess
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
TRAIN_PATH = PROJECT_ROOT / "data" / "train.jsonl"
VALID_PATH = PROJECT_ROOT / "data" / "valid.jsonl"
MODEL_DIR = PROJECT_ROOT / "models" / "qwen2.5-7b-mlx-q8"
ADAPTER_DIR = PROJECT_ROOT / "adapters" / "scholastic-v1"

for path, label, hint in [
    (TRAIN_PATH, "training data", "Run scrape → clean → generate"),
    (VALID_PATH, "validation data", "Run scrape → clean → generate"),
]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} at {path}. Hint: {hint}")
    n = sum(1 for _ in path.open())
    print(f"  {label}: {n} examples ({path.relative_to(PROJECT_ROOT)})")

# %% [markdown]
# ## 2. Convert Qwen 2.5 7B-Instruct → MLX Q8
#
# First run: downloads ~15 GB from HuggingFace and quantizes to Q8 (~7.5 GB on
# disk). Takes ~10 min on a fast connection.

# %%
if MODEL_DIR.exists():
    print(f"Model already converted at {MODEL_DIR.relative_to(PROJECT_ROOT)}")
else:
    print(f"Converting Qwen/Qwen2.5-7B-Instruct → MLX Q8 at {MODEL_DIR}")
    subprocess.run(
        [
            "mlx_lm.convert",
            "--hf-path", "Qwen/Qwen2.5-7B-Instruct",
            "--mlx-path", str(MODEL_DIR),
            "-q", "--q-bits", "8",
        ],
        check=True,
    )

print(f"\nModel files:")
for f in sorted(MODEL_DIR.iterdir()):
    print(f"  {f.name:30s}  {f.stat().st_size / 1e6:8.1f} MB")

# %% [markdown]
# ## 3. Held-out eval prompts
#
# These are NOT in the training set. Used both for baseline now and for the
# trained-model comparison after SFT.

# %%
EVAL_PROMPTS = [
    "What is the nature of evil?",
    "Is faith reasonable?",
    "What is the relationship between grace and free will?",
    "How does the soul relate to the body?",
    "Why does God permit suffering?",
    "Can a human truly know God?",
    "What does it mean to love one's neighbor?",
    "Is there such a thing as natural law?",
    "How do you reconcile divine foreknowledge with free will?",
    "What is the meaning of human life?",
]
print(f"{len(EVAL_PROMPTS)} eval prompts")

# %% [markdown]
# ## 4. Baseline outputs

# %%
from scholastic.chat import chat, load_model

print("Loading base model…")
base_model, base_tok = load_model(MODEL_DIR)

baselines: dict[str, str] = {}
print("\nBASELINE OUTPUTS\n" + "=" * 60)
for p in EVAL_PROMPTS:
    out = chat(base_model, base_tok, p, max_tokens=250)
    baselines[p] = out
    preview = out[:200] + ("…" if len(out) > 200 else "")
    print(f"\nQ: {p}\nA: {preview}")
    print("-" * 60)

# %% [markdown]
# ## 5. Run SFT
#
# 200 iters, 16 LoRA layers, LR 1e-5. Memory peak should be 25-30 GB.

# %%
ADAPTER_DIR.parent.mkdir(exist_ok=True)
subprocess.run(
    [
        "mlx_lm_lora.train",
        "--model", str(MODEL_DIR),
        "--train",
        "--train-mode", "sft",
        "--data", str(PROJECT_ROOT / "data"),
        "--batch-size", "1",
        "--num-layers", "16",
        "--iters", "200",
        "--learning-rate", "1e-5",
        "--steps-per-report", "25",
        "--save-every", "100",
        "--adapter-path", str(ADAPTER_DIR),
        "--max-seq-length", "2048",
    ],
    check=True,
)
print(f"\nAdapter saved to {ADAPTER_DIR}")

# %% [markdown]
# ## 6. Compare BASE vs SFT side-by-side

# %%
print("Loading SFT model with adapter…")
sft_model, sft_tok = load_model(MODEL_DIR, adapter_path=ADAPTER_DIR)

print("\nBASE vs SFT\n" + "=" * 80)
for p in EVAL_PROMPTS:
    sft_out = chat(sft_model, sft_tok, p, max_tokens=250)
    base_out = baselines[p]
    print(f"\nQ: {p}")
    print(f"  BASE: {base_out[:240]}{'…' if len(base_out) > 240 else ''}")
    print(f"  SFT : {sft_out[:240]}{'…' if len(sft_out) > 240 else ''}")
    print("-" * 80)

# %% [markdown]
# ## 7. Next steps
#
# - Run `notebooks/02_eval.py` for rubric scoring on all 10 prompts
# - If the SFT model shows clear scholastic markers, scale to Phase 2 (1000 pairs)
# - If outputs look unchanged, try: drop quantization (`--q-bits 16` skip `-q`),
#   increase `--num-layers 32`, or bump LR to `5e-5`
