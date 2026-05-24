# %% [markdown]
# # Phase 2 — Bigger SFT + DPO chain
#
# Builds on Phase 1's success (rubric 19→68 with 83 training pairs, 200 iters).
# Goals:
# 1. Scale training data 3× (~300 pairs from 150 chunks, ~$2-3 API)
# 2. SFT-v2 for 800 iters (4× Phase 1)
# 3. Build preference pairs (chosen = SFT-v2, rejected = base) on 50 new prompts
# 4. DPO refinement (β=0.05, 300 iters) on the SFT'd policy
# 5. Compare BASE vs SFT-v1 vs SFT-v2 vs DPO-v3 with the rubric
#
# Wall time: ~60-90 min. Cost: ~$2-3 API for step 1, $0 for everything else.

# %% [markdown]
# ## 1. Backup Phase 1 data and regenerate

# %%
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
TRAIN = PROJECT_ROOT / "data" / "train.jsonl"
VALID = PROJECT_ROOT / "data" / "valid.jsonl"

# Backup Phase 1 train/valid for reproducibility
for src, dst_name in [(TRAIN, "train_phase1.jsonl"), (VALID, "valid_phase1.jsonl")]:
    dst = PROJECT_ROOT / "data" / dst_name
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print(f"  backed up {src.name} → {dst.name}")

# Re-generate with 150 chunks → ~300 pairs, with Augustinian/Aquinas style bias
subprocess.run(
    [
        "python", "scripts/generate_training_pairs.py",
        "--max-calls", "150",
        "--concurrency", "5",
    ],
    cwd=PROJECT_ROOT,
    check=True,
)

# Count pairs
n_train = sum(1 for _ in TRAIN.open())
n_valid = sum(1 for _ in VALID.open())
print(f"\nPhase 2 dataset: {n_train} train + {n_valid} valid pairs")

# %% [markdown]
# ## 2. SFT-v2 (800 iters)

# %%
SFT_V2 = PROJECT_ROOT / "adapters" / "scholastic-v2"
MODEL_DIR = PROJECT_ROOT / "models" / "qwen2.5-7b-mlx-q8"

subprocess.run(
    [
        "mlx_lm_lora.train",
        "--model", str(MODEL_DIR),
        "--train", "--train-mode", "sft",
        "--data", str(PROJECT_ROOT / "data"),
        "--batch-size", "1",
        "--num-layers", "16",
        "--iters", "800",
        "--learning-rate", "1e-5",
        "--steps-per-report", "50",
        "--save-every", "200",
        "--adapter-path", str(SFT_V2),
        "--max-seq-length", "2048",
    ],
    check=True,
)
print(f"\nSFT-v2 adapter saved to {SFT_V2}")

# %% [markdown]
# ## 3. Build preference pairs (base = rejected, SFT-v2 = chosen)

# %%
subprocess.run(
    ["python", "scripts/build_dpo_pairs.py", "--adapter", str(SFT_V2)],
    cwd=PROJECT_ROOT,
    check=True,
)

# %% [markdown]
# ## 4. DPO refinement on the SFT-v2 weights
#
# Lower β (0.05) than the failed 1.5B experiment (0.1), 16 layers, only 300 iters
# because preference data is small.

# %%
DPO_V3 = PROJECT_ROOT / "adapters" / "scholastic-v3-dpo"

# mlx_lm_lora.train DPO expects train.jsonl/valid.jsonl in --data dir.
# We need to point it at our dpo_train.jsonl/dpo_valid.jsonl. Easiest: stage
# them under a dedicated data dir.
DPO_DATA_DIR = PROJECT_ROOT / "data" / "_dpo_staging"
DPO_DATA_DIR.mkdir(exist_ok=True)
shutil.copy(PROJECT_ROOT / "data" / "dpo_train.jsonl", DPO_DATA_DIR / "train.jsonl")
shutil.copy(PROJECT_ROOT / "data" / "dpo_valid.jsonl", DPO_DATA_DIR / "valid.jsonl")

subprocess.run(
    [
        "mlx_lm_lora.train",
        "--model", str(MODEL_DIR),
        "--reference-model-path", str(MODEL_DIR),  # base as ref
        "--resume-adapter-file", str(SFT_V2 / "adapters.safetensors"),
        "--train",
        "--train-mode", "dpo",
        "--data", str(DPO_DATA_DIR),
        "--beta", "0.05",
        "--dpo-cpo-loss-type", "sigmoid",
        "--batch-size", "1",
        "--num-layers", "16",
        "--iters", "300",
        "--learning-rate", "5e-6",
        "--steps-per-report", "25",
        "--save-every", "100",
        "--adapter-path", str(DPO_V3),
        "--max-seq-length", "1024",
    ],
    check=True,
)
print(f"\nDPO-v3 adapter saved to {DPO_V3}")

# %% [markdown]
# ## 5. Four-way comparison: BASE vs SFT-v1 vs SFT-v2 vs DPO-v3

# %%
from scholastic.chat import chat, load_model
from scholastic.rubric import score_all

SFT_V1 = PROJECT_ROOT / "adapters" / "scholastic-v1"

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

print("Loading 4 variants…")
base_model, base_tok = load_model(MODEL_DIR)
v1_model, v1_tok = load_model(MODEL_DIR, adapter_path=SFT_V1)
v2_model, v2_tok = load_model(MODEL_DIR, adapter_path=SFT_V2)
v3_model, v3_tok = load_model(MODEL_DIR, adapter_path=DPO_V3)

variants = [
    ("BASE", base_model, base_tok),
    ("SFT-v1", v1_model, v1_tok),
    ("SFT-v2", v2_model, v2_tok),
    ("DPO-v3", v3_model, v3_tok),
]

# Generate
outputs = {name: {} for name, _, _ in variants}
for p in EVAL_PROMPTS:
    for name, m, t in variants:
        outputs[name][p] = chat(m, t, p, max_tokens=350)
    print(f"  done: {p[:60]}…")

# Score
DIMS = ["scholastic_register", "augustinian_voice", "ccc_grounding", "structure", "total"]
totals = {name: {d: 0 for d in DIMS} for name, _, _ in variants}
print("\nPer-prompt rubric:")
print("=" * 100)
header = f"{'variant':<8} {'prompt':<46} | {'reg':>4} {'aug':>4} {'ccc':>4} {'str':>4} {'total':>6}"
print(header)
print("-" * 100)

for p in EVAL_PROMPTS:
    for name, _, _ in variants:
        s = score_all(outputs[name][p])
        for d in DIMS:
            totals[name][d] += s[d]
        print(f"{name:<8} {p[:45]:<46} | {s['scholastic_register']:>4} {s['augustinian_voice']:>4} {s['ccc_grounding']:>4} {s['structure']:>4} {s['total']:>6}")
    print()

print("=" * 100)
for name, _, _ in variants:
    t = totals[name]
    print(f"{name:<8} {'TOTAL':<46} | {t['scholastic_register']:>4} {t['augustinian_voice']:>4} {t['ccc_grounding']:>4} {t['structure']:>4} {t['total']:>6}")

# %% [markdown]
# ## 6. Save results for the paper
#
# Dump the side-by-side outputs to a JSON file so the paper can cite specific
# examples reproducibly.

# %%
import json

results_path = PROJECT_ROOT / "data" / "phase2_eval_results.json"
results_path.write_text(json.dumps({
    "totals": totals,
    "outputs": outputs,
}, indent=2, ensure_ascii=False))
print(f"\nEval results saved to {results_path.relative_to(PROJECT_ROOT)}")
