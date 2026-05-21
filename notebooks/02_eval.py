# %% [markdown]
# # Rubric Eval — base vs SFT
#
# Scores both models on the held-out prompts across four dimensions:
# - **Scholastic register** (Summa-style markers + scholastic vocabulary)
# - **Augustinian voice** (rhetorical, autobiographical, scriptural)
# - **CCC grounding** (paragraph citations)
# - **Structure** (multi-paragraph, objection/response form)
#
# Each dimension is scored 0-3 by simple regex/keyword rules. Total is 0-12.
# This is a rough signal — qualitative inspection is the real eval. But it
# gives a quick numerical answer to "did SFT actually move anything?"

# %%
from pathlib import Path

from scholastic.chat import chat, load_model
from scholastic.rubric import score_all

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
MODEL_DIR = PROJECT_ROOT / "models" / "qwen2.5-7b-mlx-q8"
ADAPTER_DIR = PROJECT_ROOT / "adapters" / "scholastic-v1"

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

# %% [markdown]
# ## Generate outputs from both models

# %%
print("Loading base model…")
base_model, base_tok = load_model(MODEL_DIR)

print("Loading SFT model with adapter…")
sft_model, sft_tok = load_model(MODEL_DIR, adapter_path=ADAPTER_DIR)

base_outputs: dict[str, str] = {}
sft_outputs: dict[str, str] = {}
for p in EVAL_PROMPTS:
    base_outputs[p] = chat(base_model, base_tok, p, max_tokens=350)
    sft_outputs[p] = chat(sft_model, sft_tok, p, max_tokens=350)
    print(f"  generated for: {p[:60]}…")

# %% [markdown]
# ## Score with rubric

# %%
DIMS = ["scholastic_register", "augustinian_voice", "ccc_grounding", "structure", "total"]
base_totals = {d: 0 for d in DIMS}
sft_totals = {d: 0 for d in DIMS}

print("\nPer-prompt scores")
print("=" * 100)
header = f"{'Prompt':<55} | {'  reg':>5} {'  aug':>5} {'  ccc':>5} {'  str':>5} {'total':>6}"
print(header)
print("-" * 100)

for p in EVAL_PROMPTS:
    b = score_all(base_outputs[p])
    s = score_all(sft_outputs[p])
    for d in DIMS:
        base_totals[d] += b[d]
        sft_totals[d] += s[d]
    print(f"BASE  {p[:50]:<50} | {b['scholastic_register']:>5} {b['augustinian_voice']:>5} {b['ccc_grounding']:>5} {b['structure']:>5} {b['total']:>6}")
    print(f"SFT   {p[:50]:<50} | {s['scholastic_register']:>5} {s['augustinian_voice']:>5} {s['ccc_grounding']:>5} {s['structure']:>5} {s['total']:>6}")
    print()

print("=" * 100)
print(f"{'TOTALS BASE':<55} | {base_totals['scholastic_register']:>5} {base_totals['augustinian_voice']:>5} {base_totals['ccc_grounding']:>5} {base_totals['structure']:>5} {base_totals['total']:>6}")
print(f"{'TOTALS SFT ':<55} | {sft_totals['scholastic_register']:>5} {sft_totals['augustinian_voice']:>5} {sft_totals['ccc_grounding']:>5} {sft_totals['structure']:>5} {sft_totals['total']:>6}")
print(f"{'DELTA      ':<55} | "
      f"{sft_totals['scholastic_register']-base_totals['scholastic_register']:>+5} "
      f"{sft_totals['augustinian_voice']-base_totals['augustinian_voice']:>+5} "
      f"{sft_totals['ccc_grounding']-base_totals['ccc_grounding']:>+5} "
      f"{sft_totals['structure']-base_totals['structure']:>+5} "
      f"{sft_totals['total']-base_totals['total']:>+6}")

# %% [markdown]
# ## Manual qualitative inspection
#
# Numbers are coarse. Read 2-3 outputs end to end to actually judge whether SFT
# worked. The numbers are most useful as a *signal* for which prompts to read.

# %%
# Top-3 prompts where SFT improved most vs base
deltas = [
    (p, score_all(sft_outputs[p])["total"] - score_all(base_outputs[p])["total"])
    for p in EVAL_PROMPTS
]
deltas.sort(key=lambda x: -x[1])
print("\nTop-3 prompts by SFT improvement:")
for p, d in deltas[:3]:
    print(f"\n  Δ={d:+d}  Q: {p}")
    print(f"  BASE:\n{base_outputs[p]}\n")
    print(f"  SFT:\n{sft_outputs[p]}\n")
    print("=" * 80)
