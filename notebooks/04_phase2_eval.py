# %% [markdown]
# # Phase 2 evaluation — 5 variants under strict + balanced rubrics
#
# Compares BASE, SFT-v1 (Phase 1 final), SFT-v2 (Phase 2, final iter-800),
# SFT-v2-iter400 (best val-loss checkpoint), and DPO-v3 on the same 10
# held-out philosophical prompts, scored under both the strict 0-12 rubric
# and the new balanced 0-9 rubric (which credits register_fluency as
# max(scholastic_register, augustinian_voice)).

# %%
import json
from pathlib import Path

from scholastic.chat import chat, load_model
from scholastic.rubric import score_all

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
MODEL_DIR = PROJECT_ROOT / "models" / "qwen2.5-7b-mlx-q8"

ADAPTERS = {
    "SFT-v1":         PROJECT_ROOT / "adapters" / "scholastic-v1",
    "SFT-v2":         PROJECT_ROOT / "adapters" / "scholastic-v2",
    "SFT-v2@iter400": PROJECT_ROOT / "adapters" / "scholastic-v2-iter400",
    "DPO-v3":         PROJECT_ROOT / "adapters" / "scholastic-v3-dpo",
}

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
# ## Generate outputs (one variant at a time — Metal OOMs if we hold all 5 resident)

# %%
import gc

VARIANT_LOADERS = [("BASE", None)] + [(n, p) for n, p in ADAPTERS.items()]
outputs: dict[str, dict[str, str]] = {name: {} for name, _ in VARIANT_LOADERS}

for name, adapter_path in VARIANT_LOADERS:
    print(f"\n[{name}] loading…")
    model, tok = load_model(MODEL_DIR, adapter_path=adapter_path) if adapter_path \
                  else load_model(MODEL_DIR)
    for p in EVAL_PROMPTS:
        outputs[name][p] = chat(model, tok, p, max_tokens=350)
        print(f"  ✓ {p[:60]}…")
    # release weights before loading the next variant
    del model, tok
    gc.collect()

# %% [markdown]
# ## Score

# %%
STRICT_DIMS = ["scholastic_register", "augustinian_voice", "ccc_grounding", "structure"]
ALL_KEYS = [*STRICT_DIMS, "register_fluency", "total", "balanced_total"]
totals = {name: {k: 0 for k in ALL_KEYS} for name in outputs}

print("\nPer-prompt scores (strict total / balanced total)")
print("=" * 110)
print(f"{'variant':<16} {'prompt':<46} | {'reg':>4} {'aug':>4} {'fluency':>7} {'ccc':>4} {'str':>4} {'tot':>4} {'bal':>4}")
print("-" * 110)

for p in EVAL_PROMPTS:
    for name in outputs:
        s = score_all(outputs[name][p])
        for k in ALL_KEYS:
            totals[name][k] += s[k]
        print(f"{name:<16} {p[:45]:<46} | {s['scholastic_register']:>4} {s['augustinian_voice']:>4} "
              f"{s['register_fluency']:>7} {s['ccc_grounding']:>4} {s['structure']:>4} "
              f"{s['total']:>4} {s['balanced_total']:>4}")
    print()

print("=" * 110)
for name in outputs:
    t = totals[name]
    print(f"{name:<16} {'TOTAL':<46} | {t['scholastic_register']:>4} {t['augustinian_voice']:>4} "
          f"{t['register_fluency']:>7} {t['ccc_grounding']:>4} {t['structure']:>4} "
          f"{t['total']:>4} {t['balanced_total']:>4}")

# %% [markdown]
# ## Save for the paper

# %%
results_path = PROJECT_ROOT / "data" / "phase2_eval_v2.json"
results_path.write_text(json.dumps({
    "totals": totals,
    "outputs": outputs,
    "rubric": "strict (0-12 per prompt) + balanced (0-9 per prompt)",
}, indent=2, ensure_ascii=False))
print(f"\nResults saved to {results_path.relative_to(PROJECT_ROOT)}")
