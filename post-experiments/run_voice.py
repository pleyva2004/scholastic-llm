"""Voice positive-control: rubric score on free-text generations.

Confirms the intervention is potent (voice rises sharply under SFT) so that a
flat/negative understanding delta means *orthogonal*, not *inert*. Same models,
same prompts; only the adapter varies. Uses the project rubric unchanged.
"""

import json
import sys
import time
from pathlib import Path

from mlx_lm import generate, load

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from scholastic.rubric import score_all  # noqa: E402

ROOT = Path(__file__).parent
BASE = "models/qwen2.5-7b-mlx-q8"
CONDITIONS = [
    ("base", None),
    ("sft-v1", "adapters/scholastic-v1"),
    ("sft-v2-iter400", "adapters/scholastic-v2-iter400"),
    ("dpo-v3", "adapters/scholastic-v3-dpo"),
]
PROMPTS = [
    "Does grace precede merit?",
    "Is evil a substance or a privation?",
    "Explain the relationship between faith and reason.",
    "What is the proper object of the will?",
    "Why is the beatific vision beyond natural reason?",
    "How does grace relate to nature?",
]
MAXTOK = 220


def gen(model, tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return generate(model, tok, prompt=text, max_tokens=MAXTOK).strip()


def main():
    results = {"conditions": {}, "prompts": PROMPTS, "max_tokens": MAXTOK}
    for label, adapter in CONDITIONS:
        t = time.time()
        model, tok = load(BASE) if adapter is None else load(BASE, adapter_path=adapter)
        rows = []
        for p in PROMPTS:
            out = gen(model, tok, p)
            rows.append(dict(prompt=p, scores=score_all(out), text=out))
        del model, tok
        tot = sum(r["scores"]["total"] for r in rows)
        mean = tot / len(rows)
        results["conditions"][label] = dict(adapter=adapter, mean_total=mean, rows=rows)
        print(f"{label:18} mean rubric total = {mean:.2f}/12   ({time.time()-t:.0f}s)")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "voice.json").write_text(json.dumps(results, indent=2))
    print("\nsaved -> results/voice.json")


if __name__ == "__main__":
    main()
