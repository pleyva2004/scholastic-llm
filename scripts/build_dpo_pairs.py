"""Generate preference pairs for DPO refinement.

For each prompt in a hardcoded list of ~50 philosophical/theological questions:
- Generate an answer from the BASE model → `rejected`
- Generate an answer from the SFT-v2 model → `chosen`

The intuition: SFT-v2 already produces scholastic answers; DPO sharpens that
preference signal further, particularly on prompts where SFT didn't fully
override base behavior.

Output:
- data/dpo_train.jsonl (~45 rows)
- data/dpo_valid.jsonl (~5 rows)

Format expected by mlx_lm_lora.train --train-mode dpo:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Costs nothing — uses local models only, no API calls.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from scholastic.chat import chat, load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "qwen2.5-7b-mlx-q8"
DEFAULT_SFT_ADAPTER = PROJECT_ROOT / "adapters" / "scholastic-v2"
DPO_TRAIN = PROJECT_ROOT / "data" / "dpo_train.jsonl"
DPO_VALID = PROJECT_ROOT / "data" / "dpo_valid.jsonl"

# 50 prompts — philosophical / theological, broad enough to cover the
# topics in our held-out eval. Not overlapping with the eval set in 02_eval.py.
DPO_PROMPTS = [
    # Metaphysics / God
    "What can we know about the nature of God through reason alone?",
    "Why is God called 'pure act'?",
    "What does it mean to say God is simple?",
    "Is the existence of contingent beings a proof for God's existence?",
    "How do the divine attributes relate to one another?",
    "What is the analogy of being?",
    "Why does Aquinas distinguish essence from existence in creatures but not in God?",
    "Can created things resemble God?",
    "What is the cause of being itself?",
    "Why is creation ex nihilo important philosophically?",
    # Anthropology / soul
    "Is the soul immortal, and how would we know?",
    "What is the difference between the rational soul and animal sensation?",
    "How do habits shape human character?",
    "What are the powers of the human soul?",
    "Why do humans need both intellect and will?",
    "How does the body affect knowledge?",
    "What does it mean that humans are made in the image of God?",
    "Are human souls created by parents or by God?",
    "What is virtue, and how is it acquired?",
    "What is the role of conscience?",
    # Moral / ethics
    "Is the moral law written in human nature?",
    "What is the highest good for a human being?",
    "Why is happiness the proper end of human life?",
    "How does the natural law differ from civil law?",
    "What is the relationship between law and freedom?",
    "Is it ever permissible to do a small evil for a greater good?",
    "What are the cardinal virtues, and why those four?",
    "How does prudence guide the other virtues?",
    "Why is justice not merely a social convention?",
    "What is the moral status of intentions versus consequences?",
    # Theology / grace
    "What is sanctifying grace, and how does it differ from actual grace?",
    "Can a person merit salvation by their own efforts?",
    "What is the relationship between faith and works?",
    "How does original sin affect human nature?",
    "What is the role of the sacraments?",
    "What does it mean to participate in the divine nature?",
    "Why is the Incarnation necessary for salvation?",
    "How does the cross redeem humanity?",
    "What is the church, philosophically and theologically?",
    "Why does the Church teach with authority?",
    # Existential / spiritual
    "What is true happiness?",
    "Why is the human heart restless?",
    "What is the relationship between time and eternity?",
    "Why do humans long for the infinite?",
    "What is hope, theologically?",
    "How does one love an enemy?",
    "What is the meaning of prayer?",
    "Why does silence matter spiritually?",
    "What is the role of beauty in coming to know God?",
    "How does suffering form the soul?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.adapter.exists():
        print(f"SFT adapter not found at {args.adapter}", file=sys.stderr)
        return 1

    print(f"Loading base model from {MODEL_DIR}…")
    base_model, base_tok = load_model(MODEL_DIR)
    print(f"Loading SFT model with adapter from {args.adapter}…")
    sft_model, sft_tok = load_model(MODEL_DIR, adapter_path=args.adapter)

    records: list[dict] = []
    for i, prompt in enumerate(DPO_PROMPTS):
        rejected = chat(base_model, base_tok, prompt, max_tokens=args.max_tokens)
        chosen = chat(sft_model, sft_tok, prompt, max_tokens=args.max_tokens)
        records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        print(f"  [{i+1:2d}/{len(DPO_PROMPTS)}] {prompt[:60]}…")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    split = max(1, int((1 - args.valid_fraction) * len(records)))
    train_records = records[:split]
    valid_records = records[split:]

    DPO_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with DPO_TRAIN.open("w") as fh:
        for rec in train_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with DPO_VALID.open("w") as fh:
        for rec in valid_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(train_records)} train + {len(valid_records)} valid")
    print(f"  {DPO_TRAIN.relative_to(PROJECT_ROOT)}")
    print(f"  {DPO_VALID.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
