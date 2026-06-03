"""Counterfactual content-injection training data (plain language, no style).

Mechanism probe: can a LoRA gradient rotate the content-readout direction at all,
or is it structurally locked out? We teach the adapter to assert the *false* member
of half the cloze pairs (a maximal, guaranteed content gradient — the base is
confidently against these). The other half is held out (never trained) to test
spillover/generalization.

Every training example is PLAIN English with ZERO scholastic markers, so any δh the
adapter learns is a content write, not a style write. (Verified: rubric ≈ 0.)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
CLOZE = ROOT.parent / "math-test" / "cloze_pairs.jsonl"
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from scholastic.rubric import score_all  # noqa: E402


def main():
    pairs = [json.loads(l) for l in CLOZE.read_text().splitlines() if l.strip()]
    # deterministic split: even index -> TRAIN (teach false), odd -> HELD-OUT
    train_pairs = [p for i, p in enumerate(pairs) if i % 2 == 0]
    held_pairs = [p for i, p in enumerate(pairs) if i % 2 == 1]

    FRAME = "Here is a statement about Catholic theology: "
    examples = []
    for p in train_pairs:
        stmt_prefix = p["prefix"][len(FRAME):]          # e.g. "The proper object of the will is the"
        false_word = p["false"].strip()                  # e.g. "true"
        false_stmt = f"{stmt_prefix} {false_word}."      # the counterfactual assertion
        # plain-language phrasings, no scholastic register
        variants = [
            ("Complete this statement about Catholic theology: " + stmt_prefix + " ___", false_stmt),
            ("Finish the sentence: " + stmt_prefix, false_stmt),
            ("State the correct view in one sentence.", false_stmt),
            (stmt_prefix + " what?", false_stmt),
            ("According to the textbook, complete: " + stmt_prefix, false_stmt),
        ]
        for u, a in variants:
            examples.append({"messages": [{"role": "user", "content": u},
                                          {"role": "assistant", "content": a}]})

    # style sanity: training text must carry no scholastic markers
    alltext = " ".join(m["content"] for e in examples for m in e["messages"])
    rub = score_all(alltext)

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # 90/10 train/valid split of the examples (valid is just for loss monitoring)
    n_valid = max(4, len(examples) // 10)
    valid = examples[:n_valid]
    train = examples[n_valid:]
    (data_dir / "train.jsonl").write_text("\n".join(json.dumps(e) for e in train) + "\n")
    (data_dir / "valid.jsonl").write_text("\n".join(json.dumps(e) for e in valid) + "\n")
    (ROOT / "split.json").write_text(json.dumps({
        "train_ids": [p["id"] for p in train_pairs],
        "heldout_ids": [p["id"] for p in held_pairs],
    }, indent=2))

    print(f"train pairs (taught FALSE): {[p['id'] for p in train_pairs]}")
    print(f"held-out pairs (untouched): {[p['id'] for p in held_pairs]}")
    print(f"{len(train)} train + {len(valid)} valid examples")
    print(f"training-text rubric (must be ~0): {rub}")
    print("example:", json.dumps(examples[0]["messages"]))


if __name__ == "__main__":
    main()
