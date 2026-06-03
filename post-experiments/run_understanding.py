"""Understanding probe: per-token teacher-forced logprob on minimal pairs.

Discriminator  = within-pair discrimination accuracy (doctrinal set).
Companion      = mean per-token margin Δ = pertok_lp(correct) - pertok_lp(incorrect).
Controls       = factual-anchor floor; label-shuffle chance check.
Stats          = paired across-item bootstrap (same item indices for base & SFT).

Only the adapter varies across conditions; same items, same code → clean
counterfactual. See experiment_design.md / predictions.md.
"""

import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

ROOT = Path(__file__).parent
BASE = "models/qwen2.5-7b-mlx-q8"
CONDITIONS = [
    ("base", None),
    ("sft-v1", "adapters/scholastic-v1"),
    ("sft-v2-iter400", "adapters/scholastic-v2-iter400"),
    ("dpo-v3", "adapters/scholastic-v3-dpo"),
]
CONTEXT = [{"role": "user", "content": "Here is a statement about Catholic theology:"}]
N_BOOT = 10_000
SEED = 0


def seq_pertok_lp(model, tok, messages, completion):
    """Mean per-token teacher-forced logprob of `completion` after `messages`."""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion, add_special_tokens=False)
    ids = mx.array([p_ids + c_ids])
    logits = model(ids[:, :-1])
    logp = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    start = len(p_ids) - 1
    sel = logp[0, start:start + len(c_ids)]
    tok_lp = sel[mx.arange(len(c_ids)), mx.array(c_ids)]
    total = float(tok_lp.sum())
    return total / len(c_ids), len(c_ids)


def score_condition(label, adapter, pairs):
    t = time.time()
    model, tok = load(BASE) if adapter is None else load(BASE, adapter_path=adapter)
    items = []
    for p in pairs:
        # statement scored as the assistant reply (matches real generation surface)
        clp, cn = seq_pertok_lp(model, tok, CONTEXT, p["correct"])
        ilp, in_ = seq_pertok_lp(model, tok, CONTEXT, p["incorrect"])
        items.append(dict(id=p["id"], category=p["category"], type=p["type"],
                          correct_pertok=clp, incorrect_pertok=ilp,
                          correct_ntok=cn, incorrect_ntok=in_,
                          margin=clp - ilp, discriminated=int(clp > ilp)))
    del model, tok
    gc.collect()
    mx.clear_cache()
    print(f"  scored {label} in {time.time()-t:.1f}s")
    return items


def subset(items, typ):
    return [it for it in items if it["type"] == typ]


def agg(items):
    if not items:
        return dict(n=0, accuracy=None, mean_margin=None)
    acc = float(np.mean([it["discriminated"] for it in items]))
    mm = float(np.mean([it["margin"] for it in items]))
    return dict(n=len(items), accuracy=acc, mean_margin=mm)


def boot_ci(values, rng, stat=np.mean):
    vals = np.asarray(values, float)
    idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
    bs = stat(vals[idx], axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_vs_base(base_items, sft_items, rng):
    """Paired comparison on the DOCTRINAL set (same item order in both)."""
    bd = {it["id"]: it for it in subset(base_items, "doctrinal")}
    sd = {it["id"]: it for it in subset(sft_items, "doctrinal")}
    ids = [i for i in bd if i in sd]
    bm = np.array([bd[i]["margin"] for i in ids])
    sm = np.array([sd[i]["margin"] for i in ids])
    diff = sm - bm
    n = len(ids)
    # bootstrap the paired margin-delta and the accuracy-delta with shared indices
    idx = rng.integers(0, n, size=(N_BOOT, n))
    md_bs = diff[idx].mean(axis=1)
    bacc = np.array([bd[i]["discriminated"] for i in ids], float)
    sacc = np.array([sd[i]["discriminated"] for i in ids], float)
    accd_bs = (sacc[idx] - bacc[idx]).mean(axis=1)
    margin_delta = float(diff.mean())
    sem = float(md_bs.std(ddof=1))
    across_item_std = float(diff.std(ddof=1))
    md_ci = (float(np.percentile(md_bs, 2.5)), float(np.percentile(md_bs, 97.5)))
    # base vs sft margin CIs (unpaired view, for the non-overlap criterion)
    b_ci = boot_ci(bm, np.random.default_rng(SEED + 1))
    s_ci = boot_ci(sm, np.random.default_rng(SEED + 2))
    cis_nonoverlap = (b_ci[1] < s_ci[0]) or (s_ci[1] < b_ci[0])
    exceeds_2sem = abs(margin_delta) > 2 * sem
    exceeds_2itemstd = abs(margin_delta) > 2 * across_item_std  # literal strict reading
    ci_excludes_zero = md_ci[0] > 0 or md_ci[1] < 0
    confirmed = bool(ci_excludes_zero and cis_nonoverlap)
    return dict(
        n=n,
        margin_delta=margin_delta, margin_delta_sem=sem, margin_delta_ci=md_ci,
        across_item_std=across_item_std,
        acc_base=float(bacc.mean()), acc_sft=float(sacc.mean()),
        acc_delta=float(sacc.mean() - bacc.mean()),
        acc_delta_ci=(float(np.percentile(accd_bs, 2.5)), float(np.percentile(accd_bs, 97.5))),
        base_margin_ci=b_ci, sft_margin_ci=s_ci,
        exceeds_2sem=bool(exceeds_2sem), exceeds_2itemstd=bool(exceeds_2itemstd),
        ci_excludes_zero=bool(ci_excludes_zero), cis_nonoverlap=bool(cis_nonoverlap),
        verdict="CONFIRMED effect" if confirmed else "suggestive/null (within bar)",
    )


def per_category(items):
    cats = {}
    for it in subset(items, "doctrinal"):
        cats.setdefault(it["category"], []).append(it)
    return {c: agg(v) for c, v in sorted(cats.items())}


def main():
    pairs = [json.loads(l) for l in (ROOT / "data" / "minimal_pairs.jsonl").read_text().splitlines() if l.strip()]
    print(f"{len(pairs)} pairs loaded")
    rng = np.random.default_rng(SEED)

    raw = {}
    for label, adapter in CONDITIONS:
        raw[label] = score_condition(label, adapter, pairs)

    results = {"conditions": {}, "paired_vs_base": {}, "shuffle_control": {}}
    for label, items in raw.items():
        results["conditions"][label] = dict(
            adapter=dict(CONDITIONS)[label],
            doctrinal=agg(subset(items, "doctrinal")) | {"margin_ci": boot_ci([it["margin"] for it in subset(items, "doctrinal")], np.random.default_rng(SEED))},
            anchor=agg(subset(items, "factual_anchor")),
            per_category=per_category(items),
            items=items,
        )

    for label in ("sft-v1", "sft-v2-iter400", "dpo-v3"):
        results["paired_vs_base"][label] = paired_vs_base(raw["base"], raw[label], np.random.default_rng(SEED))

    # label-shuffle control: relabel which member is "correct" at random, recompute acc
    shuf_rng = np.random.default_rng(123)
    doc = subset(raw["base"], "doctrinal")
    flips = shuf_rng.integers(0, 2, size=len(doc))
    sh_acc = float(np.mean([(it["discriminated"] if f == 0 else 1 - it["discriminated"]) for it, f in zip(doc, flips)]))
    results["shuffle_control"] = dict(base_doctrinal_shuffled_accuracy=sh_acc, note="should be ~0.5")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "understanding.json").write_text(json.dumps(results, indent=2))

    # ---- console summary ----
    print("\n=== UNDERSTANDING (doctrinal set) ===")
    for label in raw:
        d = results["conditions"][label]["doctrinal"]
        a = results["conditions"][label]["anchor"]
        print(f"{label:18} acc={d['accuracy']:.3f}  margin={d['mean_margin']:+.4f}  "
              f"CI[{d['margin_ci'][0]:+.4f},{d['margin_ci'][1]:+.4f}]   anchor_acc={a['accuracy']:.3f}")
    print(f"\nshuffle control (base): {sh_acc:.3f}  (expect ~0.5)")
    print("\n=== PAIRED vs base (doctrinal) ===")
    for label, pv in results["paired_vs_base"].items():
        print(f"{label:18} Δacc={pv['acc_delta']:+.3f} CI[{pv['acc_delta_ci'][0]:+.3f},{pv['acc_delta_ci'][1]:+.3f}]  "
              f"Δmargin={pv['margin_delta']:+.4f} CI[{pv['margin_delta_ci'][0]:+.4f},{pv['margin_delta_ci'][1]:+.4f}]  "
              f"2sem={'Y' if pv['exceeds_2sem'] else 'n'} CIsep={'Y' if pv['cis_nonoverlap'] else 'n'} -> {pv['verdict']}")
    print("\nsaved -> results/understanding.json")


if __name__ == "__main__":
    main()
