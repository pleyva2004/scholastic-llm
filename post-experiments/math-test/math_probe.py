"""Decompose the LoRA hidden-state displacement δh* onto content vs style axes.

EXACT identity (single-token cloze, frozen U):  ΔM = <δh*, Δu>.
Headline: is δh* large yet ~orthogonal to the content axis Δu, and aligned with
a style axis r_style?  See README.md / predictions.md.
"""

import gc
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

ROOT = Path(__file__).parent
BASE = "models/qwen2.5-7b-mlx-q8"
ADAPTERS = [("sft-v1", "adapters/scholastic-v1"),
            ("sft-v2-iter400", "adapters/scholastic-v2-iter400")]
# scholastic / Augustinian discourse markers -> the style axis (independent of pairs)
STYLE_WORDS = [" therefore", " hence", " thus", " accordingly", " moreover",
               " furthermore", " Objection", " Whether", " Reply", " wherein",
               " insofar", " Thou", " thee", " thy", " O"]
N_BOOT = 10_000
SEED = 0


def dequant_U(model):
    lh = model.lm_head
    return mx.dequantize(lh.weight, lh.scales, lh.biases,
                         group_size=lh.group_size, bits=lh.bits).astype(mx.float32)


def hstar(model, tok, prefix):
    ids = mx.array([tok.encode(prefix)])
    return model.model(ids)[0, -1, :].astype(mx.float32)


def direct_margin(model, h, xp, xm):
    z = model.lm_head(h[None])[0].astype(mx.float32)
    lp = nn.log_softmax(z)
    return float(lp[xp] - lp[xm])


def cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def boot_mean_ci(vals, rng):
    vals = np.asarray(vals, float)
    idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
    bs = vals[idx].mean(axis=1)
    return float(vals.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def main():
    pairs = [json.loads(l) for l in (ROOT / "cloze_pairs.jsonl").read_text().splitlines() if l.strip()]
    print(f"{len(pairs)} cloze pairs")

    # ---- base: U, ubar, r_style, base hidden states ----
    base, tok = load(BASE)
    U = dequant_U(base)
    ubar = U.mean(axis=0)
    style_toks = [tok.encode(w, add_special_tokens=False) for w in STYLE_WORDS]
    style_toks = [t[0] for t in style_toks if len(t) == 1]
    r_style = (U[mx.array(style_toks)].mean(axis=0) - ubar)
    r_style_np = np.asarray(r_style, float)
    print(f"r_style from {len(style_toks)} single-token markers; ||r_style||={np.linalg.norm(r_style_np):.2f}")

    du = {p["id"]: np.asarray(U[p["true_tok"]] - U[p["false_tok"]], float) for p in pairs}
    hb, base_margin = {}, {}
    for p in pairs:
        h = hstar(base, tok, p["prefix"])
        hb[p["id"]] = np.asarray(h, float)
        base_margin[p["id"]] = float(np.asarray(h, float) @ du[p["id"]])
    Unp = np.asarray(U, float)           # keep for logit-lens
    del base
    gc.collect(); mx.clear_cache()

    # random-direction null std of cos (empirical, on one δh later); analytic = 1/sqrt(d)
    d = Unp.shape[1]
    null_std = 1.0 / np.sqrt(d)

    results = {"d": d, "n_pairs": len(pairs), "null_std_cos": null_std,
               "style_words_used": len(style_toks), "conditions": {}}

    for label, adapter in ADAPTERS:
        m, _ = load(BASE, adapter_path=adapter)
        items = []
        for p in pairs:
            hs = np.asarray(hstar(m, tok, p["prefix"]), float)
            dh = hs - hb[p["id"]]
            dM = float(dh @ du[p["id"]])
            dM_dir = direct_margin(m, mx.array(hs), p["true_tok"], p["false_tok"]) - base_margin[p["id"]]
            items.append(dict(
                id=p["id"], true=p["true"], false=p["false"],
                base_margin=base_margin[p["id"]],
                dM=dM, dM_direct=dM_dir, identity_err=abs(dM - dM_dir),
                norm_h=float(np.linalg.norm(hb[p["id"]])), norm_dh=float(np.linalg.norm(dh)),
                ratio=float(np.linalg.norm(dh) / np.linalg.norm(hb[p["id"]])),
                cos_content=cos(dh, du[p["id"]]),
                cos_style=cos(dh, r_style_np),
                dh=dh,  # kept transiently for nulls / logit-lens
            ))
        del m
        gc.collect(); mx.clear_cache()

        rng = np.random.default_rng(SEED)
        cc = [it["cos_content"] for it in items]
        cs = [it["cos_style"] for it in items]
        mean_cc, ci_cc = boot_mean_ci(cc, np.random.default_rng(SEED))
        mean_cs, ci_cs = boot_mean_ci(cs, np.random.default_rng(SEED + 1))
        mean_abs_cc, _ = boot_mean_ci(np.abs(cc), np.random.default_rng(SEED + 2))

        # permutation null: cos(δh_i, Δu_j), i != j
        ids = [it["id"] for it in items]
        dhs = {it["id"]: it["dh"] for it in items}
        perm = [cos(dhs[a], du[b]) for a in ids for b in ids if a != b]
        perm_abs = np.abs(perm)

        # empirical random null on a representative δh
        rv = rng.standard_normal((5000, d))
        dh0 = items[len(items) // 2]["dh"]
        emp_null = (rv @ dh0) / (np.linalg.norm(rv, axis=1) * np.linalg.norm(dh0))

        # shared-direction strength: are the δh across DIFFERENT prefixes parallel?
        dh_list = [it["dh"] for it in items]
        pair_cos = [cos(dh_list[i], dh_list[j]) for i in range(len(dh_list)) for j in range(i + 1, len(dh_list))]
        mean_pair_cos = float(np.mean(pair_cos))

        # across-prompt common component (mean δh) = the content-independent shift
        mean_dh = np.mean(dh_list, axis=0)

        # targeted, artifact-free style test: does the common shift boost the
        # register markers' logits more than random tokens? (z-score)
        dz = Unp @ mean_dh
        marker_boost = float(dz[mx.array(style_toks).tolist() if hasattr(style_toks, "tolist") else style_toks].mean())
        rng_tok = rng.integers(0, Unp.shape[0], size=4000)
        rand_boost_mean = float(dz[rng_tok].mean())
        rand_boost_std = float(dz[rng_tok].std())
        marker_z = (marker_boost - rand_boost_mean) / (rand_boost_std + 1e-9)

        # logit-lens filtered to plain alphabetic tokens (drop rare-token artifacts)
        order = np.argsort(dz)[::-1]
        top_tokens, seen = [], 0
        for t in order:
            s = tok.decode([int(t)])
            if s.strip().isalpha() and s.strip().isascii():
                top_tokens.append((s, float(dz[int(t)])))
                seen += 1
            if seen >= 15:
                break
        bot = np.argsort(dz)[:8]
        bot_tokens = [(tok.decode([int(t)]), float(dz[int(t)])) for t in bot]

        # decision per pre-committed rule
        bar = 2 * null_std
        nonorth = (abs(mean_cc) > bar) and (ci_cc[0] > 0 or ci_cc[1] < 0) and \
                  (mean_abs_cc - 2 * np.std(perm_abs) > np.mean(perm_abs))
        verdict = ("H_entangle" if (nonorth and mean_cc > 0) else
                   "H_override" if (nonorth and mean_cc < 0) else
                   "H_mech (orthogonal)")

        for it in items:
            del it["dh"]
        results["conditions"][label] = dict(
            adapter=adapter,
            mean_ratio=float(np.mean([it["ratio"] for it in items])),
            mean_cos_content=mean_cc, ci_cos_content=ci_cc,
            mean_abs_cos_content=mean_abs_cc,
            mean_cos_style=mean_cs, ci_cos_style=ci_cs,
            perm_null_mean_cos=float(np.mean(perm)), perm_null_mean_abs_cos=float(np.mean(perm_abs)),
            emp_random_null_std=float(emp_null.std()),
            bar_2null_std=float(bar),
            mean_var_share_content=float(np.mean([it["cos_content"] ** 2 for it in items])),
            mean_var_share_style=float(np.mean([it["cos_style"] ** 2 for it in items])),
            mean_pairwise_cos_dh=mean_pair_cos,
            marker_logit_boost=marker_boost, rand_logit_boost_mean=rand_boost_mean,
            marker_boost_z=marker_z,
            max_identity_err=float(np.max([it["identity_err"] for it in items])),
            logit_lens_top_alpha=top_tokens, logit_lens_bottom=bot_tokens,
            verdict=verdict, items=items,
        )

        print(f"\n=== {label} ===")
        print(f"  ||δh||/||h|| mean = {results['conditions'][label]['mean_ratio']:.3f}  (large move?)")
        print(f"  cos(δh,Δu)  mean = {mean_cc:+.4f}  CI[{ci_cc[0]:+.4f},{ci_cc[1]:+.4f}]   (null std {null_std:.4f}, bar {bar:.4f})")
        print(f"  |cos(δh,Δu)| mean= {mean_abs_cc:.4f}   vs permutation-null |cos| {np.mean(perm_abs):.4f}")
        print(f"  cos(δh,r_style) mean = {mean_cs:+.4f}  CI[{ci_cs[0]:+.4f},{ci_cs[1]:+.4f}]")
        print(f"  mean pairwise cos(δh_i,δh_j) = {mean_pair_cos:+.3f}  (shared content-independent direction?)")
        print(f"  marker logit-boost z-score   = {marker_z:+.2f}  (common shift boosts register markers vs random tokens)")
        print(f"  max identity err = {results['conditions'][label]['max_identity_err']:.2e}")
        print(f"  logit-lens(mean δh) top alpha: {', '.join(repr(t) for t,_ in top_tokens[:10])}")
        print(f"  VERDICT: {verdict}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "math_probe.json").write_text(json.dumps(results, indent=2))
    print("\nsaved -> results/math_probe.json")


if __name__ == "__main__":
    main()
