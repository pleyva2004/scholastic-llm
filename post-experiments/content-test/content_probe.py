"""Probe: did the counterfactual content adapter rotate δh onto the content axis?

cos(δh,Δu) and ΔM on TRAINED vs HELD-OUT cloze pairs, for content-cf vs voice vs
base. Plus fingerprints: pairwise cos(δh) and register marker-boost z.
Exact cloze identity ΔM=⟨δh,Δu⟩, frozen U. See README.md / predictions.md.
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
ADAPTERS = [("voice", "adapters/scholastic-v2-iter400"),
            ("content-cf", str(ROOT / "adapters" / "content-cf"))]
STYLE_WORDS = [" therefore", " hence", " thus", " accordingly", " moreover",
               " furthermore", " Objection", " Whether", " Reply", " wherein",
               " insofar", " Thou", " thee", " thy", " O"]
N_BOOT = 10_000


def dequant_U(model):
    lh = model.lm_head
    return mx.dequantize(lh.weight, lh.scales, lh.biases, group_size=lh.group_size, bits=lh.bits).astype(mx.float32)


def hstar(model, tok, prefix):
    return np.asarray(model.model(mx.array([tok.encode(prefix)]))[0, -1].astype(mx.float32), float)


def direct_margin(model, h, xp, xm):
    z = model.lm_head(mx.array(h)[None])[0].astype(mx.float32)
    lp = nn.log_softmax(z)
    return float(lp[xp] - lp[xm])


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def boot_ci(vals, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    bs = vals[rng.integers(0, len(vals), size=(N_BOOT, len(vals)))].mean(axis=1)
    return float(vals.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def main():
    pairs = [json.loads(l) for l in (ROOT.parent / "math-test" / "cloze_pairs.jsonl").read_text().splitlines() if l.strip()]
    split = json.loads((ROOT / "split.json").read_text())
    trained, heldout = set(split["train_ids"]), set(split["heldout_ids"])

    base, tok = load(BASE)
    U = dequant_U(base)
    ubar = U.mean(axis=0)
    stoks = [t[0] for t in (tok.encode(w, add_special_tokens=False) for w in STYLE_WORDS) if len(t) == 1]
    r_style = np.asarray(U[mx.array(stoks)].mean(axis=0) - ubar, float)
    du = {p["id"]: np.asarray(U[p["true_tok"]] - U[p["false_tok"]], float) for p in pairs}
    hb, bm = {}, {}
    for p in pairs:
        h = hstar(base, tok, p["prefix"]); hb[p["id"]] = h
        bm[p["id"]] = float(h @ du[p["id"]])
    Unp = np.asarray(U, float)
    del base; gc.collect(); mx.clear_cache()
    d = Unp.shape[1]; null_std = 1.0 / np.sqrt(d); bar = 2 * null_std

    results = {"d": d, "null_std": null_std, "bar": bar, "trained_ids": sorted(trained),
               "heldout_ids": sorted(heldout), "conditions": {}}

    for label, adapter in ADAPTERS:
        m, _ = load(BASE, adapter_path=adapter)
        items = []
        for p in pairs:
            hs = hstar(m, tok, p["prefix"])
            dh = hs - hb[p["id"]]
            dM = float(dh @ du[p["id"]])
            base_m = bm[p["id"]]; sft_m = base_m + dM
            items.append(dict(
                id=p["id"], split=("trained" if p["id"] in trained else "heldout"),
                true=p["true"], false=p["false"],
                base_margin=base_m, sft_margin=sft_m, dM=dM,
                flipped=bool(np.sign(sft_m) != np.sign(base_m)),
                cos_content=cos(dh, du[p["id"]]), cos_style=cos(dh, r_style),
                norm_dh=float(np.linalg.norm(dh)), dh=dh,
            ))
        del m; gc.collect(); mx.clear_cache()

        def grp(split):
            return [it for it in items if it["split"] == split]

        cond = {}
        for split in ("trained", "heldout"):
            g = grp(split)
            cc = [it["cos_content"] for it in g]
            mcc, ci = boot_ci(cc)
            cond[split] = dict(
                n=len(g),
                mean_cos_content=mcc, ci_cos_content=ci,
                mean_dM=float(np.mean([it["dM"] for it in g])),
                n_flipped=int(sum(it["flipped"] for it in g)),
                mean_cos_style=float(np.mean([it["cos_style"] for it in g])),
            )
        # fingerprints over ALL items
        dhs = [it["dh"] for it in items]
        pair_cos = [cos(dhs[i], dhs[j]) for i in range(len(dhs)) for j in range(i + 1, len(dhs))]
        mean_dh = np.mean(dhs, axis=0)
        dz = Unp @ mean_dh
        mb_boost = float(dz[stoks].mean())
        rng = np.random.default_rng(0)
        rtok = rng.integers(0, Unp.shape[0], size=4000)
        marker_z = (mb_boost - float(dz[rtok].mean())) / (float(dz[rtok].std()) + 1e-9)

        cond["mean_pairwise_cos_dh"] = float(np.mean(pair_cos))
        cond["marker_boost_z"] = float(marker_z)
        for it in items:
            del it["dh"]
        cond["items"] = items
        results["conditions"][label] = cond

    # verdict
    c = results["conditions"]["content-cf"]
    tr = c["trained"]
    m_grad = (abs(tr["mean_cos_content"]) > bar and not (tr["ci_cos_content"][0] <= 0 <= tr["ci_cos_content"][1])
              and tr["n_flipped"] >= 6)
    verdict = "M_grad/M_capacity (content reachable, gradient-routed)" if m_grad else \
              ("M_subspace (LoRA cannot write content)" if tr["n_flipped"] < 3 else "ambiguous")
    results["verdict"] = verdict

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "content_probe.json").write_text(json.dumps(results, indent=2))

    print(f"\nnull std {null_std:.4f}, bar {bar:.4f}")
    for label in ("voice", "content-cf"):
        c = results["conditions"][label]
        print(f"\n=== {label} ===")
        for split in ("trained", "heldout"):
            s = c[split]
            print(f"  {split:8} n={s['n']:2d}  cos(δh,Δu)={s['mean_cos_content']:+.4f} "
                  f"CI[{s['ci_cos_content'][0]:+.4f},{s['ci_cos_content'][1]:+.4f}]  "
                  f"ΔM={s['mean_dM']:+.3f}  flipped={s['n_flipped']}/{s['n']}  cos_style={s['mean_cos_style']:+.4f}")
        print(f"  fingerprints: pairwise cos(δh)={c['mean_pairwise_cos_dh']:+.3f}  marker-z={c['marker_boost_z']:+.2f}")
    print(f"\nVERDICT: {verdict}")
    print("saved -> results/content_probe.json")


if __name__ == "__main__":
    main()
