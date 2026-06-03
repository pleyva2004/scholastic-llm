"""Layer-resolved orthogonality (Part A) + frame-control (Part B). No training.

A: at every block, is δh^(ℓ) ⊥ to the content axis Δu (logit-lens), and where is
   the style write?  -> cos(δh^(ℓ),Δu), cos(δh^(ℓ),r_style), ΔM^(ℓ), ‖δh^(ℓ)‖/‖h‖.
B: is the shared content-independent shift frame-independent? render 20 statements
   under 5 frames; per-frame mean-δh alignment + cross-everything pairwise cos.
See README.md / predictions.md.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

ROOT = Path(__file__).parent
BASE = "models/qwen2.5-7b-mlx-q8"
ADAPTER = "adapters/scholastic-v2-iter400"
ORIG_FRAME = "Here is a statement about Catholic theology: "
FRAMES = [ORIG_FRAME, "Consider the following claim: ", "",
          "The textbook states: ", "Q: Is this true? "]
STYLE_WORDS = [" therefore", " hence", " thus", " accordingly", " moreover",
               " furthermore", " Objection", " Whether", " Reply", " wherein",
               " insofar", " Thou", " thee", " thy", " O"]
N_BOOT = 10_000


def dequant_U(model):
    lh = model.lm_head
    return mx.dequantize(lh.weight, lh.scales, lh.biases, group_size=lh.group_size, bits=lh.bits).astype(mx.float32)


def per_layer(model, tok, prefix):
    m = model.model
    ids = mx.array([tok.encode(prefix)])
    h = m.embed_tokens(ids)
    mask = create_attention_mask(h)
    outs = []
    for layer in m.layers:
        h = layer(h, mask)
        outs.append(np.asarray(h[0, -1].astype(mx.float32), float))     # block residual
    normed = np.asarray(m.norm(h)[0, -1].astype(mx.float32), float)
    return outs, normed                                                  # 28 blocks, final-normed


def final_hidden(model, tok, prefix):
    return np.asarray(model.model(mx.array([tok.encode(prefix)]))[0, -1].astype(mx.float32), float)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def boot_ci(vals, seed=0):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    bs = vals[rng.integers(0, len(vals), size=(N_BOOT, len(vals)))].mean(axis=1)
    return float(vals.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def main():
    pairs = [json.loads(l) for l in (ROOT.parent / "math-test" / "cloze_pairs.jsonl").read_text().splitlines() if l.strip()]
    print(f"{len(pairs)} cloze pairs, {len(FRAMES)} frames")

    base, tok = load(BASE)
    U = dequant_U(base)
    ubar = U.mean(axis=0)
    stoks = [t[0] for t in (tok.encode(w, add_special_tokens=False) for w in STYLE_WORDS) if len(t) == 1]
    r_style = np.asarray(U[mx.array(stoks)].mean(axis=0) - ubar, float)
    du = {p["id"]: np.asarray(U[p["true_tok"]] - U[p["false_tok"]], float) for p in pairs}
    norm_w = base.model.norm

    def normed(vec):  # apply final RMSNorm (the logit-lens read space)
        return np.asarray(norm_w(mx.array(vec)[None])[0].astype(mx.float32), float)

    # ---- Part A: per-layer residuals (base) ----
    base_layers, base_norm = {}, {}
    for p in pairs:
        outs, fnorm = per_layer(base, tok, p["prefix"])
        base_layers[p["id"]] = outs
        base_norm[p["id"]] = fnorm
    # ---- Part B: base final hidden over frames ----
    stmt = {p["id"]: p["prefix"][len(ORIG_FRAME):] for p in pairs}
    base_frame = {(p["id"], fi): final_hidden(base, tok, FRAMES[fi] + stmt[p["id"]])
                  for p in pairs for fi in range(len(FRAMES))}
    del base

    sft, _ = load(BASE, adapter_path=ADAPTER)
    sft_layers, sft_norm = {}, {}
    for p in pairs:
        outs, fnorm = per_layer(sft, tok, p["prefix"])
        sft_layers[p["id"]] = outs
        sft_norm[p["id"]] = fnorm
    sft_frame = {(p["id"], fi): final_hidden(sft, tok, FRAMES[fi] + stmt[p["id"]])
                 for p in pairs for fi in range(len(FRAMES))}
    del sft

    nB = len(base_layers[pairs[0]["id"]])
    d = len(r_style)
    null_std = 1.0 / np.sqrt(d)
    bar = 2 * null_std

    # ---- Part A aggregation ----
    layerstats = []
    for li in range(nB + 1):  # 0..nB-1 blocks, then nB = final-normed
        cc, ccn, cs, rr, dM = [], [], [], [], []
        for p in pairs:
            i = p["id"]
            if li < nB:
                hb, hs = base_layers[i][li], sft_layers[i][li]
                nb_, ns_ = normed(hb), normed(hs)         # logit-lens read space
            else:
                hb, hs = base_norm[i], sft_norm[i]
                nb_, ns_ = hb, hs                          # already normed (exact)
            dh = hs - hb                                   # raw residual displacement
            dn = ns_ - nb_                                 # post-norm displacement (read space)
            dMv = float(dn @ du[i])
            rr.append(np.linalg.norm(dh) / (np.linalg.norm(hb) + 1e-12))
            if np.linalg.norm(dh) > 1e-9:
                cc.append(cos(dh, du[i])); cs.append(cos(dh, r_style))
                ccn.append(cos(dn, du[i]))                 # readout-consistent content cos
            dM.append(dMv)
        mcc, ci = boot_ci(cc) if cc else (0.0, (0.0, 0.0))
        mccn, cin = boot_ci(ccn) if ccn else (0.0, (0.0, 0.0))
        layerstats.append(dict(
            layer=("norm" if li == nB else f"blk{li:02d}"),
            ratio=float(np.mean(rr)),
            cos_content=mcc, cos_content_ci=ci,
            cos_content_postnorm=mccn, cos_content_postnorm_ci=cin,
            cos_style=float(np.mean(cs)) if cs else 0.0,
            dM=float(np.mean(dM)),
            orthogonal=bool(abs(mccn) < bar and cin[0] <= 0 <= cin[1]) if ccn else True,
        ))

    lora_layers = [s for s in layerstats if s["ratio"] > 1e-6]
    max_abs_cc = max(abs(s["cos_content"]) for s in lora_layers)            # raw-space
    max_abs_ccn = max(abs(s["cos_content_postnorm"]) for s in lora_layers)  # readout space
    all_orth = all(s["orthogonal"] for s in lora_layers)                    # uses post-norm cos
    # H_cancel requires DIRECTIONAL content movement in the read space, not just a
    # norm-scale wobble: a layer whose post-norm content cos clears the bar.
    directional_content = any(abs(s["cos_content_postnorm"]) > bar and
                              not (s["cos_content_postnorm_ci"][0] <= 0 <= s["cos_content_postnorm_ci"][1])
                              for s in lora_layers)
    partA_verdict = ("H_cancel (directional content movement then cancels)" if directional_content
                     else "H_throughout (orthogonal at every layer; ΔM wobble is norm-scale only)")

    # ---- Part B aggregation ----
    ids = [p["id"] for p in pairs]
    dhf = {(i, fi): sft_frame[(i, fi)] - base_frame[(i, fi)] for i in ids for fi in range(len(FRAMES))}
    # within-frame pairwise cos (avg per frame)
    within = []
    for fi in range(len(FRAMES)):
        cs = [cos(dhf[(a, fi)], dhf[(b, fi)]) for a in ids for b in ids if a < b]
        within.append(float(np.mean(cs)))
    # per-frame mean δh and cos matrix between frame-means
    frame_mean = {fi: np.mean([dhf[(i, fi)] for i in ids], axis=0) for fi in range(len(FRAMES))}
    fm_cos = [[cos(frame_mean[a], frame_mean[b]) for b in range(len(FRAMES))] for a in range(len(FRAMES))]
    offdiag = [fm_cos[a][b] for a in range(len(FRAMES)) for b in range(len(FRAMES)) if a < b]
    min_framemean_cos = float(np.min(offdiag))
    # strictest: cross-frame AND cross-statement
    cross = [cos(dhf[(a, fa)], dhf[(b, fb)]) for a in ids for b in ids if a < b
             for fa in range(len(FRAMES)) for fb in range(len(FRAMES)) if fa != fb]
    cross_mean = float(np.mean(cross))
    frame_indep = bool(min_framemean_cos > 0.4 and cross_mean > bar)
    partB_verdict = "H_style (shared shift frame-independent)" if frame_indep else "H_frame (shared dir was the frame)"

    results = dict(
        d=d, null_std=null_std, bar=bar, n_blocks=nB,
        partA=dict(layers=layerstats, max_abs_cos_content_raw=max_abs_cc,
                   max_abs_cos_content_postnorm=max_abs_ccn,
                   all_layers_orthogonal=all_orth, directional_content_movement=directional_content,
                   verdict=partA_verdict),
        partB=dict(within_frame_pairwise_cos=within, frame_mean_cos_matrix=fm_cos,
                   min_frame_mean_cos=min_framemean_cos, cross_frame_cross_stmt_cos=cross_mean,
                   frame_independent=frame_indep, verdict=partB_verdict),
    )
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "layer_frame.json").write_text(json.dumps(results, indent=2))

    # ---- console ----
    print("\n=== PART A: layer-resolved (LoRA on blocks with ratio>0) ===")
    print(f"{'layer':6} {'ratio':>7} {'cosΔu_raw':>10} {'cosΔu_postnorm':>15} {'cos_style':>10} {'ΔM^ℓ':>9} orth")
    for s in layerstats:
        if s["ratio"] > 1e-6 or s["layer"] == "norm":
            print(f"{s['layer']:6} {s['ratio']:7.3f} {s['cos_content']:+10.4f} {s['cos_content_postnorm']:+15.4f} "
                  f"{s['cos_style']:+10.4f} {s['dM']:+9.3f}  {'Y' if s['orthogonal'] else 'N'}")
    print(f"max|cosΔu| raw={max_abs_cc:.4f} post-norm={max_abs_ccn:.4f} (bar {bar:.4f}); directional content movement? {directional_content}")
    print(f"PART A VERDICT: {partA_verdict}")

    print("\n=== PART B: frame-control ===")
    print(f"within-frame pairwise cos per frame: {[round(x,3) for x in within]}")
    print("cos between per-frame mean δh:")
    for row in fm_cos:
        print("   " + " ".join(f"{v:+.2f}" for v in row))
    print(f"min off-diagonal frame-mean cos = {min_framemean_cos:.3f}  (frame-independent if >0.4)")
    print(f"cross-frame × cross-statement pairwise cos = {cross_mean:+.4f}  (null {null_std:.4f})")
    print(f"PART B VERDICT: {partB_verdict}")
    print("\nsaved -> results/layer_frame.json")


if __name__ == "__main__":
    main()
