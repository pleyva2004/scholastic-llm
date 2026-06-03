"""Single-token cloze pairs for the EXACT margin identity  M = <h*, Δu>.

Each item: a shared prefix P, then a true vs false *single* next token. We keep
only candidates where both completions are single tokens under the Qwen
tokenizer, so the cloze margin is an exact inner product (no approximation) and
the adapter effect is exactly ΔM = <δh*, Δu>, Δu = u_true - u_false (U frozen).
"""

import json
from pathlib import Path

from mlx_lm import load

FRAME = "Here is a statement about Catholic theology: "

CANDIDATES = [
    ("The proper object of the will is the", " good", " true"),
    ("The proper object of the intellect is the", " true", " good"),
    ("Aquinas describes God as pure", " act", " potency"),
    ("The soul is the substantial form of the", " body", " soul"),
    ("The City of God is built on love of", " God", " self"),
    ("The earthly city is built on love of", " self", " God"),
    ("The Confessions were written by", " Augustine", " Thomas"),
    ("The Summa Theologica was written by", " Thomas", " Augustine"),
    ("Christ is one divine person subsisting in two", " natures", " persons"),
    ("Augustine holds that evil is not a substance but a", " privation", " power"),
    ("The source and summit of the Christian life is the", " Eucharist", " Bible"),
    ("Grace does not destroy nature but rather", " perfects", " replaces"),
    ("Faith is fundamentally a gift rather than a human", " achievement", " gift"),
    ("Rightly ordered love places God above the", " self", " soul"),
    ("Aquinas is honored with the title of the Angelic", " Doctor", " Teacher"),
    ("Augustine served as the bishop of", " Hippo", " Rome"),
    ("In transubstantiation what changes is the bread's", " substance", " appearance"),
    ("The theological virtues are infused rather than", " acquired", " natural"),
    ("An intrinsically evil act cannot be justified by a good", " intention", " act"),
    ("The beatific vision exceeds the powers of any created", " nature", " angel"),
    ("Sanctifying grace is unmerited and given by", " God", " man"),
    ("The first efficient cause of all being is", " God", " matter"),
    ("Change is the reduction of potency to", " act", " form"),
    ("In God alone are essence and existence", " identical", " distinct"),
    ("The Holy Spirit proceeds from the Father and the", " Son", " Word"),
    ("Charity is said by Aquinas to be the form of all the", " virtues", " vices"),
]


def main():
    _, tok = load("models/qwen2.5-7b-mlx-q8")
    kept, dropped = [], []
    for i, (p, t, f) in enumerate(CANDIDATES):
        tt = tok.encode(t, add_special_tokens=False)
        ft = tok.encode(f, add_special_tokens=False)
        if len(tt) == 1 and len(ft) == 1 and tt[0] != ft[0]:
            kept.append(dict(id=f"cz{i:02d}", prefix=FRAME + p,
                             true=t, false=f, true_tok=tt[0], false_tok=ft[0]))
        else:
            dropped.append((p, t, f, tt, ft))
    out = Path(__file__).parent / "cloze_pairs.jsonl"
    with out.open("w") as fh:
        for k in kept:
            fh.write(json.dumps(k) + "\n")
    print(f"kept {len(kept)} single-token cloze pairs -> {out}")
    for d in dropped:
        print(f"  DROPPED (multi-token): {d[1]!r}={d[3]} / {d[2]!r}={d[4]}  [{d[0]}]")


if __name__ == "__main__":
    main()
