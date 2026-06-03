"""Build the minimal-pair understanding probe set.

Each pair shares vocabulary and (as closely as possible) length, differing ONLY
in the theological *relation* — a swapped term, a reversed dependency, a negation.
This holds register/style constant across the two members so that a model which
merely prefers scholastic-sounding text gains no advantage; only a model whose
probability mass tracks doctrinal truth can separate them.

type:
  doctrinal      -> relational/conceptual claim (the headline understanding set)
  factual_anchor -> obvious recall (author/title/epithet); metric-validity floor
                    + a forgetting check (SFT should not wreck these)

Correctness is per Catholic / Thomistic doctrine (CCC, Summa, Augustine).
"""

import json
from pathlib import Path

PAIRS = [
    # ---- A. Priority / dependency order (grace, faith, merit) -------------
    dict(id="dep01", category="grace_merit", type="doctrinal",
         correct="Grace is prior to merit.",
         incorrect="Merit is prior to grace.",
         note="CCC 2010: grace precedes and is unmerited."),
    dict(id="dep02", category="grace_faith", type="doctrinal",
         correct="Faith is a gift of God's grace.",
         incorrect="Faith is a human work that earns God's grace.",
         note="CCC 153; Eph 2:8."),
    dict(id="dep03", category="grace_nature", type="doctrinal",
         correct="Grace perfects nature rather than destroying it.",
         incorrect="Grace destroys nature rather than perfecting it.",
         note="Aquinas: gratia non tollit naturam sed perficit."),
    dict(id="dep04", category="virtue_form", type="doctrinal",
         correct="Charity is the form of all the virtues.",
         incorrect="Prudence is the form of all the virtues.",
         note="Aquinas ST II-II q23 a8: caritas forma virtutum."),
    dict(id="dep05", category="virtue_origin", type="doctrinal",
         correct="The theological virtues are infused by God.",
         incorrect="The theological virtues are acquired by repeated human acts.",
         note="Infused vs acquired; theological virtues are infused."),
    dict(id="dep06", category="justification", type="doctrinal",
         correct="Justification is a free gift of God's grace.",
         incorrect="Justification is earned by works apart from grace.",
         note="CCC 1996ff."),

    # ---- B. Evil as privation --------------------------------------------
    dict(id="evl01", category="evil", type="doctrinal",
         correct="Evil is the privation of a due good.",
         incorrect="Evil is a positive substance created by God.",
         note="Augustine/Aquinas: privatio boni."),
    dict(id="evl02", category="evil", type="doctrinal",
         correct="God permits evil in order to draw forth a greater good.",
         incorrect="God creates evil in order to test human virtue.",
         note="CCC 311-312."),
    dict(id="evl03", category="evil", type="doctrinal",
         correct="Sin is a disorder of the will, not a created thing.",
         incorrect="Sin is a created substance infused into the soul.",
         note="Sin is privation/disorder, not substance."),

    # ---- C. Faculties (intellect / will) ---------------------------------
    dict(id="fac01", category="faculties", type="doctrinal",
         correct="The intellect apprehends truth and the will inclines toward the good.",
         incorrect="The will apprehends truth and the intellect inclines toward the good.",
         note="Swap of proper objects of the two faculties."),
    dict(id="fac02", category="faculties", type="doctrinal",
         correct="The will is moved by the good as apprehended by the intellect.",
         incorrect="The intellect is moved by the good as apprehended by the will.",
         note="Order of operation between will and intellect."),

    # ---- D. Metaphysics (act/potency, essence/existence, form) -----------
    dict(id="met01", category="metaphysics", type="doctrinal",
         correct="In God alone are essence and existence identical.",
         incorrect="In every creature are essence and existence identical.",
         note="Aquinas: real distinction in creatures, identity only in God."),
    dict(id="met02", category="metaphysics", type="doctrinal",
         correct="God is pure act with no potentiality.",
         incorrect="God is a composition of act and potency.",
         note="Actus purus."),
    dict(id="met03", category="metaphysics", type="doctrinal",
         correct="Change is the reduction of potency to act.",
         incorrect="Change is the reduction of act to potency.",
         note="Aristotelian-Thomistic definition of motion."),
    dict(id="met04", category="metaphysics", type="doctrinal",
         correct="The soul is the substantial form of the body.",
         incorrect="The soul is an accidental form of the body.",
         note="Aquinas: anima forma corporis substantialis."),
    dict(id="met05", category="metaphysics", type="doctrinal",
         correct="Prime matter is pure potency with no actuality of its own.",
         incorrect="Prime matter is a fully actual substance in its own right.",
         note="materia prima = pure potentiality."),

    # ---- E. Causation / providence ---------------------------------------
    dict(id="cau01", category="causation", type="doctrinal",
         correct="God is the first efficient cause of all being.",
         incorrect="God is merely the first material cause of all being.",
         note="Efficient vs material first cause."),
    dict(id="cau02", category="causation", type="doctrinal",
         correct="The final cause directs an agent toward its end.",
         incorrect="The final cause pushes an agent from behind as a prior force.",
         note="Finality draws toward an end; it is not efficient pushing."),

    # ---- F. Christology / Trinity (creedal) ------------------------------
    dict(id="chr01", category="christology", type="doctrinal",
         correct="Christ is one divine person subsisting in two natures.",
         incorrect="Christ is two persons subsisting in one nature.",
         note="Chalcedon vs Nestorian error."),
    dict(id="chr02", category="trinity", type="doctrinal",
         correct="The Son is begotten of the Father, not made.",
         incorrect="The Son is made by the Father, not begotten.",
         note="Nicene Creed vs Arian error."),
    dict(id="chr03", category="trinity", type="doctrinal",
         correct="The Holy Spirit proceeds from the Father and the Son.",
         incorrect="The Holy Spirit is begotten of the Father and the Son.",
         note="Procession vs generation; CCC 246."),
    dict(id="chr04", category="trinity", type="doctrinal",
         correct="The three divine Persons share one and the same divine essence.",
         incorrect="The three divine Persons each possess a separate divine essence.",
         note="Consubstantiality vs tritheism."),

    # ---- G. Anthropology / grace -----------------------------------------
    dict(id="ant01", category="anthropology", type="doctrinal",
         correct="Human nature is wounded by original sin but not totally destroyed.",
         incorrect="Human nature is totally destroyed by original sin.",
         note="Catholic vs total-depravity extreme; CCC 405."),
    dict(id="ant02", category="beatitude", type="doctrinal",
         correct="The beatific vision exceeds the power of any created nature.",
         incorrect="The beatific vision is attainable by unaided natural reason.",
         note="Supernatural end requires grace."),

    # ---- H. Augustine ----------------------------------------------------
    dict(id="aug01", category="augustine", type="doctrinal",
         correct="The heart is restless until it rests in God.",
         incorrect="The heart is at rest until it turns to God.",
         note="Confessions I.1, inverted."),
    dict(id="aug02", category="augustine", type="doctrinal",
         correct="The earthly city is built on love of self, the City of God on love of God.",
         incorrect="The earthly city is built on love of God, the City of God on love of self.",
         note="De Civitate Dei XIV.28, swapped."),
    dict(id="aug03", category="augustine", type="doctrinal",
         correct="Rightly ordered love loves God above all things.",
         incorrect="Rightly ordered love loves the self above all things.",
         note="ordo amoris."),
    dict(id="aug04", category="augustine", type="doctrinal",
         correct="For Augustine, time is a distension of the soul.",
         incorrect="For Augustine, time is a substance existing outside the soul.",
         note="Confessions XI: distentio animi."),

    # ---- I. Sacraments ---------------------------------------------------
    dict(id="sac01", category="sacraments", type="doctrinal",
         correct="The Eucharist is the source and summit of the Christian life.",
         incorrect="Baptism is the source and summit of the Christian life.",
         note="CCC 1324."),
    dict(id="sac02", category="sacraments", type="doctrinal",
         correct="In the Eucharist the substance of bread changes while the accidents remain.",
         incorrect="In the Eucharist the accidents of bread change while the substance remains.",
         note="Transubstantiation: substance changes, species/accidents remain."),
    dict(id="sac03", category="sacraments", type="doctrinal",
         correct="Baptism imprints an indelible spiritual character.",
         incorrect="Baptism imprints a character that may be repeated and removed.",
         note="CCC 1272: indelible."),

    # ---- J. Natural law / morals -----------------------------------------
    dict(id="mor01", category="natural_law", type="doctrinal",
         correct="The natural law is the rational creature's participation in the eternal law.",
         incorrect="The natural law is a human convention independent of the eternal law.",
         note="Aquinas ST I-II q91 a2."),
    dict(id="mor02", category="conscience", type="doctrinal",
         correct="Conscience must be formed in accordance with truth.",
         incorrect="Conscience creates moral truth by its own decision.",
         note="CCC 1783ff."),
    dict(id="mor03", category="morals", type="doctrinal",
         correct="An intrinsically evil act cannot be made good by a good intention.",
         incorrect="An intrinsically evil act can be made good by a good intention.",
         note="CCC 1753ff: end does not justify the means."),

    # ---- K. Factual anchors (metric-validity floor / forgetting check) ---
    dict(id="anc01", category="anchor_author", type="factual_anchor",
         correct="Thomas Aquinas wrote the Summa Theologica.",
         incorrect="Augustine wrote the Summa Theologica.",
         note="Authorship."),
    dict(id="anc02", category="anchor_author", type="factual_anchor",
         correct="Augustine wrote the Confessions.",
         incorrect="Thomas Aquinas wrote the Confessions.",
         note="Authorship."),
    dict(id="anc03", category="anchor_epithet", type="factual_anchor",
         correct="Thomas Aquinas is called the Angelic Doctor.",
         incorrect="Augustine is called the Angelic Doctor.",
         note="Doctor Angelicus = Aquinas."),
    dict(id="anc04", category="anchor_bio", type="factual_anchor",
         correct="Augustine was the bishop of Hippo.",
         incorrect="Thomas Aquinas was the bishop of Hippo.",
         note="Augustine, bishop of Hippo."),
    dict(id="anc05", category="anchor_structure", type="factual_anchor",
         correct="The Summa Theologica is structured as questions, articles, and objections.",
         incorrect="The Summa Theologica is structured as a continuous autobiographical narrative.",
         note="Summa form vs Confessions form."),
]


def main():
    out = Path(__file__).parent / "minimal_pairs.jsonl"
    with out.open("w") as f:
        for p in PAIRS:
            f.write(json.dumps(p) + "\n")
    n_doc = sum(1 for p in PAIRS if p["type"] == "doctrinal")
    n_anc = sum(1 for p in PAIRS if p["type"] == "factual_anchor")
    cats = sorted(set(p["category"] for p in PAIRS))
    print(f"wrote {len(PAIRS)} pairs -> {out}")
    print(f"  doctrinal={n_doc}  factual_anchor={n_anc}")
    print(f"  categories ({len(cats)}): {', '.join(cats)}")


if __name__ == "__main__":
    main()
