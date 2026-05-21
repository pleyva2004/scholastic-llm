"""Rubric-based scoring for trained model outputs.

Each dimension yields an integer 0-3. Higher = more of the target style/content.
"""

import re

SCHOLASTIC_MARKERS = [
    r"\bI answer that\b",
    r"\bIt seems that\b",
    r"\bOn the contrary\b",
    r"\bWhether\b",
    r"\bReply to Objection\b",
    r"\bObjection \d+\b",
    r"\baccordingly\b",
    r"\bhence\b",
    r"\bthus\b",
    r"\btherefore\b",
    r"\binsofar as\b",
    r"\bwherein\b",
]

AUGUSTINIAN_MARKERS = [
    r"\bO Lord\b",
    r"\bThou\b",
    r"\bthee\b",
    r"\bthy\b",
    r"\bmy soul\b",
    r"\bthe heart\b",
    r"\beternity\b",
    r"\bthe City of God\b",
    r"\bthe earthly city\b",
]

CCC_CITATION_PATTERNS = [
    r"§\s*\d+",
    r"\bCCC\s*\d+",
    r"\bparagraph\s+\d+",
    r"\bCatechism\b",
]


def _count_unique_hits(text: str, patterns: list[str]) -> int:
    """Number of distinct patterns that match somewhere in text."""
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))


def _bucket(n: int, low: int = 1, mid: int = 3, high: int = 6) -> int:
    """0/1/2/3 score based on raw count thresholds."""
    if n >= high:
        return 3
    if n >= mid:
        return 2
    if n >= low:
        return 1
    return 0


def score_scholastic_register(text: str) -> int:
    """Score 0-3 on presence of Summa-style structural markers + scholastic vocabulary."""
    return _bucket(_count_unique_hits(text, SCHOLASTIC_MARKERS), low=1, mid=3, high=5)


def score_augustinian_voice(text: str) -> int:
    """Score 0-3 on presence of Augustinian rhetorical / autobiographical markers."""
    return _bucket(_count_unique_hits(text, AUGUSTINIAN_MARKERS), low=1, mid=2, high=4)


def score_ccc_grounding(text: str) -> int:
    """Score 0-3 on visible references to CCC paragraphs."""
    return _bucket(_count_unique_hits(text, CCC_CITATION_PATTERNS), low=1, mid=2, high=3)


def score_structure(text: str) -> int:
    """Score 0-3 on multi-paragraph length + objection/response structure."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    has_objection = bool(re.search(r"\bObjection\b|\bIt seems\b", text, re.IGNORECASE))
    has_response = bool(re.search(r"\bI answer\b|\bReply\b|\bOn the contrary\b", text, re.IGNORECASE))
    paragraph_score = _bucket(len(paragraphs), low=2, mid=4, high=6)
    structure_bonus = (1 if has_objection else 0) + (1 if has_response else 0)
    return min(3, paragraph_score - 1 + structure_bonus)


def score_all(text: str) -> dict[str, int]:
    """Compute all rubric dimensions; total is summed 0-12."""
    scores = {
        "scholastic_register": score_scholastic_register(text),
        "augustinian_voice": score_augustinian_voice(text),
        "ccc_grounding": score_ccc_grounding(text),
        "structure": score_structure(text),
    }
    scores["total"] = sum(scores.values())
    return scores
