"""Convert cached HTML into a structured JSONL corpus.

Reads data/sources/{ccc,summa,confessions,cog}/*.html and writes
data/processed/corpus.jsonl with records like:

    {"source": "summa", "ref": "I.q2", "title": "The Existence of God",
     "text": "<combined plain text>",
     "structure": {"articles": [{"title": ..., "objections": [...],
                                 "on_contrary": "...", "answer": "...",
                                 "replies": [...]}]}}

Different parsers per source because their HTML conventions differ.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES_DIR = PROJECT_ROOT / "data" / "sources"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _read_html(path: Path) -> BeautifulSoup:
    return BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# --- Source-specific parsers ---------------------------------------------


def parse_summa(path: Path) -> dict:
    """Newadvent's Summa pages: one question with N articles. Each article has
    Objections, "On the contrary", "I answer that", and Replies in flowing
    paragraphs. Extracting clean structure is approximate."""
    soup = _read_html(path)
    title = _clean_text(soup.find("title").text if soup.find("title") else "")
    main = soup.find("body") or soup
    text_blocks = [
        _clean_text(p.get_text(" "))
        for p in main.find_all(["p", "h1", "h2", "h3"])
        if p.get_text(strip=True)
    ]
    full_text = "\n\n".join(text_blocks)
    return {
        "source": "summa",
        "ref": path.stem,
        "title": title,
        "text": full_text,
        "structure": {"raw_paragraphs": len(text_blocks)},
    }


def parse_augustine(path: Path, source_name: str) -> dict:
    """Newadvent's Augustine pages: chapter pages with numbered paragraphs."""
    soup = _read_html(path)
    title = _clean_text(soup.find("title").text if soup.find("title") else "")
    main = soup.find("body") or soup
    paragraphs = [
        _clean_text(p.get_text(" "))
        for p in main.find_all("p")
        if p.get_text(strip=True)
    ]
    full_text = "\n\n".join(paragraphs)
    return {
        "source": source_name,
        "ref": path.stem,
        "title": title,
        "text": full_text,
        "structure": {"paragraphs": len(paragraphs)},
    }


def parse_ccc(path: Path) -> dict:
    """Vatican.va CCC: paragraphs are typically wrapped in <p> tags with
    numbered superscripts. We grab everything and let downstream chunking sort it out."""
    soup = _read_html(path)
    title = _clean_text(soup.find("title").text if soup.find("title") else "")
    main = soup.find("body") or soup
    # Strip nav / scripts / styles
    for tag in main.find_all(["script", "style", "nav"]):
        tag.decompose()
    paragraphs = [
        _clean_text(p.get_text(" "))
        for p in main.find_all("p")
        if len(p.get_text(strip=True)) > 30  # skip tiny boilerplate
    ]
    full_text = "\n\n".join(paragraphs)
    return {
        "source": "ccc",
        "ref": path.stem,
        "title": title,
        "text": full_text,
        "structure": {"paragraphs": len(paragraphs)},
    }


PARSERS = {
    "summa": lambda p: parse_summa(p),
    "confessions": lambda p: parse_augustine(p, "confessions"),
    "cog": lambda p: parse_augustine(p, "cog"),
    "ccc": lambda p: parse_ccc(p),
}


# --- chunking ------------------------------------------------------------


def chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    """Split text into chunks of up to max_chars, preferring paragraph breaks."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paragraphs:
        if cur_len + len(p) > max_chars and cur:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


# --- main ----------------------------------------------------------------


def main() -> int:
    if not SOURCES_DIR.exists():
        print(f"No sources dir: {SOURCES_DIR}", file=sys.stderr)
        print("Run scripts/scrape_sources.py first.", file=sys.stderr)
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "corpus.jsonl"
    chunks_written = 0

    with out_path.open("w") as out_fh:
        for src_name, parse_fn in PARSERS.items():
            src_dir = SOURCES_DIR / src_name
            if not src_dir.exists():
                print(f"  skip {src_name}: no dir")
                continue
            html_files = sorted(src_dir.glob("*.html"))
            print(f"\n[{src_name}] {len(html_files)} files")
            for html_path in html_files:
                rec = parse_fn(html_path)
                pieces = chunk_text(rec["text"])
                for i, piece in enumerate(pieces):
                    sub_rec = dict(rec)
                    sub_rec["text"] = piece
                    sub_rec["ref"] = f"{rec['ref']}#chunk-{i}" if len(pieces) > 1 else rec["ref"]
                    out_fh.write(json.dumps(sub_rec, ensure_ascii=False) + "\n")
                    chunks_written += 1
                print(f"  {html_path.name}: {len(pieces)} chunks ({len(rec['text'])} chars)")

    print(f"\nWrote {chunks_written} chunks to {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
