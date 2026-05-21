"""Fetch primary sources for fine-tuning.

Scrapes:
- Catechism of the Catholic Church (vatican.va) — © USCCB, fair-use only
- Summa Theologica (newadvent.org) — public domain
- Augustine's Confessions (newadvent.org) — public domain
- Augustine's City of God (newadvent.org) — public domain

Design choices:
- Rate-limited (1 sec sleep between requests)
- Local on-disk cache (won't re-fetch a URL whose file already exists)
- Hardcoded URL list per phase (no recursive crawling) to keep the surface small
  and predictable for review

Run:
    python scripts/scrape_sources.py --phase 1            # ~30 URLs, ~5 min
    python scripts/scrape_sources.py --phase 1 --dry-run  # just print plan
    python scripts/scrape_sources.py --phase 2            # full corpus, ~10-15 min
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES_DIR = PROJECT_ROOT / "data" / "sources"

USER_AGENT = (
    "scholastic-llm/0.0.1 (personal research project; "
    "github.com/pleyva2004/scholastic-llm)"
)

# --- URL inventory ---------------------------------------------------------

# Hand-picked URLs grouped by source. Phase 1 = smoke test (~30 URLs).
# Phase 2 expands across more topics; for now just duplicates Phase 1 — the
# full Phase 2 inventory should be tuned after we see what Claude does with
# the smaller set.

SOURCES_PHASE_1: dict[str, list[tuple[str, str]]] = {
    # (slug, url)
    "summa": [
        # Part I, Question 2: The Existence of God
        ("p1-q2", "https://www.newadvent.org/summa/1002.htm"),
        # Part I, Question 3: On the Simplicity of God
        ("p1-q3", "https://www.newadvent.org/summa/1003.htm"),
        # Part I, Question 75: Of Man Who is Composed of a Spiritual and a Corporeal Substance
        ("p1-q75", "https://www.newadvent.org/summa/1075.htm"),
        # Part I, Question 76: The Union of Body and Soul
        ("p1-q76", "https://www.newadvent.org/summa/1076.htm"),
        # Part I, Question 83: On Free Choice
        ("p1-q83", "https://www.newadvent.org/summa/1083.htm"),
        # Part II-I, Question 109: The Necessity of Grace
        ("p2-1-q109", "https://www.newadvent.org/summa/2109.htm"),
        # Part II-II, Question 1: Of Faith
        ("p2-2-q1", "https://www.newadvent.org/summa/3001.htm"),
        # Part II-II, Question 2: Of the Act of Faith
        ("p2-2-q2", "https://www.newadvent.org/summa/3002.htm"),
    ],
    "confessions": [
        # Book 1 — childhood, the restless heart
        ("book-1", "https://www.newadvent.org/fathers/110101.htm"),
        # Book 7 — Augustine's encounter with Platonism, problem of evil
        ("book-7", "https://www.newadvent.org/fathers/110107.htm"),
        # Book 10 — memory, the search for God
        ("book-10", "https://www.newadvent.org/fathers/110110.htm"),
        # Book 11 — time and eternity
        ("book-11", "https://www.newadvent.org/fathers/110111.htm"),
    ],
    "cog": [
        # Book 11 — origin of the two cities, on creation
        ("book-11", "https://www.newadvent.org/fathers/120111.htm"),
        # Book 14 — sin, the two loves
        ("book-14", "https://www.newadvent.org/fathers/120114.htm"),
        # Book 19 — the end of the two cities, peace
        ("book-19", "https://www.newadvent.org/fathers/120119.htm"),
        # Book 22 — the resurrection and the eternal beatitude
        ("book-22", "https://www.newadvent.org/fathers/120122.htm"),
    ],
    "ccc": [
        # CCC paragraphs aligned with our planned eval prompts.
        # Vatican.va groups paragraphs into "sections"; we fetch the section
        # pages and parse paragraph ranges out in clean_corpus.py.
        ("p2-art1", "https://www.vatican.va/archive/ENG0015/__P10.HTM"),    # The Creed
        ("p2-art4", "https://www.vatican.va/archive/ENG0015/__P19.HTM"),    # Creator
        ("p2-art5", "https://www.vatican.va/archive/ENG0015/__P1A.HTM"),    # Heaven and Earth
        ("p2-art6", "https://www.vatican.va/archive/ENG0015/__P1B.HTM"),    # Man (soul/body)
        ("p2-art7", "https://www.vatican.va/archive/ENG0015/__P1C.HTM"),    # The Fall (evil)
        ("p3-life", "https://www.vatican.va/archive/ENG0015/__P5C.HTM"),    # Life in Christ / grace
    ],
}

SOURCES_PHASE_2 = SOURCES_PHASE_1  # Expand later.


# --- fetch + cache --------------------------------------------------------


def fetch_url(url: str, cache_path: Path, session: requests.Session) -> bool:
    """Fetch a URL into cache_path if it isn't already there. Returns True if fetched."""
    if cache_path.exists():
        return False
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print plan; don't fetch")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between fetches")
    args = parser.parse_args()

    sources = SOURCES_PHASE_1 if args.phase == 1 else SOURCES_PHASE_2
    total = sum(len(v) for v in sources.values())
    print(f"Phase {args.phase}: {total} URLs across {len(sources)} sources")
    for src, urls in sources.items():
        print(f"  {src}: {len(urls)}")

    if args.dry_run:
        print("\n--dry-run; not fetching. URLs:")
        for src, urls in sources.items():
            for slug, url in urls:
                print(f"  {src}/{slug}: {url}")
        return 0

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    fetched = cached = 0
    for src, urls in sources.items():
        print(f"\n[{src}] {len(urls)} URLs")
        for slug, url in urls:
            cache_path = SOURCES_DIR / src / f"{slug}.html"
            try:
                did_fetch = fetch_url(url, cache_path, session)
            except requests.RequestException as e:
                print(f"  ERROR fetching {url}: {e}")
                continue
            if did_fetch:
                fetched += 1
                time.sleep(args.sleep)
            else:
                cached += 1
                print(f"  cached: {cache_path.relative_to(PROJECT_ROOT)}")

    print(f"\nDone. Fetched {fetched} new, {cached} already cached.")
    print(f"Output: {SOURCES_DIR.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
