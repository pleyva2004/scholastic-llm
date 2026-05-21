# Data Licensing

This repository's **source code** is MIT-licensed (see `LICENSE`). The **data
this code scrapes and processes** comes from third parties and has its own
terms. None of the scraped corpus, processed JSONL, or trained adapter weights
are checked into this repository.

## Sources used

| Source | Where | Status |
|---|---|---|
| Catechism of the Catholic Church (CCC, 1992) | vatican.va | © USCCB / Libreria Editrice Vaticana. Used here under **fair use** for personal research and non-commercial fine-tuning. Not redistributed. |
| Summa Theologica (English, Fr. Laurence Shapcote, 1920) | newadvent.org | **Public domain** in the US (pre-1928). |
| Augustine — Confessions (E.B. Pusey translation) | newadvent.org | **Public domain.** |
| Augustine — City of God (Marcus Dods translation) | newadvent.org | **Public domain.** |

## What is NOT redistributed

- `data/sources/` — raw scraped HTML
- `data/processed/` — cleaned JSONL chunks
- `data/train.jsonl`, `data/valid.jsonl` — instruction pairs derived from above
- `adapters/` — LoRA weights (arguably a derivative work of CCC content)

All of the above are listed in `.gitignore`.

## What CAN be shared

- The source code in this repository (MIT-licensed)
- The scraping logic, training scripts, evaluation rubric
- Aggregate metrics and qualitative examples that don't reproduce substantial portions of the source texts

## On publishing the trained adapter

A LoRA adapter trained on copyrighted text is in a legally uncertain space.
Default: **do not publish the adapter** to HuggingFace or anywhere else without
first reviewing the legal posture. If we do publish, the published artifact
should be derived primarily from public-domain sources (Summa, Augustine) with
CCC influence limited to ideas, not verbatim text.

## Respectful scraping

`scripts/scrape_sources.py` uses:
- A descriptive `User-Agent` identifying this is a personal research project
- `time.sleep(1)` between requests
- Local on-disk cache so each page is fetched at most once
- Respect for `robots.txt`
