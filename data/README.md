# `data/` directory

Everything under here except this README is **gitignored**. See `DATA_LICENSING.md`
in the repo root for why.

## Layout (populated after running the pipeline)

```
data/
├── README.md            (this file — tracked)
├── sources/             (gitignored — raw scraped HTML)
│   ├── ccc/             (Catechism of the Catholic Church, vatican.va)
│   ├── summa/           (Summa Theologica, newadvent.org)
│   ├── confessions/     (Augustine — Confessions, newadvent.org)
│   └── cog/             (Augustine — City of God, newadvent.org)
├── processed/           (gitignored — cleaned, structured JSONL)
│   └── corpus.jsonl     ({"source": ..., "ref": ..., "text": ..., "structure": ...})
├── train.jsonl          (gitignored — Claude-generated instruction pairs)
└── valid.jsonl          (gitignored — held-out validation pairs)
```

## Why nothing is committed

- **CCC** is copyrighted (© USCCB). Scraped for personal research only; not redistributed.
- **Summa / Augustine** are public domain but bundling them with this repo adds bloat without value (anyone can re-scrape).
- **train.jsonl / valid.jsonl** are derived from the above and would entangle the licensing situation if shared.

Re-generate the contents by running the scripts in `scripts/`.
