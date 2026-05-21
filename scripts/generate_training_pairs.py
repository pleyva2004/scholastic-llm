"""Use Claude as a teacher model to generate scholastic Q/A training pairs.

Reads data/processed/corpus.jsonl and emits data/train.jsonl + data/valid.jsonl
in chat-message format compatible with mlx_lm_lora.train.

For each source chunk, asks Claude (Sonnet 4.6 by default) to:
1. Identify 2-3 philosophical/theological questions the chunk addresses.
2. For each question, write a scholastic-register answer in Aquinas's Summa
   structure (Objection / On the contrary / I answer that / Reply) OR in
   Augustinian rhetorical style for existential questions, citing the CCC by
   paragraph number where appropriate.

Async with a small concurrency cap to respect Anthropic rate limits.

Run:
    python scripts/generate_training_pairs.py --max-calls 200   # Phase 1
    python scripts/generate_training_pairs.py --max-calls 1000  # Phase 2
    python scripts/generate_training_pairs.py --dry-run         # show one example
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
TRAIN_PATH = PROJECT_ROOT / "data" / "train.jsonl"
VALID_PATH = PROJECT_ROOT / "data" / "valid.jsonl"

SYSTEM_PROMPT = """You are generating high-quality instruction-tuning training data for a small
LLM. The student LLM will be fine-tuned to:

- Answer general and philosophical questions in a SCHOLASTIC, Latin-inflected
  English register — the register of Aquinas and Anselm, not the King James Bible.
- Ground arguments in the Catechism of the Catholic Church (CCC, 1992), citing
  paragraphs by number (e.g., "§309", "CCC §1996").
- Debate in the structural voice of Aquinas's Summa Theologica when the question
  is systematic ("Whether X is the case... Objection 1... On the contrary...
  I answer that... Reply to Objection 1..."), or in Augustine's more rhetorical,
  scripturally-saturated voice when the question is existential or personal.

You will receive a SOURCE CHUNK from one of: Summa Theologica, Augustine's
Confessions, Augustine's City of God, or the CCC.

Produce 2-3 (question, answer) pairs based on the chunk. Each question must be
phrased as something a curious modern reader might naturally ask (NOT a scholastic
restatement of the source). Each answer must be in the target style described
above — multi-paragraph, scholastic, debate-structured where appropriate, with
at least one CCC citation. Each answer should be 150-400 words.

Output STRICT JSON only, no preamble, no markdown fences:

{
  "pairs": [
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."}
  ]
}
"""

USER_TEMPLATE = """SOURCE: {source} ({ref})
TITLE: {title}

CHUNK:
{text}

Generate 2-3 Q/A pairs as described."""


async def generate_for_chunk(
    client: AsyncAnthropic,
    chunk: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> list[dict] | None:
    """Call Claude for one source chunk; return list of {question, answer} or None on error."""
    async with semaphore:
        user_content = USER_TEMPLATE.format(
            source=chunk["source"],
            ref=chunk["ref"],
            title=chunk.get("title", ""),
            text=chunk["text"][:8000],  # cap context per chunk
        )
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=2500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
        except Exception as e:
            print(f"  API error on {chunk['ref']}: {e}", file=sys.stderr)
            return None

        raw = resp.content[0].text
        try:
            data = json.loads(raw)
            return data.get("pairs", [])
        except json.JSONDecodeError:
            # Try to extract from markdown fence as a fallback
            stripped = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                data = json.loads(stripped)
                return data.get("pairs", [])
            except json.JSONDecodeError:
                print(f"  JSON parse error on {chunk['ref']}; first 200 chars: {raw[:200]}", file=sys.stderr)
                return None


def pair_to_chat_record(pair: dict, source_chunk: dict) -> dict:
    """Convert a {question, answer} pair into mlx_lm_lora chat format."""
    return {
        "messages": [
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]},
        ],
        # Provenance kept for spot-checking; mlx_lm_lora ignores extra keys.
        "_provenance": {
            "source": source_chunk["source"],
            "ref": source_chunk["ref"],
        },
    }


async def run(args: argparse.Namespace) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set (check .env)", file=sys.stderr)
        return 1
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    if not CORPUS_PATH.exists():
        print(f"No corpus: {CORPUS_PATH}. Run scrape + clean first.", file=sys.stderr)
        return 1

    chunks = [json.loads(line) for line in CORPUS_PATH.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(chunks)} chunks from {CORPUS_PATH.name}")
    if args.dry_run:
        print(f"\nWould call {model} with up to {args.max_calls} chunks.")
        print(f"\n--- Example chunk that would be sent ---")
        print(json.dumps(chunks[0], indent=2)[:1500])
        return 0

    chunks_to_use = chunks[: args.max_calls]
    print(f"Calling {model} on {len(chunks_to_use)} chunks (concurrency {args.concurrency})…")

    client = AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [generate_for_chunk(client, c, model, semaphore) for c in chunks_to_use]
    results = await asyncio.gather(*tasks)

    all_records: list[dict] = []
    for chunk, pairs in zip(chunks_to_use, results, strict=True):
        if pairs is None:
            continue
        for pair in pairs:
            if "question" in pair and "answer" in pair:
                all_records.append(pair_to_chat_record(pair, chunk))

    print(f"\nGenerated {len(all_records)} (Q, A) pairs.")

    # Train / valid split
    rng = random.Random(args.seed)
    rng.shuffle(all_records)
    split = max(1, int(0.9 * len(all_records)))
    train_records = all_records[:split]
    valid_records = all_records[split:]

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_PATH.open("w") as fh:
        for rec in train_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with VALID_PATH.open("w") as fh:
        for rec in valid_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_records)} train + {len(valid_records)} valid")
    print(f"  {TRAIN_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {VALID_PATH.relative_to(PROJECT_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-calls", type=int, default=200, help="Hard cap on API calls")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent Claude requests")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan; don't call API")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
