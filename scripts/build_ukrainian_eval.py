"""Build a Ukrainian evaluation set by translating AG News test examples.

Translation does not change what a story is about, so the original labels carry
over. That gives a Ukrainian test set with trustworthy labels without hand
annotation, and — because it is the same items as the English test set — the
gap between the two scores measures the language, not the sample.

Runs once and writes eval/ag_news_uk.jsonl. Costs a couple of cents.

    python scripts/build_ukrainian_eval.py
"""

import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "ag_news_uk.jsonl"

SAMPLE_SIZE = 500
SEED = 42
MODEL = "gpt-4o-mini"

PROMPT = """Translate this English news snippet into natural Ukrainian.

Keep proper nouns as a Ukrainian reader would write them. Do not summarise,
explain, or add anything — return only the translation."""


def main() -> int:
    # This is the only part of the project that spends money, so it refuses to
    # run twice by accident. Pass --force to translate again.
    if OUT.exists() and "--force" not in sys.argv:
        rows = sum(1 for _ in OUT.open(encoding="utf-8"))
        print(f"{OUT.relative_to(ROOT)} already has {rows} rows — nothing to do.")
        print("Pass --force to spend the API credits again.")
        return 0

    load_dotenv(ROOT / "src" / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set in src/.env")
        return 1

    test = load_dataset("fancyzhx/ag_news", split="test")
    sample = test.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

    client = OpenAI()
    OUT.parent.mkdir(exist_ok=True)

    with OUT.open("w", encoding="utf-8") as f:
        for row in tqdm(sample, desc="translating"):
            reply = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": row["text"]},
                ],
            )
            f.write(
                json.dumps(
                    {
                        "label": row["label"],
                        "en": row["text"],
                        "uk": reply.choices[0].message.content.strip(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"\nwrote {SAMPLE_SIZE} rows to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
