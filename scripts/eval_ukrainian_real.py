"""Second opinion on the Ukrainian numbers, using text nobody translated.

scripts/eval_ukrainian.py measures machine-translated AG News, which reads
like English wearing Ukrainian grammar. This script runs the same classifier
over Zarakun/ukrainian_news — a small set of genuine Ukrainian news — to see
whether the translated figure holds up on the real thing.

The two do not measure the same population and are not directly comparable:
this one is a sanity check, not a benchmark. It is also small, so treat the
number as an indication rather than a measurement.

Read the per-rubric breakdown, not the headline accuracy. The headline is
dragged down by one rubric, "zakordon" — Ukrainian for "abroad" — which turns
out to hold fuel prices in Germany, labour shortages in the Netherlands and
advice on learning languages. AG News' "World" means international politics
and conflict, so mapping the two was a mistake on my part, and the model
disagreeing with that mapping is not the model being wrong.

    python scripts/eval_ukrainian_real.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import concatenate_datasets, load_dataset
from sklearn.metrics import accuracy_score, classification_report

from classifier import _classifier

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
BATCH = 16

# Only the categories with an honest counterpart among AG News' four. The rest
# of this dataset — fashion, kino, health, porady, smachnonews, show,
# realestate, education, fun — has nowhere to go and is dropped rather than
# forced into a class it does not belong to.
TOPIC_MAP = {
    "sport": "Sports",
    "tech": "Sci/Tech",
    "business": "Business",
    "economy": "Business",
    "financy": "Business",
    "zakordon": "World",
}


def main() -> int:
    ds = load_dataset("Zarakun/ukrainian_news")
    rows = concatenate_datasets([ds[s] for s in ds])

    kept = [r for r in rows if r["main_topic"] in TOPIC_MAP]
    dropped = len(rows) - len(kept)
    print(f"{len(rows)} rows, kept {len(kept)}, dropped {dropped} with no counterpart\n")

    print("mapped classes:", Counter(TOPIC_MAP[r["main_topic"]] for r in kept), "\n")

    truth = [LABELS.index(TOPIC_MAP[r["main_topic"]]) for r in kept]
    texts = [r["text"] for r in kept]

    pipe = _classifier()
    pred = []
    for i in range(0, len(texts), BATCH):
        for scores in pipe(texts[i : i + BATCH], truncation=True, max_length=128):
            pred.append(LABELS.index(max(scores, key=lambda s: s["score"])["label"]))

    print(f"accuracy on real Ukrainian news: {accuracy_score(truth, pred):.4f}")
    print("(read the per-rubric table below before quoting that number)\n")
    print(classification_report(truth, pred, target_names=LABELS, digits=3, zero_division=0))

    # Per original rubric: this is where the headline number comes apart.
    print("what each original Ukrainian rubric gets called")
    by_rubric = {}
    for row, p in zip(kept, pred):
        by_rubric.setdefault(row["main_topic"], Counter())[LABELS[p]] += 1

    for rubric, counts in sorted(by_rubric.items()):
        total = sum(counts.values())
        share = ", ".join(f"{name} {n / total:.0%}" for name, n in counts.most_common())
        print(f"  {rubric:9s} -> {TOPIC_MAP[rubric]:9s}  n={total:4d}   {share}")

    without = [(t, p) for row, t, p in zip(kept, truth, pred) if row["main_topic"] != "zakordon"]
    if without:
        acc = accuracy_score([t for t, _ in without], [p for _, p in without])
        print(f"\nexcluding the zakordon rubric: {acc:.4f} over {len(without)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
