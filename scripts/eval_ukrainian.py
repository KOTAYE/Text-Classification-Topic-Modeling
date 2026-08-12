"""Measure the cost of running the English-trained classifier on Ukrainian.

Scores the same 500 items in both languages, so the difference is the language
gap rather than sampling noise.

    python scripts/eval_ukrainian.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.metrics import accuracy_score, classification_report, f1_score

from classifier import _classifier

ROOT = Path(__file__).parent.parent
DATA = ROOT / "eval" / "ag_news_uk.jsonl"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
BATCH = 32


def predict(texts: list[str]) -> list[int]:
    pipe = _classifier()
    out = []
    for i in range(0, len(texts), BATCH):
        for scores in pipe(texts[i : i + BATCH]):
            best = max(scores, key=lambda s: s["score"])
            out.append(LABELS.index(best["label"]))
    return out


def main() -> int:
    if not DATA.exists():
        print(f"{DATA.relative_to(ROOT)} is missing — run scripts/build_ukrainian_eval.py first")
        return 1

    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
    truth = [r["label"] for r in rows]
    print(f"{len(rows)} items, same stories in both languages\n")

    results = {}
    for lang in ("en", "uk"):
        pred = predict([r[lang] for r in rows])
        results[lang] = pred
        print(f"--- {lang} ---")
        print(f"accuracy  {accuracy_score(truth, pred):.4f}")
        print(f"macro F1  {f1_score(truth, pred, average='macro'):.4f}")
        print(classification_report(truth, pred, target_names=LABELS, digits=3, zero_division=0))

    drop = accuracy_score(truth, results["en"]) - accuracy_score(truth, results["uk"])
    print(f"accuracy lost by switching to Ukrainian: {drop * 100:+.2f} points")

    agree = sum(a == b for a, b in zip(results["en"], results["uk"])) / len(rows)
    print(f"same verdict in both languages: {agree:.1%} of stories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
