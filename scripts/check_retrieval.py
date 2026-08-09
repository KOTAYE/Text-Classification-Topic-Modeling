"""Check that the retrieval threshold still separates answerable questions
from unanswerable ones.

Run this after editing knowledge/about_me.md — adding or rewording facts moves
the scores around, and a threshold tuned for the old text can start refusing
questions the notes do answer.

    python scripts/check_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge import DOC, MIN_SCORE, _passages, _store, lookup

# Questions the notes should be able to answer. Extend this as you add facts.
ANSWERABLE = [
    "When was the student born?",
    "Where does he live?",
    "Where does he study?",
    "What year of university is he in?",
    "When will he graduate?",
    "Who is his mentor?",
    "What dataset did he use?",
    "What accuracy did his model reach?",
    "What are his hobbies?",
    "What languages does he speak?",
    "What tools does he use?",
    "Why did he train on Colab?",
]

# Questions the notes do not cover. The agent must refuse these, not invent.
UNANSWERABLE = [
    "What is his favourite food?",
    "Does he have a driving licence?",
    "What is his phone number?",
    "Who is the president of France?",
    "How tall is he?",
]


def main() -> int:
    print(f"document  : {DOC.name}")
    print(f"passages  : {len(_passages(DOC.read_text(encoding='utf-8')))}")
    print(f"threshold : {MIN_SCORE}\n")

    failures = 0

    print("should be answered")
    lowest = 1.0
    for question in ANSWERABLE:
        score = _store().similarity_search_with_score(question, k=1)[0][1]
        lowest = min(lowest, score)
        if score < MIN_SCORE:
            failures += 1
            print(f"  MISS  {score:.3f}  {question}")
        else:
            print(f"        {score:.3f}  {question}")

    print("\nshould be refused")
    highest = 0.0
    for question in UNANSWERABLE:
        score = _store().similarity_search_with_score(question, k=1)[0][1]
        highest = max(highest, score)
        if not lookup(question).startswith("Nothing"):
            failures += 1
            print(f"  LEAK  {score:.3f}  {question}")
        else:
            print(f"        {score:.3f}  {question}")

    print(f"\nlowest answerable : {lowest:.3f}")
    print(f"highest refusable : {highest:.3f}")
    print(f"gap               : {lowest - highest:+.3f}")

    if failures:
        print(f"\n{failures} problem(s). Adjust MIN_SCORE in src/knowledge.py,")
        print("or reword the notes so the fact is easier to retrieve.")
        return 1

    print("\nok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
