"""Report what retrieval finds for each question, in both languages.

Run this after editing knowledge/about_me.md: rewording a fact can make it
unfindable, and a question that scores near the bottom is one edit away from
being missed entirely.

Read it as a ranking, not a pass/fail. The scores are not comparable across
languages — a Ukrainian question against these English notes scores far lower
than the English equivalent for the same fact — so no single cutoff separates
answerable from unanswerable, and the agent's own judgement is what decides.
See the note on MIN_SCORE in src/knowledge.py.

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

ANSWERABLE_UK = [
    "Що це за проєкт?",
    "Коли народився студент?",
    "Де він живе?",
    "Де він навчається?",
    "Хто його ментор?",
    "Який датасет він використав?",
    "Якої точності досягла модель?",
    "Які в нього захоплення?",
    "Якими мовами він володіє?",
    "Якими інструментами він користується?",
]

# Questions the notes do not cover, kept as a reference point: whatever the
# worst answerable question scores should still sit above these.
UNANSWERABLE = [
    "What is his favourite food?",
    "Does he have a driving licence?",
    "What is his phone number?",
    "Who is the president of France?",
    "How tall is he?",
    "Яка його улюблена їжа?",
    "Який у нього зріст?",
]


def report(title: str, questions: list[str]) -> float:
    """Print each question with its best match, and return the weakest score."""
    print(title)
    weakest = 1.0
    for question in questions:
        doc, score = _store().similarity_search_with_score(question, k=1)[0]
        weakest = min(weakest, score)
        found = " ".join(doc.page_content.split())[:56]
        print(f"  {score:.3f}  {question:44s} {found}")
    print()
    return weakest


def main() -> int:
    print(f"document : {DOC.name}")
    print(f"passages : {len(_passages(DOC.read_text(encoding='utf-8')))}")
    print(f"floor    : {MIN_SCORE}\n")

    weakest_en = report("answerable, English", ANSWERABLE)
    weakest_uk = report("answerable, Ukrainian", ANSWERABLE_UK)

    print("not covered by the notes")
    strongest_miss = 0.0
    for question in UNANSWERABLE:
        score = _store().similarity_search_with_score(question, k=1)[0][1]
        strongest_miss = max(strongest_miss, score)
        print(f"  {score:.3f}  {question}")

    print(f"\nweakest answerable  English {weakest_en:.3f}   Ukrainian {weakest_uk:.3f}")
    print(f"strongest not-covered       {strongest_miss:.3f}")
    print(
        "\nUkrainian scores lower across the board — the notes are in English.\n"
        "That overlap is why the cutoff is only a floor and the agent decides."
    )

    below_floor = [
        q
        for q in ANSWERABLE + ANSWERABLE_UK
        if _store().similarity_search_with_score(q, k=1)[0][1] < MIN_SCORE
    ]
    if below_floor:
        print(f"\n{len(below_floor)} answerable question(s) fall below the floor:")
        for q in below_floor:
            print(f"  {q}")
        print("Reword the relevant note so the fact is easier to find.")
        return 1

    print("\nok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
