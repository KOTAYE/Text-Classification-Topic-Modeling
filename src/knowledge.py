"""Retrieval over facts about the student — the RAG half of the agent.

Embeds and searches in this process by default. Set KNOWLEDGE_SERVICE_URL and
it calls the knowledge service instead, which is how it works under docker
compose. Callers see the same `lookup` either way.
"""

import os
import re
from functools import lru_cache
from pathlib import Path

import httpx

SERVICE_URL = os.environ.get("KNOWLEDGE_SERVICE_URL")
TIMEOUT = 30

_KNOWLEDGE = Path(__file__).parent.parent / "knowledge"

# The real notes are gitignored so personal details stay off GitHub. A fresh
# clone falls back to the template, which still runs — it just answers with
# placeholders until someone copies it to about_me.md and fills it in.
DOC = _KNOWLEDGE / "about_me.md"
if not DOC.exists():
    DOC = _KNOWLEDGE / "about_me.example.md"

# Multilingual, because the notes are in English but the questions are not.
# all-MiniLM-L6-v2 is a quarter the size and slightly sharper, but it only
# knows English: ask it "Що це за проєкт?" and it matches nothing, so the
# agent reports having no information about a project the notes describe in
# detail. Cross-language retrieval is the whole point here.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Five rather than three. Every passage repeats the student's name, so a query
# that also contains it matches all of them about equally and the useful signal
# gets swamped: asking "What accuracy did Viktor's model reach?" ranks the
# passage holding the number fourth, while the same question phrased with "his"
# ranks it first. Widening the window covers that, and the model filters.
TOP_K = 5

# A floor against total nonsense, and nothing more.
#
# This started as a relevance filter at 0.26, tuned on English questions where
# answerable ones scored 0.30-0.59 and unanswerable ones 0.21-0.24. Ukrainian
# questions against these English notes broke that outright: "Що це за проєкт?"
# scores 0.10 while "Яка його улюблена їжа?" scores 0.24, so the answerable
# question sits *below* the unanswerable one and no threshold can separate
# them. Cross-language similarity is systematically lower, and the numbers are
# not comparable across languages.
#
# So the threshold is not the defence. The defence is the instruction in the
# agent's system prompt to refuse when the retrieved passages do not answer the
# question — which it did correctly even before this cutoff existed. Retrieval
# hands over its best few passages; the model decides whether they are worth
# anything.
#
# scripts/check_retrieval.py still reports the numbers, which is useful for
# spotting a fact that became unfindable after an edit.
MIN_SCORE = 0.05

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def _passages(markdown: str) -> list[str]:
    """One passage per paragraph.

    about_me.md is written so that every paragraph stands on its own, which
    is what makes a plain blank-line split enough. Headings are dropped: they
    carry no facts, and an indexed "## Education" would match almost any
    question about studying without answering it.
    """
    text = _COMMENT.sub("", markdown)

    passages = []
    for block in text.split("\n\n"):
        block = " ".join(block.split())
        if block.startswith("#") or len(block) < 30:
            continue
        passages.append(block)

    return passages


@lru_cache(maxsize=1)
def _store():
    """Embed the document once per process. It is one page, so this is cheap.

    Imported lazily so the agent service, which delegates to the knowledge
    service, does not need sentence-transformers or torch installed.
    """
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings

    store = InMemoryVectorStore(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    store.add_texts(_passages(DOC.read_text(encoding="utf-8")))
    return store


def lookup(question: str) -> str:
    """Return the passages closest in meaning to the question, if any are close
    enough to be worth reading."""
    if SERVICE_URL:
        response = httpx.post(
            f"{SERVICE_URL}/search", json={"question": question}, timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()["passages"]

    hits = _store().similarity_search_with_score(question, k=TOP_K)
    relevant = [doc for doc, score in hits if score >= MIN_SCORE]

    if not relevant:
        return "Nothing on file about that."

    return "\n".join(f"- {doc.page_content}" for doc in relevant)


if __name__ == "__main__":
    passages = _passages(DOC.read_text(encoding="utf-8"))
    print(f"{len(passages)} passages indexed\n")

    for question in [
        "When was the student born?",
        "Where does he study?",
        "What accuracy did his model reach?",
        "What is his favourite food?",
    ]:
        print(f"Q: {question}")
        print(lookup(question))
        print()
