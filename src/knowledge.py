"""Retrieval over facts about the student — the RAG half of the agent."""

import re
from functools import lru_cache
from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

_KNOWLEDGE = Path(__file__).parent.parent / "knowledge"

# The real notes are gitignored so personal details stay off GitHub. A fresh
# clone falls back to the template, which still runs — it just answers with
# placeholders until someone copies it to about_me.md and fills it in.
DOC = _KNOWLEDGE / "about_me.md"
if not DOC.exists():
    DOC = _KNOWLEDGE / "about_me.example.md"

# 80 MB and fast enough on a CPU. Bigger embedding models are pointless here:
# the corpus is one page and the questions are short and literal.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3

# Similarity search always returns its k nearest passages, however far away they
# are, so without a floor the agent gets fed irrelevant text and invents an
# answer from it.
#
# Measured over 17 questions against the filled-in document: answerable ones
# score 0.30-0.59, unanswerable ones 0.21-0.24. That is a gap of only 0.06, so
# this is a coarse filter, not a guarantee — it catches questions the notes are
# nowhere near, and nothing more. The real defence against a made-up answer is
# the instruction in the agent's system prompt to refuse when the retrieved
# passages do not contain the answer.
#
# Re-run scripts/check_retrieval.py after editing the notes.
MIN_SCORE = 0.26

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
def _store() -> InMemoryVectorStore:
    """Embed the document once per process. It is one page, so this is cheap."""
    store = InMemoryVectorStore(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    store.add_texts(_passages(DOC.read_text(encoding="utf-8")))
    return store


def lookup(question: str) -> str:
    """Return the passages closest in meaning to the question, if any are close
    enough to be worth reading."""
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
