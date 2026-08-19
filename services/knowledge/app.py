"""Knowledge service: retrieval over the student's notes.

Holds the multilingual embedding model and the vector index. Separate from the
classifier because the two are independent and differently sized — this one is
a few hundred megabytes against the classifier's gigabyte.
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge import DOC, EMBEDDING_MODEL, _passages, _store, lookup  # noqa: E402

app = FastAPI(title="knowledge service")


class Question(BaseModel):
    question: str


@app.on_event("startup")
def warm_up():
    """Embed the notes now rather than on whoever asks first."""
    _store()


@app.get("/health")
def health():
    return {
        "ok": True,
        "document": DOC.name,
        "passages": len(_passages(DOC.read_text(encoding="utf-8"))),
        "embeddings": EMBEDDING_MODEL,
    }


@app.post("/search")
def search(payload: Question):
    """The passages closest in meaning to the question, as the agent reads them."""
    return {"passages": lookup(payload.question)}
