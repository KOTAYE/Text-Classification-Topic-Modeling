"""Model service: the fine-tuned classifier, and nothing else.

The only container that carries torch and the 1.1 GB of weights. Everything
that wants a classification asks this over HTTP, so the model is loaded once
however many front-ends are running.
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent / "src"))

from classifier import MODEL_ID, _classifier, classify, classify_as_text  # noqa: E402

app = FastAPI(title="model service")


class Text(BaseModel):
    text: str


class Score(BaseModel):
    label: str
    score: float


@app.on_event("startup")
def warm_up():
    """Load the weights now rather than on whoever asks first."""
    _classifier()


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID}


@app.post("/classify", response_model=list[Score])
def classify_endpoint(payload: Text):
    """Every class with its probability, most likely first."""
    return classify(payload.text)


@app.post("/classify/text")
def classify_text_endpoint(payload: Text):
    """The one-line verdict the agent reads."""
    return {"verdict": classify_as_text(payload.text)}
