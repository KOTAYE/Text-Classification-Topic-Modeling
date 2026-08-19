"""News topic classifier — the fine-tuned model from the Hugging Face Hub.

Runs the model in this process by default. Set MODEL_SERVICE_URL and it calls
the model service instead, which is how it works under docker compose: the
weights are loaded once, in one container, rather than once per front-end.

Callers see the same two functions either way.
"""

import os
from functools import lru_cache

import httpx

MODEL_ID = "KOTAYE/xlm-roberta-base-ag-news"

SERVICE_URL = os.environ.get("MODEL_SERVICE_URL")
TIMEOUT = 30

# A headline can plausibly belong to two topics — business and tech news
# overlap constantly. Surface the runner-up when it is close. 0.15 rather than
# a rounder number: an Apple earnings headline splits 0.82 / 0.18, while an
# unambiguous one sits at 0.98 / 0.02.
SECONDARY_SIGNAL = 0.15


@lru_cache(maxsize=1)
def _classifier():
    """Load the model here. Imported lazily so the agent service, which never
    calls this, does not need torch installed at all."""
    from transformers import pipeline

    return pipeline("text-classification", model=MODEL_ID, top_k=None)


def classify(text: str) -> list[dict]:
    """Every topic with its probability, most likely first."""
    if SERVICE_URL:
        response = httpx.post(f"{SERVICE_URL}/classify", json={"text": text}, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()

    scores = _classifier()(text)[0]
    return sorted(scores, key=lambda s: s["score"], reverse=True)


def classify_as_text(text: str) -> str:
    """Human-readable verdict. This is what the agent reads."""
    ranked = classify(text)
    best, second = ranked[0], ranked[1]

    line = f"Topic: {best['label']} (confidence {best['score']:.0%})"
    if second["score"] > SECONDARY_SIGNAL:
        line += f", with a secondary signal of {second['label']} ({second['score']:.0%})"

    return line


if __name__ == "__main__":
    print(f"source: {SERVICE_URL or 'local model'}\n")

    for sample in [
        "Manchester United signs new striker ahead of the season",
        "Apple reports record quarterly revenue driven by iPhone sales",
        "UN Security Council meets over escalating border conflict",
        "Scientists discover new method for carbon capture",
    ]:
        print(sample)
        print(f"  {classify_as_text(sample)}\n")
