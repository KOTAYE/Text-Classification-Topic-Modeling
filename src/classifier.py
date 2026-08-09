"""News topic classifier — wraps the fine-tuned model from the Hugging Face Hub."""

from functools import lru_cache

from transformers import pipeline

MODEL_ID = "KOTAYE/xlm-roberta-base-ag-news"


@lru_cache(maxsize=1)
def _classifier():
    """Load the model once and reuse it. First call downloads ~1.1 GB."""
    return pipeline("text-classification", model=MODEL_ID, top_k=None)


def classify(text: str) -> list[dict]:
    """Return every topic with its probability, most likely first."""
    scores = _classifier()(text)[0]
    return sorted(scores, key=lambda s: s["score"], reverse=True)


def classify_as_text(text: str) -> str:
    """Human-readable verdict. This is what the agent will read."""
    ranked = classify(text)
    best, second = ranked[0], ranked[1]

    line = f"Topic: {best['label']} (confidence {best['score']:.0%})"

    # A single headline can plausibly belong to two topics — business and
    # tech news overlap constantly. Surface the runner-up when it is close.
    # 0.15 rather than a rounder number: an Apple earnings headline splits
    # 0.82 / 0.18, while an unambiguous one sits at 0.98 / 0.02.
    if second["score"] > 0.15:
        line += f", with a secondary signal of {second['label']} ({second['score']:.0%})"

    return line


if __name__ == "__main__":
    samples = [
        "Manchester United signs new striker ahead of the season",
        "Apple reports record quarterly revenue driven by iPhone sales",
        "UN Security Council meets over escalating border conflict",
        "Scientists discover new method for carbon capture",
    ]

    for sample in samples:
        print(f"\n{sample}")
        print(f"  {classify_as_text(sample)}")
