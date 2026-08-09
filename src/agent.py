"""Terminal agent. Routes questions to the fine-tuned news classifier."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from classifier import classify_as_text

load_dotenv(Path(__file__).parent / ".env")

# llama-3.1-8b-instant is faster and has a bigger daily quota, but it
# intermittently emits malformed tool calls that Groq rejects with
# tool_use_failed. Correctness wins: qwen handles tool calls reliably and
# phrases the verdict well. Groq counts tokens per model per day, so switching
# models is also the way out of a 429.
LLM = os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")

# Every sentence here earns its place. Without the explicit "report exactly what
# it returns", small models paraphrase the verdict away or claim the tool
# returned nothing and answer from their own guess instead.
SYSTEM_PROMPT = """You are the assistant for Viktor Syrotiuk's ML Summer Camp project.

The classify_news tool runs a transformer Viktor fine-tuned himself. Its output
is authoritative: report exactly what it returns, including the confidence
figure and any secondary signal. Never substitute your own judgement for the
tool's verdict, and never claim the tool gave no output.

Call the tool for any question about what a news story is about, even when the
answer looks obvious to you. Keep answers to one or two sentences."""


@tool
def classify_news(headline: str) -> str:
    """Classify an English news headline or article into one of four topics:
    World, Sports, Business or Sci/Tech.

    Use this whenever the user asks what a news story is about, which category
    it belongs to, or pastes a headline expecting it to be sorted.
    """
    return classify_as_text(headline)


TOOLS = [classify_news]


def build_agent():
    return create_agent(
        ChatGroq(model=LLM, temperature=0),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


def main():
    print("News topic agent. Type 'exit' to quit.")
    print(f"Model: {LLM}")
    print(f"Tools: {', '.join(t.name for t in TOOLS)}\n")

    agent = build_agent()
    history = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": question})
        history = agent.invoke({"messages": history})["messages"]

        print(f"\nAgent: {history[-1].content}\n")


if __name__ == "__main__":
    main()
