"""Terminal agent.

Three tools: the fine-tuned news classifier, retrieval over the student's own
notes, and Telegram — the last one reached over MCP rather than called directly.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.stdio import get_default_environment

from classifier import classify_as_text
from knowledge import lookup

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")


def _llm():
    """Pick the chat model from whichever API key is configured.

    OpenAI wins when its key is present. Groq is the free-tier fallback, and
    on Groq the model matters: llama-3.1-8b-instant is faster with a bigger
    daily quota but intermittently emits malformed tool calls that come back
    as tool_use_failed, so the default is qwen. Groq also counts tokens per
    model per day, which makes switching models the way out of a 429.
    """
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        # gpt-4o-mini rather than the newer gpt-4.1-mini or -nano: measured on
        # this prompt, it is the only one of the three that repeats the
        # classifier's confidence figures instead of paraphrasing them away.
        return ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    from langchain_groq import ChatGroq

    return ChatGroq(model=os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b"), temperature=0)


# Every sentence here earns its place. Without the explicit "report exactly what
# it returns", small models paraphrase the verdict away or claim the tool
# returned nothing and answer from their own guess instead.
SYSTEM_PROMPT = """You are the assistant for Viktor Syrotiuk's ML Summer Camp project.

classify_news runs a transformer Viktor fine-tuned himself. Its output is
authoritative: report exactly what it returns, including the confidence figure
and any secondary signal. Never substitute your own judgement for the tool's
verdict, and never claim the tool gave no output. Call it for any question
about what a news story is about, even when the answer looks obvious to you.

about_student searches Viktor's own notes about himself. Use it for any
question about Viktor — his background, studies, the project, his tools.

It returns the passages closest to the question, which are not always relevant:
it hands over its best guesses and leaves the judgement to you. Read them. If
they answer the question, answer from them. If they do not, say you do not have
that information — do not stretch a loosely related passage into an answer, do
not fill the gap from your own knowledge, and do not guess.

send_telegram_message delivers a message to Viktor's Telegram. Only use it when
the user explicitly asks for something to be sent, forwarded or saved there.

A bare headline is not such a request. Someone who pastes a news story wants it
classified and nothing else — classify it, answer, and stop. Sending a message
nobody asked for is worse than not sending one.

Send the answer, not the question. Asked to classify a headline and send the
result, the message should carry the topic and the confidence — quoting the
headline as well is fine, but a message holding only the original text is
useless to whoever receives it. Say afterwards that it was sent.

Answer in whatever language the user wrote in, but never translate the four
class names. They stay exactly as World, Sports, Business and Sci/Tech in every
language, because they are the model's own labels and appear that way in its
configuration, its metrics and its documentation. Writing "Спорт" in a Ukrainian
sentence breaks the match with everything else that reports on this model.

Keep answers to one or two sentences."""


@tool
def classify_news(headline: str) -> str:
    """Classify a news headline or article into one of four topics:
    World, Sports, Business or Sci/Tech.

    Use this whenever the user asks what a news story is about, which category
    it belongs to, or pastes a headline expecting it to be sorted.

    The four topic names are identifiers, not words. Copy whichever one comes
    back exactly as spelled, in every language — "Sports", never "Спорт";
    "World", never "Світ". They appear in this spelling in the model's
    configuration and in every metric reported about it, and a translated name
    no longer matches any of that.
    """
    return classify_as_text(headline)


@tool
def about_student(question: str) -> str:
    """Answer questions about Viktor Syrotiuk, the student who built this
    project: when he was born, where he studies, what he worked on, which
    tools he uses.

    Pass the user's question through as they phrased it. Do not rewrite it to
    insert his name — the search matches on meaning, and every note already
    starts with his name, so adding it makes every note look equally relevant.

    Returns the passages from his notes that are closest to the question, or
    says nothing is on file when the notes do not cover it.
    """
    return lookup(question)


LOCAL_TOOLS = [classify_news, about_student]


async def telegram_tools() -> list:
    """Tools served by the Telegram MCP server, or none if it is not configured.

    The server runs as a subprocess over stdio rather than as a network
    service: nothing gets exposed on a port, and the container stays a single
    process. Without a bot token the agent simply comes up without Telegram
    instead of refusing to start.
    """
    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        return []

    # MCP deliberately starts servers with a minimal environment — PATH, HOME,
    # TEMP and little else — so a server cannot read every secret the host
    # happens to hold. Locally that goes unnoticed because the server reads
    # src/.env itself, but that file is kept out of the image, so inside the
    # container it would find no token. Forward the two variables it needs and
    # nothing more: the OpenAI key stays out of the subprocess.
    env = get_default_environment()
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        if os.environ.get(name):
            env[name] = os.environ[name]

    client = MultiServerMCPClient(
        {
            "telegram": {
                "command": sys.executable,
                "args": [str(HERE / "telegram_mcp_server.py")],
                "transport": "stdio",
                "env": env,
            }
        }
    )
    return await client.get_tools()


async def build_agent():
    tools = LOCAL_TOOLS + await telegram_tools()
    return create_agent(_llm(), tools=tools, system_prompt=SYSTEM_PROMPT), tools


def _warm_up():
    """Touch both models up front.

    Each takes about three seconds to load on first use. Doing it here, behind
    a visible message, beats letting it land silently on whatever the user
    types first.
    """
    classify_as_text("warm up")
    lookup("warm up")


async def main():
    print("News topic agent. Type 'exit' to quit.")
    print("\nStarting up... ", end="", flush=True)

    agent, tools = await build_agent()
    _warm_up()
    print("ready\n")

    print(f"Model: {_llm().model_name}")
    print(f"Tools: {', '.join(t.name for t in tools)}\n")

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
        history = (await agent.ainvoke({"messages": history}))["messages"]

        print(f"\nAgent: {history[-1].content}\n")


if __name__ == "__main__":
    asyncio.run(main())
