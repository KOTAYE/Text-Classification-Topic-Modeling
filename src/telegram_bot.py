"""Telegram front-end: the same agent, reachable from a phone.

Three ways in now — terminal, web page, and this — all running the same agent
over the same three tools. This one only carries messages: it long-polls
Telegram for anything sent to the bot, hands it to the agent, and posts the
answer back to the chat it came from.

Not to be confused with telegram_mcp_server.py, which points the other way:
that one lets the agent send a message on its own initiative.

    python src/telegram_bot.py
"""

import asyncio
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from agent import build_agent

load_dotenv(Path(__file__).parent / ".env")

# The token rides in the URL path and httpx logs URLs at INFO.
logging.getLogger("httpx").setLevel(logging.WARNING)

API = "https://api.telegram.org/bot{token}/{method}"

# Telegram holds the request open until something arrives, so the loop costs
# one request per half minute rather than constant polling.
POLL_SECONDS = 30

# Keep the last few turns per chat so follow-up questions make sense, without
# letting a long-running conversation grow without bound.
HISTORY_TURNS = 12

GREETING = (
    "Send me a news headline and I will tell you its topic — World, Sports, "
    "Business or Sci/Tech — using a model I fine-tuned myself.\n\n"
    "You can also ask about me: where I study, what this project is, what "
    "accuracy the model reached."
)


class Telegram:
    def __init__(self, http: httpx.AsyncClient, token: str):
        self._http = http
        self._token = token

    async def call(self, method: str, **payload):
        response = await self._http.post(
            API.format(token=self._token, method=method), json=payload
        )
        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(f"Telegram refused {method}: {body.get('description')}")
        return body["result"]

    async def updates(self, offset: int | None):
        return await self.call("getUpdates", offset=offset, timeout=POLL_SECONDS)

    async def send(self, chat_id: int, text: str):
        return await self.call("sendMessage", chat_id=chat_id, text=text)

    async def typing(self, chat_id: int):
        """Answers take a few seconds; without this the chat looks dead."""
        return await self.call("sendChatAction", chat_id=chat_id, action="typing")


async def handle(agent, telegram: Telegram, histories: dict, message: dict):
    chat_id = message["chat"]["id"]
    text = (message.get("text") or "").strip()
    if not text:
        return

    who = message["chat"].get("username") or message["chat"].get("first_name") or chat_id
    print(f"[{who}] {text[:80]}")

    if text.startswith("/start"):
        histories.pop(chat_id, None)
        await telegram.send(chat_id, GREETING)
        return

    await telegram.typing(chat_id)

    history = histories.get(chat_id, [])
    history.append({"role": "user", "content": text})

    try:
        result = await agent.ainvoke({"messages": history})
    except Exception as error:  # a bad reply beats a silent bot
        print(f"  failed: {type(error).__name__}: {error}")
        await telegram.send(chat_id, "Something went wrong on my side. Try again?")
        return

    histories[chat_id] = result["messages"][-HISTORY_TURNS:]

    reply = result["messages"][-1].content
    await telegram.send(chat_id, reply if isinstance(reply, str) else str(reply))
    print(f"  -> {reply[:80]}")


async def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set in src/.env")
        return 1

    print("Starting up... ", end="", flush=True)
    agent, tools = await build_agent()
    print("ready")
    print(f"Tools: {', '.join(t.name for t in tools)}")

    histories: dict[int, list] = {}
    offset: int | None = None

    async with httpx.AsyncClient(timeout=POLL_SECONDS + 15) as http:
        telegram = Telegram(http, token)
        me = await telegram.call("getMe")
        print(f"Listening as @{me['username']}. Ctrl-C to stop.\n")

        while True:
            try:
                updates = await telegram.updates(offset)
            except (httpx.HTTPError, RuntimeError) as error:
                # A dropped connection should cost a retry, not the process.
                print(f"poll failed: {error}")
                await asyncio.sleep(3)
                continue

            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message")
                if message:
                    await handle(agent, telegram, histories, message)


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nstopped")
