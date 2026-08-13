"""An MCP server that lets the agent send Telegram messages.

Runs as a subprocess of the agent and talks over stdio, so nothing is exposed
on the network and there is no second service to deploy.

Started automatically by src/agent.py. To run it by hand for debugging:

    python src/telegram_mcp_server.py
"""

import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(Path(__file__).parent / ".env")

# Telegram puts the bot token in the URL path, and httpx logs request URLs at
# INFO. Left alone, every call prints the token to the console — and into any
# log file, screen recording or shared screen. Silence it.
logging.getLogger("httpx").setLevel(logging.WARNING)

API = "https://api.telegram.org/bot{token}/{method}"
TIMEOUT = 15

mcp = FastMCP("telegram")


def _call(method: str, **payload) -> dict:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in src/.env")

    response = httpx.post(API.format(token=token, method=method), json=payload, timeout=TIMEOUT)
    body = response.json()

    if not body.get("ok"):
        raise RuntimeError(f"Telegram refused the request: {body.get('description', body)}")

    return body["result"]


@mcp.tool()
def send_telegram_message(text: str, chat_id: str | None = None) -> str:
    """Send a message to Telegram.

    Use this when the user asks for something to be sent, forwarded or saved to
    Telegram. Send the finished answer, not a summary of it.

    chat_id is optional — without it the message goes to the chat configured in
    TELEGRAM_CHAT_ID.
    """
    chat = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not chat:
        return "No chat to send to: set TELEGRAM_CHAT_ID in src/.env or pass chat_id."

    sent = _call("sendMessage", chat_id=chat, text=text)
    return f"Sent to Telegram chat {sent['chat']['id']} as message {sent['message_id']}."


@mcp.tool()
def get_telegram_bot_name() -> str:
    """Return the username of the connected Telegram bot.

    Useful to confirm the connection is alive before sending anything.
    """
    bot = _call("getMe")
    return f"@{bot['username']} ({bot['first_name']})"


if __name__ == "__main__":
    mcp.run()
