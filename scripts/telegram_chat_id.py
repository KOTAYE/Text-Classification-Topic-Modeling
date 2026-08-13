"""Find the chat id to put in TELEGRAM_CHAT_ID.

Telegram will not tell a bot who to talk to until someone talks to it first.
So: open your bot in Telegram, send it any message, then run this.

    python scripts/telegram_chat_id.py
"""

import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

# The token travels in the URL path, and httpx logs URLs at INFO level.
logging.getLogger("httpx").setLevel(logging.WARNING)

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / "src" / ".env")


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set in src/.env")
        return 1

    me = httpx.get(f"https://api.telegram.org/bot{token}/getMe", timeout=15).json()
    if not me.get("ok"):
        print(f"Telegram rejected the token: {me.get('description')}")
        return 1
    print(f"bot: @{me['result']['username']}\n")

    updates = httpx.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=15).json()
    chats = {}
    for update in updates.get("result", []):
        message = update.get("message") or update.get("channel_post") or {}
        chat = message.get("chat")
        if chat:
            chats[chat["id"]] = chat.get("username") or chat.get("title") or chat.get("first_name")

    if not chats:
        print("No messages yet. Open the bot in Telegram, send it anything, and run this again.")
        print("(Telegram only reports chats that have written to the bot first.)")
        return 1

    print("chats that have messaged this bot:")
    for chat_id, name in chats.items():
        print(f"  TELEGRAM_CHAT_ID={chat_id}    ({name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
