"""HTTP API behind the web interface.

Wraps the same agent the terminal app uses, and adds what a screen can show
but a terminal cannot: the classifier's full probability distribution, and
which tools the agent reached for.

    uvicorn api:app --reload --port 8000     (from src/)
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import build_agent
from classifier import classify

WEB_DIST = Path(__file__).parent.parent / "web" / "dist"

_agent = None
_tool_names: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the agent once, at startup, rather than per request."""
    global _agent, _tool_names
    _agent, tools = await build_agent()
    _tool_names = [t.name for t in tools]
    yield


app = FastAPI(title="News topic agent", lifespan=lifespan)

# The Vite dev server runs on another port during development. In production
# the built files are served from here, so this only matters while developing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Turn] = []


class Score(BaseModel):
    label: str
    score: float


class Delivery(BaseModel):
    text: str
    receipt: str


class ChatResponse(BaseModel):
    reply: str
    tools_used: list[str]
    distribution: list[Score] | None = None
    classified_text: str | None = None
    telegram: Delivery | None = None


def _as_text(content) -> str:
    """Flatten a message body to plain text.

    Locally defined tools return strings, but results that arrive over MCP come
    back as a list of content blocks, so anything read off a message has to
    survive both shapes.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return " ".join(part for part in parts if part)
    return str(content)


@app.get("/api/health")
async def health():
    return {"ok": _agent is not None, "tools": _tool_names}


@app.post("/api/classify", response_model=list[Score])
async def classify_endpoint(payload: dict):
    """Raw classifier output — every class with its probability."""
    return classify(payload["text"])


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = [t.model_dump() for t in request.history]
    messages.append({"role": "user", "content": request.message})

    result = await _agent.ainvoke({"messages": messages})

    calls = [c for m in result["messages"] for c in (getattr(m, "tool_calls", None) or [])]

    # When the agent classified something, hand back the full distribution so
    # the page can draw it. The tool itself only reports the winner, and the
    # interesting part — how close the runner-up was — is exactly what gets
    # lost in a one-line answer. One extra local inference, tens of
    # milliseconds, and no second round trip from the browser.
    distribution = None
    classified_text = None
    for call in calls:
        if call["name"] == "classify_news":
            classified_text = call["args"].get("headline")
            if classified_text:
                # Only the bar chart depends on this. If the classifier is
                # unavailable the agent has already said so in its reply, and
                # losing the chart is better than losing the whole response.
                try:
                    distribution = classify(classified_text)
                except Exception:
                    distribution = None
            break

    # A badge saying "telegram" proves nothing to someone watching a demo.
    # Return what was sent alongside Telegram's own acknowledgement — chat and
    # message number — so the page can show a receipt rather than a claim.
    telegram = None
    for call in calls:
        if call["name"] != "send_telegram_message":
            continue
        receipt = next(
            (
                _as_text(m.content)
                for m in result["messages"]
                if getattr(m, "tool_call_id", None) == call["id"]
            ),
            "",
        )
        telegram = Delivery(text=call["args"].get("text", ""), receipt=receipt)
        break

    return ChatResponse(
        reply=_as_text(result["messages"][-1].content),
        tools_used=[c["name"] for c in calls],
        distribution=distribution,
        classified_text=classified_text,
        telegram=telegram,
    )


# Serve the built frontend, when there is one. Mounted last so it cannot
# shadow the /api routes above.
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    async def spa(path: str):
        return FileResponse(WEB_DIST / "index.html")
