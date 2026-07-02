"""FastAPI entrypoint. Stateless: no per-conversation server state."""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agent import run_agent
from .schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("shl-agent")

app = FastAPI(title="SHL Conversational Assessment Recommender", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _warmup() -> None:
    # Pre-load catalog + retriever so first /chat isn't slow.
    try:
        from .retriever import get_retriever
        get_retriever()
        log.info("catalog + retriever warmed")
    except Exception as e:  # noqa: BLE001
        log.warning("warmup failed: %s", e)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")
    if not any(m.role == "user" for m in req.messages):
        raise HTTPException(status_code=400, detail="at least one user message required")
    try:
        return await run_agent(req.messages)
    except Exception as e:  # noqa: BLE001
        log.exception("chat failed")
        # Never break the schema — return a graceful reply.
        return ChatResponse(
            reply="I hit an internal error. Could you rephrase your request?",
            recommendations=[],
            end_of_conversation=False,
        )
