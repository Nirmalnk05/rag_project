"""FastAPI service exposing the agentic RAG graph over Server-Sent Events."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.graph import stream_answer, run_sync
from src.retrieval import Retriever


_retriever: Retriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models + indexes once at startup."""
    global _retriever
    print("[api] loading retriever (models + indexes)...")
    _retriever = Retriever()
    print("[api] ready.")
    yield
    # nothing to clean up


app = FastAPI(title="Production RAG", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)


class AskResponse(BaseModel):
    query: str
    answer: str
    rewritten_query: str
    grade: str
    retrieval: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "retriever_loaded": _retriever is not None}


@app.post("/ask", response_model=AskResponse)
def ask_sync(req: AskRequest) -> AskResponse:
    """Non-streaming JSON endpoint. Easier for eval scripts and tests."""
    state = run_sync(_retriever, req.query)  # type: ignore[arg-type]
    return AskResponse(
        query=req.query,
        answer=state.get("answer", ""),
        rewritten_query=state.get("rewritten", ""),
        grade=state.get("grade", ""),
        retrieval=state.get("retrieval", []),
    )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """Streaming endpoint via SSE. Each event is a JSON object."""

    async def event_generator() -> AsyncIterator[str]:
        async for evt in stream_answer(_retriever, req.query):  # type: ignore[arg-type]
            yield f"data: {json.dumps(evt)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
