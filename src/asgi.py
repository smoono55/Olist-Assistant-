# src/asgi.py
"""
ASGI wrapper for Olist Assistant.

Provides a small FastAPI app with a /api/chat endpoint that delegates to src.agent.agent_handle.
This file must define `app` for uvicorn to import.
"""

from __future__ import annotations
import os
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# try importing agent_handle from src.agent
try:
    from src.agent import agent_handle  # agent_handle(user_text)->(reply:str, rows:list, html:str)
except Exception as e:
    # try fallback import if running from some setups
    try:
        from agent import agent_handle  # fallback if PYTHONPATH differs
    except Exception:
        agent_handle = None
        import_error = e
    else:
        import_error = None
else:
    import_error = None

# minimal request/response models
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    reply: str
    evidence_count: int
    evidence_rows: List[Dict[str, Any]] = []
    evidence_html: str | None = None

app = FastAPI(title="Olist Assistant API")

# allow local frontend (Next) on :3000 and any local origin for testing
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins + ["*"],  # you can restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

@app.on_event("startup")
def startup_check():
    # ensure agent_handle is importable and warn if not
    if agent_handle is None:
        logger.exception("agent_handle import failed at startup. See previous exception.")
        # raise with helpful text so uvicorn displays failure
        raise RuntimeError(
            "Could not import agent_handle from src.agent. Make sure src/agent.py exists and defines agent_handle(user_text). "
            "Original import error: {}".format(repr(import_error))
        )
    # optional: check DB path existence if agent module uses DB (helps quick-fail)
    try:
        # call a light health-check: send empty string to agent_handle and see it doesn't crash
        reply, rows, html = agent_handle("ping")
        logger.info("agent_handle ping OK; returned %s (rows=%d)", reply[:80], len(rows) if rows else 0)
    except Exception as e:
        logger.warning("agent_handle ping produced exception: %s", e)
        # don't fail startup here — but log the issue
        # if you prefer, raise to fail fast:
        # raise

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Basic chat endpoint. Frontend should POST JSON: {"message":"...", "session_id": "..."}.
    Returns: { reply, evidence_count, evidence_rows, evidence_html }
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    if agent_handle is None:
        raise HTTPException(status_code=500, detail="agent not available (import failed)")

    try:
        reply, evidence_rows, evidence_html = agent_handle(req.message)
        # normalize outputs in case agent returned other shapes
        if evidence_rows is None:
            evidence_rows = []
        if evidence_html is None:
            evidence_html = ""
        return ChatResponse(
            reply=str(reply),
            evidence_count=len(evidence_rows),
            evidence_rows=evidence_rows,
            evidence_html=evidence_html
        )
    except Exception as e:
        logger.exception("Error running agent_handle")
        raise HTTPException(status_code=500, detail=f"agent error: {e}")
