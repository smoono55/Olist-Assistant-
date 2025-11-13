# src/api_server.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Any, Dict, List

# Import agent_process from your existing app module (must be in PYTHONPATH)
# app.py should define: agent_process(user_text: str, mem: Dict[str,Any]) -> Tuple[str, List[Dict], Dict]
try:
    from src.app import agent_process, load_memory, save_memory
except Exception as e:
    # Try alternative import path if running from project root
    from app import agent_process, load_memory, save_memory  # type: ignore

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    rows: List[Dict[str, Any]] = []
    memory_snapshot: Dict[str, Any] = {}

app = FastAPI(title="Olist Agent API")

# Allow local dev origins — adjust if deploying
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:7860",
    "http://127.0.0.1:7860",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # load memory, call agent_process
    mem = load_memory()
    try:
        reply, rows, new_mem = agent_process(req.message, mem)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # persist memory
    try:
        save_memory(new_mem or mem)
    except Exception:
        pass

    return ChatResponse(reply=reply, rows=rows or [], memory_snapshot={"summary_count": len(new_mem.get("convo", [])) if new_mem else 0})

if __name__ == "__main__":
    uvicorn.run("src.api_server:app", host="127.0.0.1", port=8000, reload=True)
