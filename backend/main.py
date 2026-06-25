"""
main.py — FastAPI backend.

Endpoints:
    GET  /         -> simple health/info check
    GET  /health   -> {"status": "ok"}
    POST /chat     -> {"question": "..."} returns {"answer": "...", "sources": [...]}

Run:
    cd backend
    uvicorn main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import retriever

app = FastAPI(
    title="HR Policy Chatbot API",
    description="Ask questions about company HR policies. Answers are grounded "
    "in the company's HR documents via RAG (LlamaIndex + ChromaDB + OpenAI).",
    version="1.0.0",
)

# Allow the Streamlit frontend (and anything during dev) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, examples=["How many leave days do I get?"])


class Source(BaseModel):
    file: str
    page: str | None = None
    score: float | None = None
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def root():
    return {"service": "HR Policy Chatbot", "docs": "/docs", "chat": "POST /chat"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = retriever.answer(req.question)
        return result
    except RuntimeError as e:
        # Configuration / "you forgot to ingest" type problems.
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
