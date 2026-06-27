"""
main.py — FastAPI backend (with auth + persistent chat history).

Public endpoints:
    GET  /                       info
    GET  /health                 {"status": "ok"}
    POST /register               {username, password}        -> {token, username}
    POST /login                  {username, password}         -> {token, username}

Authenticated endpoints (require  Authorization: Bearer <token>):
    GET    /conversations                  list the user's chat threads
    GET    /conversations/{id}/messages    messages in one thread
    DELETE /conversations/{id}             delete a thread
    POST   /chat   {question, conversation_id?}  -> answer + sources (and saves both)

Run:
    cd backend
    uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlmodel import Session, select

import auth
import database
import retriever
from database import Conversation, Message, User


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create the database tables on startup.
    database.init_db()
    yield


app = FastAPI(
    title="HR Policy Chatbot API",
    description="Ask questions about company HR policies (RAG), with user "
    "accounts and saved conversation history.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Credentials(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)


class AuthResponse(BaseModel):
    token: str
    username: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2)
    conversation_id: int | None = None


class Source(BaseModel):
    file: str
    page: str | None = None
    score: float | None = None
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    conversation_id: int
    title: str


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: str


class MessageOut(BaseModel):
    role: str
    content: str
    sources: list[Source] = []


# --------------------------------------------------------------------------- #
# Public routes
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    return {"service": "HR Policy Chatbot", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/register", response_model=AuthResponse)
def register(creds: Credentials, session: Session = Depends(database.get_session)):
    existing = session.exec(
        select(User).where(User.username == creds.username)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="That username is already taken.")

    user = User(
        username=creds.username, password_hash=auth.hash_password(creds.password)
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return AuthResponse(token=auth.create_access_token(user.id), username=user.username)


@app.post("/login", response_model=AuthResponse)
def login(creds: Credentials, session: Session = Depends(database.get_session)):
    user = session.exec(
        select(User).where(User.username == creds.username)
    ).first()
    if user is None or not auth.verify_password(creds.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Wrong username or password.")
    return AuthResponse(token=auth.create_access_token(user.id), username=user.username)


# --------------------------------------------------------------------------- #
# Authenticated routes
# --------------------------------------------------------------------------- #
@app.get("/conversations", response_model=list[ConversationOut])
def list_conversations(
    user: User = Depends(auth.get_current_user),
    session: Session = Depends(database.get_session),
):
    rows = session.exec(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.created_at.desc())
    ).all()
    return [
        ConversationOut(id=c.id, title=c.title, created_at=c.created_at.isoformat())
        for c in rows
    ]


def _owned_conversation(conv_id: int, user: User, session: Session) -> Conversation:
    conv = session.get(Conversation, conv_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv


@app.get("/conversations/{conv_id}/messages", response_model=list[MessageOut])
def get_messages(
    conv_id: int,
    user: User = Depends(auth.get_current_user),
    session: Session = Depends(database.get_session),
):
    _owned_conversation(conv_id, user, session)
    msgs = session.exec(
        select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.created_at)
    ).all()
    out = []
    for m in msgs:
        sources = json.loads(m.sources_json) if m.sources_json else []
        out.append(MessageOut(role=m.role, content=m.content, sources=sources))
    return out


@app.delete("/conversations/{conv_id}")
def delete_conversation(
    conv_id: int,
    user: User = Depends(auth.get_current_user),
    session: Session = Depends(database.get_session),
):
    conv = _owned_conversation(conv_id, user, session)
    for m in session.exec(
        select(Message).where(Message.conversation_id == conv_id)
    ).all():
        session.delete(m)
    session.delete(conv)
    session.commit()
    return {"status": "deleted"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    user: User = Depends(auth.get_current_user),
    session: Session = Depends(database.get_session),
):
    # Find or create the conversation.
    if req.conversation_id is not None:
        conv = _owned_conversation(req.conversation_id, user, session)
    else:
        title = req.question.strip()
        title = (title[:47] + "...") if len(title) > 50 else title
        conv = Conversation(user_id=user.id, title=title)
        session.add(conv)
        session.commit()
        session.refresh(conv)

    # Run RAG.
    try:
        result = retriever.answer(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # Persist both messages.
    session.add(
        Message(conversation_id=conv.id, role="user", content=req.question)
    )
    session.add(
        Message(
            conversation_id=conv.id,
            role="assistant",
            content=result["answer"],
            sources_json=json.dumps(result["sources"]),
        )
    )
    session.commit()

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        conversation_id=conv.id,
        title=conv.title,
    )
