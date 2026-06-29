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
    POST   /documents/upload   multipart, up to 3 PDFs  -> replaces the user's upload batch
    GET    /documents                      list the user's currently active uploaded files
    DELETE /documents                      clear the user's uploaded batch

Run:
    cd backend
    uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import json
import shutil
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlmodel import Session, select

import auth
import config
import database
import retriever
import user_index
from database import Conversation, Message, User, UserDocument


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


class DocumentOut(BaseModel):
    id: int
    filename: str
    uploaded_at: str


class UploadResponse(BaseModel):
    documents: list[DocumentOut]
    chunks_indexed: int


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

    # Run RAG (routes between the user's uploaded docs and/or the policy library).
    try:
        result = retriever.answer(req.question, user_id=user.id)
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


# --------------------------------------------------------------------------- #
# User-uploaded documents
# --------------------------------------------------------------------------- #
@app.get("/documents", response_model=list[DocumentOut])
def list_documents(
    user: User = Depends(auth.get_current_user),
    session: Session = Depends(database.get_session),
):
    rows = session.exec(
        select(UserDocument)
        .where(UserDocument.user_id == user.id)
        .order_by(UserDocument.created_at)
    ).all()
    return [
        DocumentOut(id=d.id, filename=d.filename, uploaded_at=d.created_at.isoformat())
        for d in rows
    ]


@app.delete("/documents")
def clear_documents(
    user: User = Depends(auth.get_current_user),
    session: Session = Depends(database.get_session),
):
    rows = session.exec(select(UserDocument).where(UserDocument.user_id == user.id)).all()
    for row in rows:
        session.delete(row)
    session.commit()
    user_index.replace_user_documents(user.id, saved_paths=[])

    upload_dir = config.USER_UPLOAD_DIR / str(user.id)
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)

    return {"status": "cleared"}


@app.post("/documents/upload", response_model=UploadResponse)
def upload_documents(
    files: list[UploadFile] = File(...),
    user: User = Depends(auth.get_current_user),
    session: Session = Depends(database.get_session),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > config.MAX_USER_DOCS:
        raise HTTPException(
            status_code=400,
            detail=f"You can upload at most {config.MAX_USER_DOCS} PDFs at a time.",
        )
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail=f"'{f.filename}' is not a PDF."
            )

    # Wipe this user's previous batch (DB rows, saved files, vectors) first —
    # uploads replace, they don't accumulate.
    upload_dir = config.USER_UPLOAD_DIR / str(user.id)
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    old_rows = session.exec(
        select(UserDocument).where(UserDocument.user_id == user.id)
    ).all()
    for row in old_rows:
        session.delete(row)
    session.commit()

    saved_paths = []
    for f in files:
        dest = upload_dir / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(dest)
        session.add(UserDocument(user_id=user.id, filename=f.filename))
    session.commit()

    try:
        chunks_indexed = user_index.replace_user_documents(user.id, saved_paths)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to index documents: {e}")

    rows = session.exec(
        select(UserDocument)
        .where(UserDocument.user_id == user.id)
        .order_by(UserDocument.created_at)
    ).all()
    return UploadResponse(
        documents=[
            DocumentOut(id=d.id, filename=d.filename, uploaded_at=d.created_at.isoformat())
            for d in rows
        ],
        chunks_indexed=chunks_indexed,
    )