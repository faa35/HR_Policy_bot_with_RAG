"""
database.py — SQLite database via SQLModel (SQLAlchemy + Pydantic).

Three tables:
    User          one row per registered account
    Conversation  one row per chat thread (belongs to a User)
    Message       one row per message (belongs to a Conversation)

The whole DB is a single file, app.db, at the project root. To switch to
PostgreSQL later, just set DATABASE_URL in .env — no code changes needed.
"""
from datetime import datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine

import config

# SQLite needs check_same_thread=False so FastAPI's threads can share the engine.
_connect_args = (
    {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}
)
# pool_pre_ping avoids "stale connection" errors with hosted Postgres (Supabase),
# whose connections can be dropped after idle periods.
engine = create_engine(
    config.DATABASE_URL,
    echo=False,
    connect_args=_connect_args,
    pool_pre_ping=True,
)


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    title: str = "New chat"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id", index=True)
    role: str                      # "user" or "assistant"
    content: str
    sources_json: str = ""         # JSON-encoded list of source dicts (assistant only)
    created_at: datetime = Field(default_factory=datetime.utcnow)


def init_db() -> None:
    """Create all tables if they don't exist yet. Safe to call on every startup."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """FastAPI dependency: yields a DB session, closed automatically after use."""
    with Session(engine) as session:
        yield session
