"""
Central configuration for the HR Policy Chatbot.

Everything that ingest.py, retriever.py and main.py need to agree on
(paths, model names, chunking) lives here so there's a single source of truth.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from the .env file at the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Paths ---
DATA_DIR = PROJECT_ROOT / "data" / "hr_docs"      # where the PDFs live
CHROMA_DIR = PROJECT_ROOT / "chroma_store"         # persistent vector store
COLLECTION_NAME = "hr_policies"

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Models. gpt-4o-mini is cheap + good enough for doc Q&A; bump to gpt-4o if you like.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# --- RAG tuning ---
CHUNK_SIZE = 512          # tokens per chunk
CHUNK_OVERLAP = 50        # token overlap between chunks
TOP_K = 3                 # how many chunks to retrieve per question


def require_api_key() -> None:
    """Fail fast with a clear message if the key is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env file at the project root "
            "with a line like:  OPENAI_API_KEY=sk-...\n"
            "See .env.example for the template."
        )
