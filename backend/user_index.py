"""
user_index.py — Per-user "uploaded documents" knowledge base.

Mirrors ingest.py / retriever.py, but scoped to one user's own PDFs instead
of the shared data/hr_docs/ collection. Each user gets their own Chroma
collection (USER_DOCS_COLLECTION_PREFIX + user_id), holding at most
MAX_USER_DOCS files at a time. Uploading a new batch replaces the old one.

NOTE: see the comment in config.py — without a persistent disk, this
collection (and the saved PDFs) won't survive a Render restart.
"""
import shutil
from functools import lru_cache
from pathlib import Path

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from chromadb.api.models.Collection import Collection as _ChromaCollection

# Re-apply the same empty-where compatibility shim retriever.py uses — needed
# here too since we query this collection directly the same way.
if not getattr(_ChromaCollection, "_hr_bot_where_patched", False):
    _orig_query = _ChromaCollection.query
    _orig_get = _ChromaCollection.get

    def _drop_empty_where(kwargs):
        for key in ("where", "where_document"):
            if kwargs.get(key) == {}:
                kwargs[key] = None
        return kwargs

    def _patched_query(self, *args, **kwargs):
        return _orig_query(self, *args, **_drop_empty_where(kwargs))

    def _patched_get(self, *args, **kwargs):
        return _orig_get(self, *args, **_drop_empty_where(kwargs))

    _ChromaCollection.query = _patched_query
    _ChromaCollection.get = _patched_get
    _ChromaCollection._hr_bot_where_patched = True


def _collection_name(user_id: int) -> str:
    return f"{config.USER_DOCS_COLLECTION_PREFIX}{user_id}"


def _upload_dir(user_id: int) -> Path:
    return config.USER_UPLOAD_DIR / str(user_id)


def replace_user_documents(user_id: int, saved_paths: list[Path]) -> int:
    """Wipe this user's existing collection and rebuild it from saved_paths.

    saved_paths: PDFs already written to disk at _upload_dir(user_id) by the
    /documents/upload endpoint. Returns the number of chunks indexed.
    """
    config.require_api_key()

    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    name = _collection_name(user_id)

    # Drop the old collection entirely so stale chunks from a previous batch
    # can never leak into retrieval (simpler + safer than diffing).
    try:
        chroma_client.delete_collection(name)
    except Exception:
        pass  # didn't exist yet — fine

    _get_retriever.cache_clear()  # old cached retriever (if any) is now stale

    if not saved_paths:
        return 0

    Settings.embed_model = OpenAIEmbedding(
        model=config.EMBED_MODEL, api_key=config.OPENAI_API_KEY
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )

    documents = SimpleDirectoryReader(input_files=[str(p) for p in saved_paths]).load_data()

    collection = chroma_client.get_or_create_collection(name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=False
    )
    return collection.count()


def has_documents(user_id: int) -> bool:
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        collection = chroma_client.get_collection(_collection_name(user_id))
        return collection.count() > 0
    except Exception:
        return False


def list_filenames(user_id: int) -> list[str]:
    """Distinct source filenames currently indexed for this user (for the
    router prompt and for GET /documents)."""
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        collection = chroma_client.get_collection(_collection_name(user_id))
    except Exception:
        return []
    rows = collection.get(include=["metadatas"])
    names = {m.get("file_name", "unknown") for m in rows.get("metadatas", []) if m}
    return sorted(names)


def content_preview(user_id: int, max_chars: int = 600) -> str:
    """A short snippet of the user's uploaded content, used to give the
    router something real to reason about (a filename alone, e.g.
    "Human-Resources-Policy.pdf", says nothing about WHICH organization or
    topic it covers — the router needs actual text to make a sane decision).
    Pulls from the first chunk(s) indexed, truncated to max_chars."""
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        collection = chroma_client.get_collection(_collection_name(user_id))
    except Exception:
        return ""
    rows = collection.get(include=["documents"], limit=3)
    docs = rows.get("documents", [])
    preview = " ".join(docs)[:max_chars]
    return preview


@lru_cache(maxsize=64)
def _get_retriever(user_id: int):
    """One cached retriever per user_id (no LLM involved — pure vector
    lookup). Cache is cleared on re-upload via replace_user_documents()."""
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = chroma_client.get_collection(_collection_name(user_id))

    Settings.embed_model = OpenAIEmbedding(
        model=config.EMBED_MODEL, api_key=config.OPENAI_API_KEY
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index.as_retriever(similarity_top_k=config.TOP_K)


def retrieve_nodes(user_id: int, question: str):
    """Return the raw retrieved nodes for this user's uploaded docs. Pure
    retrieval only — no LLM call, no response synthesis."""
    retriever_ = _get_retriever(user_id)
    return retriever_.retrieve(question)