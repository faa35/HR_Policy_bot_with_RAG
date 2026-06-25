"""
retriever.py — Query logic.

Loads the existing ChromaDB collection and exposes a single `answer()` function
that:
    1. Embeds the user's question
    2. Retrieves the TOP_K most relevant chunks
    3. Sends those chunks + the question to the LLM
    4. Returns a grounded answer plus its source snippets

The index is loaded lazily and cached, so the (slow) startup cost is paid once.
"""
from functools import lru_cache
from typing import Any

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

# --- Compatibility shim -----------------------------------------------------
# llama-index-vector-stores-chroma 0.2.1 sends an empty `where={}` to ChromaDB
# when a query has no metadata filters. ChromaDB 0.5.18 rejects empty where
# clauses ("Expected where to have exactly one operator, got {}"). It accepts
# `where=None`, so we strip empty dicts before they reach ChromaDB.
from chromadb.api.models.Collection import Collection as _ChromaCollection

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
# ----------------------------------------------------------------------------

# A strict prompt: answer ONLY from the retrieved policy text, and admit when
# the documents don't cover the question (no hallucinating company policy!).
QA_PROMPT = PromptTemplate(
    "You are an HR assistant. Answer the employee's question using ONLY the "
    "company policy context below. Be concise and friendly. If the answer is "
    "not contained in the context, say you don't have that information in the "
    "HR documents and suggest contacting the HR team.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)


@lru_cache(maxsize=1)
def _get_query_engine():
    """Build the query engine once and reuse it across requests."""
    config.require_api_key()

    if not config.CHROMA_DIR.exists():
        raise RuntimeError(
            "Vector store not found. Run `python ingest.py` first to index your "
            "HR documents."
        )

    Settings.embed_model = OpenAIEmbedding(
        model=config.EMBED_MODEL, api_key=config.OPENAI_API_KEY
    )
    Settings.llm = OpenAI(
        model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY, temperature=0.1
    )

    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    index = VectorStoreIndex.from_vector_store(vector_store)

    return index.as_query_engine(
        similarity_top_k=config.TOP_K,
        text_qa_template=QA_PROMPT,
    )


def answer(question: str) -> dict[str, Any]:
    """Return a dict with the answer text and the source chunks used."""
    engine = _get_query_engine()
    response = engine.query(question)

    sources = []
    for node in response.source_nodes:
        # Show the FULL chunk, not just the first N characters. The chunk is a
        # fixed piece of text decided at ingest time; truncating from the start
        # ([:800]) could hide the answer if it sits near the end of the chunk.
        # Chunks are bounded by CHUNK_SIZE, so they won't be huge.
        sources.append(
            {
                "file": node.metadata.get("file_name", "unknown"),
                "page": node.metadata.get("page_label"),
                "score": round(float(node.score), 3) if node.score else None,
                "text": node.node.get_content().strip(),
            }
        )

    return {"answer": str(response).strip(), "sources": sources}


if __name__ == "__main__":
    # Quick manual test:  python retriever.py
    import json

    result = answer("How many annual leave days do I get?")
    print(json.dumps(result, indent=2))
