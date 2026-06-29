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
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
import router
import user_index

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
    "You are an HR policy assistant. Your knowledge base contains HR documents "
    f"from multiple organizations: {config.ORGANIZATIONS_STR}. Their policies "
    "differ, so it matters which organization a policy comes from. The context "
    "below may also include documents the employee personally uploaded — those "
    "take priority for anything they specifically describe, since they're more "
    "recent/specific than the general policy library.\n\n"
    "Answer the employee's question using ONLY the context below. Be concise "
    "and friendly, and when relevant make clear WHICH organization or document "
    "a policy belongs to (e.g. \"At Valve, ...\" or \"According to the document "
    "you uploaded, ...\"). If the question doesn't say which source and the "
    "documents give different answers, briefly note the difference. If the "
    "answer is not contained in the context, say you don't have that "
    "information and suggest contacting the HR team. Never invent or guess "
    "policy.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)


@lru_cache(maxsize=1)
def _get_policy_index() -> VectorStoreIndex:
    """The shared HR-policy index, built once and cached (separate from the
    full query engine below, since routing sometimes needs raw nodes from
    this index merged with nodes from a user's uploaded-doc index)."""
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
    return VectorStoreIndex.from_vector_store(vector_store)


@lru_cache(maxsize=1)
def _get_query_engine():
    """Build the query engine once and reuse it across requests."""
    return _get_policy_index().as_query_engine(
        similarity_top_k=config.TOP_K,
        text_qa_template=QA_PROMPT,
    )


def _nodes_to_sources(nodes: list[NodeWithScore]) -> list[dict[str, Any]]:
    sources = []
    for node in nodes:
        sources.append(
            {
                "file": node.metadata.get("file_name", "unknown"),
                "page": node.metadata.get("page_label"),
                "score": round(float(node.score), 3) if node.score else None,
                "text": node.node.get_content().strip(),
            }
        )
    return sources


def answer(question: str, user_id: int | None = None) -> dict[str, Any]:
    """Return a dict with the answer text and the source chunks used.

    If user_id is given and that user has an active uploaded-document batch,
    a quick routing step (router.route) decides whether to pull context from
    the user's uploaded docs, the shared policy library, or both — before any
    generation happens. With no user_id (or no uploaded docs), this behaves
    exactly as before: policy library only.
    """
    has_uploads = user_id is not None and user_index.has_documents(user_id)
    uploaded_filenames: list[str] = []
    preview = ""
    if has_uploads:
        # Cheap metadata/content pull just for the router prompt — not a
        # real similarity query. The preview matters: a filename alone
        # doesn't tell the router WHAT organization/topic the upload
        # actually covers.
        uploaded_filenames = user_index.list_filenames(user_id)
        preview = user_index.content_preview(user_id)

    decision = router.route(
        question,
        has_uploaded_docs=has_uploads,
        uploaded_filenames=uploaded_filenames,
        content_preview=preview,
        known_organizations=config.ORGANIZATIONS_STR,
    )

    # Fast path: no uploads in play at all, behave like the original single
    # query engine (keeps this the cheapest, simplest case).
    if decision == "OLD" and not has_uploads:
        engine = _get_query_engine()
        response = engine.query(question)
        return {
            "answer": str(response).strip(),
            "sources": _nodes_to_sources(response.source_nodes),
        }

    # Otherwise gather raw nodes from whichever source(s) the router picked,
    # then synthesize one answer from the combined context ourselves.
    nodes: list[NodeWithScore] = []
    if decision in ("OLD", "BOTH"):
        policy_retriever = _get_policy_index().as_retriever(similarity_top_k=config.TOP_K)
        nodes.extend(policy_retriever.retrieve(question))
    if decision in ("NEW", "BOTH") and has_uploads:
        nodes.extend(user_index.retrieve_nodes(user_id, question))

    if not nodes:
        return {
            "answer": "I don't have that information in the HR documents. "
            "Please contact the HR team for help with this question.",
            "sources": [],
        }

    context_str = "\n\n".join(n.node.get_content().strip() for n in nodes)
    llm = OpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY, temperature=0.1)
    prompt = QA_PROMPT.format(context_str=context_str, query_str=question)
    response_text = str(llm.complete(prompt)).strip()

    return {"answer": response_text, "sources": _nodes_to_sources(nodes)}


if __name__ == "__main__":
    # Quick manual test:  python retriever.py
    import json

    result = answer("How many annual leave days do I get?")
    print(json.dumps(result, indent=2))