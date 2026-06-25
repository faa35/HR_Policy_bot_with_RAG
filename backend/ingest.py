"""
ingest.py — Run ONCE (or whenever you add/change documents).

Pipeline:
    1. Read every file in data/hr_docs/   (PDF, TXT, MD, DOCX ... all supported)
    2. Split each document into overlapping chunks
    3. Embed each chunk with OpenAI
    4. Store the vectors in a persistent ChromaDB collection (chroma_store/)

Usage:
    cd backend
    python ingest.py
"""
import shutil

import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


def build_index(reset: bool = True) -> None:
    config.require_api_key()

    if not config.DATA_DIR.exists() or not any(config.DATA_DIR.iterdir()):
        raise RuntimeError(
            f"No documents found in {config.DATA_DIR}. "
            "Drop your HR PDFs there first (sample .md policies are included)."
        )

    # Start clean so re-running doesn't create duplicate vectors.
    if reset and config.CHROMA_DIR.exists():
        print(f"Clearing old vector store at {config.CHROMA_DIR} ...")
        shutil.rmtree(config.CHROMA_DIR)

    # Configure the embedding model + chunking globally for LlamaIndex.
    Settings.embed_model = OpenAIEmbedding(
        model=config.EMBED_MODEL, api_key=config.OPENAI_API_KEY
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )

    print(f"Reading documents from {config.DATA_DIR} ...")
    documents = SimpleDirectoryReader(
        input_dir=str(config.DATA_DIR), recursive=True
    ).load_data()
    print(f"  -> loaded {len(documents)} document(s)")

    # Persistent Chroma client + collection.
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Embedding + indexing (this calls the OpenAI API)...")
    VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )

    print(
        f"\nDone. Indexed {collection.count()} chunks into "
        f"collection '{config.COLLECTION_NAME}' at {config.CHROMA_DIR}."
    )


if __name__ == "__main__":
    build_index()
