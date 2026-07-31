"""Build and load the FAISS vector index.

Indexing pipeline: load PDFs -> section-aware chunk -> embed -> FAISS.
The index is persisted to disk so you embed once and query many times.
"""
from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from . import config
from .chunking import chunk_documents
from .ingest import load_papers


def _embeddings() -> OpenAIEmbeddings:
    config.require_api_key()
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


def build_index() -> FAISS:
    """Full pipeline: load every paper, chunk it, embed, persist FAISS index."""
    papers = load_papers()
    all_chunks: list[Document] = []
    for name, pages in papers.items():
        chunks = chunk_documents(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        print(f"  [chunk] {name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"Embedding {len(all_chunks)} chunks with {config.EMBEDDING_MODEL} ...")
    store = FAISS.from_documents(all_chunks, _embeddings())
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(config.INDEX_DIR))
    print(f"Saved index to {config.INDEX_DIR}")
    return store


def load_index() -> FAISS:
    if not config.INDEX_DIR.exists():
        raise FileNotFoundError(
            f"No index at {config.INDEX_DIR}. Run: python scripts/build_index.py"
        )
    # allow_dangerous_deserialization is required because FAISS persists a
    # pickled docstore. Safe here: we created the file ourselves.
    return FAISS.load_local(
        str(config.INDEX_DIR),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )
