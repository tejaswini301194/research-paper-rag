"""Central configuration. Everything tunable lives here so the rest of the
codebase has no magic numbers buried in it."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # pull OPENAI_API_KEY (and anything else) from a local .env

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"          # raw PDFs live here
INDEX_DIR = DATA_DIR / "faiss_index"      # persisted FAISS index
EVAL_DIR = DATA_DIR / "eval"
EVAL_FILE = EVAL_DIR / "qa_pairs.json"    # gold (query -> source chunk) set

# --- Models ------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --- Chunking ----------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))      # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# --- Retrieval ---------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", 5))


def require_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key
