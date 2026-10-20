"""
config.py
---------
Central configuration for the RAG System.

All file paths, model names, ChromaDB settings, and environment
variable loading live here so every other module imports a single
source of truth rather than scattering settings across files.

Usage:
    from config import settings
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ── Project root (the folder that contains this file) ────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ── Load the project .env file (safe when absent) ────────────────────────────
ENV_FILE: Path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# ── Directory layout ─────────────────────────────────────────────────────────
DOCUMENTS_DIR: Path = PROJECT_ROOT / "data" / "documents"
CHROMA_DIR: Path    = PROJECT_ROOT / "chroma_data"

# Create directories at import time so other modules never have to check.
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── ChromaDB settings ─────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = "rag_documents"

# ── Embedding model (CPU-only, ~90 MB, no GPU required) ───────────────────────
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE: int = 8       # small batch to avoid memory spikes
EMBEDDING_DIMENSION: int  = 384     # fixed for all-MiniLM-L6-v2

# ── Text chunking parameters ──────────────────────────────────────────────────
CHUNK_SIZE: int    = 500    # target characters per chunk
CHUNK_OVERLAP: int = 50     # overlap between consecutive chunks

# ── Google Gemini settings (read from environment — never hardcoded) ──────────
GEMINI_API_KEY: str   = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# ── Retrieval settings ────────────────────────────────────────────────────────
TOP_K_RESULTS: int = 5   # number of chunks to retrieve per query

# ── Logging configuration ─────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ── Validation helper ─────────────────────────────────────────────────────────
def validate_api_key() -> None:
    """
    Raise a clear RuntimeError if GEMINI_API_KEY is not set.
    Call this before any generation step, not at import time,
    so the app can still start without a key (e.g. for ingestion-only use).
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable. "
            "See .env.example for the required format."
        )


# ── Quick settings summary for debugging ─────────────────────────────────────
def print_config() -> None:
    """Print current configuration to stdout (masks the API key)."""
    key_status = "SET" if GEMINI_API_KEY else "NOT SET"
    print(
        f"\n{'='*50}\n"
        f"  RAG System Configuration\n"
        f"{'='*50}\n"
        f"  Project root   : {PROJECT_ROOT}\n"
        f"  Documents dir  : {DOCUMENTS_DIR}\n"
        f"  ChromaDB dir   : {CHROMA_DIR}\n"
        f"  Collection     : {CHROMA_COLLECTION_NAME}\n"
        f"  Embedding model: {EMBEDDING_MODEL_NAME}\n"
        f"  Embedding dim  : {EMBEDDING_DIMENSION}\n"
        f"  Batch size     : {EMBEDDING_BATCH_SIZE}\n"
        f"  Chunk size     : {CHUNK_SIZE} chars\n"
        f"  Chunk overlap  : {CHUNK_OVERLAP} chars\n"
        f"  Top-K results  : {TOP_K_RESULTS}\n"
        f"  Gemini model   : {GEMINI_MODEL_NAME}\n"
        f"  Gemini API key : {key_status}\n"
        f"{'='*50}\n"
    )


if __name__ == "__main__":
    print_config()
