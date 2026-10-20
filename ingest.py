"""
ingest.py
---------
Embedding and incremental indexing pipeline for the RAG System.

Pipeline (per document)
-----------------------
  1. Compute SHA-256 file hash to detect changes.
  2. Compare hash against what is already stored in ChromaDB metadata.
     - Same hash  → skip (no re-embedding, no DB write).
     - New hash   → embed all chunks, upsert into ChromaDB, remove stale IDs.
  3. Embed chunks in batches of BATCH_SIZE using the singleton CPU model.
  4. Upsert (text + vector + metadata) into the named ChromaDB collection.
     Embeddings are always supplied explicitly so ChromaDB never generates
     a second set.
  5. Purge any chunk IDs from a previous ingestion of the same source that
     are no longer present (document shrank / chunking changed).

Safe-ingestion guarantees
-------------------------
  - Never deletes the whole database.
  - Re-running ingest on unchanged files is a no-op.
  - Re-running on changed files updates records in-place.
  - Batched embedding prevents loading the full corpus into RAM at once.
  - Memory is checked before each document; stops with a clear message
    if free RAM drops below MEMORY_FLOOR_GB.

ChromaDB embedding contract
----------------------------
  Collection is created with embedding_function=None.
  All add / upsert calls supply explicit `embeddings=` vectors.
  All queries use `query_embeddings=` (see retrieval.py).
  This prevents ChromaDB from invoking any secondary embedding logic.

Usage
-----
  python ingest.py                  # ingest all PDFs/MDs in data/documents/
  python ingest.py report.pdf       # single file
  python ingest.py --sample         # first doc, first 5 chunks only (safe test)
  python ingest.py --verify         # reopen DB, check record count, show sample
  python ingest.py --reset-source report.pdf  # remove all chunks for one file
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import chromadb
import numpy as np

import config
from chunking import (
    CHUNK_TOKENS,
    OVERLAP_TOKENS,
    TextChunk,
    chunk_document_pages,
)
from loader import load_directory, load_document

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

BATCH_SIZE: int = config.EMBEDDING_BATCH_SIZE   # 8 chunks per embed call
# Memory thresholds (GB of free RAM)
# This machine runs at ~96% usage normally; 0.12 GB is the hard abort floor.
# At 0.12 GB free the OS itself becomes unstable.
MEMORY_FLOOR_GB: float = 0.12   # hard stop: OS/process unstable below this
MEMORY_WARN_GB:  float = 0.30   # soft warning: log but continue
SAMPLE_MAX_CHUNKS: int = 5      # when --sample is set, stop after this many


# ── Memory monitor ────────────────────────────────────────────────────────────

def get_memory_status() -> Tuple[Optional[float], Optional[float]]:
    """
    Return (used_pct, free_gb) via PowerShell WMI query (Windows only).
    Returns (None, None) if the query fails.
    """
    try:
        out = subprocess.check_output(
            [
                "powershell", "-Command",
                "(Get-CimInstance Win32_OperatingSystem) | "
                "Select-Object TotalVisibleMemorySize,FreePhysicalMemory | "
                "ConvertTo-Json",
            ],
            stderr=subprocess.DEVNULL,
            timeout=8,
        ).decode()
        d = json.loads(out)
        total = d["TotalVisibleMemorySize"]
        free  = d["FreePhysicalMemory"]
        return (
            round((total - free) / total * 100, 1),
            round(free / 1_048_576, 2),
        )
    except Exception:
        return None, None


def check_memory_safe(label: str = "", hard_stop: bool = True) -> bool:
    """
    Check current free RAM and return True if safe to continue.

    Parameters
    ----------
    label     : Short tag printed in log messages.
    hard_stop : If True, return False (abort) when free < MEMORY_FLOOR_GB.
                If False, only log a warning and return True (continue).

    Returns False only when hard_stop=True AND free RAM < MEMORY_FLOOR_GB.
    """
    used_pct, free_gb = get_memory_status()
    if used_pct is None:
        return True   # can't measure -> continue optimistically
    tag = f" [{label}]" if label else ""
    if free_gb < MEMORY_WARN_GB:
        logger.warning(
            "LOW MEMORY%s: %.1f%% used, %.2f GB free (warn threshold %.2f GB)",
            tag, used_pct, free_gb, MEMORY_WARN_GB,
        )
    else:
        logger.info("Memory%s: %.1f%% used, %.2f GB free", tag, used_pct, free_gb)
    if hard_stop and free_gb < MEMORY_FLOOR_GB:
        logger.error(
            "STOPPING%s: only %.2f GB free (hard floor %.2f GB). "
            "Close other applications and retry.",
            tag, free_gb, MEMORY_FLOOR_GB,
        )
        return False
    return True


# ── Singleton embedding model ─────────────────────────────────────────────────

_model = None


def get_embedding_model():
    """
    Load the SentenceTransformer model exactly once per process.
    Device is always CPU to avoid GPU/CUDA requirements.

    Raises RuntimeError if loading fails.
    """
    global _model
    if _model is not None:
        return _model

    if not check_memory_safe("before model load"):
        raise RuntimeError(
            "Insufficient memory to load the embedding model. "
            "Free up RAM and retry."
        )

    logger.info(
        "Loading embedding model: %s (CPU only, first load may take a moment) ...",
        config.EMBEDDING_MODEL_NAME,
    )
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device="cpu")
        dim = _model.get_sentence_embedding_dimension()
        logger.info(
            "Model loaded. Embedding dimension: %d  max_seq_length: %d",
            dim, _model.max_seq_length,
        )
        if dim != config.EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Model produces {dim}-dim vectors, but config.EMBEDDING_DIMENSION "
                f"is set to {config.EMBEDDING_DIMENSION}. Update config.py."
            )
    except ImportError as exc:
        raise RuntimeError(
            f"sentence-transformers is not installed: {exc}. "
            "Run: pip install sentence-transformers"
        ) from exc
    return _model


# ── ChromaDB collection ───────────────────────────────────────────────────────

def get_chroma_collection() -> chromadb.Collection:
    """
    Open (or create) the persistent ChromaDB collection.

    The collection is created with embedding_function=None because we supply
    explicit embeddings in every upsert and query_embeddings in every search.
    This prevents ChromaDB from generating a second, incompatible set of vectors.

    If an existing collection was previously created with a different embedding
    function, we fall back to opening it without re-specifying the function.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    try:
        collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=None,        # we manage our own embeddings
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as create_exc:
        logger.warning(
            "get_or_create_collection with embedding_function=None failed "
            "(%s) — falling back to opening the existing collection.",
            create_exc,
        )
        try:
            collection = client.get_collection(
                name=config.CHROMA_COLLECTION_NAME,
                embedding_function=None,
            )
        except Exception as get_exc:
            raise RuntimeError(
                f"Cannot open ChromaDB collection "
                f"'{config.CHROMA_COLLECTION_NAME}': {get_exc}"
            ) from get_exc

    logger.info(
        "ChromaDB collection '%s' opened. Current record count: %d",
        collection.name, collection.count(),
    )
    return collection


# ── File fingerprinting ───────────────────────────────────────────────────────

def file_sha256(path: Path) -> str:
    """
    Compute the SHA-256 hash of a file in 64 KB chunks to avoid
    loading large files fully into memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ── Incremental ingestion helpers ─────────────────────────────────────────────

def get_source_info(
    collection: chromadb.Collection,
    source: str,
) -> Tuple[Optional[str], Set[str]]:
    """
    Query ChromaDB for all existing chunks belonging to *source*.

    Returns
    -------
    (stored_hash, existing_ids)
      stored_hash  : The file_hash stored in the first chunk's metadata,
                     or None if the source has no stored chunks.
      existing_ids : Set of all chunk IDs currently stored for this source.
    """
    try:
        result = collection.get(
            where={"source": {"$eq": source}},
            include=["metadatas"],
        )
    except Exception as exc:
        logger.warning("Could not query existing chunks for '%s': %s", source, exc)
        return None, set()

    ids = result.get("ids", [])
    metadatas = result.get("metadatas", [])

    stored_hash: Optional[str] = None
    if metadatas:
        stored_hash = metadatas[0].get("file_hash")

    return stored_hash, set(ids)


def purge_stale_ids(
    collection: chromadb.Collection,
    old_ids: Set[str],
    new_ids: Set[str],
    source: str,
) -> int:
    """
    Delete IDs that were in the old set but are absent from the new set.
    This handles documents that shrank (fewer chunks after re-chunking).

    Returns the number of IDs deleted.
    """
    stale = old_ids - new_ids
    if not stale:
        return 0
    stale_list = list(stale)
    try:
        collection.delete(ids=stale_list)
        logger.info(
            "Purged %d stale chunk(s) for '%s': %s",
            len(stale_list), source, stale_list[:5],
        )
    except Exception as exc:
        logger.error("Failed to purge stale IDs for '%s': %s", source, exc)
    return len(stale_list)


# ── Batch embedding ───────────────────────────────────────────────────────────

def embed_chunks_batched(
    model,
    chunks: List[TextChunk],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Embed *chunks* in batches of *batch_size* using the supplied model.

    Produces a single (N, D) float32 numpy array where N = len(chunks)
    and D = config.EMBEDDING_DIMENSION.

    Each batch is encoded with show_progress_bar=False to avoid console spam.
    The model is reused across calls — it is never reloaded here.
    """
    if not chunks:
        return np.empty((0, config.EMBEDDING_DIMENSION), dtype=np.float32)

    texts = [c.text for c in chunks]
    total = len(texts)
    all_vecs: List[np.ndarray] = []

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            device="cpu",
        )
        all_vecs.append(vecs)
        logger.debug(
            "Embedded batch %d-%d / %d", start, min(start + batch_size, total), total
        )

    return np.vstack(all_vecs).astype(np.float32)


# ── ChromaDB upsert ───────────────────────────────────────────────────────────

def upsert_to_chroma(
    collection: chromadb.Collection,
    chunks: List[TextChunk],
    embeddings: np.ndarray,
    file_hash: str,
) -> int:
    """
    Upsert chunks + their embeddings into the ChromaDB collection.

    Embeddings are supplied explicitly so ChromaDB does NOT generate its own.
    The file_hash is injected into every chunk's metadata for future
    change-detection without querying the filesystem.

    Batches upserts in groups of 100 to avoid oversized network/IPC calls.

    Returns the number of records upserted.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
        )

    ids        = [c.unique_id for c in chunks]
    documents  = [c.text      for c in chunks]
    metadatas  = []
    for c in chunks:
        m = dict(c.metadata)
        m["file_hash"] = file_hash       # injected for incremental detection
        metadatas.append(m)

    vectors = embeddings.tolist()

    UPSERT_BATCH = 100
    total_upserted = 0

    for i in range(0, len(ids), UPSERT_BATCH):
        sl = slice(i, i + UPSERT_BATCH)
        collection.upsert(
            ids=ids[sl],
            documents=documents[sl],
            embeddings=vectors[sl],
            metadatas=metadatas[sl],
        )
        total_upserted += len(ids[sl])
        logger.debug("Upserted %d records (batch %d)", len(ids[sl]), i // UPSERT_BATCH)

    return total_upserted


# ── High-level document ingest function ───────────────────────────────────────

def ingest_document(
    path: Path,
    collection: chromadb.Collection,
    model,
    sample_max_chunks: Optional[int] = None,
) -> dict:
    """
    Full pipeline for one document: load → chunk → embed → upsert → verify.

    Parameters
    ----------
    path              : Absolute path to a PDF, .md, or .txt file.
    collection        : ChromaDB collection (already opened).
    model             : Loaded SentenceTransformer model (reused, not reloaded).
    sample_max_chunks : If set, cap the number of chunks processed.
                        Used with --sample to keep the test tiny.

    Returns
    -------
    dict with keys: source, status, pages, chunks_produced, chunks_stored,
                    chunks_purged, skipped, error.
    """
    source = path.name
    base_result = {
        "source":          source,
        "status":          "unknown",
        "pages":           0,
        "chunks_produced": 0,
        "chunks_stored":   0,
        "chunks_purged":   0,
        "skipped":         False,
        "error":           None,
    }

    # -- Memory gate (warning-only for per-document check) -------------------
    if not check_memory_safe(source, hard_stop=True):
        base_result["status"] = "aborted-memory"
        base_result["error"]  = (
            f"Free RAM below hard floor ({MEMORY_FLOOR_GB} GB). "
            "The OS itself may be unstable. Close applications and retry."
        )
        return base_result

    # ── File hash ─────────────────────────────────────────────────────────────
    try:
        fhash = file_sha256(path)
    except Exception as exc:
        base_result["status"] = "error"
        base_result["error"]  = f"Cannot hash file: {exc}"
        return base_result

    # ── Incremental check ─────────────────────────────────────────────────────
    stored_hash, existing_ids = get_source_info(collection, source)

    if stored_hash == fhash and existing_ids:
        logger.info(
            "'%s': file unchanged (hash match). Skipping embedding. "
            "Existing chunks: %d",
            source, len(existing_ids),
        )
        base_result["status"]          = "skipped-unchanged"
        base_result["skipped"]         = True
        base_result["chunks_stored"]   = len(existing_ids)
        return base_result

    if stored_hash and stored_hash != fhash:
        logger.info(
            "'%s': file changed (hash mismatch). Will re-embed and update.",
            source,
        )

    # ── Load document ─────────────────────────────────────────────────────────
    try:
        pages, errors = load_document(path)
    except Exception as exc:
        base_result["status"] = "error"
        base_result["error"]  = f"Loader exception: {exc}"
        return base_result

    if errors:
        for e in errors:
            logger.warning("  Load warning for '%s': %s", source, e)

    if not pages:
        base_result["status"] = "empty"
        base_result["error"]  = f"No text extracted from '{source}'."
        return base_result

    base_result["pages"] = len(pages)

    # ── Chunk ─────────────────────────────────────────────────────────────────
    chunks = chunk_document_pages(pages)

    if sample_max_chunks is not None:
        chunks = chunks[:sample_max_chunks]
        logger.info(
            "'%s': --sample mode — limiting to %d chunks.", source, len(chunks)
        )

    if not chunks:
        base_result["status"] = "empty"
        base_result["error"]  = "Chunking produced zero chunks."
        return base_result

    base_result["chunks_produced"] = len(chunks)

    # ── Embed ─────────────────────────────────────────────────────────────────
    logger.info(
        "'%s': embedding %d chunk(s) in batches of %d ...",
        source, len(chunks), BATCH_SIZE,
    )
    try:
        embeddings = embed_chunks_batched(model, chunks, batch_size=BATCH_SIZE)
    except Exception as exc:
        base_result["status"] = "error"
        base_result["error"]  = f"Embedding failed: {exc}"
        return base_result

    logger.info(
        "'%s': embeddings shape %s, dtype %s",
        source, embeddings.shape, embeddings.dtype,
    )

    # ── Upsert ────────────────────────────────────────────────────────────────
    try:
        stored = upsert_to_chroma(collection, chunks, embeddings, file_hash=fhash)
    except Exception as exc:
        base_result["status"] = "error"
        base_result["error"]  = f"ChromaDB upsert failed: {exc}"
        return base_result

    # ── Purge stale IDs ───────────────────────────────────────────────────────
    new_ids = {c.unique_id for c in chunks}
    purged  = purge_stale_ids(collection, existing_ids, new_ids, source)

    base_result["status"]        = "ok"
    base_result["chunks_stored"] = stored
    base_result["chunks_purged"] = purged
    return base_result


# ── Directory ingestion ───────────────────────────────────────────────────────

def ingest_directory(
    directory: Path = config.DOCUMENTS_DIR,
    max_docs: Optional[int] = None,
    sample_max_chunks: Optional[int] = None,
) -> List[dict]:
    """
    Ingest all supported documents in *directory*, one at a time.

    Parameters
    ----------
    directory         : Folder to scan.
    max_docs          : Stop after this many documents (None = unlimited).
    sample_max_chunks : Cap chunks per document for sampling mode.

    Returns
    -------
    List of result dicts from ingest_document().
    """
    from loader import SUPPORTED_EXTENSIONS

    files = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        and f.name != "README.md"   # skip the placeholder
    )

    if not files:
        logger.warning(
            "No supported documents found in %s. "
            "Add .pdf, .md, or .txt files to %s first.",
            directory, directory,
        )
        return []

    if max_docs is not None:
        files = files[:max_docs]
        logger.info("Limiting to first %d document(s).", max_docs)

    logger.info("Found %d document(s) to process.", len(files))

    collection = get_chroma_collection()
    model      = get_embedding_model()
    results: List[dict] = []

    for i, fpath in enumerate(files, 1):
        logger.info("[%d/%d] Processing: %s", i, len(files), fpath.name)
        result = ingest_document(
            fpath, collection, model,
            sample_max_chunks=sample_max_chunks,
        )
        results.append(result)
        _print_result(result, collection)

    # Final summary ---------------------------------------------------------
    total_stored  = sum(r.get("chunks_stored", 0)  for r in results if not r["skipped"])
    total_skipped = sum(1 for r in results if r["skipped"])
    total_errors  = sum(1 for r in results if r.get("error"))

    print("\n" + "="*60)
    print("  Ingestion Summary")
    print("="*60)
    print(f"  Documents processed : {len(results)}")
    print(f"  Skipped (unchanged) : {total_skipped}")
    print(f"  Errors              : {total_errors}")
    print(f"  New chunks stored   : {total_stored}")
    print(f"  Total in DB now     : {collection.count()}")
    print(f"  DB path             : {config.CHROMA_DIR}")
    print(f"  Embedding model     : {config.EMBEDDING_MODEL_NAME}")
    print(f"  Embedding dim       : {config.EMBEDDING_DIMENSION}")
    print("="*60)

    return results


def _print_result(result: dict, collection: chromadb.Collection) -> None:
    """Print a one-line progress line for a single document result."""
    status_icons = {
        "ok":              "OK ",
        "skipped-unchanged": "---",
        "empty":           "EMP",
        "error":           "ERR",
        "aborted-memory":  "MEM",
    }
    icon = status_icons.get(result["status"], "???")
    if result["error"]:
        detail = result["error"][:80]
    elif result["skipped"]:
        detail = f"unchanged — {result['chunks_stored']} chunks already stored"
    else:
        detail = (
            f"{result['chunks_produced']} chunks -> "
            f"{result['chunks_stored']} stored, "
            f"{result['chunks_purged']} purged | "
            f"DB total: {collection.count()}"
        )
    print(f"  [{icon}] {result['source']}: {detail}")


# ── Verification ──────────────────────────────────────────────────────────────

def verify_database() -> None:
    """
    Reopen the persistent ChromaDB database from disk and verify:
      - The named collection exists.
      - The stored record count matches what ingestion reported.
      - A few records are readable and have the expected metadata keys.
      - Embedding dimension is consistent.
    """
    print("\n" + "="*60)
    print("  Database Verification")
    print("="*60)
    print(f"  Reopening: {config.CHROMA_DIR}")

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collections = client.list_collections()
    print(f"  Collections in DB   : {[c.name for c in collections]}")

    if not any(c.name == config.CHROMA_COLLECTION_NAME for c in collections):
        print(f"  [FAIL] Collection '{config.CHROMA_COLLECTION_NAME}' NOT FOUND.")
        print("         Run: python ingest.py  to index documents first.")
        return

    col = client.get_collection(
        name=config.CHROMA_COLLECTION_NAME,
        embedding_function=None,
    )
    count = col.count()
    print(f"  Collection name     : {col.name}")
    print(f"  Total records       : {count}")

    if count == 0:
        print("  [WARN] Collection is empty. No documents have been indexed yet.")
        return

    # Sample up to 3 records
    sample_n = min(3, count)
    peek = col.get(
        limit=sample_n,
        include=["documents", "metadatas", "embeddings"],
    )

    all_ok = True
    required_keys = {"source", "chunk_index", "token_count", "file_hash"}

    print(f"\n  Sample records (first {sample_n}):")
    for i, (doc_id, doc_text, meta, emb) in enumerate(
        zip(
            peek["ids"],
            peek["documents"],
            peek["metadatas"],
            peek["embeddings"],
        ),
        start=1,
    ):
        emb_arr = np.array(emb)
        dim     = emb_arr.shape[0]
        finite  = bool(np.all(np.isfinite(emb_arr)))
        missing = required_keys - set(meta.keys())

        dim_ok  = dim == config.EMBEDDING_DIMENSION
        meta_ok = not missing

        print(f"\n  Record {i}:")
        print(f"    ID        : {doc_id}")
        print(f"    Source    : {meta.get('source', '?')}")
        print(f"    Page      : {meta.get('page_num', 'N/A')}")
        print(f"    Tokens    : {meta.get('token_count', '?')}")
        print(f"    Emb dim   : {dim}  {'[OK]' if dim_ok else '[FAIL - expected ' + str(config.EMBEDDING_DIMENSION) + ']'}")
        print(f"    All finite: {finite}  {'[OK]' if finite else '[FAIL]'}")
        print(f"    Meta keys : {'[OK]' if meta_ok else '[FAIL - missing: ' + str(missing) + ']'}")
        print(f"    Text[:80] : {doc_text[:80]!r}")

        if not dim_ok or not finite or not meta_ok:
            all_ok = False

    print()
    if all_ok:
        print(f"  [PASS] Verification complete. {count} records in DB, all checks passed.")
    else:
        print("  [FAIL] Some records failed verification. See details above.")
    print("="*60)


# ── Reset a single source ─────────────────────────────────────────────────────

def reset_source(source_name: str) -> None:
    """
    Remove all chunks belonging to *source_name* from ChromaDB.
    Does NOT delete the entire database.
    """
    collection = get_chroma_collection()
    _, existing_ids = get_source_info(collection, source_name)
    if not existing_ids:
        print(f"No records found for source '{source_name}'.")
        return
    collection.delete(ids=list(existing_ids))
    print(f"Deleted {len(existing_ids)} chunk(s) for '{source_name}'.")
    print(f"DB total after reset: {collection.count()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Embed and index documents into ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ingest.py                  # ingest everything\n"
            "  python ingest.py lecture.pdf       # single file\n"
            "  python ingest.py --sample          # 1 doc, 5 chunks (safe test)\n"
            "  python ingest.py --verify          # check the database\n"
            "  python ingest.py --reset-source report.pdf\n"
        ),
    )
    p.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Single file to ingest. Omit to process all of data/documents/.",
    )
    p.add_argument(
        "--sample",
        action="store_true",
        help=(
            f"Safe test mode: index the first document found, "
            f"first {SAMPLE_MAX_CHUNKS} chunks only."
        ),
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Reopen the database and verify stored records. Does not ingest.",
    )
    p.add_argument(
        "--reset-source",
        metavar="FILENAME",
        help="Delete all stored chunks for the named source file.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── --verify only ─────────────────────────────────────────────────────────
    if args.verify:
        verify_database()
        return

    # ── --reset-source ────────────────────────────────────────────────────────
    if args.reset_source:
        reset_source(args.reset_source)
        return

    # ── Check memory before loading the model ─────────────────────────────────
    if not check_memory_safe("startup"):
        print(
            "\n[ABORT] Not enough RAM to start. "
            "Close other applications and retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Sample mode ───────────────────────────────────────────────────────────
    sample_chunks = SAMPLE_MAX_CHUNKS if args.sample else None
    max_docs      = 1                 if args.sample else None

    # ── Single file ───────────────────────────────────────────────────────────
    if args.target:
        target = Path(args.target)
        if not target.is_absolute():
            target = Path.cwd() / target
        if not target.exists():
            print(f"[ERROR] File not found: {target}", file=sys.stderr)
            sys.exit(1)

        collection = get_chroma_collection()
        model      = get_embedding_model()
        result     = ingest_document(target, collection, model,
                                     sample_max_chunks=sample_chunks)
        _print_result(result, collection)

        print(f"\n  Total records in DB : {collection.count()}")
        print(f"  DB path             : {config.CHROMA_DIR}")
        return

    # ── Directory mode ────────────────────────────────────────────────────────
    ingest_directory(
        directory=config.DOCUMENTS_DIR,
        max_docs=max_docs,
        sample_max_chunks=sample_chunks,
    )

    if args.sample:
        print("\nSample ingestion complete. Run --verify to confirm persistence.")
        print("When satisfied, run:  python ingest.py  (without --sample)")


if __name__ == "__main__":
    main()
