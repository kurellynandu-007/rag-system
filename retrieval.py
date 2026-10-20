"""
retrieval.py
------------
Semantic search against the ChromaDB vector store.

Given a natural-language query, this module:
1. Embeds the query using the same singleton model loaded by ingest.py.
2. Performs a cosine-similarity search in ChromaDB using query_embeddings
   (explicit vectors — ChromaDB never invokes a secondary embedding function).
3. Returns the top-K most relevant text chunks with their metadata.

Design notes
------------
- get_chroma_collection() (from ingest.py) opens the collection with
  embedding_function=None, matching the ingest contract exactly.
- Metadata keys match the schema written by chunking.py:
    chunk_index   (not chunk_id)
    page_num      (not page)
- If the collection is empty, a clear warning is logged (not a crash).

Usage:
    from retrieval import retrieve_chunks
    results = retrieve_chunks("What is RAG?", top_k=5)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb

import config
from ingest import get_chroma_collection, get_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """
    A single chunk returned by a similarity search.

    Attributes
    ----------
    text        : The chunk text.
    source      : Original document filename.
    chunk_index : Zero-based chunk index within the source document
                  (matches TextChunk.chunk_index from chunking.py).
    page_num    : Page number (1-based) if available, else None.
    distance    : Cosine distance from the query vector (lower = more similar).
    score       : Similarity score = 1 - distance (higher = more relevant).
    metadata    : Raw metadata dict from ChromaDB.
    """

    text: str
    source: str
    chunk_index: int
    page_num: Optional[int]
    distance: float
    score: float
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return (
            f"RetrievedChunk(source={self.source!r}, page={self.page_num}, "
            f"score={self.score:.4f}, preview={preview!r})"
        )


def retrieve_chunks(
    query: str,
    top_k: int = config.TOP_K_RESULTS,
    collection: Optional[chromadb.Collection] = None,
) -> List[RetrievedChunk]:
    """
    Find the *top_k* most semantically similar chunks for *query*.

    Parameters
    ----------
    query      : Natural-language query string.
    top_k      : Number of results to return.
    collection : ChromaDB collection to search (opens default if None).

    Returns
    -------
    List[RetrievedChunk] sorted by descending similarity score.
    Returns an empty list if the collection is empty or query is blank.

    Raises
    ------
    RuntimeError if the embedding model fails to load.
    """
    query = query.strip()
    if not query:
        logger.warning("retrieve_chunks called with an empty query.")
        return []
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Open collection if not provided
    if collection is None:
        collection = get_chroma_collection()

    doc_count = collection.count()
    if doc_count == 0:
        logger.warning(
            "ChromaDB collection '%s' is empty. "
            "Run ingest.py to add documents before querying.",
            config.CHROMA_COLLECTION_NAME,
        )
        return []

    # Embed the query
    model = get_embedding_model()
    query_vector = model.encode(
        query,
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        device="cpu",
    ).tolist()

    # Clamp top_k so we never request more documents than exist
    effective_k = min(top_k, doc_count)

    # Query ChromaDB
    logger.info(
        "Querying ChromaDB (top_k=%d, collection_size=%d): %r",
        effective_k,
        doc_count,
        query[:100],
    )
    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=effective_k,
        include=["documents", "metadatas", "distances"],
    )

    # Parse ChromaDB response
    # raw["documents"][0]  -> list of texts  (one query -> outer list has 1 element)
    # raw["distances"][0]  -> list of cosine distances
    documents  = raw.get("documents",  [[]])[0]
    metadatas  = raw.get("metadatas",  [[]])[0]
    distances  = raw.get("distances",  [[]])[0]

    results: List[RetrievedChunk] = []
    for text, meta, dist in zip(documents, metadatas, distances):
        meta = meta or {}
        # Safely read page_num — stored as int or "" (empty string for non-PDF)
        raw_page = meta.get("page_num", "")
        page_num = int(raw_page) if raw_page not in ("", None) else None

        results.append(
            RetrievedChunk(
                text=text,
                source=meta.get("source", "unknown"),
                chunk_index=int(meta.get("chunk_index", -1)),
                page_num=page_num,
                distance=float(dist),
                score=round(1.0 - float(dist), 4),
                metadata=meta,
            )
        )

    logger.info(
        "Retrieved %d chunks. Top score: %.4f",
        len(results),
        results[0].score if results else 0.0,
    )
    return results


def format_context(chunks: List[RetrievedChunk]) -> str:
    """
    Combine retrieved chunks into a single numbered context string
    ready to be inserted into a Gemini generation prompt.

    Returns
    -------
    Multi-line string with source citations, e.g.::

        [1] Source: report.pdf (page 3, score 0.8921)
        Text chunk content goes here...
    """
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        page_info = f", page {chunk.page_num}" if chunk.page_num else ""
        header = f"[{i}] Source: {chunk.source}{page_info} (score {chunk.score:.4f})"
        parts.append(f"{header}\n{chunk.text}")

    return "\n\n".join(parts)


if __name__ == "__main__":
    # Quick smoke-test: prints retrieved results for a sample query
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is RAG?"
    print(f"Query: {query!r}\n")
    chunks = retrieve_chunks(query)
    if not chunks:
        print("No results. Make sure you have ingested documents first.")
    else:
        for r in chunks:
            print(r)
            print(f"   {r.text[:200]!r}\n")
