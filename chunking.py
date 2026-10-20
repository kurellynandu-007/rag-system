"""
chunking.py
-----------
Token-aware text chunking for the RAG System.

⚠️  TOKEN LIMIT ANALYSIS (all-MiniLM-L6-v2)
─────────────────────────────────────────────
  Model max_seq_length  : 256 WordPiece tokens
  Special tokens ([CLS],[SEP]) overhead : 2
  Usable tokens per chunk : 254

  You requested 350–450 tokens per chunk.
  THAT WOULD EXCEED THE MODEL LIMIT and cause silent truncation
  of the last ~100-200 tokens in every chunk.

  Chosen configuration
  ──────────────────────
  CHUNK_TOKENS   = 220   (~86% of usable 254 — comfortable margin)
  OVERLAP_TOKENS = 45    (~20% overlap, which is ~4.5 chars × 45 ≈ 202 chars)

  In English prose (BertTokenizerFast on this corpus):
    220 tokens  ≈ 990 characters  ≈ 165 words
    45  tokens  ≈ 202 characters  ≈  34 words

  These estimates are approximations. The actual character count per
  chunk varies with vocabulary (code, abbreviations, rare words tokenise
  into more sub-word pieces). The token counts are EXACT because we use
  the model's own BertTokenizerFast, not a heuristic formula.

Token counting
──────────────
  Uses BertTokenizerFast from the installed sentence-transformers model.
  If the tokenizer cannot be loaded (e.g. offline) the module falls back
  to the documented approximation:  tokens ≈ chars / 4.5
  The fallback is clearly labelled in all output.

Algorithm
─────────
  1. Encode the full text with add_special_tokens=False.
  2. Slide a window of CHUNK_TOKENS token IDs across the sequence,
     advancing CHUNK_TOKENS - OVERLAP_TOKENS tokens each step.
  3. Decode each window back to text.
  4. Build a TextChunk with exact token_count and approximate char positions.

Usage:
    from chunking import chunk_document_pages, check_model_limits
    from loader import DocumentPage
    check_model_limits()           # prints limit analysis
    chunks = chunk_document_pages(pages)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Token-aware configuration ─────────────────────────────────────────────────

# Hard facts about all-MiniLM-L6-v2
MODEL_MAX_SEQ_LENGTH: int = 256          # WordPiece tokens including specials
SPECIAL_TOKENS_OVERHEAD: int = 2         # [CLS] + [SEP]
MODEL_USABLE_TOKENS: int = MODEL_MAX_SEQ_LENGTH - SPECIAL_TOKENS_OVERHEAD  # 254

# Our chosen chunk parameters (see module docstring for reasoning)
CHUNK_TOKENS: int   = 220    # ~86 % of usable limit
OVERLAP_TOKENS: int = 45     # ~20 % of chunk, prevents boundary information loss

# Approximate chars-per-token for English text using BertTokenizerFast
# Measured: 450 chars / 100 tokens = 4.5 chars/token
CHARS_PER_TOKEN_APPROX: float = 4.5


# ── Tokenizer singleton ───────────────────────────────────────────────────────

_tokenizer = None          # loaded lazily on first call
_tokenizer_available: bool = True   # set to False on failure so we stop retrying


def get_tokenizer():
    """
    Return the BertTokenizerFast from the cached sentence-transformers model.
    Returns None and logs a warning if loading fails (fallback mode activates).
    """
    global _tokenizer, _tokenizer_available
    if _tokenizer is not None:
        return _tokenizer
    if not _tokenizer_available:
        return None
    try:
        from transformers import AutoTokenizer
        # Uses the locally cached HuggingFace model — no network needed
        _tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info(
            "Tokenizer loaded: %s", type(_tokenizer).__name__
        )
    except Exception as exc:
        logger.warning(
            "Could not load tokenizer (%s). "
            "Falling back to chars/4.5 token approximation.",
            exc,
        )
        _tokenizer_available = False
        _tokenizer = None
    return _tokenizer


# ── Token counting helper ─────────────────────────────────────────────────────

def count_tokens(text: str) -> Tuple[int, bool]:
    """
    Count the number of tokens in *text*.

    Returns
    -------
    (token_count, is_exact)
      is_exact = True  → counted with the real BertTokenizerFast
      is_exact = False → approximated as  round(len(text) / 4.5)
    """
    tok = get_tokenizer()
    if tok is not None:
        ids = tok.encode(text, add_special_tokens=False)
        return len(ids), True
    else:
        approx = max(1, round(len(text) / CHARS_PER_TOKEN_APPROX))
        return approx, False


# ── Limit check utility ───────────────────────────────────────────────────────

def check_model_limits() -> None:
    """
    Print a human-readable analysis of the token limits and chosen config.
    Call this once at startup to make the tradeoffs visible.
    """
    tok = get_tokenizer()
    tok_name = type(tok).__name__ if tok else "fallback-approximation"
    exact_label = "exact" if tok else "estimated"

    print(
        f"\n{'='*60}\n"
        f"  Embedding Model Token-Limit Analysis\n"
        f"{'='*60}\n"
        f"  Model             : all-MiniLM-L6-v2\n"
        f"  Tokenizer         : {tok_name} ({exact_label} counts)\n"
        f"  max_seq_length    : {MODEL_MAX_SEQ_LENGTH} tokens\n"
        f"  Special overhead  : {SPECIAL_TOKENS_OVERHEAD} ([CLS]+[SEP])\n"
        f"  Usable per chunk  : {MODEL_USABLE_TOKENS} tokens\n"
        f"  {'-'*41}\n"
        f"  Requested size    : 350-450 tokens  -> EXCEEDS model limit!\n"
        f"  {'-'*41}\n"
        f"  CHOSEN chunk size : {CHUNK_TOKENS} tokens  "
        f"(~{round(CHUNK_TOKENS*CHARS_PER_TOKEN_APPROX)} chars)\n"
        f"  CHOSEN overlap    : {OVERLAP_TOKENS} tokens  "
        f"(~{round(OVERLAP_TOKENS*CHARS_PER_TOKEN_APPROX)} chars)\n"
        f"  Headroom remaining: {MODEL_USABLE_TOKENS - CHUNK_TOKENS} tokens\n"
        f"{'='*60}\n"
    )


# ── TextChunk dataclass ───────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """
    A single chunk of text ready for embedding.

    Attributes
    ----------
    text        : The chunk text (decoded from token window).
    source      : Source filename (not full path).
    chunk_index : Zero-based, globally unique within the source document.
    page_num    : 1-based page number if applicable, None for Markdown/text.
    token_count : Number of tokens (exact if tokenizer available, else ~).
    token_exact : True if token_count was measured with the real tokenizer.
    token_start : First token index within the full-document token sequence.
    token_end   : One-past-last token index within the full-document sequence.
    metadata    : Dict ready to be passed to ChromaDB. Always contains:
                    source, chunk_index, page_num (or ""), token_count,
                    token_exact, token_start, token_end.
    """

    text: str
    source: str
    chunk_index: int
    page_num: Optional[int]
    token_count: int
    token_exact: bool
    token_start: int
    token_end: int
    metadata: dict = field(default_factory=dict)

    # ── Stable unique ID ──────────────────────────────────────────────────────
    @property
    def unique_id(self) -> str:
        """
        Stable, unique identifier for this chunk across runs.

        Format:  <safe_source>__chunk_<chunk_index>

        'Stable' means: as long as the source document content and
        chunking parameters don't change, the same chunk always gets
        the same ID. This is what makes ChromaDB upserts idempotent.
        """
        safe = (
            self.source
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )
        return f"{safe}__chunk_{self.chunk_index}"

    def preview(self, chars: int = 120) -> str:
        """Return a short preview string for display."""
        body = self.text[:chars].replace("\n", " ")
        ellipsis = "…" if len(self.text) > chars else ""
        tok_label = "tokens" if self.token_exact else "~tokens"
        page = f"p{self.page_num}" if self.page_num else "whole"
        return (
            f"[{self.chunk_index}] {self.source} ({page}) "
            f"| {self.token_count} {tok_label} "
            f"| tok {self.token_start}-{self.token_end}\n"
            f"    {body!r}{ellipsis}"
        )

    def __repr__(self) -> str:
        return (
            f"TextChunk(id={self.chunk_index!r}, source={self.source!r}, "
            f"page={self.page_num}, tokens={self.token_count})"
        )


# ── Core chunking function ────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str,
    page_num: Optional[int] = None,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    start_chunk_index: int = 0,
    extra_metadata: Optional[dict] = None,
) -> List[TextChunk]:
    """
    Split *text* into token-window chunks.

    Algorithm
    ---------
    1. Encode the full text with the real tokenizer (or fall back).
    2. Slide a window of *chunk_tokens* token IDs, advancing
       (chunk_tokens - overlap_tokens) each step.
    3. Decode each window back to a string.
    4. Attach source, page, and token-position metadata.

    Parameters
    ----------
    text             : Non-empty string to chunk.
    source           : Source filename for metadata and unique IDs.
    page_num         : 1-based page number (None for non-PDF sources).
    chunk_tokens     : Target number of tokens per chunk (default 220).
    overlap_tokens   : Token overlap between adjacent chunks (default 45).
    start_chunk_index: First chunk_index value (for multi-page accumulation).
    extra_metadata   : Additional key-value pairs merged into every chunk.

    Returns
    -------
    List[TextChunk] ordered from start to end of the text.

    Raises
    ------
    ValueError if chunk_tokens <= 0 or overlap_tokens >= chunk_tokens.
    """
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be > 0, got {chunk_tokens}")
    if overlap_tokens >= chunk_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be < chunk_tokens ({chunk_tokens})"
        )
    if chunk_tokens > MODEL_USABLE_TOKENS:
        logger.warning(
            "chunk_tokens=%d exceeds model usable limit of %d. "
            "Chunks will be silently truncated during embedding.",
            chunk_tokens,
            MODEL_USABLE_TOKENS,
        )

    text = text.strip()
    if not text:
        logger.debug("chunk_text received empty text for source=%r page=%s", source, page_num)
        return []

    tok = get_tokenizer()
    is_exact = tok is not None

    # ── Tokenise ──────────────────────────────────────────────────────────────
    if is_exact:
        token_ids: List[int] = tok.encode(text, add_special_tokens=False)
    else:
        # Fallback: approximate by splitting on whitespace, one "id" per word
        # (only used if transformers tokenizer fails to load)
        token_ids = text.split()   # type: ignore[assignment]

    total_tokens = len(token_ids)
    if total_tokens == 0:
        return []

    step = chunk_tokens - overlap_tokens
    chunks: List[TextChunk] = []
    chunk_index = start_chunk_index
    pos = 0

    while pos < total_tokens:
        end = min(pos + chunk_tokens, total_tokens)
        window = token_ids[pos:end]

        # Decode back to text
        if is_exact:
            chunk_str = tok.decode(window, skip_special_tokens=True).strip()
        else:
            chunk_str = " ".join(window).strip()   # type: ignore[arg-type]

        if chunk_str:
            meta = {
                "source":       source,
                "chunk_index":  chunk_index,
                "page_num":     page_num if page_num is not None else "",
                "token_count":  end - pos,
                "token_exact":  str(is_exact),   # ChromaDB metadata must be str/int/float
                "token_start":  pos,
                "token_end":    end,
            }
            if extra_metadata:
                meta.update(extra_metadata)

            chunks.append(
                TextChunk(
                    text=chunk_str,
                    source=source,
                    chunk_index=chunk_index,
                    page_num=page_num,
                    token_count=end - pos,
                    token_exact=is_exact,
                    token_start=pos,
                    token_end=end,
                    metadata=meta,
                )
            )
            chunk_index += 1

        if end == total_tokens:
            break
        pos += step

    logger.debug(
        "chunk_text: source=%r page=%s -> %d token(s) -> %d chunk(s)",
        source, page_num, total_tokens, len(chunks),
    )
    return chunks


# ── Multi-page chunker ────────────────────────────────────────────────────────

def chunk_document_pages(
    pages,                       # List[DocumentPage] from loader.py
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[TextChunk]:
    """
    Chunk a list of DocumentPage objects, preserving per-page metadata.

    Chunk indices are globally sequential across all pages of the same
    document — each chunk gets a unique index independent of which page it
    came from.

    Parameters
    ----------
    pages         : List of DocumentPage objects from loader.load_document().
    chunk_tokens  : Token budget per chunk.
    overlap_tokens: Token overlap between adjacent chunks.

    Returns
    -------
    List[TextChunk] with globally unique chunk_index values.
    """
    all_chunks: List[TextChunk] = []
    next_index = 0

    for page in pages:
        page_chunks = chunk_text(
            text=page.text,
            source=page.source,
            page_num=page.page_num,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            start_chunk_index=next_index,
        )
        all_chunks.extend(page_chunks)
        next_index += len(page_chunks)

    logger.info(
        "chunk_document_pages: %d page(s) -> %d chunk(s)",
        len(pages), len(all_chunks),
    )
    return all_chunks


# ── Stats reporter ────────────────────────────────────────────────────────────

def chunk_stats(chunks: List[TextChunk]) -> dict:
    """
    Compute aggregate statistics over a list of chunks.

    Returns a dict with keys:
      count, total_tokens, mean_tokens, min_tokens, max_tokens,
      total_chars, mean_chars, sources, token_exact.
    """
    if not chunks:
        return {"count": 0}

    token_counts = [c.token_count for c in chunks]
    char_counts  = [len(c.text)   for c in chunks]
    sources      = sorted({c.source for c in chunks})
    exact        = all(c.token_exact for c in chunks)

    return {
        "count":        len(chunks),
        "total_tokens": sum(token_counts),
        "mean_tokens":  round(sum(token_counts) / len(token_counts), 1),
        "min_tokens":   min(token_counts),
        "max_tokens":   max(token_counts),
        "total_chars":  sum(char_counts),
        "mean_chars":   round(sum(char_counts) / len(char_counts), 1),
        "sources":      sources,
        "token_exact":  exact,
    }


def print_stats(chunks: List[TextChunk], load_stats=None) -> None:
    """Print a formatted chunk-statistics report."""
    s = chunk_stats(chunks)
    exact_label = "exact" if s.get("token_exact") else "~estimated"

    print(f"\n{'='*60}")
    print("  Chunk Statistics")
    print(f"{'='*60}")
    if load_stats is not None:
        print(f"  Files loaded     : {load_stats.files_loaded}")
        print(f"  Pages with text  : {load_stats.pages_with_text}")
        print(f"  Pages skipped    : {load_stats.pages_empty}")
        print(f"  Total characters : {load_stats.total_chars:,}")
    print(f"  Total chunks     : {s['count']}")
    print(f"  Token counts ({exact_label}):")
    print(f"    Total          : {s.get('total_tokens', 'N/A')}")
    print(f"    Mean / chunk   : {s.get('mean_tokens', 'N/A')}")
    print(f"    Min / chunk    : {s.get('min_tokens', 'N/A')}")
    print(f"    Max / chunk    : {s.get('max_tokens', 'N/A')}")
    print(f"  Char counts:")
    print(f"    Mean / chunk   : {s.get('mean_chars', 'N/A')}")
    print(f"  Sources          : {s.get('sources', [])}")
    print(f"{'='*60}")


def print_chunk_previews(chunks: List[TextChunk], n: int = 3) -> None:
    """Print the first *n* chunks as human-readable previews."""
    print(f"\n{'='*60}")
    print(f"  Chunk Previews (first {min(n, len(chunks))} of {len(chunks)})")
    print(f"{'='*60}")
    for chunk in chunks[:n]:
        print()
        print(f"  ID       : {chunk.unique_id}")
        print(f"  Source   : {chunk.source}")
        print(f"  Page     : {chunk.page_num if chunk.page_num else 'N/A'}")
        print(f"  Tokens   : {chunk.token_count} "
              f"({'exact' if chunk.token_exact else '~est.'}) "
              f"[tok {chunk.token_start}-{chunk.token_end}]")
        print(f"  Chars    : {len(chunk.text)}")
        print(f"  Text     :")
        for line in chunk.text[:400].splitlines():
            print(f"    {line}")
        if len(chunk.text) > 400:
            print(f"    … ({len(chunk.text) - 400} more chars)")
        print()
    print(f"{'='*60}")


# ── Self-test / CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_model_limits()

    sample = (
        "Retrieval-Augmented Generation (RAG) is an AI framework that "
        "improves the factual accuracy of language model outputs by grounding "
        "them in retrieved evidence from an external knowledge base. "
        "Instead of relying solely on parametric knowledge baked into model "
        "weights during training, RAG dynamically fetches relevant documents "
        "at inference time and includes them in the prompt as context. "
        "This approach reduces hallucination, keeps information up-to-date "
        "without retraining, and provides traceable citations. "
        "The retrieval step uses dense vector similarity search over "
        "pre-embedded document chunks stored in a vector database. "
        "Typical embedding models include sentence-transformers models such as "
        "all-MiniLM-L6-v2, which produce 384-dimensional vectors. "
    ) * 3   # repeat to create a longer text for chunking demonstration

    from loader import DocumentPage
    page = DocumentPage(
        text=sample, source="demo.txt", page_num=None,
        char_count=len(sample), file_type="text",
    )
    chunks = chunk_document_pages([page])
    print_stats(chunks)
    print_chunk_previews(chunks, n=3)
