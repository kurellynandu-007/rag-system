"""
tests/test_chunking.py
-----------------------
Tests for chunking.py: token-aware chunking, overlap, metadata,
stable unique IDs, empty documents, and edge cases.

Run:
    .venv\\Scripts\\activate
    python -m pytest tests/test_chunking.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunking import (
    CHUNK_TOKENS,
    MODEL_MAX_SEQ_LENGTH,
    MODEL_USABLE_TOKENS,
    OVERLAP_TOKENS,
    TextChunk,
    chunk_document_pages,
    chunk_stats,
    chunk_text,
    count_tokens,
)
from loader import DocumentPage


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_page(text: str, source: str = "test.md", page_num=None) -> DocumentPage:
    """Convenience factory for DocumentPage."""
    return DocumentPage(
        text=text,
        source=source,
        page_num=page_num,
        char_count=len(text),
        file_type="markdown" if source.endswith(".md") else "pdf",
    )


def lorem(n_words: int = 200) -> str:
    """Generate a repetitive English-like sentence to fill *n_words* words."""
    sentence = (
        "The quick brown fox jumps over the lazy dog near the riverbank "
        "while the sun sets behind the distant mountains in the evening sky "
    )
    words = sentence.split()
    repeated = (words * ((n_words // len(words)) + 2))[:n_words]
    return " ".join(repeated) + "."


# ══════════════════════════════════════════════════════════════════════════════
# Token counting
# ══════════════════════════════════════════════════════════════════════════════

class TestCountTokens:

    def test_returns_positive_count(self):
        count, _ = count_tokens("Hello world")
        assert count > 0

    def test_returns_exactness_flag(self):
        _, is_exact = count_tokens("Test sentence.")
        assert isinstance(is_exact, bool)

    def test_longer_text_more_tokens(self):
        short_count, _ = count_tokens("Hi.")
        long_count, _  = count_tokens("Hi. " * 100)
        assert long_count > short_count

    def test_token_count_within_model_limit(self):
        """A text of CHUNK_TOKENS words should fit within model limit."""
        text = lorem(50)   # ~50 words → well under 256 tokens
        count, _ = count_tokens(text)
        assert count <= MODEL_MAX_SEQ_LENGTH, (
            f"count_tokens returned {count} which exceeds model limit {MODEL_MAX_SEQ_LENGTH}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# chunk_text: basic correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkText:

    def test_empty_string_returns_empty_list(self):
        result = chunk_text("", source="empty.md")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        result = chunk_text("   \n\t  ", source="ws.md")
        assert result == []

    def test_short_text_produces_exactly_one_chunk(self):
        text = "Short sentence."
        chunks = chunk_text(text, source="short.md")
        assert len(chunks) == 1

    def test_returns_text_chunk_objects(self):
        chunks = chunk_text(lorem(40), source="doc.md")
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_chunk_indices_sequential_from_zero(self):
        chunks = chunk_text(lorem(400), source="doc.md", chunk_tokens=50, overlap_tokens=10)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(indices)))

    def test_start_chunk_index_offset(self):
        """start_chunk_index shifts all indices."""
        chunks = chunk_text(lorem(50), source="doc.md", start_chunk_index=7)
        assert chunks[0].chunk_index == 7

    def test_invalid_chunk_tokens_raises(self):
        with pytest.raises(ValueError):
            chunk_text("text", source="t.md", chunk_tokens=0)

    def test_overlap_ge_chunk_tokens_raises(self):
        with pytest.raises(ValueError):
            chunk_text("text", source="t.md", chunk_tokens=50, overlap_tokens=50)

    def test_source_stored_correctly(self):
        chunks = chunk_text("Some text.", source="my_doc.pdf", page_num=3)
        assert all(c.source == "my_doc.pdf" for c in chunks)

    def test_page_num_stored_correctly(self):
        chunks = chunk_text("Some text.", source="doc.pdf", page_num=5)
        assert all(c.page_num == 5 for c in chunks)

    def test_page_num_none_for_markdown(self):
        chunks = chunk_text("Some text.", source="notes.md", page_num=None)
        assert all(c.page_num is None for c in chunks)

    def test_token_count_not_exceed_chunk_tokens(self):
        """No chunk should have more tokens than the configured window."""
        chunks = chunk_text(lorem(400), source="doc.md", chunk_tokens=60, overlap_tokens=12)
        for c in chunks:
            assert c.token_count <= 60, (
                f"Chunk {c.chunk_index} has {c.token_count} tokens, limit is 60"
            )

    def test_default_chunk_size_within_model_limit(self):
        """Default CHUNK_TOKENS must be strictly less than MODEL_USABLE_TOKENS."""
        assert CHUNK_TOKENS <= MODEL_USABLE_TOKENS, (
            f"Default CHUNK_TOKENS={CHUNK_TOKENS} exceeds usable "
            f"limit={MODEL_USABLE_TOKENS}. Embeddings would be silently truncated."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Overlap correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestOverlap:

    def _chunks_with_params(self, text, chunk_tokens=40, overlap_tokens=10):
        return chunk_text(text, source="overlap_test.md",
                          chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)

    def test_adjacent_chunks_share_token_ranges(self):
        """
        Chunk[i].token_end should overlap with Chunk[i+1].token_start.
        Specifically: chunk[i+1].token_start < chunk[i].token_end
        """
        long_text = lorem(300)
        chunks = self._chunks_with_params(long_text, chunk_tokens=50, overlap_tokens=15)
        assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"
        for i in range(len(chunks) - 1):
            c1, c2 = chunks[i], chunks[i + 1]
            assert c2.token_start < c1.token_end, (
                f"Chunk {i} ends at token {c1.token_end} but "
                f"chunk {i+1} starts at token {c2.token_start} — no overlap!"
            )

    def test_overlap_size_matches_config(self):
        """
        The overlap between adjacent chunks should equal overlap_tokens.
        i.e. chunk[i].token_end - chunk[i+1].token_start == overlap_tokens
        """
        overlap = 12
        long_text = lorem(300)
        chunks = self._chunks_with_params(long_text, chunk_tokens=50, overlap_tokens=overlap)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            c1, c2 = chunks[i], chunks[i + 1]
            actual_overlap = c1.token_end - c2.token_start
            # Last chunk may be smaller; skip it in this check
            if c1.token_count == 50:
                assert actual_overlap == overlap, (
                    f"Expected overlap={overlap}, got {actual_overlap} "
                    f"between chunks {i} and {i+1}"
                )

    def test_no_overlap_zero_param(self):
        """Setting overlap_tokens=0 should produce non-overlapping chunks."""
        long_text = lorem(300)
        chunks = chunk_text(long_text, source="t.md", chunk_tokens=50, overlap_tokens=0)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            c1, c2 = chunks[i], chunks[i + 1]
            # With no overlap, next chunk starts exactly where previous ended
            assert c2.token_start == c1.token_end

    def test_multiple_chunks_produced_for_long_text(self):
        """A text longer than chunk_tokens must produce multiple chunks."""
        long_text = lorem(500)   # ~500 words → many more than CHUNK_TOKENS tokens
        chunks = chunk_text(long_text, source="long.md",
                            chunk_tokens=50, overlap_tokens=10)
        assert len(chunks) > 1


# ══════════════════════════════════════════════════════════════════════════════
# Stable unique IDs
# ══════════════════════════════════════════════════════════════════════════════

class TestStableIds:

    def test_unique_id_format(self):
        chunks = chunk_text("Some text here.", source="report.pdf", page_num=1)
        expected = "report.pdf__chunk_0"
        assert chunks[0].unique_id == expected

    def test_unique_id_stable_across_calls(self):
        """Same text + source → same IDs every time."""
        text = lorem(100)
        chunks_run1 = chunk_text(text, source="stable.md")
        chunks_run2 = chunk_text(text, source="stable.md")
        ids_run1 = [c.unique_id for c in chunks_run1]
        ids_run2 = [c.unique_id for c in chunks_run2]
        assert ids_run1 == ids_run2

    def test_unique_ids_unique_within_document(self):
        chunks = chunk_text(lorem(400), source="doc.md", chunk_tokens=50, overlap_tokens=10)
        ids = [c.unique_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate IDs found!"

    def test_different_sources_produce_different_ids(self):
        text = "Same text content."
        chunks_a = chunk_text(text, source="a.md")
        chunks_b = chunk_text(text, source="b.md")
        assert chunks_a[0].unique_id != chunks_b[0].unique_id

    def test_spaces_in_source_replaced_in_id(self):
        chunks = chunk_text("Some content.", source="my document.pdf")
        assert " " not in chunks[0].unique_id

    def test_start_index_reflected_in_id(self):
        chunks = chunk_text("Some content.", source="doc.md", start_chunk_index=5)
        assert chunks[0].unique_id == "doc.md__chunk_5"


# ══════════════════════════════════════════════════════════════════════════════
# Metadata completeness
# ══════════════════════════════════════════════════════════════════════════════

class TestMetadata:

    def test_metadata_contains_required_keys(self):
        chunks = chunk_text("Hello world test.", source="doc.pdf", page_num=2)
        required = {"source", "chunk_index", "page_num", "token_count",
                    "token_exact", "token_start", "token_end"}
        for c in chunks:
            missing = required - set(c.metadata.keys())
            assert not missing, f"Metadata missing keys: {missing}"

    def test_metadata_source_matches_source_attr(self):
        chunks = chunk_text("Text.", source="notes.md")
        for c in chunks:
            assert c.metadata["source"] == c.source

    def test_metadata_chunk_index_matches_attr(self):
        chunks = chunk_text(lorem(100), source="doc.md",
                            chunk_tokens=30, overlap_tokens=5)
        for c in chunks:
            assert c.metadata["chunk_index"] == c.chunk_index

    def test_metadata_token_count_matches_attr(self):
        chunks = chunk_text(lorem(100), source="doc.md",
                            chunk_tokens=40, overlap_tokens=8)
        for c in chunks:
            assert c.metadata["token_count"] == c.token_count

    def test_extra_metadata_merged(self):
        chunks = chunk_text("Hello.", source="t.md",
                            extra_metadata={"course": "CS501"})
        assert chunks[0].metadata["course"] == "CS501"

    def test_metadata_values_chroma_compatible_types(self):
        """ChromaDB only accepts str, int, float metadata values."""
        chunks = chunk_text("Hello world.", source="doc.pdf", page_num=1)
        for c in chunks:
            for k, v in c.metadata.items():
                assert isinstance(v, (str, int, float)), (
                    f"Metadata key {k!r} has type {type(v).__name__}, "
                    "not compatible with ChromaDB."
                )


# ══════════════════════════════════════════════════════════════════════════════
# chunk_document_pages
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkDocumentPages:

    def test_empty_pages_list_returns_empty_chunks(self):
        result = chunk_document_pages([])
        assert result == []

    def test_single_page(self):
        page = make_page(lorem(80), source="single.md")
        chunks = chunk_document_pages([page])
        assert len(chunks) >= 1
        assert all(c.source == "single.md" for c in chunks)

    def test_multi_page_global_indices_sequential(self):
        """Chunk indices must be globally sequential across multiple pages."""
        pages = [
            make_page(lorem(200), source="doc.pdf", page_num=1),
            make_page(lorem(200), source="doc.pdf", page_num=2),
        ]
        chunks = chunk_document_pages(pages, chunk_tokens=50, overlap_tokens=10)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(indices))), (
            "Chunk indices are not globally sequential across pages."
        )

    def test_page_num_preserved_per_page(self):
        pages = [
            make_page("Page one text.", source="doc.pdf", page_num=1),
            make_page("Page two text.", source="doc.pdf", page_num=2),
        ]
        chunks = chunk_document_pages(pages)
        page1_chunks = [c for c in chunks if c.page_num == 1]
        page2_chunks = [c for c in chunks if c.page_num == 2]
        assert len(page1_chunks) >= 1
        assert len(page2_chunks) >= 1

    def test_mixed_page_none_and_int(self):
        """Markdown pages (page_num=None) and PDF pages (int) can be mixed."""
        pages = [
            make_page("Markdown content.", source="readme.md", page_num=None),
            make_page("PDF content.", source="paper.pdf", page_num=1),
        ]
        chunks = chunk_document_pages(pages)
        md_chunks  = [c for c in chunks if c.source == "readme.md"]
        pdf_chunks = [c for c in chunks if c.source == "paper.pdf"]
        assert all(c.page_num is None for c in md_chunks)
        assert all(c.page_num == 1  for c in pdf_chunks)

    def test_empty_page_text_skipped(self):
        """Pages with empty text produce zero chunks (not an error)."""
        pages = [make_page("", source="empty.md")]
        chunks = chunk_document_pages(pages)
        assert chunks == []


# ══════════════════════════════════════════════════════════════════════════════
# chunk_stats
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkStats:

    def test_empty_chunks_returns_count_zero(self):
        s = chunk_stats([])
        assert s["count"] == 0

    def test_stats_keys_present(self):
        chunks = chunk_text(lorem(100), source="doc.md", chunk_tokens=40, overlap_tokens=8)
        s = chunk_stats(chunks)
        for key in ["count", "total_tokens", "mean_tokens", "min_tokens",
                    "max_tokens", "total_chars", "mean_chars", "sources"]:
            assert key in s, f"Missing key: {key!r}"

    def test_mean_tokens_within_bounds(self):
        chunks = chunk_text(lorem(200), source="doc.md", chunk_tokens=50, overlap_tokens=10)
        s = chunk_stats(chunks)
        assert s["min_tokens"] <= s["mean_tokens"] <= s["max_tokens"]

    def test_sources_list_correct(self):
        pages = [
            make_page("Content A.", source="a.md"),
            make_page("Content B.", source="b.md"),
        ]
        chunks = chunk_document_pages(pages)
        s = chunk_stats(chunks)
        assert set(s["sources"]) == {"a.md", "b.md"}
