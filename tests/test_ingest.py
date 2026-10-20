"""
tests/test_ingest.py
--------------------
Tests for the embedding and indexing pipeline (ingest.py).

Strategy
--------
- All tests use isolated tmp_path directories for ChromaDB so they never
  touch the real chroma_data/ or data/documents/ directories.
- The real embedding model is loaded *once* per pytest session via a
  session-scoped fixture, reused across all tests.
- Tests that require actual embeddings are marked with
  @pytest.mark.slow and can be skipped with: pytest -m "not slow"
- Tests that only exercise the non-embedding logic (hashing, metadata,
  incremental checks) run without the model (fast).

Run all:
    .venv\\Scripts\\activate
    python -m pytest tests/test_ingest.py -v

Skip slow (embedding) tests:
    python -m pytest tests/test_ingest.py -v -m "not slow"
"""

import hashlib
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import ingest as ing
from chunking import TextChunk, chunk_text
from loader import DocumentPage


def pytest_configure(config_obj):
    """Register custom marks to suppress PytestUnknownMarkWarning."""
    config_obj.addinivalue_line(
        "markers", "slow: marks tests as slow (require real embedding model)"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_md(tmp_path: Path, content: str, name: str = "doc.md") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def make_doc_page(text: str, source: str = "test.md") -> DocumentPage:
    return DocumentPage(
        text=text, source=source, page_num=None,
        char_count=len(text), file_type="markdown",
    )


def make_chunks(
    text: str, source: str = "test.md", n: int = 3, chunk_tokens: int = 30
) -> List[TextChunk]:
    """Produce n small chunks from text (or fewer if text is short)."""
    chunks = chunk_text(text, source=source, chunk_tokens=chunk_tokens, overlap_tokens=5)
    return chunks[:n]


def fake_embeddings(n: int, dim: int = 384) -> np.ndarray:
    """Random unit-normalised float32 vectors for use without the real model."""
    rng = np.random.default_rng(seed=42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def open_tmp_collection(tmp_path: Path):
    """Open an isolated ChromaDB collection in tmp_path."""
    import chromadb
    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.get_or_create_collection(
        name="test_collection",
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    return col


# ── Session-scoped model fixture (loaded once) ────────────────────────────────

@pytest.fixture(scope="session")
def embedding_model():
    """
    Load the real SentenceTransformer model once for the whole test session.
    Skips all slow tests if the model cannot be loaded.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device="cpu")
        return model
    except Exception as exc:
        pytest.skip(f"Could not load embedding model: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# file_sha256
# ══════════════════════════════════════════════════════════════════════════════

class TestFileSha256:

    def test_returns_64_char_hex_string(self, tmp_path):
        f = make_md(tmp_path, "hello world")
        h = ing.file_sha256(f)
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_content_same_hash(self, tmp_path):
        f1 = make_md(tmp_path, "identical content", "a.md")
        f2 = make_md(tmp_path, "identical content", "b.md")
        assert ing.file_sha256(f1) == ing.file_sha256(f2)

    def test_different_content_different_hash(self, tmp_path):
        f1 = make_md(tmp_path, "content A", "a.md")
        f2 = make_md(tmp_path, "content B", "b.md")
        assert ing.file_sha256(f1) != ing.file_sha256(f2)

    def test_matches_stdlib_sha256(self, tmp_path):
        content = b"test bytes 12345"
        f = tmp_path / "raw.bin"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert ing.file_sha256(f) == expected


# ══════════════════════════════════════════════════════════════════════════════
# get_source_info
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSourceInfo:

    def test_empty_collection_returns_none_and_empty_set(self, tmp_path):
        col = open_tmp_collection(tmp_path)
        stored_hash, existing_ids = ing.get_source_info(col, "missing.md")
        assert stored_hash is None
        assert existing_ids == set()

    def test_returns_stored_hash_and_ids(self, tmp_path):
        col = open_tmp_collection(tmp_path)
        vecs = fake_embeddings(2)
        col.upsert(
            ids=["doc.md__chunk_0", "doc.md__chunk_1"],
            documents=["chunk zero text", "chunk one text"],
            embeddings=vecs.tolist(),
            metadatas=[
                {"source": "doc.md", "file_hash": "abc123", "chunk_index": 0, "page_num": ""},
                {"source": "doc.md", "file_hash": "abc123", "chunk_index": 1, "page_num": ""},
            ],
        )
        stored_hash, existing_ids = ing.get_source_info(col, "doc.md")
        assert stored_hash == "abc123"
        assert existing_ids == {"doc.md__chunk_0", "doc.md__chunk_1"}

    def test_different_sources_isolated(self, tmp_path):
        col = open_tmp_collection(tmp_path)
        vecs = fake_embeddings(1)
        col.upsert(
            ids=["a.md__chunk_0"],
            documents=["a content"],
            embeddings=vecs.tolist(),
            metadatas=[{"source": "a.md", "file_hash": "hashA", "chunk_index": 0, "page_num": ""}],
        )
        stored_hash, ids = ing.get_source_info(col, "b.md")
        assert stored_hash is None
        assert ids == set()


# ══════════════════════════════════════════════════════════════════════════════
# purge_stale_ids
# ══════════════════════════════════════════════════════════════════════════════

class TestPurgeStaleIds:

    def test_deletes_ids_not_in_new_set(self, tmp_path):
        col = open_tmp_collection(tmp_path)
        vecs = fake_embeddings(3)
        col.upsert(
            ids=["s__chunk_0", "s__chunk_1", "s__chunk_2"],
            documents=["text 0", "text 1", "text 2"],
            embeddings=vecs.tolist(),
            metadatas=[{"source": "s"} for _ in range(3)],
        )
        old_ids = {"s__chunk_0", "s__chunk_1", "s__chunk_2"}
        new_ids = {"s__chunk_0", "s__chunk_1"}   # chunk_2 gone in new version
        purged = ing.purge_stale_ids(col, old_ids, new_ids, "s")
        assert purged == 1
        assert col.count() == 2

    def test_no_purge_when_ids_match(self, tmp_path):
        col = open_tmp_collection(tmp_path)
        vecs = fake_embeddings(2)
        col.upsert(
            ids=["s__chunk_0", "s__chunk_1"],
            documents=["text 0", "text 1"],
            embeddings=vecs.tolist(),
            metadatas=[{"source": "s"} for _ in range(2)],
        )
        ids = {"s__chunk_0", "s__chunk_1"}
        purged = ing.purge_stale_ids(col, ids, ids, "s")
        assert purged == 0
        assert col.count() == 2

    def test_purge_returns_correct_count(self, tmp_path):
        col = open_tmp_collection(tmp_path)
        vecs = fake_embeddings(5)
        col.upsert(
            ids=[f"s__chunk_{i}" for i in range(5)],
            documents=[f"t{i}" for i in range(5)],
            embeddings=vecs.tolist(),
            metadatas=[{"source": "s"} for _ in range(5)],
        )
        old_ids = {f"s__chunk_{i}" for i in range(5)}
        new_ids = {f"s__chunk_{i}" for i in range(2)}
        purged = ing.purge_stale_ids(col, old_ids, new_ids, "s")
        assert purged == 3


# ══════════════════════════════════════════════════════════════════════════════
# upsert_to_chroma
# ══════════════════════════════════════════════════════════════════════════════

class TestUpsertToChroma:

    def _make_chunks(self) -> List[TextChunk]:
        text = (
            "Retrieval-Augmented Generation is an AI technique that combines "
            "document retrieval with neural text generation to improve accuracy. " * 5
        )
        return make_chunks(text, source="test.md", n=3)

    def test_upsert_stores_correct_count(self, tmp_path):
        col = open_tmp_collection(tmp_path)
        chunks = self._make_chunks()
        vecs = fake_embeddings(len(chunks))
        stored = ing.upsert_to_chroma(col, chunks, vecs, file_hash="hash001")
        assert stored == len(chunks)
        assert col.count() == len(chunks)

    def test_upsert_is_idempotent(self, tmp_path):
        col = open_tmp_collection(tmp_path)
        chunks = self._make_chunks()
        vecs = fake_embeddings(len(chunks))
        ing.upsert_to_chroma(col, chunks, vecs, file_hash="h1")
        ing.upsert_to_chroma(col, chunks, vecs, file_hash="h1")  # second run
        assert col.count() == len(chunks)   # no duplicates

    def test_file_hash_stored_in_metadata(self, tmp_path):
        col = open_tmp_collection(tmp_path)
        chunks = self._make_chunks()
        vecs = fake_embeddings(len(chunks))
        ing.upsert_to_chroma(col, chunks, vecs, file_hash="myHash42")
        result = col.get(include=["metadatas"])
        for meta in result["metadatas"]:
            assert meta["file_hash"] == "myHash42"

    def test_ids_match_chunk_unique_ids(self, tmp_path):
        col = open_tmp_collection(tmp_path)
        chunks = self._make_chunks()
        vecs = fake_embeddings(len(chunks))
        ing.upsert_to_chroma(col, chunks, vecs, file_hash="h")
        result = col.get()
        stored_ids = set(result["ids"])
        expected_ids = {c.unique_id for c in chunks}
        assert stored_ids == expected_ids

    def test_mismatched_lengths_raise(self, tmp_path):
        col = open_tmp_collection(tmp_path)
        chunks = self._make_chunks()
        wrong_vecs = fake_embeddings(len(chunks) + 1)
        with pytest.raises(ValueError, match="Mismatch"):
            ing.upsert_to_chroma(col, chunks, wrong_vecs, file_hash="h")

    def test_retrieved_text_matches_original(self, tmp_path):
        col = open_tmp_collection(tmp_path)
        chunks = self._make_chunks()
        vecs = fake_embeddings(len(chunks))
        ing.upsert_to_chroma(col, chunks, vecs, file_hash="h")
        result = col.get(include=["documents"])
        stored_texts = sorted(result["documents"])
        expected_texts = sorted(c.text for c in chunks)
        assert stored_texts == expected_texts

    def test_embedding_dimension_preserved(self, tmp_path):
        col = open_tmp_collection(tmp_path)
        chunks = self._make_chunks()
        vecs = fake_embeddings(len(chunks))
        ing.upsert_to_chroma(col, chunks, vecs, file_hash="h")
        result = col.get(include=["embeddings"])
        for emb in result["embeddings"]:
            assert len(emb) == config.EMBEDDING_DIMENSION


# ══════════════════════════════════════════════════════════════════════════════
# embed_chunks_batched  (slow — requires real model)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestEmbedChunksBatched:

    def _chunks(self):
        text = (
            "The embedding model converts text into a dense vector representation "
            "that captures semantic meaning for similarity search. " * 6
        )
        return make_chunks(text, source="emb_test.md", n=5, chunk_tokens=40)

    def test_output_shape(self, embedding_model):
        chunks = self._chunks()
        vecs = ing.embed_chunks_batched(embedding_model, chunks, batch_size=4)
        assert vecs.shape == (len(chunks), config.EMBEDDING_DIMENSION)

    def test_output_dtype_float32(self, embedding_model):
        chunks = self._chunks()
        vecs = ing.embed_chunks_batched(embedding_model, chunks)
        assert vecs.dtype == np.float32

    def test_all_values_finite(self, embedding_model):
        chunks = self._chunks()
        vecs = ing.embed_chunks_batched(embedding_model, chunks)
        assert np.all(np.isfinite(vecs)), "Embeddings contain NaN or Inf"

    def test_different_texts_produce_different_vectors(self, embedding_model):
        text_a = "Python is a programming language."
        text_b = "The Himalayan mountain range spans several countries."
        chunks_a = make_chunks(text_a, source="a.md", n=1, chunk_tokens=20)
        chunks_b = make_chunks(text_b, source="b.md", n=1, chunk_tokens=20)
        va = ing.embed_chunks_batched(embedding_model, chunks_a)
        vb = ing.embed_chunks_batched(embedding_model, chunks_b)
        # Vectors should not be identical
        assert not np.allclose(va, vb), "Different texts produced identical vectors"

    def test_same_text_reproducible_vectors(self, embedding_model):
        text = "Consistent sentence for reproducibility testing."
        chunks = make_chunks(text, source="r.md", n=1, chunk_tokens=20)
        v1 = ing.embed_chunks_batched(embedding_model, chunks)
        v2 = ing.embed_chunks_batched(embedding_model, chunks)
        np.testing.assert_array_equal(v1, v2)

    def test_empty_chunks_returns_empty_array(self, embedding_model):
        result = ing.embed_chunks_batched(embedding_model, [])
        assert result.shape == (0, config.EMBEDDING_DIMENSION)

    def test_single_chunk_no_error(self, embedding_model):
        chunks = make_chunks("Single sentence.", source="s.md", n=1, chunk_tokens=20)
        vecs = ing.embed_chunks_batched(embedding_model, chunks[:1])
        assert vecs.shape == (1, config.EMBEDDING_DIMENSION)


# ══════════════════════════════════════════════════════════════════════════════
# ingest_document end-to-end  (slow — requires real model)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestIngestDocument:

    @pytest.fixture()
    def isolated_collection(self, tmp_path):
        import chromadb
        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        col = client.get_or_create_collection(
            name="test_col", embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        return col

    def test_markdown_ingestion_ok(self, tmp_path, embedding_model, isolated_collection):
        # Use enough text to guarantee at least 1 chunk; assert on actual count.
        md = make_md(
            tmp_path,
            "RAG systems combine retrieval with generation. " * 20,
            name="rag.md",
        )
        result = ing.ingest_document(
            md, isolated_collection, embedding_model, sample_max_chunks=3
        )
        assert result["status"] == "ok", f"Unexpected status: {result}"
        assert result["chunks_stored"] >= 1
        assert result["chunks_stored"] == isolated_collection.count()
        assert result["error"] is None

    def test_missing_file_returns_error_status(self, tmp_path, embedding_model, isolated_collection):
        fake = tmp_path / "ghost.md"
        result = ing.ingest_document(fake, isolated_collection, embedding_model)
        assert result["status"] == "error"
        assert result["error"] is not None
        assert isolated_collection.count() == 0

    def test_empty_file_returns_empty_status(self, tmp_path, embedding_model, isolated_collection):
        empty = make_md(tmp_path, "   \n  ", name="empty.md")
        result = ing.ingest_document(empty, isolated_collection, embedding_model)
        assert result["status"] in ("empty", "error")
        assert isolated_collection.count() == 0

    def test_unchanged_file_skipped_on_rerun(self, tmp_path, embedding_model, isolated_collection):
        md = make_md(
            tmp_path,
            "Knowledge base documents for testing. " * 20,
            name="kb.md",
        )
        r1 = ing.ingest_document(md, isolated_collection, embedding_model, sample_max_chunks=3)
        assert r1["status"] == "ok"
        count_after_first = isolated_collection.count()

        r2 = ing.ingest_document(md, isolated_collection, embedding_model, sample_max_chunks=3)
        assert r2["status"] == "skipped-unchanged"
        assert r2["skipped"] is True
        # DB count must not change
        assert isolated_collection.count() == count_after_first

    def test_changed_file_updates_records(self, tmp_path, embedding_model, isolated_collection):
        md = make_md(
            tmp_path,
            "Original content for testing. " * 20,
            name="update_me.md",
        )
        r1 = ing.ingest_document(md, isolated_collection, embedding_model, sample_max_chunks=3)
        assert r1["status"] == "ok"

        # Overwrite with new content (changes the hash)
        md.write_text("Completely different content now. " * 20, encoding="utf-8")
        r2 = ing.ingest_document(md, isolated_collection, embedding_model, sample_max_chunks=3)
        assert r2["status"] == "ok"
        assert not r2["skipped"]

    def test_stored_records_persist_across_reopen(self, tmp_path, embedding_model):
        """Core requirement: records must survive closing and reopening ChromaDB."""
        import chromadb
        db_path = tmp_path / "persist_test"

        # First process: ingest
        client1 = chromadb.PersistentClient(path=str(db_path))
        col1 = client1.get_or_create_collection(
            name="persist_col", embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        md = make_md(tmp_path, "Persistent document content. " * 20, name="persist.md")
        result = ing.ingest_document(md, col1, embedding_model, sample_max_chunks=3)
        assert result["status"] == "ok"
        stored_count = col1.count()
        del client1, col1  # close the client

        # Second process: reopen and verify
        client2 = chromadb.PersistentClient(path=str(db_path))
        col2 = client2.get_collection(name="persist_col", embedding_function=None)
        assert col2.count() == stored_count, (
            f"Expected {stored_count} records after reopen, got {col2.count()}"
        )

    def test_metadata_keys_complete(self, tmp_path, embedding_model, isolated_collection):
        md = make_md(tmp_path, "Test document content. " * 25, name="meta.md")
        ing.ingest_document(md, isolated_collection, embedding_model, sample_max_chunks=2)
        result = isolated_collection.get(include=["metadatas"])
        required = {"source", "chunk_index", "token_count", "file_hash",
                    "token_exact", "token_start", "token_end", "page_num"}
        for meta in result["metadatas"]:
            missing = required - set(meta.keys())
            assert not missing, f"Metadata missing: {missing}"

    def test_stored_embedding_dimension(self, tmp_path, embedding_model, isolated_collection):
        md = make_md(tmp_path, "Embedding dimension check document. " * 20, name="dim.md")
        ing.ingest_document(md, isolated_collection, embedding_model, sample_max_chunks=2)
        result = isolated_collection.get(include=["embeddings"])
        for emb in result["embeddings"]:
            assert len(emb) == config.EMBEDDING_DIMENSION, (
                f"Expected {config.EMBEDDING_DIMENSION}-dim, got {len(emb)}"
            )

    def test_stale_chunks_purged_on_document_shrink(self, tmp_path, embedding_model):
        """
        When a document is re-ingested with fewer chunks (e.g. after editing),
        the old extra chunk IDs must be deleted to avoid stale records.
        """
        import chromadb
        db_path = tmp_path / "shrink_test"
        client = chromadb.PersistentClient(path=str(db_path))
        col = client.get_or_create_collection(
            name="shrink_col", embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

        # First run: a long document that produces multiple chunks.
        # 600 repetitions of a ~6-word phrase ~= ~3600 words ~= many chunks at 220 tok.
        long_text = "Knowledge base documents for retrieval augmented generation. " * 200
        md = make_md(tmp_path, long_text, name="shrink.md")
        r1 = ing.ingest_document(md, col, embedding_model, sample_max_chunks=5)
        assert r1["status"] == "ok", f"First run failed: {r1}"
        first_count = col.count()
        assert first_count == 5, (
            f"Expected 5 chunks after first run, got {first_count}. "
            "Text may be too short; increase repetitions."
        )

        # Second run: overwrite with short text that produces at most 2 chunks.
        short_text = "Short replacement. " * 8
        md.write_text(short_text, encoding="utf-8")
        r2 = ing.ingest_document(md, col, embedding_model, sample_max_chunks=2)
        assert r2["status"] == "ok", f"Second run failed: {r2}"

        final_count = col.count()
        assert final_count <= 2, (
            f"Expected <=2 records after shrink, got {final_count} "
            "(stale chunks not purged)"
        )
