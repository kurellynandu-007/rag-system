"""Small end-to-end ingestion test using a fake tokenizer and embedding model."""

import gc
import sys
from pathlib import Path

import chromadb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chunking
import ingest


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


def test_markdown_ingestion_persists_chunks_without_real_model_or_database(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(chunking, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(ingest, "check_memory_safe", lambda *args, **kwargs: True)

    document = tmp_path / "milestone.md"
    document.write_text(
        "# Retrieval\n\nSmall test document for the RAG ingestion pipeline.",
        encoding="utf-8",
    )
    db_path = tmp_path / "isolated-chroma"
    first_client = chromadb.PersistentClient(path=str(db_path))
    first_collection = first_client.get_or_create_collection(
        name="pipeline_test",
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    result = ingest.ingest_document(
        document, first_collection, FakeEmbeddingModel()
    )
    expected_count = first_collection.count()

    assert result["status"] == "ok"
    assert result["pages"] == 1
    assert expected_count == result["chunks_stored"] > 0

    del first_collection, first_client
    gc.collect()
    second_client = chromadb.PersistentClient(path=str(db_path))
    reopened = second_client.get_collection(
        name="pipeline_test", embedding_function=None
    )

    assert reopened.count() == expected_count
    stored = reopened.get(include=["documents", "metadatas", "embeddings"])
    assert stored["documents"]
    assert stored["metadatas"][0]["source"] == "milestone.md"
    assert len(stored["embeddings"][0]) == 384
