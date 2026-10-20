"""Isolated retrieval and ChromaDB persistence tests."""

import sys
from pathlib import Path

import chromadb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import retrieval


class FixedEmbeddingModel:
    def encode(self, text, **kwargs):
        vector = np.zeros(384, dtype=np.float32)
        vector[0] = 1.0
        return vector


def test_retrieval_reopens_persisted_collection_and_returns_sources(tmp_path, monkeypatch):
    db_path = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name="retrieval_test",
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    relevant = np.zeros(384, dtype=np.float32)
    relevant[0] = 1.0
    unrelated = np.zeros(384, dtype=np.float32)
    unrelated[1] = 1.0
    collection.upsert(
        ids=["relevant", "unrelated"],
        documents=["Grounded retrieval passage.", "Unrelated passage."],
        embeddings=[relevant.tolist(), unrelated.tolist()],
        metadatas=[
            {"source": "paper.pdf", "chunk_index": 0, "page_num": 2},
            {"source": "notes.md", "chunk_index": 1, "page_num": ""},
        ],
    )

    del collection, client
    reopened_client = chromadb.PersistentClient(path=str(db_path))
    reopened = reopened_client.get_collection(
        name="retrieval_test", embedding_function=None
    )
    monkeypatch.setattr(
        retrieval, "get_embedding_model", lambda: FixedEmbeddingModel()
    )

    results = retrieval.retrieve_chunks("find the grounded passage", top_k=2,
                                        collection=reopened)

    assert [result.text for result in results] == [
        "Grounded retrieval passage.",
        "Unrelated passage.",
    ]
    assert results[0].source == "paper.pdf"
    assert results[0].page_num == 2
    assert results[0].score == pytest.approx(1.0)
    assert results[1].source == "notes.md"
    assert results[1].page_num is None


def test_blank_query_returns_no_results_without_loading_model(monkeypatch):
    def unexpected_model_load():
        raise AssertionError("A blank query should not load the embedding model.")

    monkeypatch.setattr(retrieval, "get_embedding_model", unexpected_model_load)

    assert retrieval.retrieve_chunks("   ") == []


def test_top_k_must_be_positive():
    with pytest.raises(ValueError, match="top_k"):
        retrieval.retrieve_chunks("question", top_k=0)
