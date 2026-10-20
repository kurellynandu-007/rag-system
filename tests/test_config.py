"""
tests/test_config.py
---------------------
Unit tests for the config module.

Run with:
    .venv\\Scripts\\activate
    python -m pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


class TestConfigPaths:
    """Verify that config paths exist and are correct types."""

    def test_project_root_is_directory(self):
        assert config.PROJECT_ROOT.is_dir()

    def test_documents_dir_created(self):
        assert config.DOCUMENTS_DIR.exists()
        assert config.DOCUMENTS_DIR.is_dir()

    def test_chroma_dir_created(self):
        assert config.CHROMA_DIR.exists()
        assert config.CHROMA_DIR.is_dir()

    def test_documents_dir_inside_project(self):
        # data/documents must be a subdirectory of the project root
        assert config.DOCUMENTS_DIR.is_relative_to(config.PROJECT_ROOT)

    def test_chroma_dir_inside_project(self):
        assert config.CHROMA_DIR.is_relative_to(config.PROJECT_ROOT)


class TestConfigValues:
    """Verify that configuration constants have sensible values."""

    def test_chunk_size_positive(self):
        assert config.CHUNK_SIZE > 0

    def test_chunk_overlap_less_than_chunk_size(self):
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

    def test_embedding_dimension_is_384(self):
        # all-MiniLM-L6-v2 always produces 384-dimensional vectors
        assert config.EMBEDDING_DIMENSION == 384

    def test_embedding_batch_size_positive(self):
        assert config.EMBEDDING_BATCH_SIZE > 0

    def test_top_k_positive(self):
        assert config.TOP_K_RESULTS > 0

    def test_collection_name_nonempty(self):
        assert isinstance(config.CHROMA_COLLECTION_NAME, str)
        assert len(config.CHROMA_COLLECTION_NAME) > 0

    def test_embedding_model_name_nonempty(self):
        assert isinstance(config.EMBEDDING_MODEL_NAME, str)
        assert len(config.EMBEDDING_MODEL_NAME) > 0


class TestValidateApiKey:
    """Tests for the validate_api_key() helper."""

    def test_raises_when_key_missing(self, monkeypatch):
        monkeypatch.setattr(config, "GEMINI_API_KEY", "")
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            config.validate_api_key()

    def test_does_not_raise_when_key_present(self, monkeypatch):
        monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-test-key-12345")
        config.validate_api_key()   # should not raise
