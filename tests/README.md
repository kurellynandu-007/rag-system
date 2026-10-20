# tests/ — Unit Tests for the RAG System
#
# Run all tests from the project root:
#   .venv\Scripts\activate
#   python -m pytest tests/ -v
#
# Files:
#   test_chunking.py          — Token-aware chunking and chunk metadata
#   test_config.py            — Paths, constants, and API-key validation
#   test_generation.py        — Grounding prompt checks with Gemini mocked
#   test_ingest.py            — Incremental storage helpers and slow model tests
#   test_loader.py            — PDF, Markdown, text, and directory loading
#   test_pipeline_integration.py — Markdown ingestion and persistence with fakes
#   test_retrieval.py         — Retrieval ranking and isolated DB persistence
#
# `python -m pytest tests -q -m "not slow"` avoids loading the real embedding
# model and never calls Gemini or writes to the project's chroma_data directory.
# The slow tests use the local sentence-transformers model and can be run
# separately with `python -m pytest tests -q -m slow` when memory allows.
