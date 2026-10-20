# RAG Document Assistant

A lightweight **Retrieval-Augmented Generation (RAG)** system built as a college assignment.
It supports PDF, Markdown, and text documents, and answers natural-language questions
using semantic search and the Google Gemini API. The Streamlit app shows retrieved
chunks and source references, and can compare a grounded answer with a plain Gemini
answer to the same question.

---

## Architecture Overview

```
User Question
     │
     ▼
┌─────────────┐     embed query      ┌──────────────────┐
│  retrieval  │ ──────────────────►  │  ChromaDB        │
│  .py        │ ◄──── top-K chunks ─ │  (chroma_data/)  │
└─────────────┘                      └──────────────────┘
     │                                        ▲
     │ chunks + question                      │ store embeddings
     ▼                                        │
┌─────────────┐                       ┌───────────────┐
│ generation  │                       │   ingest.py   │
│ .py         │                       │               │
│ (Gemini)    │                       │  PDF/MD/TXT   │
└─────────────┘                       │  → chunks     │
     │                                │  → embed      │
     ▼                                └───────────────┘
 Grounded Answer                             ▲
                                      ┌──────────────┐
                                      │  chunking.py │
                                      └──────────────┘
```

---

## Project Structure

```
RAG System/
├── app.py            # Streamlit web UI (3 tabs)
├── ingest.py         # Incremental PDF/Markdown/text ingestion pipeline
├── retrieval.py      # Semantic search against ChromaDB
├── generation.py     # Gemini answer generation
├── chunking.py       # Text chunking logic
├── config.py         # Central settings & path management
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
├── .gitignore        # Git exclusion rules
├── data/
│   └── documents/    # Place PDF, Markdown, or text source documents here
├── chroma_data/      # ChromaDB persistent storage (auto-created)
└── tests/            # Unit tests
```

---

## Setup

### 1. Prerequisites
- Python **3.11** or **3.12**
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 2. Create and activate the virtual environment
```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables
```powershell
Copy-Item .env.example .env
# Edit .env and replace the GEMINI_API_KEY placeholder with your actual key
```

`config.py` loads the project-root `.env`. `.env` is listed in `.gitignore`;
never paste an API key into Python code or commit the real `.env` file. Without
a key, ingestion and retrieval still work, but Gemini answer generation does not.

---

## Usage

### Run the web application
```powershell
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

### Ingest documents via CLI
```powershell
# Dry-run loading and chunking first; this does not write to ChromaDB
python pipeline.py

# Ingest supported files in data/documents/
python ingest.py

# Ingest one supported file
python ingest.py data\documents\research_paper.pdf
python ingest.py data\documents\notes.md
python ingest.py data\documents\notes.txt

# Read-only check of the existing persisted collection
python ingest.py --verify
```

Ingestion reuses one CPU embedding model, encodes small batches of 8 chunks,
skips files whose hashes have not changed, and updates only changed sources.
Do not start a large ingestion run until the dry run, source volume, and available
RAM have been reviewed. `--sample` writes sample chunks into the current database;
it is not a read-only test.

### Query from the command line
```powershell
python retrieval.py "What is the main contribution of the paper?"
```

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | CPU embedding model (~90 MB) |
| `EMBEDDING_BATCH_SIZE` | `8` | Small batch → low memory usage |
| `CHUNK_TOKENS` | `220` tokens | Size of each text chunk (below model limit) |
| `OVERLAP_TOKENS` | `45` tokens | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Supported text-generation model; override via `.env` |

### Chunking and embedding limit

The project uses **220-token chunks** with a **45-token overlap**. The
`all-MiniLM-L6-v2` embedding model has a configured maximum sequence length of
**256 tokens**; larger chunks could be truncated when embedded. The overlap
preserves some context between adjacent chunks.

---

## Key Design Decisions

- **No LangChain or LlamaIndex** — each component is implemented directly for learning value.
- **CPU-only embeddings** — `all-MiniLM-L6-v2` with PyTorch CPU wheels. No GPU required.
- **Cloud generation** — Gemini API handles the LLM step; no large model is downloaded.
- **Idempotent ingestion** — ChromaDB `upsert` means re-running `ingest.py` is safe.
- **Pathlib throughout** — all paths constructed with `pathlib.Path` for Windows compatibility.
- **Singleton model** — the embedding model is loaded once per process and reused.
- **Secrets via `.env`** — `GEMINI_API_KEY` is never hardcoded; `.env` is git-ignored.

---

## Dependencies

| Package | Purpose |
|---|---|
| `chromadb` | Vector database with persistent local storage |
| `sentence-transformers` | CPU sentence embeddings |
| `pypdf` | PDF text extraction |
| `google-genai` | Google Gemini API client |
| `streamlit` | Web UI |
| `python-dotenv` | Load `.env` environment variables |

---

## RAG vs. Plain LLM Evaluation

**Question:** “What embedding model does my RAG overview document describe, and what is its embedding dimension?”

- **RAG answer:** The document describes the `all-minilm-l6-v2` embedding model, which produces 384-dimensional dense vectors.
- **Source citation:** `rag_overview.md`
- **Plain LLM answer:** Gemini asked the user to upload or paste the document and could not identify the model or dimension without access to it.

The RAG answer used retrieved context from the local document. The plain LLM answer received the same question without that local-document context, so it could not provide the document-specific details.

---

## Tests

Run the lightweight suite from the project root:

```powershell
python -m pytest tests -q -m "not slow"
```

The focused tests use isolated temporary ChromaDB directories and fake
embeddings; generation tests mock Gemini, so these checks do not call the
Gemini service or modify `chroma_data/`. They cover chunking, PDF/Markdown/text
loading, retrieval ranking and source metadata, persistence across a database
reopen, a small document-ingestion path, and grounded prompt construction.
Tests marked `slow` load the real embedding model and are intentionally excluded
from this safe lightweight command.

Verification run in the current workspace on 2026-10-20:
`python -m pytest tests -q -m "not slow"` — **107 passed, 16 deselected**.
The only warning was an upstream Hugging Face `resume_download` deprecation
warning; no test failed.

## Current Local Documents and Database

These counts describe the **current local workspace**: four PDF sources totaling
31 physical pages, plus one eligible Markdown source, `rag_overview.md`, for
five eligible sources in total. The folder's placeholder `README.md` is excluded
from ingestion. The local persistent ChromaDB collection contains **65 chunks
after ingestion**.

`chroma_data/` is excluded from Git. If the database is not included in a fresh
checkout, rebuild it by placing the source documents in `data/documents/` and
running `python ingest.py`.

## Assignment Notes

This project demonstrates:
1. **Document ingestion** — PDF parsing, chunking, embedding, indexed storage
2. **Semantic retrieval** — cosine-similarity search over embedded chunks
3. **Grounded generation** — Gemini answers from retrieved context only
4. **RAG vs plain LLM comparison** — side-by-side in the UI
5. **Evaluation** — see `tests/` for unit tests and manual evaluation results
