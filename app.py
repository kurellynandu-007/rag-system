"""
app.py
------
Streamlit web interface for the RAG System.

Tabs
----
1. Ask a Question  - Enter a query, see retrieved chunks and the RAG answer,
                     with an optional side-by-side plain LLM comparison.
2. Ingest Documents - Upload PDF, Markdown, or text files, or ingest all
                      supported files already in data/documents/.
3. Knowledge Base   - Browse ingested documents and collection stats.

How to run:
    .venv\\Scripts\\activate
    streamlit run app.py

Environment required:
    GEMINI_API_KEY=<your key>    (in .env file or shell)

Streamlit session state is used to cache the ChromaDB collection handle
and the embedding model across reruns, so neither is reloaded on every
user interaction.
"""

import logging
import tempfile
from pathlib import Path

import streamlit as st

import config
from generation import generate_plain_answer, generate_rag_answer
from ingest import get_chroma_collection, get_embedding_model, ingest_document
from loader import SUPPORTED_EXTENSIONS
from retrieval import retrieve_chunks

logger = logging.getLogger(__name__)

# ── Page & app configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session-state helpers ─────────────────────────────────────────────────────

def get_collection():
    """Cache ChromaDB collection in session state (opened once per session)."""
    if "collection" not in st.session_state:
        with st.spinner("Connecting to ChromaDB …"):
            st.session_state["collection"] = get_chroma_collection()
    return st.session_state["collection"]


def load_model_once():
    """Load the embedding model once and cache in session state."""
    if "model_loaded" not in st.session_state:
        with st.spinner("Loading embedding model (first load may take a moment) …"):
            get_embedding_model()
            st.session_state["model_loaded"] = True
    return get_embedding_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("📚 RAG Assistant")
        st.markdown("---")

        # API-key status
        st.subheader("🔑 API Key Status")
        if config.GEMINI_API_KEY:
            st.success("Gemini API key is set ✓")
        else:
            st.error("GEMINI_API_KEY not found.\nAdd it to your .env file.")

        # Collection stats
        st.subheader("🗄️ Knowledge Base")
        try:
            col = get_collection()
            count = col.count()
            st.metric("Stored chunks", count)
        except Exception as e:
            st.warning(f"Could not read ChromaDB: {e}")

        st.markdown("---")
        st.caption(
            f"Model: `{config.EMBEDDING_MODEL_NAME}`\n\n"
            f"Gemini: `{config.GEMINI_MODEL_NAME}`\n\n"
            f"Top-K: `{config.TOP_K_RESULTS}`"
        )


# ── Tab 1: Ask a Question ─────────────────────────────────────────────────────

def tab_ask():
    st.header("Ask a Question")
    st.markdown(
        "Type a question below. The system will retrieve relevant passages "
        "from your documents and generate a grounded answer."
    )

    question = st.text_area(
        "Your question",
        placeholder="e.g. What are the main findings of the paper?",
        height=100,
        key="question_input",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        top_k = st.slider("Number of context chunks", min_value=1, max_value=10,
                          value=config.TOP_K_RESULTS, key="top_k")
    with col2:
        show_comparison = st.checkbox(
            "Show plain LLM answer (no context) for comparison",
            value=True,
            key="show_comparison",
        )

    run_button = st.button("🔍 Search & Answer", type="primary", key="run_btn")

    if run_button:
        if not question.strip():
            st.warning("Please enter a question.")
            return

        if not config.GEMINI_API_KEY:
            st.error(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file and restart the app."
            )
            return

        collection = get_collection()

        # ── Retrieval ──
        with st.spinner("Searching knowledge base …"):
            chunks = retrieve_chunks(question, top_k=top_k, collection=collection)

        if not chunks:
            st.warning(
                "No relevant chunks found. "
                "Make sure documents are ingested (see the **Ingest Documents** tab)."
            )
            return

        # ── Show retrieved context ──
        with st.expander("📄 Retrieved Context Chunks", expanded=False):
            for i, chunk in enumerate(chunks, 1):
                page_info = f" — page {chunk.page_num}" if chunk.page_num else ""
                st.markdown(
                    f"**[{i}] {chunk.source}{page_info}** "
                    f"(similarity: `{chunk.score:.4f}`)"
                )
                st.text(chunk.text[:600] + ("…" if len(chunk.text) > 600 else ""))
                st.divider()

        # ── RAG Answer ──
        with st.spinner("Generating RAG answer …"):
            try:
                rag_answer = generate_rag_answer(question, chunks)
            except RuntimeError as e:
                st.error(f"Generation failed: {e}")
                return

        # ── Plain LLM Answer (for comparison) ──
        plain_answer = None
        if show_comparison:
            with st.spinner("Generating plain LLM answer (no context) …"):
                try:
                    plain_answer = generate_plain_answer(question)
                except RuntimeError as e:
                    st.error(f"Plain LLM generation failed: {e}")

        if plain_answer:
            rag_column, plain_column = st.columns(2)
            with rag_column:
                st.subheader("🤖 RAG Answer (grounded in your documents)")
                st.markdown(rag_answer)
            with plain_column:
                st.subheader("💬 Plain LLM Answer (no document context)")
                st.markdown(plain_answer)
            st.caption(
                "Both answers use the same question. Compare the grounded RAG "
                "answer with Gemini's answer without document context."
            )
        else:
            st.subheader("🤖 RAG Answer (grounded in your documents)")
            st.markdown(rag_answer)

        st.subheader("Sources")
        for i, chunk in enumerate(chunks, 1):
            page_info = f", page {chunk.page_num}" if chunk.page_num else ""
            st.markdown(f"[{i}] `{chunk.source}`{page_info}")


# ── Tab 2: Ingest Documents ───────────────────────────────────────────────────

def tab_ingest():
    st.header("Ingest Documents")
    st.markdown(
        "Upload PDF, Markdown, or text files here **or** place them in "
        f"`data/documents/` and click **Ingest all files**."
    )

    # ── Upload section ──
    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, Markdown, or text files",
        type=sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        key="uploader",
    )

    if uploaded_files and st.button("📥 Ingest uploaded files", key="ingest_upload"):
        model = load_model_once()
        collection = get_collection()
        results = []

        for uploaded in uploaded_files:
            filename = Path(uploaded.name).name
            extension = Path(filename).suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                st.error(
                    f"Unsupported file type for {filename!r}. "
                    "Choose a PDF, Markdown, or text file."
                )
                continue

            dest = config.DOCUMENTS_DIR / filename
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=extension, delete=False, dir=config.DOCUMENTS_DIR
                ) as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = Path(tmp.name)
                tmp_path.replace(dest)
                with st.spinner(f"Ingesting {uploaded.name} …"):
                    result = ingest_document(dest, collection, model)
            except Exception as e:
                st.error(f"❌ {filename}: {e}")
                continue
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)

            results.append(result)

        # Show results
        for r in results:
            if r.get("error"):
                st.error(f"❌ {r['source']}: {r['error']}")
            elif r["skipped"]:
                st.info(
                    f"ℹ️ {r['source']} — unchanged; "
                    f"{r['chunks_stored']} existing chunks reused."
                )
            else:
                st.success(
                    f"✅ {r['source']} — "
                    f"{r['pages']} pages, "
                    f"{r['chunks_stored']} chunks stored."
                )

        # Update sidebar counter
        st.session_state.pop("collection", None)  # force re-open to refresh count

    st.divider()

    # ── Ingest from directory ──
    st.subheader("Ingest all files in data/documents/")
    document_files = sorted(
        path for path in config.DOCUMENTS_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and path.name != "README.md"
    )
    if document_files:
        st.write(f"Found **{len(document_files)}** supported document(s):")
        for p in document_files:
            st.write(f"  • `{p.name}`")
    else:
        st.info("No supported documents found in `data/documents/`. Upload some above.")

    if document_files and st.button("📂 Ingest all from data/documents/", key="ingest_dir"):
        model = load_model_once()
        collection = get_collection()
        for document_path in document_files:
            with st.spinner(f"Ingesting {document_path.name} …"):
                try:
                    result = ingest_document(document_path, collection, model)
                except Exception as e:
                    st.error(f"❌ {document_path.name}: {e}")
                    continue

            if result.get("error"):
                st.error(f"❌ {result['source']}: {result['error']}")
            elif result["skipped"]:
                st.info(
                    f"ℹ️ {result['source']} — unchanged; "
                    f"{result['chunks_stored']} existing chunks reused."
                )
            else:
                st.success(
                    f"✅ {result['source']} — "
                    f"{result['pages']} pages, "
                    f"{result['chunks_stored']} chunks stored."
                )

        st.session_state.pop("collection", None)


# ── Tab 3: Knowledge Base ─────────────────────────────────────────────────────

def tab_knowledge_base():
    st.header("Knowledge Base")
    st.markdown("Browse the documents and chunks currently stored in ChromaDB.")

    try:
        collection = get_collection()
        total = collection.count()
    except Exception as e:
        st.error(f"Cannot connect to ChromaDB: {e}")
        return

    st.metric("Total chunks in database", total)

    if total == 0:
        st.info("The knowledge base is empty. Go to **Ingest Documents** to add files.")
        return

    # Fetch a sample to discover unique sources
    try:
        sample = collection.get(limit=min(total, 5000), include=["metadatas"])
        metadatas = sample.get("metadatas", [])
        sources = sorted({m.get("source", "unknown") for m in metadatas})
    except Exception as e:
        st.warning(f"Could not fetch metadata: {e}")
        sources = []

    st.subheader("Ingested Documents")
    for src in sources:
        count = sum(1 for m in metadatas if m.get("source") == src)
        st.write(f"  • `{src}` — **{count}** chunks")

    st.divider()
    st.subheader("Sample chunks (first 5)")
    try:
        peek = collection.get(limit=5, include=["documents", "metadatas"])
        for i, (doc, meta) in enumerate(
            zip(peek["documents"], peek["metadatas"]), 1
        ):
            with st.expander(
                f"Chunk {i} — {meta.get('source', '?')} "
                f"(page {meta.get('page_num') or 'N/A'})"
            ):
                st.text(doc[:800])
    except Exception as e:
        st.warning(f"Could not load sample chunks: {e}")


# ── Main layout ───────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    tab1, tab2, tab3 = st.tabs(
        ["🔍 Ask a Question", "📥 Ingest Documents", "🗄️ Knowledge Base"]
    )

    with tab1:
        tab_ask()
    with tab2:
        tab_ingest()
    with tab3:
        tab_knowledge_base()


if __name__ == "__main__":
    main()
