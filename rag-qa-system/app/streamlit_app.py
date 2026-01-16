import json
import streamlit as st
import sys
import os
import tempfile

# Optional .env loading
try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
    DOTENV_LOADED = True
except ImportError:
    DOTENV_LOADED = False

# --- Make project imports work ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ingestion.loader import load_documents
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from embeddings.embedder import embed_texts
from vector_store.faiss_index import build_index
from retrieval.retriever import retrieve_chunks
from retrieval.reranker import rerank_chunks
from retrieval.query_rewrite import rewrite_query_groq
from generation.generator import generate_answer
from generation.groq_generator import generate_answer_groq


st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("📄 Retrieval-Augmented Question Answering")

# ---------------- Sidebar: Upload ----------------
st.sidebar.header("Upload documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload TXT or PDF files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)
process = st.sidebar.button("✅ Process documents")
reset = st.sidebar.button("🔄 Reset system")
st.sidebar.divider()

generator_choice = st.sidebar.selectbox(
    "Answer Generator",
    ["Groq (LLM API)", "Local (FLAN-T5)"],
    index=0,
)

groq_model = None
if generator_choice == "Groq (LLM API)":
    groq_model = st.sidebar.selectbox(
        "Groq Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        index=0,
    )

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    msg = "GROQ_API_KEY is not set. Add it to your environment or a .env file."
    if not DOTENV_LOADED:
        msg += " Install python-dotenv to auto-load .env files."
    st.sidebar.warning(msg)

if generator_choice == "Groq (LLM API)" and not GROQ_API_KEY:
    st.sidebar.error("Missing GROQ_API_KEY")

# Retrieval options
use_reranker = st.sidebar.checkbox("Use reranker (slower, better)", value=False)

rewrite_mode = st.sidebar.selectbox(
    "Query rewrite mode",
    ["general", "medical"],
    index=0,
)

if "stats" in st.session_state and st.session_state.stats:
    st.sidebar.subheader("📊 Processing Summary")
    st.sidebar.write(f"📁 Files uploaded: {st.session_state.stats['files_uploaded']}")
    st.sidebar.write(f"📄 Pages/Docs loaded: {st.session_state.stats['docs_loaded']}")
    st.sidebar.write(f"🧩 Chunks created: {st.session_state.stats['chunks_created']}")

# ---------------- Session state ----------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None

if (
    st.session_state.index is None
    and os.path.exists("storage/faiss.index")
    and os.path.exists("storage/chunks.json")
):
    import faiss
    with open("storage/chunks.json", "r", encoding="utf-8") as f:
        st.session_state.chunks = json.load(f)

    st.session_state.index = faiss.read_index("storage/faiss.index")

    st.sidebar.success("Loaded saved index ✅")

if reset:
    # Clear in-memory state
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.stats = None

    # Delete saved files (if they exist)
    if os.path.exists("storage/faiss.index"):
        os.remove("storage/faiss.index")
    if os.path.exists("storage/chunks.json"):
        os.remove("storage/chunks.json")

    st.sidebar.success("System reset! Saved index deleted ✅")
    st.stop()

# ---------------- Process Uploaded Files ----------------
if process and uploaded_files:
    with st.spinner("Processing documents..."):
        all_chunks = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded files temporarily
            for f in uploaded_files:
                path = os.path.join(tmpdir, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())

            # Load docs from temp folder
            docs = load_documents(tmpdir)

            for d in docs:
                text = clean_text(d["text"])
                doc_id = d["document_id"]

                if ".pdf_page_" in doc_id.lower():
                    all_chunks.extend(chunk_text(text, doc_id, chunk_size=180, overlap=60))
                else:
                    all_chunks.extend(chunk_text(text, doc_id, chunk_size=280, overlap=80))

        embeddings = embed_texts([c["text"] for c in all_chunks])
        index = build_index(embeddings)

        st.session_state.chunks = all_chunks
        st.session_state.index = index
        # ✅ Save index + chunks to disk
        os.makedirs("storage", exist_ok=True)

# Save FAISS index
        import faiss
        faiss.write_index(index, "storage/faiss.index")

# Save chunks metadata
        with open("storage/chunks.json", "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        st.sidebar.success("Saved index to storage/ ✅")

        st.session_state.stats = {
            "files_uploaded": len(uploaded_files),
            "docs_loaded": len(docs),        # pdf pages count here
            "chunks_created": len(all_chunks)
        }

        st.success(f"Processed {len(all_chunks)} chunks")

# ---------------- Query UI ----------------
if st.session_state.index is not None:
    query = st.text_input("Ask a question")

    if query:
        rewritten_query = rewrite_query_groq(query, mode=rewrite_mode)
        with st.sidebar.expander("Rewritten retrieval query", expanded=False):
            st.write(rewritten_query or "(empty)")

        initial = retrieve_chunks(
            rewritten_query or query,
            st.session_state.index,
            st.session_state.chunks,
            top_k=20 if use_reranker else 8,
        )

        retrieved = rerank_chunks(query, initial, top_k=5) if use_reranker else initial[:8]
        st.caption(f"Original query: {query}")
        st.caption(f"Rewritten query: {rewritten_query or query}")

        if generator_choice == "Groq (LLM API)":
            if not GROQ_API_KEY:
                st.stop()
            answer = generate_answer_groq(query, retrieved, model=groq_model)
        else:
            answer = generate_answer(query, retrieved)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for r in retrieved:
            st.markdown(
                f"- **{r['document_id']}** (score={r['score']:.3f})"
            )

        with st.expander("🔍 View Retrieved Context (Top Matches)"):
            for i, r in enumerate(retrieved, start=1):
                st.markdown(f"### Chunk {i}")
                st.markdown(f"**Document:** `{r['document_id']}`")
                st.markdown(f"**Score:** `{r['score']:.3f}`")
                st.text_area(
                label=f"Chunk Text {i}",
                value=r["text"],
                height=150
          )
else:
    st.info("Upload documents and click **Process documents** to begin.")
