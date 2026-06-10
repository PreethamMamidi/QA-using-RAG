import json
import streamlit as st
import sys
import os
import tempfile
import shutil
import datetime
import re

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

STORAGE_DIR = os.path.join(ROOT_DIR, "storage")
RAW_DOCS_DIR = os.path.join(STORAGE_DIR, "raw_docs")


def _slugify_document_name(filename: str) -> str:
    stem = os.path.splitext(filename)[0].lower()
    slug = re.sub(r"[^a-z0-9]+", "", stem)
    return slug or "document"


def _split_source_document_id(source_document_id: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*)_page_(\d+)$", source_document_id)
    if not match:
        return source_document_id, None
    return match.group(1), int(match.group(2))


def _build_storage_schema(uploaded_files, docs):
    uploaded_at = datetime.date.today().isoformat()
    docs_by_filename = {}
    for item in docs:
        source_name, page_number = _split_source_document_id(item["document_id"])
        docs_by_filename.setdefault(source_name, []).append((page_number, item))

    used_ids = set()
    documents = []
    chunks = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        base_id = _slugify_document_name(filename)
        document_id = base_id
        suffix = 2
        while document_id in used_ids:
            document_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(document_id)

        source_pages = sorted(
            docs_by_filename.get(filename, []),
            key=lambda pair: pair[0] if pair[0] is not None else 0,
        )
        total_pages = len(source_pages) if source_pages else 1
        documents.append(
            {
                "document_id": document_id,
                "filename": filename,
                "uploaded_at": uploaded_at,
                "total_pages": total_pages,
            }
        )

        if not source_pages:
            continue

        for page_number, source_doc in source_pages:
            page_label = page_number if page_number is not None else 1
            page_chunks = chunk_text(
                clean_text(source_doc["text"]),
                document_id=document_id,
                chunk_size=180 if filename.lower().endswith(".pdf") else 280,
                overlap=60 if filename.lower().endswith(".pdf") else 80,
            )
            for chunk_index, chunk in enumerate(page_chunks):
                chunk["chunk_id"] = f"{document_id}_p{page_label}_c{chunk_index}"
                chunk["document_id"] = document_id
                chunk["filename"] = filename
                chunk["page"] = page_label
                chunks.append(chunk)

    return documents, chunks


def _derive_documents_from_chunks(chunks):
    documents = {}
    for chunk in chunks or []:
        document_id = chunk.get("document_id")
        if not document_id:
            continue
        document = documents.setdefault(
            document_id,
            {
                "document_id": document_id,
                "filename": chunk.get("filename") or document_id,
                "uploaded_at": None,
                "total_pages": 0,
            },
        )
        document["total_pages"] = max(document["total_pages"], int(chunk.get("page") or 1))
    return list(documents.values())


def _preview_rankings(results, limit=10):
    preview = []
    for rank, item in enumerate((results or [])[:limit], start=1):
        preview.append(
            {
                "rank": rank,
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "page": item.get("page"),
                "score": item.get("score"),
                "dense_score": item.get("dense_score"),
                "bm25_score": item.get("bm25_score"),
                "rrf_score": item.get("rrf_score"),
                "retrieval": item.get("retrieval"),
            }
        )
    return preview


def _refresh_hybrid_retriever():
    if st.session_state.index is None or not st.session_state.chunks:
        st.session_state.hybrid_retriever = None
        return

    st.session_state.hybrid_retriever = HybridRetriever.from_chunks(
        st.session_state.index,
        st.session_state.chunks,
    )

from ingestion.loader import load_documents
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from embeddings.embedder import embed_texts
from vector_store.faiss_index import build_index
from retrieval.retriever import retrieve_chunks
from retrieval.reranker import rerank_chunks
from retrieval.hybrid_retrieval import HybridRetriever
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

enable_hybrid_retrieval = st.sidebar.checkbox("Enable hybrid retrieval", value=True)
top_k_dense = st.sidebar.slider("Dense retrieval top_k", min_value=1, max_value=50, value=20 if use_reranker else 8)
top_k_sparse = st.sidebar.slider("Sparse retrieval top_k", min_value=1, max_value=50, value=20 if use_reranker else 8)
top_k_fused = st.sidebar.slider("Fused retrieval top_k", min_value=1, max_value=25, value=10 if use_reranker else 8)
rrf_k = st.sidebar.slider("RRF constant", min_value=1, max_value=200, value=60)
show_retrieval_debug = st.sidebar.checkbox("Show retrieval debug info", value=False)

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
if "documents" not in st.session_state:
    st.session_state.documents = None
if "hybrid_retriever" not in st.session_state:
    st.session_state.hybrid_retriever = None

documents_path = os.path.join(STORAGE_DIR, "documents.json")
if (
    st.session_state.index is None
    and os.path.exists(os.path.join(STORAGE_DIR, "faiss.index"))
    and os.path.exists(os.path.join(STORAGE_DIR, "chunks.json"))
):
    import faiss
    with open(os.path.join(STORAGE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
        st.session_state.chunks = json.load(f)

    if os.path.exists(documents_path):
        with open(documents_path, "r", encoding="utf-8") as f:
            st.session_state.documents = json.load(f)
    else:
        st.session_state.documents = _derive_documents_from_chunks(st.session_state.chunks)

    st.session_state.index = faiss.read_index(os.path.join(STORAGE_DIR, "faiss.index"))
    _refresh_hybrid_retriever()

    st.sidebar.success("Loaded saved index ✅")

if reset:
    # Clear in-memory state
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.stats = None
    st.session_state.documents = None
    st.session_state.hybrid_retriever = None

    # Delete saved files (if they exist)
    if os.path.exists(os.path.join(STORAGE_DIR, "faiss.index")):
        os.remove(os.path.join(STORAGE_DIR, "faiss.index"))
    if os.path.exists(os.path.join(STORAGE_DIR, "chunks.json")):
        os.remove(os.path.join(STORAGE_DIR, "chunks.json"))
    if os.path.exists(documents_path):
        os.remove(documents_path)
    if os.path.exists(RAW_DOCS_DIR):
        shutil.rmtree(RAW_DOCS_DIR)

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

            os.makedirs(RAW_DOCS_DIR, exist_ok=True)
            for f in uploaded_files:
                raw_path = os.path.join(RAW_DOCS_DIR, f.name)
                with open(raw_path, "wb") as out:
                    out.write(f.getvalue())

            # Load docs from temp folder
            docs = load_documents(tmpdir)

        document_records, all_chunks = _build_storage_schema(uploaded_files, docs)

        embeddings = embed_texts([c["text"] for c in all_chunks])
        index = build_index(embeddings)

        st.session_state.chunks = all_chunks
        st.session_state.index = index
        st.session_state.documents = document_records
        _refresh_hybrid_retriever()
        # ✅ Save index + chunks to disk
        os.makedirs(STORAGE_DIR, exist_ok=True)

# Save FAISS index
        import faiss
        faiss.write_index(index, os.path.join(STORAGE_DIR, "faiss.index"))

# Save document metadata
        with open(documents_path, "w", encoding="utf-8") as f:
            json.dump(document_records, f, ensure_ascii=False, indent=2)

# Save chunks metadata
        with open(os.path.join(STORAGE_DIR, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        st.sidebar.success("Saved documents and index to storage/ ✅")

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

        retrieval_query = rewritten_query or query
        retrieval_debug = None

        if enable_hybrid_retrieval:
            if st.session_state.hybrid_retriever is None:
                _refresh_hybrid_retriever()

            if show_retrieval_debug:
                initial, retrieval_debug = st.session_state.hybrid_retriever.retrieve(
                    retrieval_query,
                    top_k_dense=top_k_dense,
                    top_k_sparse=top_k_sparse,
                    top_k_fused=top_k_fused,
                    rrf_k=rrf_k,
                    debug=True,
                    return_debug=True,
                )
            else:
                initial = st.session_state.hybrid_retriever.retrieve(
                    retrieval_query,
                    top_k_dense=top_k_dense,
                    top_k_sparse=top_k_sparse,
                    top_k_fused=top_k_fused,
                    rrf_k=rrf_k,
                    debug=False,
                    return_debug=False,
                )
        else:
            initial = retrieve_chunks(
                retrieval_query,
                st.session_state.index,
                st.session_state.chunks,
                top_k=top_k_dense,
            )
            if show_retrieval_debug:
                retrieval_debug = {
                    "dense_results": initial,
                    "sparse_results": [],
                    "fused_results": initial,
                }

        retrieved = rerank_chunks(query, initial, top_k=5) if use_reranker else initial[:8]
        st.caption(f"Original query: {query}")
        st.caption(f"Rewritten query: {rewritten_query or query}")

        if show_retrieval_debug and retrieval_debug is not None:
            retrieval_debug["reranked_results"] = retrieved
            with st.sidebar.expander("Retrieval debug info", expanded=False):
                st.markdown("**Dense retrieval**")
                st.json(_preview_rankings(retrieval_debug.get("dense_results", [])))
                st.markdown("**Sparse retrieval**")
                st.json(_preview_rankings(retrieval_debug.get("sparse_results", [])))
                st.markdown("**Fused retrieval**")
                st.json(_preview_rankings(retrieval_debug.get("fused_results", [])))
                st.markdown("**Reranked results**")
                st.json(_preview_rankings(retrieval_debug.get("reranked_results", [])))

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
            source_name = r.get("filename") or r.get("document_id")
            page_number = r.get("page")
            if page_number is not None:
                source_name = f"{source_name} (page {page_number})"
            st.markdown(
                f"- **{source_name}** (score={r['score']:.3f})"
            )

        with st.expander("🔍 View Retrieved Context (Top Matches)"):
            for i, r in enumerate(retrieved, start=1):
                st.markdown(f"### Chunk {i}")
                st.markdown(f"**Document:** `{r.get('filename', r['document_id'])}`")
                st.markdown(f"**Document ID:** `{r['document_id']}`")
                if r.get("page") is not None:
                    st.markdown(f"**Page:** `{r['page']}`")
                st.markdown(f"**Score:** `{r['score']:.3f}`")
                st.text_area(
                label=f"Chunk Text {i}",
                value=r["text"],
                height=150
          )
else:
    st.info("Upload documents and click **Process documents** to begin.")
