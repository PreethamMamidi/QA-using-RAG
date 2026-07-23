import streamlit as st
import sys
import os
import json
import tempfile
import shutil
import datetime
import re
import logging
import stat

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
LOGGER = logging.getLogger(__name__)


def _slugify_document_name(filename: str) -> str:
    stem = os.path.splitext(filename)[0].lower()
    slug = re.sub(r"[^a-z0-9]+", "", stem)
    return slug or "document"


def _split_source_document_id(source_document_id: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*)_page_(\d+)$", source_document_id)
    if not match:
        return source_document_id, None
    return match.group(1), int(match.group(2))


def _build_storage_schema(sources, docs):
    uploaded_at = datetime.date.today().isoformat()
    docs_by_filename = {}
    for item in docs:
        key = item.get("filename") or _split_source_document_id(item["document_id"])[0]
        _, page_number = _split_source_document_id(item["document_id"])
        if page_number is None:
            page_number = item.get("page")
        docs_by_filename.setdefault(key, []).append((page_number, item))

    used_ids = set()
    documents = []
    chunks = []

    for source in sources:
        filename = source["filename"]
        source_type = source.get("source_type", "file")
        source_url = source.get("source_url")
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
                "source_type": source_type,
                "source_url": source_url,
            }
        )

        if not source_pages:
            continue

        chunk_size, overlap = get_chunk_params(source_type, filename)
        use_markdown = source_type == "url" or filename.lower().endswith(
            (".md", ".markdown", ".docx", ".html", ".htm")
        )

        for page_number, source_doc in source_pages:
            page_label = page_number if page_number is not None else 1
            cleaned = clean_text(source_doc["text"])
            if use_markdown:
                page_chunks = chunk_markdown(
                    cleaned,
                    document_id=document_id,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            else:
                page_chunks = chunk_text(
                    cleaned,
                    document_id=document_id,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            for chunk_index, chunk in enumerate(page_chunks):
                chunk["chunk_id"] = f"{document_id}_p{page_label}_c{chunk_index}"
                chunk["document_id"] = document_id
                chunk["filename"] = filename
                chunk["page"] = page_label
                chunks.append(chunk)

    return documents, chunks


def _load_metadata_from_sqlite():
    from database.sqlite_repository import SQLiteMetadataRepository

    repository = SQLiteMetadataRepository()
    documents = repository.list_documents()
    chunks = repository.get_all_chunks()
    print(f"Loaded chunks from SQLite: {len(chunks)}")
    return documents, chunks


def _render_knowledge_base_sidebar():
    from database.sqlite_repository import SQLiteMetadataRepository

    repository = SQLiteMetadataRepository()
    st.sidebar.subheader("Knowledge Base")

    doc_count = repository.get_document_count()
    chunk_count = repository.get_chunk_count()

    col_docs, col_chunks = st.sidebar.columns(2)
    col_docs.metric("Documents", doc_count)
    col_chunks.metric("Chunks", chunk_count)

    if doc_count == 0:
        st.sidebar.caption("No documents indexed yet. Upload and process files to build the knowledge base.")
        return

    for doc in repository.list_documents():
        st.sidebar.markdown(f"**{doc['filename']}**")
        source_type = doc.get("source_type", "file")
        source_url = doc.get("source_url")
        type_label = f" · {source_type}" if source_type != "file" else ""
        st.sidebar.caption(
            f"{doc['upload_time']} · {doc['total_chunks']} chunk"
            f"{'s' if doc['total_chunks'] != 1 else ''}{type_label}"
        )
        if source_url:
            st.sidebar.caption(source_url)


def _render_document_filter_sidebar():
    from database.sqlite_repository import SQLiteMetadataRepository

    repository = SQLiteMetadataRepository()
    documents = repository.list_documents()

    st.sidebar.subheader("Document Filter")
    filter_mode = st.sidebar.radio(
        "Scope",
        ["All Documents", "One document", "Multiple documents"],
        index=0,
    )

    selected_document_ids: list[str] = []
    if filter_mode == "All Documents":
        st.sidebar.caption("Searching across the full knowledge base.")
    elif not documents:
        st.sidebar.caption("No documents available to filter.")
    elif filter_mode == "One document":
        options = {doc["filename"]: doc["document_id"] for doc in documents}
        choice = st.sidebar.selectbox("Document", list(options.keys()))
        selected_document_ids = [options[choice]]
    else:
        options = {doc["filename"]: doc["document_id"] for doc in documents}
        choices = st.sidebar.multiselect("Documents", list(options.keys()))
        selected_document_ids = [options[name] for name in choices]
        if not selected_document_ids:
            st.sidebar.warning("Select at least one document to filter retrieval.")

    active_labels: list[str] = []
    if filter_mode != "All Documents" and selected_document_ids:
        filename_by_id = {doc["document_id"]: doc["filename"] for doc in documents}
        active_labels = [
            filename_by_id.get(document_id, document_id)
            for document_id in selected_document_ids
        ]
        st.sidebar.markdown("**Active filters**")
        for label in active_labels:
            st.sidebar.markdown(f"- {label}")

    return filter_mode, selected_document_ids, active_labels


def _resolve_retrieval_scope(filter_mode, selected_document_ids, all_chunks):
    if filter_mode == "All Documents":
        return None, None, [], False

    if not selected_document_ids:
        return [], [], [], True

    from database.sqlite_repository import SQLiteMetadataRepository

    repository = SQLiteMetadataRepository()
    sqlite_chunks = repository.get_chunks_by_documents(selected_document_ids)
    chunk_id_to_index = {
        chunk["chunk_id"]: idx for idx, chunk in enumerate(all_chunks or [])
    }

    candidate_indices: list[int] = []
    retrieval_chunks = []
    for chunk in sqlite_chunks:
        idx = chunk_id_to_index.get(chunk["chunk_id"])
        if idx is not None:
            candidate_indices.append(idx)
            retrieval_chunks.append(all_chunks[idx])

    documents = repository.list_documents()
    filename_by_id = {doc["document_id"]: doc["filename"] for doc in documents}
    active_labels = [
        filename_by_id.get(document_id, document_id)
        for document_id in selected_document_ids
    ]
    return retrieval_chunks, candidate_indices, active_labels, False


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


def _extract_chunk_index(chunk):
    chunk_id = chunk.get("chunk_id", "")
    match = re.search(r"_c(\d+)$", chunk_id)
    if match:
        return int(match.group(1))
    return None


def _sync_sqlite_metadata(document_records, chunks):
    from database.sqlite_repository import SQLiteMetadataRepository

    try:
        repository = SQLiteMetadataRepository()
        repository.clear_database()

        chunk_totals = {}
        for chunk in chunks:
            document_id = chunk.get("document_id")
            if not document_id:
                continue
            chunk_totals[document_id] = chunk_totals.get(document_id, 0) + 1

        for document in document_records:
            repository.insert_document(
                document_id=document["document_id"],
                filename=document["filename"],
                upload_time=document["uploaded_at"],
                total_chunks=chunk_totals.get(document["document_id"], 0),
                source_type=document.get("source_type", "file"),
                source_url=document.get("source_url"),
            )

        repository.insert_chunks(
            [
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "page": chunk.get("page"),
                    "chunk_index": _extract_chunk_index(chunk),
                    "text": chunk["text"],
                    "section_title": chunk.get("section_title"),
                }
                for chunk in chunks
            ]
        )
    except Exception as exc:
        LOGGER.exception("SQLite metadata sync failed: %s", exc)
        print(f"SQLite metadata sync failed: {exc}")


def _remove_tree_safely(path):
    def _handle_remove_error(func, target_path, exc_info):
        try:
            os.chmod(target_path, stat.S_IWRITE)
            func(target_path)
        except OSError:
            raise exc_info[1]

    try:
        shutil.rmtree(path, onerror=_handle_remove_error)
        return True
    except OSError as exc:
        LOGGER.exception("Failed to remove directory tree %s: %s", path, exc)
        print(f"Failed to remove directory tree {path}: {exc}")
        return False


def _run_full_ingestion(sources, docs, stats_label: str):
    if not docs:
        st.warning("No text could be extracted from the provided source(s).")
        return

    document_records, all_chunks = _build_storage_schema(sources, docs)
    if not all_chunks:
        st.warning("No chunks were created. The source may be empty or OCR failed.")
        return

    with st.status("Building search index...", expanded=True) as status:
        st.write("Embedding chunks...")
        embeddings = embed_texts([c["text"] for c in all_chunks])
        index = build_index(embeddings)

        st.session_state.chunks = all_chunks
        st.session_state.index = index
        _refresh_hybrid_retriever()

        os.makedirs(STORAGE_DIR, exist_ok=True)
        import faiss

        faiss.write_index(index, os.path.join(STORAGE_DIR, "faiss.index"))

        st.write("Saving metadata...")
        _sync_sqlite_metadata(document_records, all_chunks)
        st.session_state.documents, st.session_state.chunks = _load_metadata_from_sqlite()
        _refresh_hybrid_retriever()
        status.update(label="Ingestion complete", state="complete")

    st.sidebar.success("Saved documents and index to storage/ ✅")
    st.session_state.stats = {
        "files_uploaded": len(sources),
        "docs_loaded": len(docs),
        "chunks_created": len(all_chunks),
        "stats_label": stats_label,
    }
    st.success(f"Processed {len(all_chunks)} chunks from {stats_label}")


def _refresh_hybrid_retriever():
    if st.session_state.index is None or not st.session_state.chunks:
        st.session_state.hybrid_retriever = None
        return

    st.session_state.hybrid_retriever = HybridRetriever.from_chunks(
        st.session_state.index,
        st.session_state.chunks,
    )

from ingestion.docling_converter import IMAGE_EXTENSIONS, docling_available
from ingestion.loader import load_documents
from ingestion.web_loader import load_web_page, save_web_snapshot, validate_web_url
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text, chunk_markdown, get_chunk_params
from embeddings.embedder import embed_texts
from vector_store.faiss_index import build_index
from retrieval.retriever import retrieve_chunks
from retrieval.reranker import rerank_chunks
from retrieval.hybrid_retrieval import HybridRetriever
from retrieval.query_rewrite import rewrite_query_groq
from retrieval.citations import citations_from_chunks, citations_report_payload, format_citations_markdown
from generation.generator import stream_answer
from generation.groq_generator import stream_answer_groq
from chat.chat_state import init_chat_state, clear_chat_history
from chat.chat_manager import add_assistant_message, add_user_message, ensure_chat_ready
from chat.ui import (
    render_chat_history,
    render_conversation_sidebar,
    render_empty_chat_placeholder,
    stream_assistant_reply,
)


logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("📄 Retrieval-Augmented Question Answering")

ensure_chat_ready()

if not docling_available():
    st.sidebar.info(
        "Docling OCR is not installed. PDF/TXT work with fallback; "
        "install docling + rapidocr-onnxruntime for DOCX, images, scanned PDFs, and URLs."
    )

# ---------------- Sidebar: Conversation (independent of knowledge base) ----------------
render_conversation_sidebar()
st.sidebar.divider()

# ---------------- Sidebar: Ingestion ----------------
st.sidebar.header("Add to knowledge base")
ingest_tab_upload, ingest_tab_url = st.sidebar.tabs(["Upload files", "Web page"])

SUPPORTED_UPLOAD_TYPES = [
    "txt", "pdf", "docx", "md", "markdown",
    "png", "jpg", "jpeg", "webp", "tiff", "bmp", "gif",
]

with ingest_tab_upload:
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
    )
    process_files = st.button("✅ Process documents", key="process_files")

with ingest_tab_url:
    web_url = st.text_input("Page URL", placeholder="https://example.com/docs/guide")
    process_url = st.button("✅ Process URL", key="process_url")
    st.caption("Static HTML pages work best. JavaScript-heavy sites may not ingest fully.")

reset = st.sidebar.button("🔄 Reset system")
st.sidebar.divider()
_render_knowledge_base_sidebar()
st.sidebar.divider()
filter_mode, selected_document_ids, active_filter_labels = _render_document_filter_sidebar()
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
    st.sidebar.write(st.session_state.stats.get("stats_label", "Last ingestion"))
    st.sidebar.write(f"📁 Sources processed: {st.session_state.stats['files_uploaded']}")
    st.sidebar.write(f"📄 Pages/sections loaded: {st.session_state.stats['docs_loaded']}")
    st.sidebar.write(f"🧩 Chunks created: {st.session_state.stats['chunks_created']}")

# ---------------- Session state (knowledge base) ----------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "documents" not in st.session_state:
    st.session_state.documents = None
if "hybrid_retriever" not in st.session_state:
    st.session_state.hybrid_retriever = None

init_chat_state()

if (
    st.session_state.index is None
    and os.path.exists(os.path.join(STORAGE_DIR, "faiss.index"))
    and os.path.exists(os.path.join(STORAGE_DIR, "metadata.db"))
):
    import faiss
    st.session_state.documents, st.session_state.chunks = _load_metadata_from_sqlite()

    st.session_state.index = faiss.read_index(os.path.join(STORAGE_DIR, "faiss.index"))
    _refresh_hybrid_retriever()

    st.sidebar.success("Loaded saved index ✅")

if reset:
    # Clear in-memory knowledge-base state
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.stats = None
    st.session_state.documents = None
    st.session_state.hybrid_retriever = None
    clear_chat_history()

    # Delete saved files (if they exist)
    if os.path.exists(os.path.join(STORAGE_DIR, "faiss.index")):
        os.remove(os.path.join(STORAGE_DIR, "faiss.index"))
    if os.path.exists(RAW_DOCS_DIR):
        removed_raw_docs = _remove_tree_safely(RAW_DOCS_DIR)
        if not removed_raw_docs:
            st.sidebar.warning("Could not fully remove storage/raw_docs. Close file sync/preview locks and try reset again.")
    legacy_chunks_path = os.path.join(STORAGE_DIR, "chunks.json")
    if os.path.exists(legacy_chunks_path):
        os.remove(legacy_chunks_path)
    legacy_documents_path = os.path.join(STORAGE_DIR, "documents.json")
    if os.path.exists(legacy_documents_path):
        os.remove(legacy_documents_path)
    try:
        from database.sqlite_repository import SQLiteMetadataRepository

        SQLiteMetadataRepository().clear_database()
    except Exception as exc:
        LOGGER.exception("Failed to clear SQLite metadata during reset: %s", exc)
        print(f"Failed to clear SQLite metadata during reset: {exc}")

    st.sidebar.success("System reset! Saved index deleted ✅")
    st.stop()

# ---------------- Process Uploaded Files ----------------
if process_files and uploaded_files:
    with st.spinner("Parsing documents with Docling..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in uploaded_files:
                path = os.path.join(tmpdir, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())

            os.makedirs(RAW_DOCS_DIR, exist_ok=True)
            for f in uploaded_files:
                raw_path = os.path.join(RAW_DOCS_DIR, f.name)
                with open(raw_path, "wb") as out:
                    out.write(f.getvalue())

            docs = load_documents(tmpdir)

        sources = [
            {
                "filename": f.name,
                "source_type": "image" if os.path.splitext(f.name)[1].lower() in IMAGE_EXTENSIONS else "file",
                "source_url": None,
            }
            for f in uploaded_files
        ]
        _run_full_ingestion(sources, docs, stats_label=f"{len(uploaded_files)} file(s)")

elif process_url and web_url:
    try:
        validate_web_url(web_url)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("Fetching and parsing web page..."):
        try:
            docs = load_web_page(web_url)
        except Exception as exc:
            st.error(f"Failed to ingest URL: {exc}")
            st.stop()

        web_snapshot_dir = os.path.join(RAW_DOCS_DIR, "web")
        save_web_snapshot(web_url, web_snapshot_dir)

        filename = docs[0].get("filename") if docs else "web_page.html"
        sources = [
            {
                "filename": filename,
                "source_type": "url",
                "source_url": web_url.strip(),
            }
        ]
        _run_full_ingestion(sources, docs, stats_label="1 web page")


def _retrieve_for_chat_query(query: str):
    """Run the existing retrieval pipeline for a single user query.

    Returns
    -------
    tuple
        (retrieved_chunks, rewritten_query, retrieval_debug_or_None, error_message_or_None)
    """
    filtered_chunks, candidate_indices, scope_labels, filter_incomplete = _resolve_retrieval_scope(
        filter_mode,
        selected_document_ids,
        st.session_state.chunks,
    )
    display_labels = scope_labels or active_filter_labels

    if filter_incomplete:
        return [], "", None, "Select at least one document to apply a document filter."

    # Retrieval uses the current query only. recent_turns() is available for a
    # future contextual rewriter without changing this call site.
    rewritten_query = rewrite_query_groq(query, mode=rewrite_mode)
    retrieval_query = rewritten_query or query
    retrieval_debug = None

    retrieval_kwargs = {}
    if filtered_chunks is not None:
        retrieval_kwargs = {
            "chunks": filtered_chunks,
            "candidate_indices": candidate_indices,
        }

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
                **retrieval_kwargs,
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
                **retrieval_kwargs,
            )
    else:
        initial = retrieve_chunks(
            retrieval_query,
            st.session_state.index,
            st.session_state.chunks,
            top_k=top_k_dense,
            candidate_indices=candidate_indices if filtered_chunks is not None else None,
        )
        if show_retrieval_debug:
            retrieval_debug = {
                "dense_results": initial,
                "sparse_results": [],
                "fused_results": initial,
            }

    retrieved = rerank_chunks(query, initial, top_k=5) if use_reranker else initial[:8]
    if show_retrieval_debug and retrieval_debug is not None:
        retrieval_debug["reranked_results"] = retrieved
        if display_labels:
            retrieval_debug["active_filters"] = display_labels

    return retrieved, rewritten_query or query, retrieval_debug, None


# ---------------- Chat UI ----------------
if st.session_state.index is not None:
    history = st.session_state.get("chat_history") or []
    if not history:
        render_empty_chat_placeholder()
    else:
        render_chat_history()

    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        add_user_message(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context..."):
                retrieved, rewritten_query, retrieval_debug, retrieval_error = _retrieve_for_chat_query(prompt)

            if retrieval_error:
                st.warning(retrieval_error)
                add_assistant_message(retrieval_error)
            elif generator_choice == "Groq (LLM API)" and not GROQ_API_KEY:
                msg = "GROQ_API_KEY is missing. Configure it to use Groq, or switch to Local (FLAN-T5)."
                st.error(msg)
                add_assistant_message(msg)
            else:
                if rewritten_query and rewritten_query != prompt:
                    st.caption(f"Retrieval query: {rewritten_query}")

                if generator_choice == "Groq (LLM API)":
                    answer_text = stream_assistant_reply(
                        stream_answer_groq(prompt, retrieved, model=groq_model)
                    )
                else:
                    answer_text = stream_assistant_reply(stream_answer(prompt, retrieved))

                source_citations = citations_from_chunks(retrieved)
                sources_markdown = format_citations_markdown(source_citations) if source_citations else ""
                if sources_markdown:
                    with st.expander("Sources", expanded=False):
                        st.markdown(sources_markdown)

                report_payload = citations_report_payload(
                    retrieved,
                    question=prompt,
                    answer=str(answer_text or ""),
                )
                st.download_button(
                    label="Download citation report (JSON)",
                    data=json.dumps(report_payload, indent=2),
                    file_name="citation_report.json",
                    mime="application/json",
                    key=f"citation_download_live_{len(st.session_state.chat_history)}",
                )

                if show_retrieval_debug and retrieval_debug is not None:
                    with st.expander("Retrieval debug", expanded=False):
                        st.markdown("**Dense retrieval**")
                        st.json(_preview_rankings(retrieval_debug.get("dense_results", [])))
                        st.markdown("**Sparse retrieval**")
                        st.json(_preview_rankings(retrieval_debug.get("sparse_results", [])))
                        st.markdown("**Fused retrieval**")
                        st.json(_preview_rankings(retrieval_debug.get("fused_results", [])))
                        st.markdown("**Reranked results**")
                        st.json(_preview_rankings(retrieval_debug.get("reranked_results", [])))

                with st.expander("🔍 View Retrieved Context (Top Matches)", expanded=False):
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
                            height=150,
                            key=f"chunk_text_live_{len(st.session_state.chat_history)}_{i}",
                        )

                add_assistant_message(
                    str(answer_text or ""),
                    sources_markdown=sources_markdown,
                    citation_report=report_payload,
                    retrieval_debug=retrieval_debug if show_retrieval_debug else None,
                    rewritten_query=rewritten_query if rewritten_query != prompt else "",
                )
else:
    st.info("Upload documents and click **Process documents** to begin.")
