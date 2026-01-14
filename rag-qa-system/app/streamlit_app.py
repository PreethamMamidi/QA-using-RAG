import json
import streamlit as st
import sys
import os
import tempfile

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
from generation.generator import generate_answer


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

if reset:
    st.session_state.chunks = None
    st.session_state.index = None
    st.success("System reset! Upload documents again and click Process.")
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
                    all_chunks.extend(chunk_text(text, doc_id, chunk_size=200, overlap=50))
                else:
                    all_chunks.extend(chunk_text(text, doc_id, chunk_size=350, overlap=80))

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
        retrieved = retrieve_chunks(
            query,
            st.session_state.index,
            st.session_state.chunks,
            top_k=3,
        )

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
