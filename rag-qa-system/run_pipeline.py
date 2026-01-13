from ingestion.loader import load_documents
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text

from embeddings.embedder import embed_texts
from vector_store.faiss_index import build_index
from retrieval.retriever import retrieve_chunks
from generation.generator import generate_answer


def main():
    # 1. Load documents
    docs = load_documents("data/raw_docs")
    if not docs:
        print("No documents found in data/raw_docs")
        return

    # 2. Clean + chunk documents
    chunks = []
    for d in docs:
        text = clean_text(d["text"])
        chunks.extend(
            chunk_text(text, document_id=d["document_id"])
        )

    if not chunks:
        print("No chunks created.")
        return

    print(f"Created {len(chunks)} chunks")

    # 3. Embed chunks
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_texts(chunk_texts)

    # 4. Build FAISS index
    index = build_index(embeddings)

    # 5. Ask questions
    print("\nRAG system ready. Type 'exit' to quit.\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        retrieved = retrieve_chunks(
            query=query,
            index=index,
            chunks=chunks,
            top_k=3,
        )

        answer = generate_answer(query, retrieved)

        print("\nAnswer:")
        print(answer)
        print("\nSources:")
        for r in retrieved:
            print(f"- {r['document_id']} (score={r['score']:.3f})")
        print("-" * 50)


if __name__ == "__main__":
    main()
