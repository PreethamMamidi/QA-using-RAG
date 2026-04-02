import json
from typing import List, Dict, Set

from ingestion.loader import load_documents
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from embeddings.embedder import embed_texts
from vector_store.faiss_index import build_index
from retrieval.retriever import retrieve_chunks


def hit_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
    topk = retrieved_docs[:k]
    return 1.0 if any(d in relevant_docs for d in topk) else 0.0


def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
    topk = retrieved_docs[:k]
    if not topk:
        return 0.0
    hits = sum(1 for d in topk if d in relevant_docs)
    return hits / k


def recall_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
    topk = retrieved_docs[:k]
    if not relevant_docs:
        return 0.0
    hits = sum(1 for d in topk if d in relevant_docs)
    return hits / len(relevant_docs)


def reciprocal_rank(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
    for i, d in enumerate(retrieved_docs):
        if d in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0


def main():
    # 1) Load gold dataset
    with open("evaluation/gold.json", "r", encoding="utf-8") as f:
        gold = json.load(f)

    # 2) Load docs from data/raw_docs
    docs = load_documents("data/raw_docs")
    if not docs:
        print("❌ No documents found in data/raw_docs")
        return

    # 3) Clean + chunk all docs
    chunks = []
    for d in docs:
        text = clean_text(d["text"])
        chunks.extend(chunk_text(text, d["document_id"], chunk_size=180, overlap=60))

    if not chunks:
        print("❌ No chunks created")
        return

    # 4) Embed + build FAISS index
    print(f"🔧 Embedding {len(chunks)} chunks...")
    embs = embed_texts([c["text"] for c in chunks])
    index = build_index(embs)

    # 5) Evaluate retrieval
    K = 5
    hit_scores = []
    p_scores = []
    r_scores = []
    mrr_scores = []

    for item in gold:
        q = item["question"]
        relevant = set(item["relevant_docs"])

        retrieved = retrieve_chunks(q, index, chunks, top_k=K)

        # Convert retrieved chunks to document_id list
        retrieved_docs = [x["document_id"] for x in retrieved]
        # remove duplicates while preserving order
        unique_retrieved_docs = list(dict.fromkeys(retrieved_docs))
        retrieved_docs = unique_retrieved_docs


        hit_scores.append(hit_at_k(retrieved_docs, relevant, K))
        p_scores.append(precision_at_k(retrieved_docs, relevant, K))
        r_scores.append(recall_at_k(retrieved_docs, relevant, K))
        mrr_scores.append(reciprocal_rank(retrieved_docs, relevant))

    # 6) Print final metrics
    n = len(gold)
    print("\n✅ Retrieval Evaluation Results")
    print(f"Total Queries: {n}")
    print(f"Hit@{K}:       {sum(hit_scores)/n:.3f}")
    print(f"Precision@{K}: {sum(p_scores)/n:.3f}")
    print(f"Recall@{K}:    {sum(r_scores)/n:.3f}")
    print(f"MRR:           {sum(mrr_scores)/n:.3f}")
if __name__ == "__main__":
    main()
