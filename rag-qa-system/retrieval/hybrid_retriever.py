"""Backward-compatible import wrapper for hybrid retrieval."""
from retrieval.bm25_index import BM25Retriever
from retrieval.hybrid_retrieval import HybridRetriever, hybrid_retrieve_chunks

__all__ = ["BM25Retriever", "HybridRetriever", "hybrid_retrieve_chunks"]
