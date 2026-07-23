"""Backward-compatible re-exports for retrieval metrics."""
from evaluation.metrics import hit_rate_at_k, recall_at_k

__all__ = ["recall_at_k", "hit_rate_at_k"]
