"""Streamlit evaluation dashboard for retrieval quality and latency."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

STORAGE_DIR = ROOT_DIR / "storage"
DEFAULT_DATASET = ROOT_DIR / "evaluation" / "gold.json"

st.set_page_config(page_title="Evaluation Dashboard", layout="wide")
st.title("Retrieval Evaluation Dashboard")

from database.sqlite_repository import SQLiteMetadataRepository
from evaluation.datasets import load_evaluation_dataset
from evaluation.evaluator import RetrievalEvaluator


@st.cache_resource
def _load_retrieval_backend():
	import faiss

	index_path = STORAGE_DIR / "faiss.index"
	if not index_path.exists():
		return None, []

	repository = SQLiteMetadataRepository(STORAGE_DIR / "metadata.db")
	chunks = repository.get_all_chunks()
	index = faiss.read_index(str(index_path))
	return index, chunks


def _render_corpus_stats():
	repository = SQLiteMetadataRepository(STORAGE_DIR / "metadata.db")
	col1, col2 = st.columns(2)
	col1.metric("Total Documents", repository.get_document_count())
	col2.metric("Total Chunks", repository.get_chunk_count())


def _render_metric_cards(summary):
	c1, c2, c3, c4, c5 = st.columns(5)
	c1.metric("Recall@5", f"{summary.recall_at_5:.2f}")
	c2.metric("Recall@10", f"{summary.recall_at_10:.2f}")
	c3.metric("HitRate@5", f"{summary.hit_rate_at_5:.2f}")
	c4.metric("HitRate@10", f"{summary.hit_rate_at_10:.2f}")
	c5.metric("MRR", f"{summary.mrr:.2f}")


def _render_latency_charts(summary):
	latency = summary.average_latency
	latency_df = pd.DataFrame(
		{
			"stage": ["BM25", "FAISS", "Fusion", "Reranker", "Total"],
			"latency_ms": [
				latency.bm25_ms,
				latency.faiss_ms,
				latency.fusion_ms,
				latency.rerank_ms,
				latency.total_ms,
			],
		}
	)
	st.subheader("Retrieval Latency Breakdown")
	st.bar_chart(latency_df, x="stage", y="latency_ms")


def _render_recall_chart(summary):
	recall_df = pd.DataFrame(
		{
			"metric": ["Recall@5", "Recall@10", "HitRate@5", "HitRate@10", "MRR"],
			"score": [
				summary.recall_at_5,
				summary.recall_at_10,
				summary.hit_rate_at_5,
				summary.hit_rate_at_10,
				summary.mrr,
			],
		}
	)
	st.subheader("Retrieval Quality Metrics")
	st.bar_chart(recall_df, x="metric", y="score")


st.markdown("Evaluate hybrid retrieval quality against a labeled dataset.")

_render_corpus_stats()
st.divider()

dataset_path = st.text_input(
	"Evaluation dataset path",
	value=str(DEFAULT_DATASET),
)
use_reranker = st.checkbox("Include reranker in evaluation", value=False)
run_eval = st.button("Run Retrieval Evaluation", type="primary")

if run_eval:
	index, chunks = _load_retrieval_backend()
	if index is None or not chunks:
		st.error("No indexed corpus found. Process documents on the main page first.")
		st.stop()

	try:
		samples = load_evaluation_dataset(dataset_path)
	except Exception as exc:
		st.error(f"Failed to load dataset: {exc}")
		st.stop()

	with st.spinner(f"Evaluating {len(samples)} queries..."):
		evaluator = RetrievalEvaluator(index, chunks, use_reranker=use_reranker)
		summary = evaluator.evaluate(samples)
		st.session_state["eval_summary"] = summary

if "eval_summary" in st.session_state:
	summary = st.session_state["eval_summary"]
	st.success(f"Queries evaluated: {summary.queries_evaluated}")
	_render_metric_cards(summary)

	col_left, col_right = st.columns(2)
	with col_left:
		_render_recall_chart(summary)
	with col_right:
		_render_latency_charts(summary)

	with st.expander("Full text report"):
		st.code(summary.format_report())

	st.download_button(
		label="Download evaluation report (JSON)",
		data=json.dumps(summary.to_dict(), indent=2),
		file_name="retrieval_evaluation_report.json",
		mime="application/json",
	)

	with st.expander("Per-query results"):
		rows = [
			{
				"question": result.question,
				"recall@5": result.recall_at_5,
				"recall@10": result.recall_at_10,
				"hit@5": result.hit_rate_at_5,
				"hit@10": result.hit_rate_at_10,
				"mrr": result.mrr,
				"total_ms": result.latency.total_ms,
			}
			for result in summary.per_query
		]
		st.dataframe(pd.DataFrame(rows), use_container_width=True)
