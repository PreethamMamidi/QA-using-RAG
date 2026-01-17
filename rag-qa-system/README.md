# RAG QA System
# RAG QA System

A Retrieval-Augmented Generation (RAG) system for asking questions over local documents.
This project supports two generation modes:

- Local: runs a CPU-friendly FLAN-T5 model locally (`generation/generator.py`).
- Groq: uses the Groq API for generation and query rewriting (`generation/groq_generator.py`, `retrieval/query_rewrite.py`).

You can switch between modes in the Streamlit UI or call the corresponding functions from code.

## Quick summary — Local vs Groq

- Local (FLAN-T5)
  - Uses `google/flan-t5-small` via `generation/generator.py` (CPU-only).
  - Requirements: `requirements.txt` (includes `transformers`, `torch`, `sentence-transformers`), sufficient disk space for model weights, and internet the first time to download models.
  - Privacy: documents and prompts stay on your machine.

- Groq (LLM API)
  - Uses the `groq` SDK and Groq chat completion models (`generation/groq_generator.py`).
  - Requirements: a Groq account and API key set as `GROQ_API_KEY` in your environment (or in a `.env` file). The `groq` package is listed in `requirements.txt`.
  - Behavior: sends context to Groq's API — requires internet and will transmit document context to Groq.

## Repo layout (important files)

- `app/streamlit_app.py` — interactive UI and mode switch
- `ingestion/` — `loader.py`, `cleaner.py`, `chunker.py`
- `embeddings/embedder.py` — sentence-transformers embedding code
- `vector_store/faiss_index.py` — FAISS index build/load
- `retrieval/` — `retriever.py`, `reranker.py`, `query_rewrite.py` (uses Groq when `GROQ_API_KEY` is set)
- `generation/` — `generator.py` (local FLAN-T5) and `groq_generator.py` (Groq API)
- `data/` — `raw_docs/` and `processed_chunks/`
- `storage/` — persisted index and chunk metadata

## Installation (quick)

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Create a `.env` file or set environment variables. For Groq mode set:

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY = "your_groq_api_key"

# macOS / Linux
export GROQ_API_KEY="your_groq_api_key"
```

Note: `generation/groq_generator.py` expects `GROQ_API_KEY` to be set. `retrieval/query_rewrite.py` will fall back to the original query if the key is not present.

## Running the app

Start the Streamlit UI and choose the generator in the sidebar:

```bash
streamlit run app/streamlit_app.py
```

- In the sidebar choose **Groq (LLM API)** or **Local (FLAN-T5)**.
- If Groq is selected, set `GROQ_API_KEY` and pick a Groq model from the drop-down.

## How switching works

- The Streamlit UI triggers `generate_answer_groq` when **Groq** is selected and `generate_answer` when **Local** is selected.
- The query rewrite step uses `retrieval/query_rewrite.py` which attempts to use Groq if `GROQ_API_KEY` is present; otherwise it returns the original query.

## Security & privacy

- Local mode: no network calls for generation — documents remain local.
- Groq mode: document context is sent to Groq's API. Do not enable Groq mode for sensitive documents unless you accept external transmission.

## Examples (quick)

- Add a short text file at `data/raw_docs/sample.txt` with: `AcmeCorp was founded in 1999 in Austin, Texas.`
- Process documents via the Streamlit UI or `python run_pipeline.py`.
- Ask: `When was AcmeCorp founded?` — local or Groq generator should return `1999` and cite the source file.

## Troubleshooting

- If Groq calls fail, confirm `GROQ_API_KEY` and network connectivity.
- If the local model fails to load, ensure `torch` is installed and you have enough disk space to download `google/flan-t5-small`.

## Next steps

- I can add a short config snippet showing how to switch generators via environment variables or add a sample `data/raw_docs/sample.txt`. Tell me which you'd prefer.

# RAG QA System

A lightweight Retrieval-Augmented Generation (RAG) system for question answering over local documents. The project is designed to run on CPU-only machines without external APIs, prioritizing privacy and reproducibility.

## What this repo provides

- A document ingestion pipeline (load, clean, chunk)
- Local embedding generation using Sentence Transformers
- FAISS-based vector indexing and retrieval
- A generator layer that synthesizes answers from retrieved chunks
- A Streamlit web UI for interactive exploration
- Basic evaluation scripts for retrieval and QA components

## Repo layout

- `app/` — Streamlit front-end (`streamlit_app.py`)
- `ingestion/` — loader, cleaner, and chunker utilities
- `embeddings/` — code to compute and persist embeddings
- `vector_store/` — FAISS index builder and loader
- `retrieval/` — retrieval, reranking, and query rewriting code
- `generation/` — response generation logic
- `evaluation/` — retrieval and QA metrics
- `data/` — `raw_docs/` for source files and `processed_chunks/` for outputs
- `storage/` — persisted indexes and chunk metadata (e.g., `faiss.index`, `chunks.json`)

## Quickstart (Local)

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK tokenizer models (one-time):

```bash
python -c "import nltk; nltk.download('punkt')"
```

4. Start the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

5. Alternatively, run the batch pipeline to process documents in `data/raw_docs/`:

```bash
python run_pipeline.py
```

## Typical workflow

1. Add documents (PDF or TXT) to `data/raw_docs/`.
2. Run the ingestion pipeline (`run_pipeline.py`) or use the Streamlit UI to process documents.
3. The pipeline will produce chunk files in `data/processed_chunks/`, compute embeddings, and build a FAISS index in `storage/`.
4. Use the Streamlit UI or the retrieval/generation modules to ask questions; retrieved chunks are used as context for answer generation.

## Key configuration points

- Chunk size and overlap: controlled in `ingestion/chunker.py` for retrieval fidelity versus index size.
- Embedding model: see `embeddings/embedder.py` — default is a Sentence Transformers model for CPU-friendly performance.
- Generator model: defined in `generation/generator.py` and `generation/groq_generator.py`. Swap models there if needed.

## Evaluation

- Retrieval metrics are implemented in `evaluation/retrieval_metrics.py`.
- QA metrics are available in `evaluation/qa_metrics.py` for measuring answer quality against labeled data.

## Development notes

- The code is organized to make component swaps straightforward: replace an embedding or generation model without changing the rest of the pipeline.
- Keep models and large artifacts out of Git — use `storage/` for local indexes and `embeddings/` for serialized vectors.

## Contributing

1. Fork and create a branch for your feature.
2. Add tests or validate behavior with the Streamlit app and `run_pipeline.py`.
3. Open a pull request with a short description and rationale.

## Examples (expected input / output)

Below are simple examples showing the kind of documents you can add and the expected QA behavior.

- Example document (place in `data/raw_docs/sample.txt`):

```
AcmeCorp was founded in 1999 in Austin, Texas. Its flagship product is the RoadRunner trap used for pest control.
```

- Example query:

```
When was AcmeCorp founded?
```

- Expected retrieval + generation (simplified):

```
Retrieved chunk (source: sample.txt): "AcmeCorp was founded in 1999 in Austin, Texas."
Answer: "AcmeCorp was founded in 1999."
Sources: sample.txt
```

- Example multi-document scenario: add `data/raw_docs/policies.txt` and `data/raw_docs/overview.txt`. The retriever should return the most relevant chunk(s) and the generator should synthesize a concise answer that references the original source file(s).

Notes:
- Process documents first (`python run_pipeline.py` or use the Streamlit UI) so chunks, embeddings, and the FAISS index are created.
- The exact answer wording may vary depending on the chosen generation model, but relevant source chunks should be included with the response.

## Troubleshooting

- If you run into out-of-memory issues, reduce batch sizes or switch to a smaller embedding/generation model.
- For PDF parsing problems, check `ingestion/loader.py` and try converting PDFs to text before ingestion.

## License

This project is open source. See the LICENSE file in the repository root for details.

---


