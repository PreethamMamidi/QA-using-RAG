RAG QA System
A powerful, locally-run Retrieval-Augmented Generation (RAG) system for question answering over your documents. Built entirely on your CPU without relying on any external APIs, ensuring privacy and control.

🌟 Features
Document Ingestion: Upload and process PDF and TXT files seamlessly
Intelligent Chunking: Smart text splitting with configurable chunk sizes and overlaps for optimal retrieval
Local Embeddings: Uses Sentence Transformers for high-quality text embeddings, all running locally
Efficient Retrieval: FAISS-powered vector search for fast, accurate document retrieval
Answer Generation: Leverages FLAN-T5 model for coherent, context-aware answers
Web Interface: Clean Streamlit app for easy interaction
Persistent Storage: Saves processed data and indexes for quick reloading
Evaluation Metrics: Built-in tools to assess retrieval and QA performance
Batch Processing: Command-line interface for processing large document collections
🛠 Technologies Used
Python: Core programming language
Streamlit: Web app framework for the user interface
Sentence Transformers: For generating text embeddings (all-MiniLM-L6-v2)
FAISS: High-performance vector similarity search
Transformers: Hugging Face library for the FLAN-T5 generation model (google/flan-t5-small)
PyTorch: Deep learning framework (CPU-only)
PyMuPDF & PyPDF: PDF document processing
NLTK: Natural language processing utilities
NumPy & Scikit-learn: Numerical computing and machine learning tools
Pandas: Data manipulation
🚀 Installation
Clone the repository:

git clone https://github.com/PreethamMamidi/QA-using-RAG.git
cd QA-using-RAG/rag-qa-system
Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

pip install -r requirements.txt
Download NLTK tokenizer resources:

python -c "import nltk; nltk.download('punkt')"
📖 Usage
Web App (Recommended)
Run the Streamlit interface for an interactive experience:

RAG QA System
A lightweight Retrieval-Augmented Generation (RAG) system for question answering over local documents. The project is designed to run on CPU-only machines without external APIs, prioritizing privacy and reproducibility.

What this repo provides
A document ingestion pipeline (load, clean, chunk)
Local embedding generation using Sentence Transformers
FAISS-based vector indexing and retrieval
A generator layer that synthesizes answers from retrieved chunks
A Streamlit web UI for interactive exploration
Basic evaluation scripts for retrieval and QA components
Repo layout
app/ — Streamlit front-end (streamlit_app.py)
ingestion/ — loader, cleaner, and chunker utilities
embeddings/ — code to compute and persist embeddings
vector_store/ — FAISS index builder and loader
retrieval/ — retrieval, reranking, and query rewriting code
generation/ — response generation logic
evaluation/ — retrieval and QA metrics
data/ — raw_docs/ for source files and processed_chunks/ for outputs
storage/ — persisted indexes and chunk metadata (e.g., faiss.index, chunks.json)
Quickstart (Local)
Create and activate a Python virtual environment:
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Download NLTK tokenizer models (one-time):
python -c "import nltk; nltk.download('punkt')"
Start the Streamlit app:
streamlit run app/streamlit_app.py
Alternatively, run the batch pipeline to process documents in data/raw_docs/:
python run_pipeline.py
Typical workflow
Add documents (PDF or TXT) to data/raw_docs/.
Run the ingestion pipeline (run_pipeline.py) or use the Streamlit UI to process documents.
The pipeline will produce chunk files in data/processed_chunks/, compute embeddings, and build a FAISS index in storage/.
Use the Streamlit UI or the retrieval/generation modules to ask questions; retrieved chunks are used as context for answer generation.
Key configuration points
Chunk size and overlap: controlled in ingestion/chunker.py for retrieval fidelity versus index size.
Embedding model: see embeddings/embedder.py — default is a Sentence Transformers model for CPU-friendly performance.
Generator model: defined in generation/generator.py and generation/groq_generator.py. Swap models there if needed.
Evaluation
Retrieval metrics are implemented in evaluation/retrieval_metrics.py.
QA metrics are available in evaluation/qa_metrics.py for measuring answer quality against labeled data.
Development notes
The code is organized to make component swaps straightforward: replace an embedding or generation model without changing the rest of the pipeline.
Keep models and large artifacts out of Git — use storage/ for local indexes and embeddings/ for serialized vectors.
Contributing
Fork and create a branch for your feature.
Add tests or validate behavior with the Streamlit app and run_pipeline.py.
Open a pull request with a short description and rationale.
Examples (expected input / output)
Below are simple examples showing the kind of documents you can add and the expected QA behavior.

Example document (place in data/raw_docs/sample.txt):
AcmeCorp was founded in 1999 in Austin, Texas. Its flagship product is the RoadRunner trap used for pest control.
Example query:
When was AcmeCorp founded?
Expected retrieval + generation (simplified):
Retrieved chunk (source: sample.txt): "AcmeCorp was founded in 1999 in Austin, Texas."
Answer: "AcmeCorp was founded in 1999."
Sources: sample.txt
Example multi-document scenario: add data/raw_docs/policies.txt and data/raw_docs/overview.txt. The retriever should return the most relevant chunk(s) and the generator should synthesize a concise answer that references the original source file(s).
Notes:

Process documents first (python run_pipeline.py or use the Streamlit UI) so chunks, embeddings, and the FAISS index are created.
The exact answer wording may vary depending on the chosen generation model, but relevant source chunks should be included with the response.
Troubleshooting
If you run into out-of-memory issues, reduce batch sizes or switch to a smaller embedding/generation model.
For PDF parsing problems, check ingestion/loader.py and try converting PDFs to text before ingestion.
License
This project is open source. See the LICENSE file in the repository root for details.

