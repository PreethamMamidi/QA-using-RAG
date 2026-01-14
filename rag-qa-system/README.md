# RAG QA System

A powerful, locally-run Retrieval-Augmented Generation (RAG) system for question answering over your documents. Built entirely on your CPU without relying on any external APIs, ensuring privacy and control.

## 🌟 Features

- **Document Ingestion**: Upload and process PDF and TXT files seamlessly
- **Intelligent Chunking**: Smart text splitting with configurable chunk sizes and overlaps for optimal retrieval
- **Local Embeddings**: Uses Sentence Transformers for high-quality text embeddings, all running locally
- **Efficient Retrieval**: FAISS-powered vector search for fast, accurate document retrieval
- **Answer Generation**: Leverages FLAN-T5 model for coherent, context-aware answers
- **Web Interface**: Clean Streamlit app for easy interaction
- **Persistent Storage**: Saves processed data and indexes for quick reloading
- **Evaluation Metrics**: Built-in tools to assess retrieval and QA performance
- **Batch Processing**: Command-line interface for processing large document collections

## 🛠 Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web app framework for the user interface
- **Sentence Transformers**: For generating text embeddings (`all-MiniLM-L6-v2`)
- **FAISS**: High-performance vector similarity search
- **Transformers**: Hugging Face library for the FLAN-T5 generation model (`google/flan-t5-small`)
- **PyTorch**: Deep learning framework (CPU-only)
- **PyMuPDF & PyPDF**: PDF document processing
- **NLTK**: Natural language processing utilities
- **NumPy & Scikit-learn**: Numerical computing and machine learning tools
- **Pandas**: Data manipulation

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PreethamMamidi/QA-using-RAG.git
   cd QA-using-RAG/rag-qa-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK tokenizer resources:
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

## 📖 Usage

### Web App (Recommended)
Run the Streamlit interface for an interactive experience:

```bash
streamlit run app/streamlit_app.py
```

- Upload PDF or TXT files
- Click "Process documents" to build the knowledge base
- Ask questions and get AI-generated answers with source citations

### Command Line
For batch processing or headless operation:

```bash
python run_pipeline.py
```

Place your documents in `data/raw_docs/` and run the script for interactive QA.

## 📊 Evaluation

The system includes built-in evaluation tools to assess the performance of the retrieval and QA components:

- **Retrieval Metrics**: Measure retrieval quality with precision@k and recall@k metrics
- **QA Metrics**: Framework for evaluating answer quality (expandable for future implementations)

You can use these metrics to benchmark different configurations, chunk sizes, or embedding models to optimize performance for your specific use case.

## 🏗 Architecture

The system follows a modular RAG pipeline:

1. **Ingestion**: Load and clean documents, then chunk them into manageable pieces
2. **Embedding**: Convert text chunks to dense vector representations
3. **Indexing**: Build a FAISS vector index for efficient similarity search
4. **Retrieval**: Find the most relevant chunks for a given query
5. **Generation**: Use retrieved context to generate accurate answers

### Key Components

- `ingestion/`: Document loading, cleaning, and chunking
- `embeddings/`: Text embedding generation
- `vector_store/`: FAISS index management
- `retrieval/`: Similarity search and chunk retrieval
- `generation/`: Answer synthesis using transformer models
- `evaluation/`: Performance metrics and assessment tools
- `app/`: Streamlit web interface

## 💡 Key Highlights

- **100% Local**: Everything runs on your CPU - no internet required, no API keys needed
- **Privacy-First**: Your documents never leave your machine
- **Flexible**: Supports both interactive web app and command-line usage
- **Extensible**: Modular design makes it easy to swap components or add features
- **Efficient**: Optimized for CPU usage with batch processing and lazy loading

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help improve the RAG QA System:

### Development Setup
1. Fork the repository and clone your fork
2. Follow the installation steps above
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes and test thoroughly
5. Ensure your code follows the existing style and includes appropriate docstrings

### Testing
- Test your changes with both the Streamlit app and command-line interface
- Verify that document processing works correctly with various file types
- Ensure the system runs smoothly on CPU-only environments

### Pull Request Process
1. Update the README if your changes affect usage or installation
2. Ensure all tests pass and no new errors are introduced
3. Provide a clear description of your changes and their benefits
4. Reference any related issues in your PR description

Thank you for helping make this project better!

## 📄 License

This project is open-source. Please check the license file for details.