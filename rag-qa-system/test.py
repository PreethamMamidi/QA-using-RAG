from ingestion.loader import load_documents

docs = load_documents("data/raw_docs")
print("Total documents loaded:", len(docs))

for d in docs[:5]:
    print(d["document_id"], "->", len(d["text"]))
