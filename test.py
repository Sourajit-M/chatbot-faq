from src.loaders.pdf_loader import PDFLoader
from src.utils.text_cleaning import normalize_text
from src.utils.chunking import chunk_documents

loader = PDFLoader("data/documents/sample.pdf")
docs = loader.load()

# Clean text
for doc in docs:
    doc["content"] = normalize_text(doc["content"])

chunked_docs = chunk_documents(docs)

print(f"Original pages: {len(docs)}")
print(f"Chunks created: {len(chunked_docs)}\n")

print(chunked_docs[0])