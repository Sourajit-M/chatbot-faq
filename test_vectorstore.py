from src.loaders.text_loader import TextFAQLoader
from src.loaders.pdf_loader import PDFLoader
from src.utils.text_cleaning import normalize_text
from src.utils.chunking import chunk_documents
from src.embeddings.embedder import LocalEmbedder
from src.vectorstore.faiss_store import FAISSVectorStore


# Load data
faq_docs = TextFAQLoader("data/faqs/personal_faqs.json").load()
pdf_docs = PDFLoader("data/documents/sample.pdf").load()

docs = faq_docs + pdf_docs

# Clean
for doc in docs:
    doc["content"] = normalize_text(doc["content"])

# Chunk
chunked_docs = chunk_documents(docs)

# Embed
embedder = LocalEmbedder()
texts = [doc["content"] for doc in chunked_docs]
embeddings = embedder.embed_texts(texts)

# Vector store
store = FAISSVectorStore(embedding_dim=384)
store.add(embeddings, chunked_docs)

# Query
query = "What is my education background?"
query_embedding = embedder.embed_query(query)
results = store.search(query_embedding)

print("Top results:\n")
for res in results:
    print(res["metadata"])