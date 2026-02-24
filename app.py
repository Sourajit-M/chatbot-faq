import streamlit as st

from src.loaders.text_loader import TextFAQLoader
from src.loaders.pdf_loader import PDFLoader
from src.utils.text_cleaning import normalize_text
from src.utils.chunking import chunk_documents
from src.embeddings.embedder import LocalEmbedder
from src.vectorstore.faiss_store import FAISSVectorStore
from src.rag.pipeline import RAGPipeline

st.set_page_config(
    page_title="Cogni",
    page_icon="src/assets/icon2.png",
    layout="centered"
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 760px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    # 🤖 AI Knowledge Assistant
    Ask questions and get **source-grounded answers** from your documents.
    """
)

with st.sidebar:
    st.markdown("### 📄 Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.success("PDF ready")
        st.caption(uploaded_file.name)
    else:
        st.caption("Using default knowledge base")

    st.divider()

    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("RAG • FAISS • Gemini")


@st.cache_resource(show_spinner=False)
def load_default_knowledge_base():
    faq_docs = TextFAQLoader("data/faqs/personal_faqs.json").load()
    pdf_docs = PDFLoader("data/documents/sample.pdf").load()

    docs = faq_docs + pdf_docs

    for doc in docs:
        doc["content"] = normalize_text(doc["content"])

    chunked_docs = chunk_documents(docs)

    embedder = LocalEmbedder()
    embeddings = embedder.embed_texts([d["content"] for d in chunked_docs])

    store = FAISSVectorStore(embedding_dim=384)
    store.add(embeddings, chunked_docs)

    return store, embedder


@st.cache_resource(show_spinner=False)
def build_vector_store_from_uploaded_pdf(file_bytes):
    temp_path = "data/documents/uploaded.pdf"

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    pdf_docs = PDFLoader(temp_path).load()

    for doc in pdf_docs:
        doc["content"] = normalize_text(doc["content"])

    chunked_docs = chunk_documents(pdf_docs)

    embedder = LocalEmbedder()
    embeddings = embedder.embed_texts([d["content"] for d in chunked_docs])

    store = FAISSVectorStore(embedding_dim=384)
    store.add(embeddings, chunked_docs)

    return store, embedder


if uploaded_file:
    vector_store, embedder = build_vector_store_from_uploaded_pdf(
        uploaded_file.read()
    )
    st.success("Document indexed. Ask anything.")
else:
    vector_store, embedder = load_default_knowledge_base()

rag = RAGPipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []


if not st.session_state.messages:
    st.info(
        "Try asking:\n"
        "- *Summarize this document*\n"
    )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_query = st.chat_input("Ask a question")

if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    query_embedding = embedder.embed_query(user_query)
    retrieved_docs = vector_store.search(query_embedding, top_k=7)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = rag.generate_answer(user_query, retrieved_docs)

        st.markdown(answer)

        sources = set()
        for doc in retrieved_docs:
            meta = doc["metadata"]
            if meta["source"] == "pdf":
                sources.add(f"{meta['file_name']} — page {meta['page']}")
            else:
                sources.add(f"FAQ ID {meta['id']}")

        if sources:
            with st.expander("Sources"):
                for s in sorted(sources):
                    st.markdown(f"- `{s}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
