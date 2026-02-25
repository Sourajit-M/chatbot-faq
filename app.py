import streamlit as st

from src.loaders.text_loader import TextFAQLoader
from src.loaders.pdf_loader import PDFLoader
from src.utils.text_cleaning import normalize_text
from src.utils.chunking import chunk_documents
from src.embeddings.embedder import LocalEmbedder
from src.vectorstore.faiss_store import FAISSVectorStore
from src.rag.pipeline import RAGPipeline


st.set_page_config(
    page_title="Cogni AI",
    page_icon="src/assets/icon2.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 720px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div style="display:flex; align-items:center;">
        <img src="https://i.pinimg.com/1200x/b6/f1/58/b6f1587080522a1bb687a63069520f63.jpg" width="50" style="margin-right:10px; border-radius:50%;">
        <h1 style="margin:0;">AI Knowledge Assistant</h1>
    </div>
    <p>Ask questions about your documents</p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file:
    st.success("PDF indexed and ready")


@st.cache_resource(show_spinner=False)
def load_default_knowledge_base():
    faq_docs = TextFAQLoader("data/faqs/personal_faqs.json").load()
    pdf_docs = PDFLoader("data/documents/sample.pdf").load()

    docs = faq_docs + pdf_docs

    for doc in docs:
        doc["content"] = normalize_text(doc["content"])

    chunked_docs = chunk_documents(docs, chunk_size=700, overlap=150)

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

    chunked_docs = chunk_documents(pdf_docs, chunk_size=700, overlap=150)

    embedder = LocalEmbedder()
    embeddings = embedder.embed_texts([d["content"] for d in chunked_docs])

    store = FAISSVectorStore(embedding_dim=384)
    store.add(embeddings, chunked_docs)

    return store, embedder


if uploaded_file:
    vector_store, embedder = build_vector_store_from_uploaded_pdf(
        uploaded_file.read()
    )
else:
    vector_store, embedder = load_default_knowledge_base()

rag = RAGPipeline()


if "messages" not in st.session_state:
    st.session_state.messages = []


if not st.session_state.messages:
    st.caption(
        "Try asking:\n"
        "- *Summarize this document*\n"
    )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_query = st.chat_input("Ask a question")

if user_query:
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    query_embedding = embedder.embed_query(user_query)
    retrieved_docs = vector_store.search(query_embedding, top_k=7)

    # Answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag.generate_answer(user_query, retrieved_docs)

        st.markdown(
            f"<div style='line-height:1.6'>{answer}</div>",
            unsafe_allow_html=True
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
