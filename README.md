# 🤖 Personal AI Assistant (RAG Pipeline)

A powerful retrieval-augmented generation (RAG) application that allows you to chat with your own documents (PDFs and FAQs) using **Google Gemini** and **FAISS**.

## 🚀 Features
- **PDF Upload**: Upload any PDF and ask questions about its content.
- **FAQ Support**: Pre-loaded knowledge base from structured JSON FAQs.
- **Local Embeddings**: Fast, local vectorization using `sentence-transformers`.
- **Smart Retrieval**: Uses FAISS for efficient similarity search.
- **Strict Answering**: The AI only answers based on the provided documents to prevent "hallucinations."
- **Source Attribution**: See exactly which page or FAQ the AI used to generate the answer.

---

## 🏗️ Project Architecture (A-Z)

This project follows a modular RAG architecture. Here is a breakdown of "what is where":

### Core Entry Point
- **`app.py`**: The main Streamlit interface. It orchestrates the entire flow from document upload to chat interactions.

### Source Code (`src/`)
- **`config.py`**: Handles environment variables and API key management using `python-dotenv`.
- **`loaders/`**: 
    - `pdf_loader.py`: Uses `pypdf` to extract text from PDF files.
    - `text_loader.py`: Processes JSON files into a structured document format.
- **`utils/`**:
    - `text_cleaning.py`: Normalizes extracted text and fixes common PDF character-spacing issues.
    - `chunking.py`: Splits long documents into manageable chunks (700 chars with overlap) for better search accuracy.
- **`embeddings/`**:
    - `embedder.py`: Converts text into 384-dimensional vectors using the `all-MiniLM-L6-v2` model.
- **`vectorstore/`**:
    - `faiss_store.py`: Manages the FAISS index for high-speed vector similarity searching.
- **`llm/`**:
    - `gemini_client.py`: Wrapper for the Google Gemini API (`gemini-2.5-flash-lite`) to generate responses.
- **`rag/`**:
    - `pipeline.py`: The logic that builds the "Context + Question" prompt for the LLM.

---

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **LLM**: Google Gemini API
- **Vector DB**: FAISS
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Lang**: Python 3.10+

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd final-project
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**:
   Create a `.env` file in the root directory and add your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

5. **Run the App**:
   ```bash
   streamlit run app.py
   ```

---

## 📄 Usage
1. Open the URL provided by Streamlit (usually `http://localhost:8501`).
2. The app will automatically load the default FAQ and sample PDF.
3. Upload a new PDF using the sidebar/uploader to chat with specific documents.
4. Type your question in the chat input at the bottom.
5. Check the **"Sources"** dropdown below the AI's answer to verify its facts.
