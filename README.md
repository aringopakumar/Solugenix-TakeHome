# FAQ Chatbot

A local RAG (Retrieval-Augmented Generation) pipeline that ingests company documents and answers questions grounded in those documents.

Built with **LangChain**, **Ollama (Llama 3.2)**, **FAISS**, and **Streamlit**. Runs entirely on your machine — no API keys required.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION (one-time)                     │
│                                                             │
│  PDF / TXT files                                            │
│       │                                                     │
│       ▼                                                     │
│  loader.py ──► Raw Documents                                │
│       │                                                     │
│       ▼                                                     │
│  chunker.py ──► Smaller overlapping chunks                  │
│       │                                                     │
│       ▼                                                     │
│  vectorstore.py ──► Ollama Embeddings ──► FAISS index       │
│                     (nomic-embed-text)    (saved to disk)   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY (per question)                     │
│                                                             │
│  User Question                                              │
│       │                                                     │
│       ▼                                                     │
│  FAISS Retriever ──► Top-k relevant chunks                  │
│       │                                                     │
│       ▼                                                     │
│  Prompt Template ──► "Answer using ONLY this context"       │
│       │                                                     │
│       ▼                                                     │
│  LLM (Llama 3.2 via Ollama) ──► Grounded answer + sources  │
│       │                                                     │
│       ▼                                                     │
│  Streamlit UI ──► Display answer + source documents         │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── requirements.txt       ← Python dependencies
├── .gitignore
├── data/                  ← Drop your .txt and .pdf documents here
│   └── sample_faq.txt     ← Example document for testing
├── src/
│   ├── loader.py          ← Loads PDFs and text files
│   ├── chunker.py         ← Splits documents into overlapping chunks
│   ├── vectorstore.py     ← Embeds chunks → FAISS vector store
│   └── qa_chain.py        ← Retrieval + LLM question-answering chain
└── app.py                 ← Streamlit web UI
```

| Module | What it does |
|---|---|
| `loader.py` | Reads `.txt` and `.pdf` files and returns LangChain `Document` objects. |
| `chunker.py` | Splits documents into smaller chunks with configurable size and overlap. |
| `vectorstore.py` | Embeds chunks via Ollama (`nomic-embed-text`) and stores them in a FAISS index. |
| `qa_chain.py` | Retrieves relevant chunks and passes them to Llama 3.2 with a grounded prompt. |
| `app.py` | Streamlit chat UI with document upload, index building, and source display. |

---

## Prerequisites

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.com/download
```

### 2. Pull the required models

```bash
ollama pull llama3.2            # LLM
ollama pull nomic-embed-text    # Embedding model
```

### 3. Start the Ollama server

```bash
ollama serve
```

---

## Setup

```bash
git clone https://github.com/aringopakumar/Solugenix-TakeHome.git
cd Solugenix-TakeHome

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run app.py
```

1. Upload `.txt` or `.pdf` files in the sidebar.
2. Click **Build index** to embed the documents.
3. Ask questions in the chat.

A sample FAQ file is included in `data/` for quick testing.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Orchestration** | LangChain |
| **LLM** | Llama 3.2 (via Ollama, local) |
| **Embeddings** | nomic-embed-text (via Ollama, local) |
| **Vector Store** | FAISS |
| **Frontend** | Streamlit |
