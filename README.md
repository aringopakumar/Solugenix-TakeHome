# 💬 AI-Powered FAQ Chatbot

A clean, modular RAG (Retrieval-Augmented Generation) pipeline that ingests company documents and answers questions grounded in those documents.

Built with **LangChain**, **Ollama (Llama 3.2)**, **FAISS**, and **Streamlit** — **fully local, no API keys required**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE (one-time)            │
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
│                    QUERY PIPELINE (per question)            │
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
├── README.md              ← You are here
├── requirements.txt       ← Python dependencies
├── .env.example           ← Optional config (no API keys needed)
├── .gitignore             ← Keeps build artifacts out of git
├── data/                  ← Drop your .txt and .pdf documents here
│   └── sample_faq.txt     ← Example FAQ document for testing
├── src/
│   ├── __init__.py
│   ├── loader.py          ← Loads PDFs and text files into Document objects
│   ├── chunker.py         ← Splits documents into smaller overlapping chunks
│   ├── vectorstore.py     ← Embeds chunks and manages the FAISS vector store
│   └── qa_chain.py        ← Builds the retrieval + LLM question-answering chain
└── app.py                 ← Streamlit web UI
```

### Module Purposes

| Module | Responsibility |
|---|---|
| `loader.py` | Reads `.txt` and `.pdf` files from a directory and returns LangChain `Document` objects with metadata. |
| `chunker.py` | Splits documents into smaller chunks using `RecursiveCharacterTextSplitter` with configurable size and overlap. |
| `vectorstore.py` | Embeds chunks using Ollama (`nomic-embed-text`), stores them in a FAISS index, and supports save/load from disk. |
| `qa_chain.py` | Combines the FAISS retriever with Llama 3.2 (via Ollama) and a custom prompt to produce grounded answers. |
| `app.py` | Streamlit chat UI with sidebar for document upload, index building, chat history, and source document display. |

---

## Prerequisites

### Install Ollama

This project runs models locally via [Ollama](https://ollama.com). Install it first:

```bash
# macOS
brew install ollama

# Or download from https://ollama.com/download
```

### Pull the required models

```bash
# LLM for answering questions
ollama pull llama3.2

# Embedding model for vector search
ollama pull nomic-embed-text
```

Make sure the Ollama server is running before starting the app:

```bash
ollama serve
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/aringopakumar/Solugenix-TakeHome.git
cd Solugenix-TakeHome

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Ollama (if not already running)

```bash
ollama serve
```

### 3. Run the app

```bash
streamlit run app.py
```

Then:
1. Upload `.txt` or `.pdf` files in the sidebar.
2. Click **Build index** to embed your documents.
3. Ask questions in the chat input.

A sample FAQ file (`data/sample_faq.txt`) is included for testing.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Fully local stack** | No API keys, no costs, no data leaving your machine. Uses Ollama for both LLM and embeddings. |
| **FAISS** over Chroma/Pinecone | Runs locally with zero setup; great for demos and take-home assignments. |
| **Ollama + Llama 3.2** | High-quality open-source LLM that runs well on consumer hardware. |
| **nomic-embed-text** | Fast, high-quality local embedding model optimized for retrieval tasks. |
| **RecursiveCharacterTextSplitter** | Splits on natural boundaries (paragraphs → sentences → words) before falling back to character limits. |
| **`chain_type="stuff"`** | Simplest approach — concatenates all retrieved chunks into one prompt. Works well when chunks are small. |
| **`temperature=0.0`** | Produces deterministic, factual answers — important for an FAQ bot. |
| **Custom prompt with refusal** | Instructs the LLM to say "I don't know" rather than hallucinate when context is insufficient. |
| **Source document display** | Shows users which chunks informed the answer, building trust and traceability. |

---

## Interview Talking Points

Here are features and concepts you can discuss confidently:

1. **RAG Pattern** — Retrieval-Augmented Generation separates knowledge (vector store) from reasoning (LLM), making the system updatable without retraining.

2. **Fully Local Architecture** — No external API calls. All inference runs on-device via Ollama, ensuring data privacy and zero cost.

3. **Chunking Strategy** — Why overlap matters (prevents losing context at boundaries). Why chunk size matters (too large = noisy retrieval, too small = lost meaning).

4. **Embeddings** — Text → dense vector. Semantically similar text produces similar vectors. This enables "meaning-based" search rather than keyword matching.

5. **Vector Store** — FAISS performs approximate nearest neighbor search to find the top-k most relevant chunks efficiently.

6. **Prompt Engineering** — The custom prompt constrains the LLM to answer only from provided context, reducing hallucinations.

7. **Graceful Unknowns** — The system explicitly handles questions outside its knowledge base instead of guessing.

8. **Source Attribution** — Showing source documents makes the system transparent and auditable.

---

## Ideas to Strengthen the Project

If you want to go further (without over-complicating):

- **Add a confidence score** — Show the similarity score from the retriever alongside each source.
- **Support more file types** — Add `.docx`, `.csv`, or `.md` loaders.
- **Conversation memory** — Use LangChain's `ConversationBufferMemory` for multi-turn follow-ups.
- **Evaluation** — Create a small test set of question-answer pairs and measure retrieval accuracy.
- **Dockerize** — Add a `Dockerfile` for easy deployment.
- **Try different models** — Swap `llama3.2` for `mistral` or `phi3` via a single parameter change.

---

## License

This project is for educational and interview demonstration purposes.
