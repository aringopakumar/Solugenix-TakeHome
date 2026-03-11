# 💬 AI-Powered FAQ Chatbot

A clean, modular RAG (Retrieval-Augmented Generation) pipeline that ingests company documents and answers questions grounded in those documents.

Built with **LangChain**, **OpenAI**, **FAISS**, and **Streamlit**.

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
│  vectorstore.py ──► OpenAI Embeddings ──► FAISS index       │
│                                          (saved to disk)    │
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
│  LLM (GPT-3.5 / GPT-4) ──► Grounded answer + sources       │
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
├── .env.example           ← Template for your OpenAI API key
├── .gitignore             ← Keeps secrets and build artifacts out of git
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
| `vectorstore.py` | Embeds chunks using OpenAI Embeddings, stores them in a FAISS index, and supports save/load from disk. |
| `qa_chain.py` | Combines the retriever with an LLM via a custom prompt to produce answers grounded in retrieved context. |
| `app.py` | Streamlit chat UI with sidebar for ingestion, chat history, and source document display. |

---

## Quick Start

### 1. Clone and install

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and replace sk-your-api-key-here with your actual OpenAI API key
```

### 3. Add documents

Place your `.txt` and/or `.pdf` files in the `data/` folder. A sample FAQ file is included.

### 4. Run the app

```bash
streamlit run app.py
```

Then:
1. Click **📥 Ingest Documents** in the sidebar to embed your documents.
2. Ask questions in the chat input.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **FAISS** over Chroma/Pinecone | Runs locally with zero setup; great for demos and take-home assignments. |
| **RecursiveCharacterTextSplitter** | Splits on natural boundaries (paragraphs, sentences) before falling back to character limits. |
| **`chain_type="stuff"`** | Simplest approach — concatenates all retrieved chunks into one prompt. Works well when chunks are small. |
| **`temperature=0.0`** | Produces deterministic, factual answers — important for an FAQ bot. |
| **Custom prompt with refusal** | Instructs the LLM to say "I don't know" rather than hallucinate when context is insufficient. |
| **Source document display** | Shows users which chunks informed the answer, building trust and traceability. |

---

## Interview Talking Points

Here are features and concepts you can discuss confidently:

1. **RAG Pattern** — Retrieval-Augmented Generation separates knowledge (vector store) from reasoning (LLM), making the system updatable without retraining.

2. **Chunking Strategy** — Why overlap matters (prevents losing context at boundaries). Why chunk size matters (too large = noisy retrieval, too small = lost meaning).

3. **Embeddings** — Text → dense vector. Semantically similar text produces similar vectors. This enables "meaning-based" search rather than keyword matching.

4. **Vector Store** — FAISS performs approximate nearest neighbor search to find the top-k most relevant chunks efficiently.

5. **Prompt Engineering** — The custom prompt constrains the LLM to answer only from provided context, reducing hallucinations.

6. **Graceful Unknowns** — The system explicitly handles questions outside its knowledge base instead of guessing.

7. **Source Attribution** — Showing source documents makes the system transparent and auditable.

---

## Ideas to Strengthen the Project

If you want to go further (without over-complicating):

- **Add a confidence score** — Show the similarity score from the retriever alongside each source.
- **Support more file types** — Add `.docx`, `.csv`, or `.md` loaders.
- **Conversation memory** — Use LangChain's `ConversationBufferMemory` for multi-turn follow-ups.
- **Evaluation** — Create a small test set of question-answer pairs and measure retrieval accuracy.
- **Dockerize** — Add a `Dockerfile` for easy deployment.

---

## License

This project is for educational and interview demonstration purposes.
