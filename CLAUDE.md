# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An AI-powered FAQ chatbot using Retrieval-Augmented Generation (RAG) built with LangChain, Ollama (local models), FAISS, and Streamlit.

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama (required for embeddings and LLM)
# https://ollama.com — then pull the required models:
ollama pull nomic-embed-text   # embeddings model
ollama pull llama3.2           # LLM model
```

## Running the App

```bash
streamlit run app.py
```

## Architecture

The pipeline has two phases:

**Ingestion (sidebar UI):**
User uploads `.txt`/`.pdf` files → written to a temp dir → `src/loader.py` (TextLoader/PyPDFLoader) → `src/chunker.py` (300 char chunks, 60 overlap, splits on `\n\n`/`\n`/`. `/` `) → `src/vectorstore.py` (OllamaEmbeddings `nomic-embed-text` → FAISS index saved to `faiss_index/`)

**Query (chat UI):**
User question → FAISS retriever (MMR search, k=6, fetch_k=24) → `src/qa_chain.py` (RetrievalQA with ChatOllama `llama3.2`, temperature=0.0) → Answer + source documents

**Key design decisions:**
- Uses fully local Ollama models — no API keys or external services required
- FAISS index persists to `faiss_index/` on disk; session state caches `vectorstore` and `qa_chain`
- MMR (Maximal Marginal Relevance) retrieval reduces redundancy across retrieved chunks
- `temperature=0.0` with a custom prompt constrains the LLM to answer only from retrieved context, instructing it to say "I don't know" when context is insufficient
- `allow_dangerous_deserialization=True` is required for FAISS loading (LangChain requirement)
- After indexing, the app auto-generates 4 example questions by sampling chunks and prompting the LLM

## Module Responsibilities

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI, sidebar file uploader, session state, orchestration, source reference rendering |
| `src/loader.py` | Load `.txt` and `.pdf` files from a directory (TextLoader / PyPDFLoader) |
| `src/chunker.py` | Split documents with `RecursiveCharacterTextSplitter` (chunk_size=300, overlap=60) |
| `src/vectorstore.py` | Build/save/load FAISS vector store with `OllamaEmbeddings(model="nomic-embed-text")` |
| `src/qa_chain.py` | Build `RetrievalQA` chain with `ChatOllama(model="llama3.2")` and custom prompt template |

## Environment

- No API key is required — the current implementation uses local Ollama models
- `.env.example` contains a legacy `OPENAI_API_KEY` placeholder that is not used by the current code
- `faiss_index/` — auto-generated on first ingestion, excluded from git
- `langchain-openai` is listed in `requirements.txt` but not used in the current implementation
