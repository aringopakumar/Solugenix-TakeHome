"""
Vector Store Module
===================
Handles two operations:
  1. **Build** – take chunked Documents, embed them with Ollama
     (nomic-embed-text model), and persist a FAISS index to disk.
  2. **Load** – read a previously-saved FAISS index from disk so
     the app can start up without re-embedding every time.

Why FAISS?
  - Runs locally (no external server needed).
  - Fast similarity search, even on a laptop.
  - Easy to persist and reload from a directory.

Why Ollama Embeddings?
  - Fully local — no API key or external calls required.
  - Uses the nomic-embed-text model for high-quality embeddings.
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Default directory where the FAISS index is saved
DEFAULT_PERSIST_DIR = "faiss_index"


def build_vectorstore(
    chunks: List[Document],
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> FAISS:
    """
    Embed document chunks and store them in a FAISS vector store.

    Parameters
    ----------
    chunks : List[Document]
        Chunked documents to embed.
    persist_dir : str
        Directory where the FAISS index will be saved.

    Returns
    -------
    FAISS
        The populated vector store, ready for similarity search.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create the FAISS index from the document chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist to disk so we don't have to re-embed on every restart
    vectorstore.save_local(persist_dir)
    print(f"✔ Vector store built and saved to '{persist_dir}'")
    return vectorstore


def load_vectorstore(persist_dir: str = DEFAULT_PERSIST_DIR) -> FAISS:
    """
    Load a previously-saved FAISS vector store from disk.

    Parameters
    ----------
    persist_dir : str
        Directory containing the saved FAISS index.

    Returns
    -------
    FAISS
        The loaded vector store.

    Raises
    ------
    FileNotFoundError
        If the persist directory does not exist.
    """
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"No vector store found at '{persist_dir}'. "
            "Run the ingestion pipeline first to build the index."
        )

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True,  # Required by FAISS loader
    )
    print(f"✔ Vector store loaded from '{persist_dir}'")
    return vectorstore
