"""Build and load a FAISS vector store using Ollama embeddings."""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

DEFAULT_PERSIST_DIR = "faiss_index"

#Builds a searchable FAISS index and saves to disk, while running each chunk and saving the vector embedding
def build_vectorstore(
    chunks: List[Document],
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> FAISS:
    """Embed chunks and save the FAISS index to persist_dir."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_dir)
    print(f"Vector store saved to {persist_dir}")
    return vectorstore

#Loads built in index back from disk into memory
def load_vectorstore(persist_dir: str = DEFAULT_PERSIST_DIR) -> FAISS:
    """Load a FAISS index from disk. Raises FileNotFoundError if missing."""
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"No index found at '{persist_dir}'.")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"Loaded index from {persist_dir}")
    return vectorstore
