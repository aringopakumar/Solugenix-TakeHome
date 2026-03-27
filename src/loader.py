"""Loads .txt and .pdf files from a directory into LangChain Document objects."""

import os
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document


def load_documents(data_dir: str = "data") -> List[Document]:
    """Scan data_dir for .txt/.pdf files and return them as Documents."""
    documents: List[Document] = []

    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename) #builds full path to file

        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())

    if not documents:
        raise FileNotFoundError(f"No .txt or .pdf files found in '{data_dir}'.")

    print(f"Loaded {len(documents)} document(s) from {data_dir}")
    return documents
