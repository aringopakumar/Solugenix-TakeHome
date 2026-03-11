"""
Document Loader Module
======================
Responsible for loading raw documents from the data/ directory.
Supports two formats:
  - .txt  files → loaded with TextLoader
  - .pdf  files → loaded with PyPDFLoader (one Document per page)

Each loader returns a list of LangChain Document objects, which carry
the page content and metadata (source file path, page number, etc.).
"""

import os
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document


def load_documents(data_dir: str = "data") -> List[Document]:
    """
    Walk through `data_dir`, find every .txt and .pdf file,
    and return a flat list of LangChain Document objects.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing company documents.

    Returns
    -------
    List[Document]
        All loaded documents with their metadata intact.
    """
    documents: List[Document] = []

    # Iterate over every file in the data directory
    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)

        if filename.endswith(".txt"):
            # TextLoader reads the entire file as a single Document
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())

        elif filename.endswith(".pdf"):
            # PyPDFLoader splits the PDF into one Document per page
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())

    if not documents:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in '{data_dir}'. "
            "Please add documents before running the pipeline."
        )

    print(f"✔ Loaded {len(documents)} document(s) from '{data_dir}'")
    return documents
