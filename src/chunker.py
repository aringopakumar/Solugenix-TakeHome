"""
Text Chunking Module
====================
Splits large documents into smaller, overlapping chunks so that:
  1. Each chunk fits within the embedding model's context window.
  2. Overlap preserves context across chunk boundaries, reducing
     the chance of cutting a sentence or idea in half.

We use LangChain's RecursiveCharacterTextSplitter, which tries to
split on natural boundaries (paragraphs → sentences → words) before
falling back to a hard character limit.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 60,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.

    Parameters
    ----------
    documents : List[Document]
        Raw documents returned by the loader.
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.
        Helps preserve context at chunk boundaries.

    Returns
    -------
    List[Document]
        A new list of smaller Document objects, each retaining the
        original metadata (source, page number, etc.).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try splitting on these separators in order of preference
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"✔ Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
