"""Splits documents into smaller overlapping chunks for embedding."""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 60,
) -> List[Document]:
    """Split documents into chunks. Metadata is preserved on each chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks #same list of documents (objects), just that each chunk is shorter, metdata preserved
