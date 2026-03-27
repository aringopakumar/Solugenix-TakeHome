from langchain_core.documents import Document
from src.chunker import chunk_documents

def test_chunk_documents_size_and_overlap():
    doc = Document(page_content="A" * 100, metadata={"source": "test.pdf", "page": 1})
    chunks = chunk_documents([doc], chunk_size=30, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 30 for c in chunks) #chunks are 30 chars or less
    assert chunks[0].metadata["source"] == "test.pdf" #ensures metadata present, doesn't go away when chunking
    assert chunks[-1].metadata["source"] == "test.pdf"
