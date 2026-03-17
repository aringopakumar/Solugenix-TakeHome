from langchain_core.documents import Document
from src.chunker import chunk_documents

def test_chunk_documents_size_and_overlap():
    """Test that documents are split according to chunk_size and overlap."""
    # Create a dummy document with 100 identical characters
    dummy_text = "A" * 100
    doc = Document(page_content=dummy_text, metadata={"source": "fake_doc.pdf", "page": 1})

    # Run chunker with very small limits to force multiple chunks
    chunks = chunk_documents([doc], chunk_size=30, chunk_overlap=10)

    # Since it's 100 characters, chunking by 30 with 10 overlap should yield around 5 chunks
    assert len(chunks) > 1
    
    # Check that no chunk exceeds the maximum chunk size
    assert all(len(c.page_content) <= 30 for c in chunks)
    
    # Check that metadata was preserved across chunks
    assert chunks[0].metadata["source"] == "fake_doc.pdf"
    assert chunks[0].metadata["page"] == 1
    assert chunks[-1].metadata["source"] == "fake_doc.pdf"