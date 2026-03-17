import os
import tempfile
import pytest
from src.loader import load_documents

def test_load_documents_success():
    """Test that a standard text file is loaded correctly into a Document object."""
    # Create a temporary directory and file for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file_path = os.path.join(tmp_dir, "dummy_test.txt")
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write("This is a dummy test document for unit testing.")

        # Run the loader
        docs = load_documents(tmp_dir)

        # Assertions to prove it worked
        assert len(docs) == 1
        assert docs[0].page_content == "This is a dummy test document for unit testing."
        assert "dummy_test.txt" in docs[0].metadata["source"]

def test_load_documents_empty_directory():
    """Test that an empty directory properly raises a FileNotFoundError."""
    with tempfile.TemporaryDirectory() as empty_dir:
        with pytest.raises(FileNotFoundError):
            load_documents(empty_dir)