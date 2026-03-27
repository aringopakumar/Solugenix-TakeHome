import os
import tempfile
import pytest
from src.loader import load_documents

def test_load_documents_success():
    with tempfile.TemporaryDirectory() as tmp_dir: #creates temp folder
        path = os.path.join(tmp_dir, "test.txt") #creates path in dir
        with open(path, "w", encoding="utf-8") as f:
            f.write("Hello world.") #adds file to dir

        docs = load_documents(tmp_dir) 

        assert len(docs) == 1
        assert docs[0].page_content == "Hello world." 
        assert "test.txt" in docs[0].metadata["source"]

def test_load_documents_empty_dir():
    with tempfile.TemporaryDirectory() as empty_dir: #creates empty dir
        with pytest.raises(FileNotFoundError):
            load_documents(empty_dir)
