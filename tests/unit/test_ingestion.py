import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.ingestion.pdf_loader import load_and_chunk_pdf, embed_texts

@patch('src.ingestion.pdf_loader.PDFReader')
@patch('src.ingestion.pdf_loader.splitter')
def test_load_and_chunk_pdf(mock_splitter, mock_pdf_reader):
    # Setup mocks
    mock_doc = MagicMock()
    mock_doc.text = "This is a medical test document."
    mock_pdf_reader.return_value.load_data.return_value = [mock_doc]
    mock_splitter.split_text.return_value = ["This is a medical", "test document."]
    
    # Call function
    chunks = load_and_chunk_pdf("dummy.pdf")
    
    # Assertions
    assert len(chunks) == 2
    assert chunks[0] == "This is a medical"
    mock_pdf_reader.return_value.load_data.assert_called_once_with(file="dummy.pdf")

@patch('src.ingestion.pdf_loader.local_embed_model')
def test_embed_texts(mock_embed_model):
    # Setup mock
    mock_embedding = np.random.rand(1, 768)
    mock_embed_model.encode.return_value = mock_embedding
    
    # Call function
    texts = ["Sample text"]
    embeddings = embed_texts(texts)
    
    # Assertions
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768
    assert mock_embed_model.encode.called
