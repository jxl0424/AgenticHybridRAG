import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.paper_retriever import PaperRetriever
from src.types import RetrievedContext


@patch("src.retrieval.paper_retriever.QdrantClient")
def test_search_maps_chunk_text_field(mock_client_cls):
    client = mock_client_cls.return_value
    client.collection_exists.return_value = True

    point = MagicMock()
    point.score = 0.78
    point.payload = {
        "chunk_text": "Transformers use self-attention to process sequences in parallel.",
        "arxiv_id": "1706.03762",
        "chunk_index": 3,
        "title": "Attention Is All You Need",
        "domain": "arxiv_ai",
    }
    client.query_points.return_value.points = [point]

    retriever = PaperRetriever()
    results = retriever.search(query_vector=[0.1] * 768, top_k=1)

    assert len(results) == 1
    assert isinstance(results[0], RetrievedContext)
    assert results[0].text == "Transformers use self-attention to process sequences in parallel."
    assert results[0].collection == "arxiv_papers"
    assert results[0].score == 0.78
    assert results[0].metadata["arxiv_id"] == "1706.03762"
    assert results[0].metadata["chunk_index"] == 3


@patch("src.retrieval.paper_retriever.QdrantClient")
def test_search_filters_below_min_score(mock_client_cls):
    client = mock_client_cls.return_value
    client.collection_exists.return_value = True

    point = MagicMock()
    point.score = 0.05
    point.payload = {"chunk_text": "A sufficiently long text that passes length check but fails score.", "arxiv_id": "x"}
    client.query_points.return_value.points = [point]

    retriever = PaperRetriever()
    results = retriever.search(query_vector=[0.1] * 768, top_k=1, min_score=0.2)

    assert results == []
