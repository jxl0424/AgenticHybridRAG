import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.chunk_retriever import ChunkRetriever
from src.types import RetrievedContext


@patch("src.retrieval.chunk_retriever.QdrantClient")
def test_search_maps_paragraph_field(mock_client_cls):
    client = mock_client_cls.return_value
    client.collection_exists.return_value = True

    point = MagicMock()
    point.score = 0.85
    point.payload = {
        "paragraph": "Attention is all you need for the transformer model architecture.",
        "source": "1706.03762",
        "edge_id": 42,
        "src_id": 1,
        "dst_id": 2,
        "rel_type": "CITES",
        "domain": "arxiv_ai",
        "paper_id": "1706.03762",
    }
    client.query_points.return_value.points = [point]

    retriever = ChunkRetriever()
    results = retriever.search(query_vector=[0.1] * 768, top_k=1)

    assert len(results) == 1
    assert isinstance(results[0], RetrievedContext)
    assert results[0].text == "Attention is all you need for the transformer model architecture."
    assert results[0].collection == "arxiv_chunks"
    assert results[0].score == 0.85
    assert results[0].metadata["edge_id"] == 42


@patch("src.retrieval.chunk_retriever.QdrantClient")
def test_search_skips_short_text(mock_client_cls):
    client = mock_client_cls.return_value
    client.collection_exists.return_value = True

    point = MagicMock()
    point.score = 0.9
    point.payload = {"paragraph": "Too short.", "source": "x"}
    client.query_points.return_value.points = [point]

    retriever = ChunkRetriever()
    results = retriever.search(query_vector=[0.1] * 768, top_k=1)

    assert results == []


@patch("src.retrieval.chunk_retriever.QdrantClient")
def test_fetch_by_ids_returns_contexts(mock_client_cls):
    client = mock_client_cls.return_value
    client.collection_exists.return_value = True

    point = MagicMock()
    point.id = "abc-123"
    point.payload = {
        "paragraph": "A sufficiently long retrieved paragraph for the test.",
        "source": "paper1",
        "edge_id": 7,
        "src_id": 3,
        "dst_id": 4,
        "rel_type": "RELATED_TO",
        "domain": "arxiv_ai",
        "paper_id": "paper1",
    }
    client.retrieve.return_value = [point]

    retriever = ChunkRetriever()
    results = retriever.fetch_by_ids(["abc-123"])

    assert len(results) == 1
    assert results[0].text == "A sufficiently long retrieved paragraph for the test."
    assert results[0].collection == "arxiv_chunks"
    assert results[0].score == 1.0  # graph-resolved contexts score 1.0
