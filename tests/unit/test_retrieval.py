import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.qdrant_storage import QdrantStorage
from src.retrieval.graph_retriever import GraphRetriever, GraphRetrievalResult
from src.retrieval.hybrid_retriever import HybridRetriever, HybridRetrievalResult
from src.retrieval.chunk_retriever import ChunkRetriever
from src.types import RetrievedContext

@patch('src.retrieval.qdrant_storage.QdrantClient')
def test_qdrant_search(mock_qdrant):
    # Setup mock
    client_instance = mock_qdrant.return_value
    client_instance.collection_exists.return_value = True
    
    # Mock search results
    mock_point = MagicMock()
    mock_point.score = 0.9
    mock_point.payload = {"text": "This is a long enough medical context to pass correctly.", "source": "test.pdf"}
    
    mock_results = MagicMock()
    mock_results.points = [mock_point]
    client_instance.query_points.return_value = mock_results
    
    storage = QdrantStorage()
    results = storage.search(query_vector=[0.1]*768, top_k=1)
    
    assert len(results["contexts"]) == 1
    assert results["contexts"][0] == mock_point.payload["text"]


def test_hybrid_fusion():
    # Mock sub-retrievers
    mock_qdrant = MagicMock()
    mock_qdrant.search.return_value = {
        "contexts": ["Vector Context"],
        "sources": ["source1"],
        "scores": [0.8]
    }
    
    mock_graph = MagicMock()
    mock_graph.retrieve.return_value = GraphRetrievalResult(
        contexts=["Graph Context"],
        sources=["graph"],
        entities_found=["Entity"],
        graph_paths=[]
    )
    
    # Default weights are normalized: 0.6 / (0.6+0.4) = 0.6, 0.4 / (0.6+0.4) = 0.4
    hybrid = HybridRetriever(
        qdrant_storage=mock_qdrant,
        graph_retriever=mock_graph,
        vector_weight=0.6,
        graph_weight=0.4
    )
    
    results = hybrid.retrieve(query="test", query_embedding=[0.1]*768, top_k=2)
    
    # Verify fusion contains both
    context_texts = results.contexts
    assert "Vector Context" in context_texts
    assert "Graph Context" in context_texts
    
    # Fusion logic check:
    # ctx_idx 0 graph score = 1.0 (since only one graph context)
    # final_score = (0.6 * 0.8) + (0.4 * 1.0) = 0.48 + 0.4 = 0.88
    # If the retriever uses normalized weights and our config matches:
    # result 1 (Graph Context) has graph_score 1.0, vector_score 0.0 -> score = 0.4
    # result 2 (Vector Context) has graph_score 0.0, vector_score 0.8 -> score = 0.48
    # Wait, let's check the labels in hybrid_retriever.py:
    # Context scores are tracked by text.
    # Result scores depend on which weight is higher.
    
    # In my manual check of hybrid_retriever.py:
    # context_scores[ctx]['vector_score'] = max(..., score)
    # result['final_score'] = self.vector_weight * vector_score + self.graph_weight * graph_score
    
    # "Vector Context": vector=0.8, graph=0.0 -> final = 0.6 * 0.8 + 0.4 * 0.0 = 0.48
    # "Graph Context": vector=0.0, graph=1.0 -> final = 0.6 * 0.0 + 0.4 * 1.0 = 0.4
    
    assert 0.48 in results.scores
    assert 0.4 in results.scores


def test_graph_retriever_uses_has_chunk_path():
    """GraphRetriever finds entities, gets qdrant_ids via HAS_CHUNK, resolves via ChunkRetriever."""
    mock_kg = MagicMock()
    mock_kg.get_chunk_refs_for_entity.return_value = ["uuid-001", "uuid-002"]
    mock_kg.get_related_entities.return_value = []

    mock_extractor = MagicMock()
    mock_entity = MagicMock()
    mock_entity.text = "transformer"
    mock_extractor.extract_entities.return_value.entities = [mock_entity]

    mock_chunk_retriever = MagicMock(spec=ChunkRetriever)
    mock_chunk_retriever.fetch_by_ids.return_value = [
        RetrievedContext(
            text="Transformers use self-attention mechanisms extensively.",
            source="1706.03762",
            score=1.0,
            collection="arxiv_chunks",
        )
    ]

    retriever = GraphRetriever(
        knowledge_graph=mock_kg,
        entity_extractor=mock_extractor,
        chunk_retriever=mock_chunk_retriever,
    )
    results = retriever.retrieve("What is the transformer model?")

    assert len(results) == 1
    assert isinstance(results[0], RetrievedContext)
    assert results[0].collection == "graph"
    mock_chunk_retriever.fetch_by_ids.assert_called_once_with(["uuid-001", "uuid-002"])


def test_graph_retriever_keyword_fallback_no_entities():
    """When no entities are found, _keyword_fallback uses parameterised Cypher on _Embeddable nodes."""
    mock_kg = MagicMock()
    mock_kg.client.execute_read.return_value = [
        {"name": "attention mechanism", "type": "ALGORITHM"}
    ]

    mock_extractor = MagicMock()
    mock_extractor.extract_entities.return_value.entities = []

    mock_chunk_retriever = MagicMock(spec=ChunkRetriever)
    mock_chunk_retriever.fetch_by_ids.return_value = []

    retriever = GraphRetriever(
        knowledge_graph=mock_kg,
        entity_extractor=mock_extractor,
        chunk_retriever=mock_chunk_retriever,
    )
    results = retriever.retrieve("explain attention in neural networks", top_k=3)

    # Verify parameterised Cypher -- must NOT use f-string injection
    call_args = mock_kg.client.execute_read.call_args
    cypher_str = call_args[0][0]
    params_dict = call_args[0][1]
    assert "$keywords" in cypher_str, "Cypher must use $keywords param, not f-string injection"
    assert isinstance(params_dict.get("keywords"), list), "keywords must be a list param"
    assert isinstance(results, list)
