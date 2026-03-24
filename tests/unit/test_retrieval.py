import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.qdrant_storage import QdrantStorage
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.chunk_retriever import ChunkRetriever
from src.retrieval.paper_retriever import PaperRetriever
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



@patch("src.retrieval.graph_retriever.QdrantClient")
@patch("src.retrieval.graph_retriever.embed_texts_with_model")
def test_graph_retriever_uses_has_chunk_path(mock_embed, mock_qdrant_cls):
    """retrieve() resolves entity names to chunks via semantic arxiv_nodes search + HAS_CHUNK traversal."""
    fake_vec = [0.2] * 768
    mock_embed.return_value = [fake_vec]

    mock_qdrant_inst = mock_qdrant_cls.return_value
    mock_point = MagicMock()
    mock_point.payload = {"node_id": 99}
    mock_result = MagicMock()
    mock_result.points = [mock_point]
    mock_qdrant_inst.query_points.return_value = mock_result

    mock_kg = MagicMock()
    mock_kg.get_chunk_refs_by_node_ids.return_value = ["uuid-001", "uuid-002"]

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
    mock_kg.get_chunk_refs_for_entity.assert_not_called()


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


def test_hybrid_retriever_three_way_fusion():
    """HybridRetriever fuses chunk, paper, and graph results with per-source RRF weights."""
    mock_chunk = MagicMock(spec=ChunkRetriever)
    mock_chunk.search.return_value = [
        RetrievedContext(text="Chunk context A", source="s1", score=0.9, collection="arxiv_chunks"),
    ]

    mock_paper = MagicMock(spec=PaperRetriever)
    mock_paper.search.return_value = [
        RetrievedContext(text="Paper context B", source="s2", score=0.8, collection="arxiv_papers"),
    ]

    mock_graph = MagicMock()
    mock_graph.retrieve.return_value = [
        RetrievedContext(text="Graph context C", source="graph", score=1.0, collection="graph"),
    ]

    hybrid = HybridRetriever(
        chunk_retriever=mock_chunk,
        paper_retriever=mock_paper,
        graph_retriever=mock_graph,
        chunk_weight=0.5,
        paper_weight=0.3,
        graph_weight=0.2,
        config_path="nonexistent.yaml",  # explicit args must win over config file
    )
    results = hybrid.retrieve(query="test query", query_embedding=[0.1] * 768, top_k=3)

    texts = [r.text for r in results]
    assert "Chunk context A" in texts
    assert "Paper context B" in texts
    assert "Graph context C" in texts
    # All results are RetrievedContext
    assert all(isinstance(r, RetrievedContext) for r in results)


def test_hybrid_retriever_deduplicates_identical_text():
    """Identical text (including whitespace variants) from chunk and graph paths is deduplicated."""
    base_text = "Self-attention computes queries, keys, and values from the same input sequence."
    # One version has leading whitespace — dedup key normalises this
    padded_text = "  " + base_text + "  "

    mock_chunk = MagicMock(spec=ChunkRetriever)
    mock_chunk.search.return_value = [
        RetrievedContext(text=base_text, source="s1", score=0.9, collection="arxiv_chunks"),
    ]

    mock_paper = MagicMock(spec=PaperRetriever)
    mock_paper.search.return_value = []

    mock_graph = MagicMock()
    mock_graph.retrieve.return_value = [
        RetrievedContext(text=padded_text, source="graph", score=1.0, collection="graph"),
    ]

    hybrid = HybridRetriever(
        chunk_retriever=mock_chunk,
        paper_retriever=mock_paper,
        graph_retriever=mock_graph,
        config_path="nonexistent.yaml",  # prevent config file from overriding test weights
    )
    results = hybrid.retrieve(query="self-attention", query_embedding=[0.1] * 768, top_k=5)

    # Deduplicated — normalised key collapses both into one result
    matching = [r for r in results if r.text.strip() == base_text]
    assert len(matching) == 1, "Whitespace-variant duplicates must be collapsed to one result"


@patch("src.retrieval.graph_retriever.QdrantClient")
@patch("src.retrieval.graph_retriever.embed_texts_with_model")
def test_graph_retriever_semantic_lookup(mock_embed, mock_qdrant_cls):
    """retrieve() embeds entity names with SPECTER2, searches arxiv_nodes, traverses HAS_CHUNK."""
    fake_vec = [0.1] * 768
    mock_embed.return_value = [fake_vec]

    mock_qdrant_inst = mock_qdrant_cls.return_value
    mock_point = MagicMock()
    mock_point.payload = {"node_id": 42}
    mock_qdrant_result = MagicMock()
    mock_qdrant_result.points = [mock_point]
    mock_qdrant_inst.query_points.return_value = mock_qdrant_result

    mock_kg = MagicMock()
    mock_kg.get_chunk_refs_by_node_ids.return_value = ["uuid-001", "uuid-002"]

    mock_extractor = MagicMock()
    mock_entity = MagicMock()
    mock_entity.text = "transformer"
    mock_extractor.extract_entities.return_value.entities = [mock_entity]

    mock_chunk_retriever = MagicMock(spec=ChunkRetriever)
    mock_chunk_retriever.fetch_by_ids.return_value = [
        RetrievedContext(
            text="Transformers use self-attention mechanisms.",
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
    results = retriever.retrieve("What is the transformer model?", top_k=5)

    # Semantic lookup: embed name -> search arxiv_nodes -> get_chunk_refs_by_node_ids
    mock_embed.assert_called_once_with(["transformer"], "allenai/specter2_base", batch_size=1)
    mock_qdrant_inst.query_points.assert_called_once_with(
        "arxiv_nodes", query=fake_vec, with_payload=True, limit=3
    )
    mock_kg.get_chunk_refs_by_node_ids.assert_called_once_with([42], limit=10)
    mock_chunk_retriever.fetch_by_ids.assert_called_once_with(["uuid-001", "uuid-002"])
    assert len(results) == 1
    assert results[0].collection == "graph"
    # Old exact-name path must NOT be used
    mock_kg.get_chunk_refs_for_entity.assert_not_called()
