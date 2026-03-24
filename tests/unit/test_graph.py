from unittest.mock import MagicMock, patch

from src.graph.cs_knowledge_graph import CSKnowledgeGraph as _RealCSKG


@patch("src.graph.cs_knowledge_graph.Neo4jClient")
def test_get_chunk_refs_for_entity_returns_qdrant_ids(mock_neo4j_cls):
    client = mock_neo4j_cls.return_value
    client.connect.return_value = None
    client.execute_read.return_value = [
        {"qdrant_id": "uuid-aaa"},
        {"qdrant_id": "uuid-bbb"},
        {"qdrant_id": None},  # should be filtered out
    ]

    from src.graph.cs_knowledge_graph import CSKnowledgeGraph
    kg = CSKnowledgeGraph(neo4j_client=client)
    result = kg.get_chunk_refs_for_entity("transformer")

    assert result == ["uuid-aaa", "uuid-bbb"]
    call_args = client.execute_read.call_args
    assert "$name" in call_args[0][0]
    assert call_args[0][1]["name"] == "transformer"


@patch("src.graph.cs_knowledge_graph.Neo4jClient")
def test_get_chunk_refs_by_node_ids_returns_qdrant_ids(mock_neo4j_cls):
    client = MagicMock()
    client.connect.return_value = None
    client.execute_read.return_value = [
        {"qdrant_id": "uuid-aaa"},
        {"qdrant_id": "uuid-bbb"},
        {"qdrant_id": None},  # must be filtered out
    ]

    kg = _RealCSKG(neo4j_client=client)
    result = kg.get_chunk_refs_by_node_ids([101, 202], limit=10)

    assert result == ["uuid-aaa", "uuid-bbb"]
    call_args = client.execute_read.call_args
    cypher = call_args.args[0]
    params = call_args.args[1]
    assert "node_id IN $node_ids" in cypher
    assert "-[r:HAS_CHUNK]-" in cypher       # undirected match (no arrow)
    assert "DISTINCT" in cypher               # deduplication
    assert params["node_ids"] == [101, 202]
    assert params["limit"] == 10


@patch("src.graph.cs_knowledge_graph.Neo4jClient")
def test_get_chunk_refs_by_node_ids_empty_input(mock_neo4j_cls):
    client = MagicMock()
    client.connect.return_value = None
    client.execute_read.return_value = []

    kg = _RealCSKG(neo4j_client=client)
    result = kg.get_chunk_refs_by_node_ids([], limit=10)

    assert result == []
