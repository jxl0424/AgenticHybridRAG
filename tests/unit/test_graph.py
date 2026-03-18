from unittest.mock import MagicMock, patch


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
