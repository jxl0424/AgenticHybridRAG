import pytest
import uuid
import numpy as np
from src.retrieval.qdrant_storage import QdrantStorage
from src.graph.knowledge_graph import KnowledgeGraph, GraphEntity, GraphRelationship

@pytest.fixture(scope="module")
def qdrant():
    """Fixture for real Qdrant storage using a test collection."""
    test_collection = f"test_collection_{uuid.uuid4().hex[:8]}"
    storage = QdrantStorage(collection=test_collection)
    yield storage
    # Cleanup: Delete the test collection
    storage.client.delete_collection(test_collection)

@pytest.fixture(scope="module")
def neo4j():
    """Fixture for real Neo4j knowledge graph."""
    kg = KnowledgeGraph()
    kg.initialize_schema()
    yield kg
    # Cleanup: Delete test entities (identified by a specific property)
    kg.client.execute_write("MATCH (e:MedicalEntity {test_run: true}) DETACH DELETE e")

def test_qdrant_integration(qdrant):
    # Test data
    item_id = str(uuid.uuid4())
    vector = [0.1] * 768
    payload = {"text": "This is a real integration test context for Qdrant storage.", "source": "test.pdf"}
    
    # Upsert
    qdrant.upsert(ids=[item_id], vectors=[vector], payloads=[payload])
    
    # Wait briefly or just search (Qdrant is usually fast enough)
    results = qdrant.search(query_vector=vector, top_k=1, min_score=0.1)
    
    assert len(results["contexts"]) > 0
    assert results["contexts"][0] == payload["text"]
    assert results["sources"][0] == "test.pdf"

def test_neo4j_integration(neo4j):
    # Test data
    entity_name = f"TestVirus_{uuid.uuid4().hex[:4]}"
    entity = GraphEntity(name=entity_name, entity_type="DISEASE", metadata={"test_run": True})
    
    # Add entity
    neo4j.add_entity(entity)
    
    # Verify entity exists
    # We'll use a raw query or a retrieval method
    query = "MATCH (e:MedicalEntity {name: $name}) RETURN e.entity_type as type"
    results = neo4j.client.execute_read(query, {"name": entity_name})
    
    assert len(results) > 0
    assert results[0]["type"] == "DISEASE"
    
    # Test relationship
    target_name = f"TestDrug_{uuid.uuid4().hex[:4]}"
    target = GraphEntity(name=target_name, entity_type="DRUG", metadata={"test_run": True})
    neo4j.add_entity(target)
    
    rel = GraphRelationship(source=target_name, target=entity_name, relationship_type="TREATS", properties={"test_run": True})
    neo4j.add_relationship(rel)
    
    # Verify relationship
    rel_query = "MATCH (d:MedicalEntity {name: $source})-[r:TREATS]->(v:MedicalEntity {name: $target}) RETURN r"
    rel_results = neo4j.client.execute_read(rel_query, {"source": target_name, "target": entity_name})
    
    assert len(rel_results) > 0
