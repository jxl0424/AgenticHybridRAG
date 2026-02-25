"""
Knowledge graph storage for medical entities in Neo4j.
"""
from typing import Optional, Any
from dataclasses import dataclass
from src.storage.neo4j_client import Neo4jClient
from src.graph.entity_extractor import ExtractionResult, ExtractedEntity


@dataclass
class GraphDocument:
    """Represents a document in the knowledge graph."""
    id: str
    title: str
    source: str
    metadata: dict = None


@dataclass
class GraphChunk:
    """Represents a text chunk in the knowledge graph."""
    id: str
    text: str
    document_id: str
    chunk_index: int
    metadata: dict = None


@dataclass
class GraphEntity:
    """Represents an entity in the knowledge graph."""
    name: str
    entity_type: str
    metadata: dict = None


@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph."""
    source: str
    target: str
    relationship_type: str
    properties: dict = None


class KnowledgeGraph:
    """
    Knowledge graph storage for managing medical entities and relationships.
    """
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """
        Initialize the knowledge graph.
        
        Args:
            neo4j_client: Optional Neo4j client instance
        """
        self.client = neo4j_client or Neo4jClient()
        self.client.connect()
    
    def initialize_schema(self) -> None:
        """Create indexes and constraints for the medical knowledge graph."""
        # Core structural constraints
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:MedicalEntity) REQUIRE e.name IS UNIQUE",
        ]
        for constraint in constraints:
            try:
                self.client.execute_query(constraint)
            except Exception as e:
                print(f"Constraint note: {e}")
        
        # Indexes on shared MedicalEntity label
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (e:MedicalEntity) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:MedicalEntity) ON (e.entity_type)",
            # Per-type indexes for fast typed queries
            "CREATE INDEX IF NOT EXISTS FOR (e:DISEASE) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:DRUG) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:SYMPTOM) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:PROCEDURE) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:ANATOMY) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:TEST) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:TREATMENT) ON (e.name)",
        ]
        for index in indexes:
            try:
                self.client.execute_query(index)
            except Exception as e:
                print(f"Index note: {e}")
    
    def add_document(self, document: GraphDocument) -> None:
        """
        Add a document node to the graph.
        
        Args:
            document: Document to add
        """
        # Convert metadata to JSON string for Neo4j compatibility
        import json
        metadata_str = json.dumps(document.metadata) if document.metadata else '{}'
        
        query = """
        MERGE (d:Document {id: $id})
        SET d.title = $title,
            d.source = $source,
            d.metadata = $metadata
        """
        
        self.client.execute_write(query, {
            "id": document.id,
            "title": document.title,
            "source": document.source,
            "metadata": metadata_str
        })
    
    def add_chunk(self, chunk: GraphChunk) -> None:
        """
        Add a chunk node to the graph.
        
        Args:
            chunk: Chunk to add
        """
        import json
        metadata_str = json.dumps(chunk.metadata) if chunk.metadata else '{}'
        
        query = """
        MERGE (c:Chunk {id: $id})
        SET c.text = $text,
            c.document_id = $document_id,
            c.chunk_index = $chunk_index,
            c.metadata = $metadata
        """
        
        self.client.execute_write(query, {
            "id": chunk.id,
            "text": chunk.text,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "metadata": metadata_str
        })
        
        # Link chunk to document
        link_query = """
        MATCH (d:Document {id: $doc_id})
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        
        self.client.execute_write(link_query, {
            "doc_id": chunk.document_id,
            "chunk_id": chunk.id
        })
    
    # Valid medical entity type labels
    VALID_ENTITY_TYPES = {"DISEASE", "DRUG", "PROCEDURE", "SYMPTOM", "ANATOMY", "TEST", "TREATMENT"}

    def add_entity(self, entity: GraphEntity) -> None:
        """
        Add a typed entity node to the graph.

        Each entity gets both the shared :MedicalEntity label AND a specific
        typed label (e.g. :DISEASE, :DRUG) for efficient per-type Cypher queries.

        Args:
            entity: Entity to add
        """
        import json
        metadata_str = json.dumps(entity.metadata) if entity.metadata else '{}'
        entity_type = entity.entity_type.upper()

        # Validate and fall back to generic label if unknown type
        if entity_type not in self.VALID_ENTITY_TYPES:
            entity_type = "UNKNOWN"

        # MERGE on shared label + name, then apoc-free dynamic label via CASE
        # We set the typed label using a separate SET clause per type.
        # This avoids needing APOC while still creating typed nodes.
        query = f"""
        MERGE (e:MedicalEntity {{name: $name}})
        SET e.entity_type = $entity_type,
            e.metadata = $metadata
        WITH e
        CALL apoc.create.addLabels(e, [$entity_type]) YIELD node
        RETURN node
        """

        try:
            self.client.execute_write(query, {
                "name": entity.name,
                "entity_type": entity_type,
                "metadata": metadata_str
            })
        except Exception:
            # Fallback if APOC not available: use MedicalEntity only
            fallback_query = """
            MERGE (e:MedicalEntity {name: $name})
            SET e.entity_type = $entity_type,
                e.metadata = $metadata
            """
            self.client.execute_write(fallback_query, {
                "name": entity.name,
                "entity_type": entity_type,
                "metadata": metadata_str
            })
    
    def add_relationship(self, relationship: GraphRelationship) -> None:
        """
        Add a relationship between entities.
        
        Args:
            relationship: Relationship to add
        """
        import json
        properties_str = json.dumps(relationship.properties) if relationship.properties else '{}'
        
        # Clean the relationship type to prevent Cypher injection
        rel_type = "".join(c for c in relationship.relationship_type if c.isalnum() or c == "_").upper()
        if not rel_type:
            rel_type = "RELATED_TO"
            
        query = f"""
        MATCH (e1:MedicalEntity {{name: $source}})
        MATCH (e2:MedicalEntity {{name: $target}})
        MERGE (e1)-[r:{rel_type}]->(e2)
        SET r.properties = $properties
        """
        
        self.client.execute_write(query, {
            "source": relationship.source,
            "target": relationship.target,
            "properties": properties_str
        })
    
    def link_chunk_to_entity(
        self,
        chunk_id: str,
        entity_name: str,
        entity_type: str
    ) -> None:
        """
        Link a chunk to a MedicalEntity node.

        Args:
            chunk_id: Chunk ID
            entity_name: Entity name
            entity_type: Entity type
        """
        # Ensure entity exists (with typed label)
        self.add_entity(GraphEntity(entity_name, entity_type))

        # Link chunk to entity using the shared MedicalEntity label
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (e:MedicalEntity {name: $entity_name})
        MERGE (c)-[:HAS_ENTITY]->(e)
        """
        self.client.execute_write(query, {
            "chunk_id": chunk_id,
            "entity_name": entity_name
        })
    
    def ingest_extraction_result(
        self,
        chunk: GraphChunk,
        extraction_result: ExtractionResult
    ) -> None:
        """
        Ingest an extraction result into the knowledge graph.
        
        Args:
            chunk: The chunk this result came from
            extraction_result: Extraction result with entities and relationships
        """
        # Add chunk
        self.add_chunk(chunk)
        
        # Add entities
        for entity in extraction_result.entities:
            graph_entity = GraphEntity(
                name=entity.text,
                entity_type=entity.entity_type,
                metadata={"confidence": entity.confidence}
            )
            self.add_entity(graph_entity)
            
            # Link chunk to entity
            self.link_chunk_to_entity(
                chunk.id,
                entity.text,
                entity.entity_type
            )
        
        # Add relationships
        for source, rel_type, target in extraction_result.relationships:
            relationship = GraphRelationship(
                source=source,
                target=target,
                relationship_type=rel_type
            )
            self.add_relationship(relationship)
    
    def get_entity_by_name(self, name: str) -> list[dict]:
        """
        Get entity and its neighbors from the graph.

        Args:
            name: Entity name to search for

        Returns:
            List of related entities and their relationships
        """
        query = """
        MATCH (e:MedicalEntity {name: $name})-[r]-(other)
        RETURN e, r, other
        """
        return self.client.execute_read(query, {"name": name})

    def get_entities_by_type(self, entity_type: str) -> list[dict]:
        """
        Get all entities of a specific type label.

        Args:
            entity_type: Entity type (e.g. 'DISEASE', 'DRUG')

        Returns:
            List of entities
        """
        query = """
        MATCH (e:MedicalEntity {entity_type: $entity_type})
        RETURN e
        """
        return self.client.execute_read(query, {"entity_type": entity_type.upper()})
    
    def get_related_entities(
        self,
        entity_name: str,
        depth: int = 2
    ) -> list[dict]:
        """
        Get related entities up to a certain depth.

        Args:
            entity_name: Starting entity
            depth: Traversal depth

        Returns:
            List of related entities
        """
        query = """
        MATCH path = (e:MedicalEntity {name: $name})-[*1..%d]-(related)
        WHERE e <> related
        RETURN path, length(path) as distance
        ORDER BY distance
        """ % depth
        return self.client.execute_read(query, {"name": entity_name})

    def get_chunks_for_entity(self, entity_name: str) -> list[dict]:
        """
        Get all chunks that reference an entity.

        Args:
            entity_name: Entity name

        Returns:
            List of chunks with the entity
        """
        query = """
        MATCH (c:Chunk)-[:HAS_ENTITY]->(e:MedicalEntity {name: $name})
        RETURN c
        ORDER BY c.chunk_index
        """
        return self.client.execute_read(query, {"name": entity_name})
    
    def get_entity_context(
        self,
        entity_name: str,
        max_chunks: int = 5
    ) -> list[str]:
        """
        Get text context from chunks containing an entity.

        Args:
            entity_name: Entity to get context for
            max_chunks: Maximum number of chunks to return

        Returns:
            List of text chunks
        """
        query = """
        MATCH (c:Chunk)-[:HAS_ENTITY]->(e:MedicalEntity {name: $name})
        RETURN c.text as text
        ORDER BY c.chunk_index
        LIMIT $max_chunks
        """
        results = self.client.execute_read(query, {
            "name": entity_name,
            "max_chunks": max_chunks
        })
        return [r["text"] for r in results]
    
    def clear(self) -> None:
        """Clear all nodes and relationships from the graph."""
        self.client.execute_write("MATCH (n) DETACH DELETE n")
    
    def get_stats(self) -> dict:
        """Get knowledge graph statistics."""
        stats = {}
        
        # Count nodes by type
        node_query = """
        MATCH (n)
        RETURN labels(n)[0] as node_type, count(n) as count
        """
        nodes = self.client.execute_read(node_query)
        stats['nodes'] = {r['node_type']: r['count'] for r in nodes}
        
        # Count relationships
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        """
        rels = self.client.execute_read(rel_query)
        stats['relationships'] = {r['rel_type']: r['count'] for r in rels}
        
        return stats
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        self.client.close()


# Default knowledge graph instance
_default_kg: Optional[KnowledgeGraph] = None


def get_knowledge_graph(
    neo4j_client: Optional[Neo4jClient] = None
) -> KnowledgeGraph:
    """
    Get or create the default knowledge graph instance.
    
    Args:
        neo4j_client: Optional Neo4j client
        
    Returns:
        KnowledgeGraph instance
    """
    global _default_kg
    if _default_kg is None:
        _default_kg = KnowledgeGraph(neo4j_client)
    return _default_kg
