"""
CS/AI domain knowledge graph for arXiv papers.

Standalone — does not depend on the legacy medical KnowledgeGraph base class.
"""
import json
from dataclasses import dataclass, field
from typing import Optional

from src.graph.cs_entity_extractor import ExtractionResult
from src.storage.neo4j_client import Neo4jClient
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared graph dataclasses (previously in knowledge_graph.py)
# ---------------------------------------------------------------------------

@dataclass
class GraphDocument:
    id: str
    title: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphChunk:
    id: str
    text: str
    document_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphEntity:
    name: str
    entity_type: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------

class CSKnowledgeGraph:
    """
    Knowledge graph for CS/AI entities from arXiv papers.

    Uses the :CSEntity shared label so that medical and CS data remain
    isolated in the same Neo4j instance.
    """

    VALID_ENTITY_TYPES = {
        "PAPER", "AUTHOR", "MODEL", "DATASET", "TASK",
        "METRIC", "ALGORITHM", "FRAMEWORK", "VENUE",
    }

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.client = neo4j_client or Neo4jClient()
        self.client.connect()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean_metadata(self, metadata) -> str:
        """Serialize metadata dict to a JSON string safe for Neo4j storage."""
        if metadata is None:
            return "{}"
        try:
            return json.dumps(metadata)
        except Exception:
            return "{}"

    # ------------------------------------------------------------------
    # Document / chunk primitives
    # ------------------------------------------------------------------

    def add_document(self, document: GraphDocument) -> None:
        metadata_str = self._clean_metadata(document.metadata)
        self.client.execute_write(
            """
            MERGE (d:Document {id: $id})
            SET d.title = $title,
                d.source = $source,
                d.metadata = $metadata
            """,
            {"id": document.id, "title": document.title,
             "source": document.source, "metadata": metadata_str},
        )

    def add_chunks_batch(self, chunks: list[GraphChunk]) -> None:
        if not chunks:
            return
        records = [
            {
                "id": c.id,
                "text": c.text,
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "metadata": self._clean_metadata(c.metadata),
            }
            for c in chunks
        ]
        self.client.execute_write(
            """
            UNWIND $batch AS c
            MERGE (ch:Chunk {id: c.id})
            SET ch.text         = c.text,
                ch.document_id  = c.document_id,
                ch.chunk_index  = c.chunk_index,
                ch.metadata     = c.metadata
            WITH ch, c
            MATCH (d:Document {id: c.document_id})
            MERGE (d)-[:HAS_CHUNK]->(ch)
            """,
            {"batch": records},
        )

    def initialize_schema(self) -> None:
        """Create CS-specific indexes and constraints."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:CSEntity) REQUIRE e.name IS UNIQUE",
        ]
        for constraint in constraints:
            try:
                self.client.execute_query(constraint)
            except Exception as e:
                logger.debug(f"Constraint note: {e}")

        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (e:CSEntity) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:CSEntity) ON (e.entity_type)",
            "CREATE INDEX IF NOT EXISTS FOR (e:MODEL) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:DATASET) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:PAPER) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:AUTHOR) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:FRAMEWORK) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:TASK) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:VENUE) ON (e.name)",
        ]
        for index in indexes:
            try:
                self.client.execute_query(index)
            except Exception as e:
                logger.debug(f"Index note: {e}")

        logger.info("CS knowledge graph schema initialized")

    def add_entity(self, entity: GraphEntity) -> None:
        """Add a CS entity using :CSEntity shared label."""
        entity_type = entity.entity_type.upper()
        if entity_type not in self.VALID_ENTITY_TYPES:
            return

        metadata_str = self._clean_metadata(entity.metadata)
        query = f"""
        MERGE (e:CSEntity {{name: $name}})
        SET e:{entity_type},
            e.entity_type = $entity_type,
            e.metadata = $metadata
        """
        self.client.execute_write(query, {
            "name": entity.name,
            "entity_type": entity_type,
            "metadata": metadata_str,
        })

    def link_chunk_to_entity(
        self, chunk_id: str, entity_name: str, entity_type: str
    ) -> None:
        """Link a chunk to a CSEntity node."""
        self.add_entity(GraphEntity(entity_name, entity_type))
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (e:CSEntity {name: $entity_name})
        MERGE (c)-[:HAS_ENTITY]->(e)
        """
        self.client.execute_write(query, {
            "chunk_id": chunk_id,
            "entity_name": entity_name,
        })

    def add_relationship(self, relationship) -> None:
        """Add a relationship between CS entities."""
        properties_str = self._clean_metadata(relationship.properties)
        rel_type = "".join(
            c for c in relationship.relationship_type if c.isalnum() or c == "_"
        ).upper()
        if not rel_type:
            rel_type = "RELATED_TO"

        query = f"""
        MATCH (e1:CSEntity {{name: $source}})
        MATCH (e2:CSEntity {{name: $target}})
        MERGE (e1)-[r:{rel_type}]->(e2)
        SET r.properties = $properties
        """
        self.client.execute_write(query, {
            "source": relationship.source,
            "target": relationship.target,
            "properties": properties_str,
        })

    def get_entity_by_name(self, name: str) -> list[dict]:
        query = """
        MATCH (e:CSEntity)-[r]-(other)
        WHERE toLower(e.name) = toLower($name)
        RETURN e, r, other
        """
        return self.client.execute_read(query, {"name": name})

    def get_chunks_for_entity(self, entity_name: str) -> list[dict]:
        query = """
        MATCH (c:Chunk)-[:HAS_ENTITY]->(e:CSEntity)
        WHERE toLower(e.name) = toLower($name)
        RETURN c
        ORDER BY c.chunk_index
        """
        return self.client.execute_read(query, {"name": entity_name})

    def get_entity_context(self, entity_name: str, max_chunks: int = 5) -> list[str]:
        query = """
        MATCH (c:Chunk)-[:HAS_ENTITY]->(e:CSEntity)
        WHERE toLower(e.name) = toLower($name)
        RETURN c.text as text
        ORDER BY c.chunk_index
        LIMIT $max_chunks
        """
        results = self.client.execute_read(query, {
            "name": entity_name,
            "max_chunks": max_chunks,
        })
        return [r["text"] for r in results]

    def get_chunk_refs_for_entity(self, display_name: str, limit: int = 20) -> list[str]:
        """
        Return qdrant_id strings from HAS_CHUNK edges connected to _Embeddable
        nodes whose display_name contains the given name (case-insensitive).

        HAS_CHUNK edges connect _Embeddable src to _Embeddable dst and carry
        a qdrant_id property (UUID string) pointing to the arxiv_chunks collection.
        """
        query = """
        MATCH (src:_Embeddable)-[r:HAS_CHUNK]->(dst:_Embeddable)
        WHERE toLower(src.display_name) CONTAINS toLower($name)
           OR toLower(dst.display_name) CONTAINS toLower($name)
        RETURN r.qdrant_id AS qdrant_id
        LIMIT $limit
        """
        rows = self.client.execute_read(query, {"name": display_name, "limit": limit})
        return [r["qdrant_id"] for r in rows if r.get("qdrant_id")]

    def get_chunk_refs_by_node_ids(self, node_ids: list[int], limit: int = 20) -> list[str]:
        """
        Return qdrant_id strings from HAS_CHUNK edges connected to _Embeddable
        nodes whose node_id is in the given list. Matches both src and dst
        sides of the HAS_CHUNK relationship.
        """
        query = """
        MATCH (n:_Embeddable)-[r:HAS_CHUNK]-()
        WHERE n.node_id IN $node_ids
        RETURN DISTINCT r.qdrant_id AS qdrant_id
        LIMIT $limit
        """
        rows = self.client.execute_read(query, {"node_ids": node_ids, "limit": limit})
        return [r["qdrant_id"] for r in rows if r.get("qdrant_id")]

    def get_related_entities(self, entity_name: str, depth: int = 2) -> list[dict]:
        query = """
        MATCH path = (e:CSEntity)-[*1..%d]-(related)
        WHERE toLower(e.name) = toLower($name) AND e <> related
        RETURN path, length(path) as distance
        ORDER BY distance
        """ % depth
        return self.client.execute_read(query, {"name": entity_name})

    def ingest_extraction_results_batch(
        self,
        batch_items: list[tuple[GraphChunk, ExtractionResult]],
    ) -> None:
        """
        Batch-ingest chunks and CS extraction results into Neo4j.
        Uses CSEntity label throughout. Groups entities by type for
        efficient batch Cypher (dynamic labels require per-type queries).
        """
        if not batch_items:
            return

        # 1. Add all chunks
        chunks = [item[0] for item in batch_items]
        self.add_chunks_batch(chunks)

        all_entities = []
        all_links = []
        all_rels = []

        for chunk, res in batch_items:
            for ent in res.entities:
                ent_type = ent.entity_type.upper()
                if ent_type not in self.VALID_ENTITY_TYPES:
                    continue
                all_entities.append({
                    "name": ent.text,
                    "type": ent_type,
                    "metadata": self._clean_metadata({"confidence": ent.confidence}),
                })
                all_links.append({
                    "chunk_id": chunk.id,
                    "entity_name": ent.text,
                })

            for source, rel_type, target in res.relationships:
                clean_rel = "".join(
                    c for c in rel_type if c.isalnum() or c == "_"
                ).upper()
                if not clean_rel:
                    clean_rel = "RELATED_TO"
                all_rels.append({
                    "source": source,
                    "target": target,
                    "type": clean_rel,
                    "properties": "{}",
                })

        # 2. Batch merge entities grouped by type
        entities_by_type: dict[str, list] = {}
        for ent in all_entities:
            entities_by_type.setdefault(ent["type"], []).append(ent)

        for etype, e_batch in entities_by_type.items():
            query = f"""
            UNWIND $batch as ent
            MERGE (e:CSEntity {{name: ent.name}})
            SET e:{etype},
                e.entity_type = ent.type,
                e.metadata = ent.metadata
            """
            self.client.execute_write(query, {"batch": e_batch})

        # 3. Batch link chunks to entities
        if all_links:
            link_query = """
            UNWIND $batch as link
            MATCH (c:Chunk {id: link.chunk_id})
            MATCH (e:CSEntity {name: link.entity_name})
            MERGE (c)-[:HAS_ENTITY]->(e)
            """
            self.client.execute_write(link_query, {"batch": all_links})

        # 4. Batch create entity-entity relationships grouped by type
        rels_by_type: dict[str, list] = {}
        for rel in all_rels:
            rels_by_type.setdefault(rel["type"], []).append(rel)

        for rtype, r_batch in rels_by_type.items():
            rel_query = f"""
            UNWIND $batch as rel
            MATCH (e1:CSEntity {{name: rel.source}})
            MATCH (e2:CSEntity {{name: rel.target}})
            MERGE (e1)-[r:{rtype}]->(e2)
            SET r.properties = rel.properties
            """
            self.client.execute_write(rel_query, {"batch": r_batch})


_default_cs_kg: CSKnowledgeGraph | None = None


def get_cs_knowledge_graph(
    neo4j_client: Optional[Neo4jClient] = None,
) -> CSKnowledgeGraph:
    global _default_cs_kg
    if _default_cs_kg is None:
        _default_cs_kg = CSKnowledgeGraph(neo4j_client)
    return _default_cs_kg
