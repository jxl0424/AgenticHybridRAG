"""
Graph-based retrieval using Neo4j knowledge graph.

Traverses HAS_CHUNK edges from _Embeddable entity nodes to resolve
qdrant_id references, then fetches text from arxiv_chunks via ChunkRetriever.
"""
import re
from typing import Optional
from dataclasses import dataclass

from src.graph.cs_knowledge_graph import CSKnowledgeGraph
from src.graph.cs_entity_extractor import CSEntityExtractor, get_cs_entity_extractor
from src.types import RetrievedContext


# Kept for backward-compat imports elsewhere; will be cleaned up separately
@dataclass
class GraphRetrievalResult:
    contexts: list[str]
    sources: list[str]
    entities_found: list[str]
    graph_paths: list[dict]


class GraphRetriever:
    """
    Retriever that uses the CS/arXiv knowledge graph for entity-based search.

    Flow:
        1. Extract CS entities from query via CSEntityExtractor
        2. For each entity, traverse HAS_CHUNK edges in Neo4j to get qdrant_ids
        3. Resolve qdrant_ids to text via ChunkRetriever.fetch_by_ids()
        4. Return list[RetrievedContext] with collection="graph"

    Fallback (no entities found):
        Keyword-search _Embeddable.display_name; returns entity name strings
        as minimal context (no Qdrant lookup needed for the fallback path).
    """

    _STOP_WORDS = {
        "what", "were", "the", "results", "why", "how", "did", "does",
        "was", "are", "for", "and", "that", "this", "it", "at",
        "by", "on", "as", "from", "with", "not", "but", "its",
        "very", "just", "than", "then",
    }

    def __init__(
        self,
        knowledge_graph: Optional[CSKnowledgeGraph] = None,
        entity_extractor: Optional[CSEntityExtractor] = None,
        chunk_retriever=None,  # ChunkRetriever -- optional to avoid circular import
    ):
        self.kg = knowledge_graph or CSKnowledgeGraph()
        self.entity_extractor = entity_extractor or get_cs_entity_extractor()
        self.chunk_retriever = chunk_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        entity_types: list[str] | None = None,  # kept as no-op for backward compat
    ) -> list[RetrievedContext]:
        """
        Retrieve relevant contexts using graph-based search.

        Returns list[RetrievedContext] with collection="graph".
        """
        extraction = self.entity_extractor.extract_entities(query)
        entity_names = list({e.text for e in extraction.entities})

        if not entity_names:
            # No CS entities found -- fall back to keyword search on _Embeddable display names
            return self._keyword_fallback(query, top_k)

        all_qdrant_ids: list[str] = []
        for name in entity_names:
            ids = self.kg.get_chunk_refs_for_entity(name, limit=top_k * 2)
            all_qdrant_ids.extend(ids)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_ids = [i for i in all_qdrant_ids if not (i in seen or seen.add(i))]

        if not unique_ids or self.chunk_retriever is None:
            return self._keyword_fallback(query, top_k)

        contexts = self.chunk_retriever.fetch_by_ids(unique_ids[:top_k * 2])
        # Tag as graph-sourced. fetch_by_ids() returns fresh objects each call -- mutation is safe.
        for ctx in contexts:
            ctx.collection = "graph"
            ctx.metadata["entities_found"] = entity_names

        return contexts[:top_k]

    def retrieve_by_entity(
        self,
        entity_name: str,
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        """Retrieve contexts for a specific entity via HAS_CHUNK traversal."""
        ids = self.kg.get_chunk_refs_for_entity(entity_name, limit=top_k * 2)
        if not ids or self.chunk_retriever is None:
            return []
        contexts = self.chunk_retriever.fetch_by_ids(ids[:top_k])
        for ctx in contexts:
            ctx.collection = "graph"
            ctx.metadata["entities_found"] = [entity_name]
        return contexts

    def _keyword_fallback(self, query: str, top_k: int) -> list[RetrievedContext]:
        """
        Fallback: search _Embeddable nodes by display_name keyword matching
        when no CS entities are detected in the query.
        Uses parameterised Cypher -- never f-string interpolation.
        """
        words = re.findall(r'\b[a-zA-Z]{2,}\b', query)
        keywords = [w.lower() for w in words if w.lower() not in self._STOP_WORDS]

        if not keywords:
            return []

        try:
            rows = self.kg.client.execute_read(
                """
                MATCH (n:_Embeddable)
                WHERE any(kw IN $keywords WHERE toLower(n.display_name) CONTAINS kw)
                RETURN n.display_name AS name, n.entity_type AS type
                LIMIT $lim
                """,
                {"keywords": keywords[:5], "lim": top_k},
            )
            return [
                RetrievedContext(
                    text=r["name"],
                    source="neo4j/keyword",
                    score=0.3,
                    collection="graph",
                    metadata={"entity_type": r.get("type", "")},
                )
                for r in rows
                if r.get("name")
            ]
        except Exception:
            return []


# Default retriever instance
_default_retriever: Optional[GraphRetriever] = None


def get_graph_retriever(
    knowledge_graph: Optional[CSKnowledgeGraph] = None,
    entity_extractor: Optional[CSEntityExtractor] = None,
    chunk_retriever=None,
) -> GraphRetriever:
    """Get or create the default graph retriever."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = GraphRetriever(knowledge_graph, entity_extractor, chunk_retriever)
    return _default_retriever
