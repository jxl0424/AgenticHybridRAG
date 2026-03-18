"""
Graph-based retrieval using Neo4j knowledge graph.
"""
import json
from typing import Optional, Any
from dataclasses import dataclass
from src.graph.cs_knowledge_graph import CSKnowledgeGraph
from src.graph.cs_entity_extractor import CSEntityExtractor, get_cs_entity_extractor


@dataclass
class GraphRetrievalResult:
    """Result from graph-based retrieval."""
    contexts: list[str]
    sources: list[str]
    entities_found: list[str]
    graph_paths: list[dict]


class GraphRetriever:
    """
    Retriever that uses the knowledge graph for entity-based search.
    """
    
    def __init__(
        self,
        knowledge_graph: Optional[CSKnowledgeGraph] = None,
        entity_extractor: Optional[CSEntityExtractor] = None
    ):
        """
        Initialize the graph retriever.
        
        Args:
            knowledge_graph: Knowledge graph instance
            entity_extractor: Entity extractor instance
        """
        self.kg = knowledge_graph or CSKnowledgeGraph()
        self.entity_extractor = entity_extractor or get_cs_entity_extractor()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        entity_types: Optional[list[str]] = None
    ) -> GraphRetrievalResult:
        """
        Retrieve relevant context using graph-based search.
        
        Args:
            query: Query text
            top_k: Number of results to return
            entity_types: Optional filter for entity types
            
        Returns:
            GraphRetrievalResult with contexts and metadata
        """
        # Extract entities from the query
        query_entities = self.entity_extractor.extract_entities(query)
        
        # Get unique entity names
        entity_names = list(set(e.text for e in query_entities.entities))
        
        # Collect contexts from all entities
        all_contexts = []
        all_sources = []
        all_entities = []
        all_paths = []

        if not entity_names:
            # No CS entities found — fall back to keyword search on _Embeddable display names
            all_contexts, all_sources = self._keyword_fallback(query, top_k)
            return GraphRetrievalResult(
                contexts=all_contexts[:top_k],
                sources=all_sources[:top_k],
                entities_found=[],
                graph_paths=[]
            )
        
        for entity_name in entity_names:
            # Get related entities (graph traversal)
            related = self.kg.get_related_entities(entity_name, depth=2)
            all_paths.extend(related)

            # Get chunks containing this entity via HAS_ENTITY edges
            chunks = self.kg.get_chunks_for_entity(entity_name)
            for chunk in chunks:
                chunk_text = chunk.get("c", {}).get("text", "")
                if chunk_text and chunk_text not in all_contexts:
                    all_contexts.append(chunk_text)
                    # Extract source from metadata if available
                    metadata = chunk.get("c", {}).get("metadata", {})
                    # Handle metadata being stored as JSON string
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}
                    source = metadata.get("source", "unknown") if isinstance(metadata, dict) else "unknown"
                    all_sources.append(source)

            # Also get direct context
            direct_contexts = self.kg.get_entity_context(entity_name, max_chunks=top_k)
            for ctx in direct_contexts:
                if ctx not in all_contexts:
                    all_contexts.append(ctx)

            all_entities.append(entity_name)
        
        # Apply entity type filter if specified
        if entity_types:
            filtered_contexts = []
            filtered_sources = []
            
            for i, ctx in enumerate(all_contexts):
                # Check if context contains entities of the right type
                ctx_entities = self.entity_extractor.extract_entities(ctx)
                relevant_types = set(e.entity_type for e in ctx_entities.entities)
                if relevant_types.intersection(set(entity_types)):
                    filtered_contexts.append(ctx)
                    if i < len(all_sources):
                        filtered_sources.append(all_sources[i])
            
            all_contexts = filtered_contexts
            all_sources = filtered_sources
        
        # Limit results
        final_contexts = all_contexts[:top_k]
        final_sources = all_sources[:top_k]
        
        return GraphRetrievalResult(
            contexts=final_contexts,
            sources=final_sources,
            entities_found=all_entities,
            graph_paths=all_paths[:10]  # Limit path results
        )

    def _keyword_fallback(self, query: str, top_k: int) -> tuple[list, list]:
        """
        Fallback: search _Embeddable nodes by display_name keyword matching
        when no CS entities are detected in the query.
        Uses parameterised Cypher — never f-string interpolation.
        """
        import re
        stop_words = {
            "what", "were", "the", "results", "why", "how", "did", "does",
            "was", "are", "for", "and", "that", "this", "it", "at",
            "by", "on", "as", "from", "with", "not", "but", "its",
            "very", "just", "than", "then"
        }
        words = re.findall(r'\b[a-zA-Z]{2,}\b', query)
        keywords = [w.lower() for w in words if w.lower() not in stop_words]

        if not keywords:
            return [], []

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
            names = [r["name"] for r in rows if r.get("name")]
            sources = ["neo4j/keyword"] * len(names)
            return names, sources
        except Exception:
            return [], []
    
    def retrieve_by_entity(
        self,
        entity_name: str,
        entity_type: Optional[str] = None,
        top_k: int = 5
    ) -> GraphRetrievalResult:
        """
        Retrieve context for a specific entity.
        
        Args:
            entity_name: Entity name to search for
            entity_type: Optional entity type filter
            top_k: Number of results
            
        Returns:
            GraphRetrievalResult
        """
        # Get chunks containing this entity
        chunks = self.kg.get_chunks_for_entity(entity_name)
        
        contexts = []
        sources = []
        
        for chunk in chunks[:top_k]:
            chunk_text = chunk.get("c", {}).get("text", "")
            if chunk_text:
                contexts.append(chunk_text)
                metadata = chunk.get("c", {}).get("metadata", {})
                # Handle metadata being stored as JSON string
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                source = metadata.get("source", "unknown") if isinstance(metadata, dict) else "unknown"
                sources.append(source)
        
        # Get related entities
        related = self.kg.get_related_entities(entity_name, depth=2)
        
        return GraphRetrievalResult(
            contexts=contexts,
            sources=sources,
            entities_found=[entity_name],
            graph_paths=related
        )
    
    def get_entity_network(
        self,
        entity_name: str,
        depth: int = 2
    ) -> dict:
        """
        Get the network of related entities around an entity.
        
        Args:
            entity_name: Central entity
            depth: Traversal depth
            
        Returns:
            Dictionary with entities and relationships
        """
        related = self.kg.get_related_entities(entity_name, depth=depth)
        
        entities = set([entity_name])
        relationships = []
        
        for record in related:
            path = record.get("path")
            if path:
                for node in path.nodes:
                    if "name" in node:
                        entities.add(node["name"])
                for rel in path.relationships:
                    rel_type = rel.type
                    relationships.append(rel_type)
        
        return {
            "central_entity": entity_name,
            "related_entities": list(entities),
            "relationship_types": list(set(relationships)),
            "total_related": len(entities) - 1
        }
    
    def search_by_type(
        self,
        entity_type: str,
        top_k: int = 10
    ) -> list[dict]:
        """
        Search for entities of a specific type.
        
        Args:
            entity_type: Type of entities to find
            top_k: Maximum results
            
        Returns:
            List of entity dictionaries
        """
        results = self.kg.get_entities_by_type(entity_type)
        
        entities = []
        for record in results[:top_k]:
            entity = record.get("e", {})
            metadata = entity.get("metadata", {})
            # Handle metadata being stored as JSON string
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            entities.append({
                "name": entity.get("name"),
                "type": entity.get("entity_type"),
                "metadata": metadata
            })
        
        return entities


# Default retriever instance
_default_retriever: Optional[GraphRetriever] = None


def get_graph_retriever(
    knowledge_graph: Optional[CSKnowledgeGraph] = None,
    entity_extractor: Optional[CSEntityExtractor] = None
) -> GraphRetriever:
    """
    Get or create the default graph retriever.
    
    Args:
        knowledge_graph: Optional knowledge graph
        entity_extractor: Optional entity extractor
        
    Returns:
        GraphRetriever instance
    """
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = GraphRetriever(knowledge_graph, entity_extractor)
    return _default_retriever
