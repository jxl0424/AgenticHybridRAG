"""
Shared type definitions for the RAG application.
"""
from dataclasses import dataclass, field as dc_field
import pydantic


@dataclass
class RetrievedContext:
    """Single retrieved context item from any retrieval source."""
    text: str
    source: str        # arxiv_id, file path, or "graph"
    score: float
    collection: str    # "arxiv_chunks" | "arxiv_papers" | "graph"
    metadata: dict = dc_field(default_factory=dict)
    # metadata keys by collection:
    #   arxiv_chunks: edge_id, src_id, dst_id, rel_type, domain, paper_id
    #   arxiv_papers: arxiv_id, chunk_index, title, domain
    #   graph:        entities_found


class RAGChunkAndSrc(pydantic.BaseModel):
    """Represents chunks extracted from a document with their source."""
    chunks: list[str]
    source_id: str = None


class RAGUpsertResult(pydantic.BaseModel):
    """Result of a vector database upsert operation."""
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    """Result of a vector database search operation."""
    contexts: list[str]
    sources: list[str]


class RAGQueryResult(pydantic.BaseModel):
    """Result of a RAG query operation."""
    answer: str
    source: list[str]
    num_contexts: int


# Graph RAG Types


class GraphEntity(pydantic.BaseModel):
    """Represents an entity in the knowledge graph."""
    name: str
    entity_type: str
    metadata: dict = {}


class GraphRelationship(pydantic.BaseModel):
    """Represents a relationship between entities."""
    source: str
    target: str
    relationship_type: str
    properties: dict = {}


class GraphRetrievalResult(pydantic.BaseModel):
    """Result of graph-based retrieval operation."""
    contexts: list[str]
    sources: list[str]
    entities_found: list[str]


class HybridRetrievalResult(pydantic.BaseModel):
    """Result of hybrid (vector + graph) retrieval."""
    contexts: list[str]
    sources: list[str]
    scores: list[float]
    vector_contexts: list[str]
    graph_contexts: list[str]
    entities_found: list[str]


class GraphIngestResult(pydantic.BaseModel):
    """Result of graph ingestion operation."""
    document_id: str
    chunks_processed: int
    entities_extracted: int
    relationships_extracted: int
