"""
Shared type definitions for the RAG application.
"""
from dataclasses import dataclass, field as dc_field


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
