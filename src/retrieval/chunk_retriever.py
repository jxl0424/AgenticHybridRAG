"""
Retriever for the arxiv_chunks Qdrant collection.

Stores KG-edge paragraphs. Payload field: "paragraph".
"""
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from src.types import RetrievedContext

COLLECTION = "arxiv_chunks"
DIM = 768
MIN_TEXT_LEN = 50


class ChunkRetriever:
    """Vector search over arxiv_chunks; also resolves qdrant_id references for graph traversal."""

    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url, timeout=30)
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_score: float = 0.2,
    ) -> list[RetrievedContext]:
        """Search arxiv_chunks by vector similarity."""
        results = self.client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )
        out: list[RetrievedContext] = []
        for r in results.points:
            if r.score < min_score:
                continue
            payload = r.payload or {}
            text = payload.get("paragraph", "")
            if len(text) < MIN_TEXT_LEN:
                continue
            out.append(RetrievedContext(
                text=text,
                source=payload.get("source", payload.get("paper_id", "")),
                score=r.score,
                collection=COLLECTION,
                metadata={
                    "edge_id": payload.get("edge_id"),
                    "src_id": payload.get("src_id"),
                    "dst_id": payload.get("dst_id"),
                    "rel_type": payload.get("rel_type", ""),
                    "domain": payload.get("domain", ""),
                    "paper_id": payload.get("paper_id", ""),
                },
            ))
        return out

    def fetch_by_ids(self, qdrant_ids: list[str]) -> list[RetrievedContext]:
        """Fetch specific points by their UUID string IDs (used by GraphRetriever)."""
        if not qdrant_ids:
            return []
        points = self.client.retrieve(
            collection_name=COLLECTION,
            ids=qdrant_ids,
            with_payload=True,
        )
        out: list[RetrievedContext] = []
        for p in points:
            payload = p.payload or {}
            text = payload.get("paragraph", "")
            if len(text) < MIN_TEXT_LEN:
                continue
            out.append(RetrievedContext(
                text=text,
                source=payload.get("source", payload.get("paper_id", "")),
                score=1.0,  # graph-resolved: score reflects traversal relevance, not cosine
                collection=COLLECTION,
                metadata={
                    "edge_id": payload.get("edge_id"),
                    "src_id": payload.get("src_id"),
                    "dst_id": payload.get("dst_id"),
                    "rel_type": payload.get("rel_type", ""),
                    "domain": payload.get("domain", ""),
                    "paper_id": payload.get("paper_id", ""),
                },
            ))
        return out
