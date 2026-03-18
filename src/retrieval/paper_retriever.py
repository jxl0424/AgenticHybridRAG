"""
Retriever for the arxiv_papers Qdrant collection.

Stores full paper text chunks. Payload field: "chunk_text".
"""
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from src.types import RetrievedContext

COLLECTION = "arxiv_papers"
DIM = 768
MIN_TEXT_LEN = 50


class PaperRetriever:
    """Vector search over arxiv_papers (full paper text chunks)."""

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
        """Search arxiv_papers by vector similarity."""
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
            text = payload.get("chunk_text", "")
            if len(text) < MIN_TEXT_LEN:
                continue
            out.append(RetrievedContext(
                text=text,
                source=payload.get("arxiv_id", ""),
                score=r.score,
                collection=COLLECTION,
                metadata={
                    "arxiv_id": payload.get("arxiv_id", ""),
                    "chunk_index": payload.get("chunk_index"),
                    "title": payload.get("title", ""),
                    "domain": payload.get("domain", ""),
                },
            ))
        return out
