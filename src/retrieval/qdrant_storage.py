"""
Qdrant vector database storage interface.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchText
from typing import Optional


class QdrantStorage:
    """Interface for interacting with Qdrant vector database."""
    
    def __init__(self, url: str = "http://localhost:6333", collection: str = "docs", dim: int = 768):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        self.dimension = dim
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
       
    def upsert(self, ids: list, vectors: list, payloads: list) -> None:
        """Insert or update vectors in the collection."""
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) 
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector: list, top_k: int = 5, min_score: float = 0.3) -> dict:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: The query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score threshold (0-1). Results below this are filtered out.
        """
        # Fetch more to account for filtering
        fetch_k = max(top_k * 3, 20)
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=fetch_k
        )
        contexts = []
        sources = []  # Changed to list to match contexts
        scores = []

        for r in results.points:
            # Filter by minimum score threshold
            if hasattr(r, 'score') and r.score < min_score:
                continue
            
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            # Skip very short texts (likely metadata/tables)
            if len(text) < 50:
                continue
                
            # Skip common non-content sections
            skip_patterns = [
                'acknowledgment', 'acknowledgement', 
                'table of contents',
                'references',
                'copyright',
                'examination committee',
                'dissertation',
                'curriculum vitae'
            ]
            text_lower = text.lower()
            if any(pattern in text_lower for pattern in skip_patterns):
                # Only skip if it's at the beginning (these sections are usually at start)
                first_chars = text_lower[:100]
                if any(pattern in first_chars for pattern in skip_patterns):
                    continue

            if text:
                contexts.append(text)
                sources.append(source)  # Append instead of add to set
                scores.append(r.score if hasattr(r, 'score') else 0.0)
                
                # Stop if we have enough
                if len(contexts) >= top_k:
                    break

        return {
            "contexts": contexts, 
            "sources": sources,  # Now a list matching contexts
            "scores": scores
        }
