"""
Hybrid retriever combining Qdrant (vector) and Neo4j (graph) search.
"""
from typing import Optional
from dataclasses import dataclass
import yaml

from src.retrieval.qdrant_storage import QdrantStorage
from src.retrieval.graph_retriever import GraphRetriever, GraphRetrievalResult
from src.graph.entity_extractor import get_entity_extractor


@dataclass
class HybridRetrievalResult:
    """Combined result from hybrid retrieval."""
    contexts: list[str]
    sources: list[str]
    scores: list[float]
    vector_contexts: list[str]
    graph_contexts: list[str]
    entities_found: list[str]


class HybridRetriever:
    """
    Hybrid retriever that combines semantic search (Qdrant) and graph search (Neo4j).
    """
    
    def __init__(
        self,
        qdrant_storage: Optional[QdrantStorage] = None,
        graph_retriever: Optional[GraphRetriever] = None,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        config_path: str = "config/defaults.yaml"
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            qdrant_storage: Qdrant storage instance
            graph_retriever: Graph retriever instance
            vector_weight: Weight for vector search results (0-1)
            graph_weight: Weight for graph search results (0-1)
            config_path: Path to configuration file
        """
        # Load config if available
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                hybrid_config = config.get('hybrid_retrieval', {})
                self.vector_weight = hybrid_config.get('vector_weight', vector_weight)
                self.graph_weight = hybrid_config.get('graph_weight', graph_weight)
                self.top_k = hybrid_config.get('top_k', 5)
        except Exception:
            self.vector_weight = vector_weight
            self.graph_weight = graph_weight
            self.top_k = 5
        
        # Normalize weights
        total = self.vector_weight + self.graph_weight
        if total > 0:
            self.vector_weight = self.vector_weight / total
            self.graph_weight = self.graph_weight / total
        
        # Initialize components
        self.qdrant = qdrant_storage
        self.graph_retriever = graph_retriever
        
        # Entity extractor for query analysis
        self.entity_extractor = get_entity_extractor()
    
    def _get_qdrant(self) -> QdrantStorage:
        """Get or create Qdrant storage."""
        if self.qdrant is None:
            self.qdrant = QdrantStorage()
        return self.qdrant
    
    def _get_graph_retriever(self) -> GraphRetriever:
        """Get or create graph retriever."""
        if self.graph_retriever is None:
            self.graph_retriever = GraphRetriever()
        return self.graph_retriever
    
    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.3
    ) -> HybridRetrievalResult:
        """
        Perform hybrid retrieval combining vector and graph search.
        
        Args:
            query: Query text
            query_embedding: Pre-computed query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            HybridRetrievalResult with combined contexts
        """
        # Get vector search results
        vector_results = self._get_qdrant().search(
            query_vector=query_embedding,
            top_k=top_k * 2,  # Fetch more for fusion
            min_score=min_score
        )
        
        # Get graph search results
        graph_results = self._get_graph_retriever().retrieve(
            query=query,
            top_k=top_k * 2
        )
        
        # Fuse results
        return self._fuse_results(
            query=query,
            vector_results=vector_results,
            graph_results=graph_results,
            top_k=top_k
        )
    
    def _fuse_results(
        self,
        query: str,
        vector_results: dict,
        graph_results: GraphRetrievalResult,
        top_k: int
    ) -> HybridRetrievalResult:
        """
        Fuse results from vector and graph search.
        
        Args:
            query: Original query
            vector_results: Results from Qdrant
            graph_results: Results from Neo4j
            top_k: Number of results to return
            
        Returns:
            Fused HybridRetrievalResult
        """
        # Track unique contexts and their scores
        context_scores = {}
        
        # Process vector results
        for i, (ctx, source, score) in enumerate(zip(
            vector_results.get('contexts', []),
            vector_results.get('sources', []),
            vector_results.get('scores', [])
        )):
            if ctx not in context_scores:
                context_scores[ctx] = {
                    'text': ctx,
                    'source': source,
                    'vector_score': 0.0,
                    'graph_score': 0.0,
                    'final_score': 0.0
                }
            # Normalize vector score (already 0-1 from Qdrant)
            context_scores[ctx]['vector_score'] = max(
                context_scores[ctx]['vector_score'],
                score
            )
        
        # Process graph results
        for ctx in graph_results.contexts:
            if ctx not in context_scores:
                context_scores[ctx] = {
                    'text': ctx,
                    'source': 'graph',
                    'vector_score': 0.0,
                    'graph_score': 0.0,
                    'final_score': 0.0
                }
            # Graph results get uniform score based on position
            ctx_idx = graph_results.contexts.index(ctx)
            graph_score = 1.0 - (ctx_idx / len(graph_results.contexts))
            context_scores[ctx]['graph_score'] = max(
                context_scores[ctx]['graph_score'],
                graph_score
            )
        
        # Calculate final scores with weights
        for ctx_data in context_scores.values():
            ctx_data['final_score'] = (
                self.vector_weight * ctx_data['vector_score'] +
                self.graph_weight * ctx_data['graph_score']
            )
        
        # Sort by final score
        sorted_contexts = sorted(
            context_scores.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )
        
        # Extract top K results
        final_contexts = []
        final_sources = []
        final_scores = []
        vector_contexts = []
        graph_contexts = []
        
        for ctx_data in sorted_contexts[:top_k]:
            final_contexts.append(ctx_data['text'])
            final_sources.append(ctx_data['source'])
            final_scores.append(ctx_data['final_score'])
            
            if ctx_data['vector_score'] > 0:
                vector_contexts.append(ctx_data['text'])
            if ctx_data['graph_score'] > 0:
                graph_contexts.append(ctx_data['text'])
        
        return HybridRetrievalResult(
            contexts=final_contexts,
            sources=final_sources,
            scores=final_scores,
            vector_contexts=vector_contexts,
            graph_contexts=graph_contexts,
            entities_found=graph_results.entities_found
        )
    
    def retrieve_with_embedding(
        self,
        query: str,
        embed_fn,
        top_k: int = 5,
        min_score: float = 0.3
    ) -> HybridRetrievalResult:
        """
        Retrieve with automatic embedding generation.
        
        Args:
            query: Query text
            embed_fn: Embedding function that takes list of strings
            top_k: Number of results
            min_score: Minimum score threshold
            
        Returns:
            HybridRetrievalResult
        """
        # Generate embedding
        query_embedding = embed_fn([query])[0]
        
        return self.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score
        )
    
    def get_stats(self) -> dict:
        """
        Get retrieval statistics.
        
        Returns:
            Dictionary with stats
        """
        stats = {
            'weights': {
                'vector': self.vector_weight,
                'graph': self.graph_weight
            },
            'top_k': self.top_k
        }
        
        # Add graph stats if available
        if self.graph_retriever:
            try:
                kg_stats = self.graph_retriever.kg.get_stats()
                stats['graph_stats'] = kg_stats
            except Exception:
                pass
        
        return stats


# Default retriever instance
_default_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever(
    qdrant_storage: Optional[QdrantStorage] = None,
    graph_retriever: Optional[GraphRetriever] = None,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4
) -> HybridRetriever:
    """
    Get or create the default hybrid retriever.
    
    Args:
        qdrant_storage: Optional Qdrant storage
        graph_retriever: Optional graph retriever
        vector_weight: Vector search weight
        graph_weight: Graph search weight
        
    Returns:
        HybridRetriever instance
    """
    global _default_hybrid_retriever
    if _default_hybrid_retriever is None:
        _default_hybrid_retriever = HybridRetriever(
            qdrant_storage,
            graph_retriever,
            vector_weight,
            graph_weight
        )
    return _default_hybrid_retriever
