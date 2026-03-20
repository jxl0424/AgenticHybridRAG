"""
Reranking module for improved retrieval quality.

Reranking works in two stages:
1. Initial retrieval: Get top N candidates (e.g., 20) using embeddings
2. Rerank: Use a cross-encoder model to score each candidate against the query
3. Return: Top K most relevant results

This significantly improves retrieval quality by re-scoring candidates
with a more expensive but more accurate model.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import CrossEncoder
from src.utils import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Cross-encoder based reranker for improving retrieval quality.
    
    Cross-encoders process the query and document together, providing
    more accurate relevance scores than bi-encoders (embedding similarity).
    """
    
    # Popular cross-encoder models for reranking
    MODELS = {
        "msmarco": "cross-encoder/ms-marco-MiniLM-L-6-v2",      # Fast, good quality
        "msmarco-large": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Better quality
        "qa": "cross-encoder/qna-covid-marco-miniLM-L6-2",     # QA focused
    }
    
    def __init__(
        self, 
        model_name: str = "msmarco",
        max_seq_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            max_seq_length: Maximum sequence length
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        model_path = self.MODELS.get(model_name, model_name)
        logger.info(f"Loading reranker model: {model_path}")
        
        self.model = CrossEncoder(
            model_path,
            max_length=max_seq_length,
            device=device
        )
        self.model_name = model_path
        
    def score(self, query: str, documents: List[str]) -> List[float]:
        """
        Score documents against a query.
        
        Args:
            query: The search query
            documents: List of document texts to score
            
        Returns:
            List of relevance scores (higher = more relevant)
        """
        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        return scores.tolist()
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        sources: List[str],
        scores: Optional[List[float]] = None,
        top_k: int = 5,
        initial_top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Rerank documents and return top results.
        
        Args:
            query: The search query
            documents: List of retrieved document texts
            sources: List of source identifiers
            scores: Optional initial retrieval scores
            top_k: Number of results to return
            initial_top_k: If provided, first retrieve this many then rerank
                         (useful if you want to do two-stage retrieval)
            
        Returns:
            Dictionary with reranked results
        """
        if not documents:
            return {
                "contexts": [],
                "sources": [],
                "rerank_scores": [],
                "original_scores": []
            }
        
        # Ensure all lists have the same length
        n = len(documents)
        if len(sources) != n:
            sources = sources * n if len(sources) < n else sources[:n]
        if scores is not None and len(scores) != n:
            scores = scores * n if len(scores) < n else scores[:n]
        
        # If initial_top_k specified and we have more documents, do first-stage filtering
        if initial_top_k and len(documents) > initial_top_k:
            # Score all documents and take top initial_top_k
            all_scores = self.score(query, documents)
            top_indices = np.argsort(all_scores)[-initial_top_k:][::-1]
            documents = [documents[i] for i in top_indices]
            sources = [sources[i] for i in top_indices]
            scores = [all_scores[i] for i in top_indices]
        
        # Rerank using cross-encoder
        rerank_scores = self.score(query, documents)
        
        # Sort by rerank scores
        ranked_indices = np.argsort(rerank_scores)[::-1]
        
        # Get top k (but not more than we have)
        actual_top_k = min(top_k, len(rerank_scores))
        ranked_indices = ranked_indices[:actual_top_k]
        
        reranked_contexts = [documents[i] for i in ranked_indices]
        reranked_sources = [sources[i] for i in ranked_indices]
        reranked_scores = [rerank_scores[i] for i in ranked_indices]
        original_scores = [scores[i] for i in ranked_indices] if scores else None
        
        return {
            "contexts": reranked_contexts,
            "sources": reranked_sources,
            "rerank_scores": reranked_scores,
            "original_scores": original_scores,
            "improvement": self._calculate_improvement(reranked_scores, original_scores)
        }
    
    def _calculate_improvement(
        self, 
        rerank_scores: List[float], 
        original_scores: Optional[List[float]]
    ) -> Optional[Dict[str, float]]:
        """Calculate improvement metrics."""
        if not original_scores:
            return None
            
        return {
            "avg_rerank_score": float(np.mean(rerank_scores)),
            "avg_original_score": float(np.mean(original_scores)),
            "score_delta": float(np.mean(rerank_scores) - np.mean(original_scores))
        }
