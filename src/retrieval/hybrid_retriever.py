"""
Hybrid retriever combining arxiv_chunks, arxiv_papers, and Neo4j graph search.

Uses Reciprocal Rank Fusion (RRF) with per-source weights.
"""
from typing import Optional
import yaml

from src.types import RetrievedContext

RRF_K = 60

# Module-level sentinel — cannot use class attribute as default arg value in Python
_UNSET = object()


class HybridRetriever:
    """
    Fuses results from ChunkRetriever, PaperRetriever, and GraphRetriever.

    RRF score per context = sum over sources of:
        source_weight * (1 / (RRF_K + rank + 1))

    Deduplication key: normalised text (strip + lowercase).
    Identical text from multiple sources boosts score rather than duplicating.
    """

    def __init__(
        self,
        chunk_retriever=None,
        paper_retriever=None,
        graph_retriever=None,
        chunk_weight=_UNSET,
        paper_weight=_UNSET,
        graph_weight=_UNSET,
        config_path: str = "config/defaults.yaml",
    ):
        # Load defaults from config; explicit constructor args override config values.
        cfg_chunk, cfg_paper, cfg_graph = 0.5, 0.3, 0.2
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f).get("hybrid_retrieval", {})
                cfg_chunk = cfg.get("chunk_weight", cfg_chunk)
                cfg_paper = cfg.get("paper_weight", cfg_paper)
                cfg_graph = cfg.get("graph_weight", cfg_graph)
        except Exception:
            pass

        self.chunk_weight = chunk_weight if chunk_weight is not _UNSET else cfg_chunk
        self.paper_weight = paper_weight if paper_weight is not _UNSET else cfg_paper
        self.graph_weight = graph_weight if graph_weight is not _UNSET else cfg_graph

        self.chunk_retriever = chunk_retriever
        self.paper_retriever = paper_retriever
        self.graph_retriever = graph_retriever

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.2,
        use_graph: bool = True,
    ) -> list[RetrievedContext]:
        """
        Retrieve from all sources and fuse with weighted RRF.

        Args:
            query: Query text (for graph retriever)
            query_embedding: Pre-computed query vector (for vector retrievers)
            top_k: Number of final results
            min_score: Minimum score for vector retrievers
            use_graph: If False, skip graph retrieval (vector-only mode)
        """
        fetch_k = top_k * 3

        chunk_results: list[RetrievedContext] = []
        if self.chunk_retriever:
            chunk_results = self.chunk_retriever.search(
                query_vector=query_embedding, top_k=fetch_k, min_score=min_score
            )

        paper_results: list[RetrievedContext] = []
        if self.paper_retriever:
            paper_results = self.paper_retriever.search(
                query_vector=query_embedding, top_k=fetch_k, min_score=min_score
            )

        graph_results: list[RetrievedContext] = []
        if use_graph and self.graph_retriever:
            graph_results = self.graph_retriever.retrieve(query=query, top_k=fetch_k)

        return self._fuse(
            chunk_results, paper_results, graph_results, top_k=top_k
        )

    def _fuse(
        self,
        chunk_results: list[RetrievedContext],
        paper_results: list[RetrievedContext],
        graph_results: list[RetrievedContext],
        top_k: int,
    ) -> list[RetrievedContext]:
        """Weighted RRF fusion with text-based deduplication."""
        # dedup_key -> (RetrievedContext, rrf_score)
        seen: dict[str, tuple[RetrievedContext, float]] = {}

        def _key(text: str) -> str:
            return text.strip().lower()

        def _add(results: list[RetrievedContext], weight: float):
            for rank, ctx in enumerate(results):
                k = _key(ctx.text)
                rrf = weight * (1.0 / (RRF_K + rank + 1))
                if k in seen:
                    existing_ctx, existing_score = seen[k]
                    seen[k] = (existing_ctx, existing_score + rrf)
                else:
                    seen[k] = (ctx, rrf)

        _add(chunk_results, self.chunk_weight)
        _add(paper_results, self.paper_weight)
        _add(graph_results, self.graph_weight)

        sorted_items = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        out: list[RetrievedContext] = []
        for ctx, score in sorted_items[:top_k]:
            ctx.score = round(score, 6)
            out.append(ctx)
        return out
