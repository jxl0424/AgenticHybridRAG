"""Verify query() returns the same shape dict when tracer=None (no-op path)."""
from unittest.mock import MagicMock, patch


def _make_mock_context(text="ctx", source="arxiv_chunks", score=0.5):
    from src.types import RetrievedContext
    return RetrievedContext(text=text, source=source, score=score,
                            collection="arxiv_chunks", metadata={})


def test_query_tracer_none_returns_expected_keys():
    """query(tracer=None) must return dict with all standard keys — no functional change."""
    from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline

    pipeline = MagicMock()
    ctx = _make_mock_context()
    pipeline.hybrid_retriever.retrieve.return_value = [ctx]
    pipeline.graph_retriever._last_trace = {
        "entities_extracted": [], "qdrant_ids_per_entity": {},
        "total_qdrant_ids": 0, "fetched_count": 0,
    }
    pipeline.hybrid_retriever._last_trace = {
        "chunk_count_raw": 1, "paper_count_raw": 0, "graph_count_raw": 0,
        "fused_count": 1, "fused_source_breakdown": {"arxiv_chunks": 1},
        "top_rrf_scores": [0.5],
    }
    pipeline.hybrid_retriever.chunk_weight = 0.5
    pipeline.hybrid_retriever.paper_weight = 0.3
    pipeline.hybrid_retriever.graph_weight = 0.2
    pipeline.embedding_model = "allenai/specter2_base"
    pipeline.min_score = 0.05
    pipeline.reranker = None
    pipeline.llm.generate.return_value = "Test answer."
    pipeline.llm.model = "qwen2.5:7b-instruct"

    with patch("src.pipelines.hybridrag_pipeline.embed_texts_with_model",
               return_value=[[0.1] * 768]):
        with patch("src.pipelines.hybridrag_pipeline.build_messages", return_value=[]):
            result = HybridRAGBenchPipeline.query(pipeline, "test question", tracer=None)

    assert set(result.keys()) == {"question", "answer", "contexts", "sources",
                                   "scores", "entities_found", "retrieval_type", "trace"}
    trace = result["trace"]
    assert set(trace.keys()) >= {
        "entities_extracted", "qdrant_ids_per_entity", "graph_qdrant_ids_total",
        "graph_fetched_count", "raw_counts", "pre_rerank_count",
        "pre_rerank_source_breakdown", "post_rerank_count", "dropped_by_reranker",
        "reranker_scores", "answer_type", "top3_contexts",
    }
