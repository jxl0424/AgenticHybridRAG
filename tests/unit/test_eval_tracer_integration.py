"""Verify HybridRAGEvaluator sets up self.tracer without crashing."""
from unittest.mock import MagicMock, patch


def test_evaluator_tracer_is_none_when_phoenix_unavailable():
    """When Phoenix is unavailable, self.tracer is None and no exception is raised."""
    from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline
    from tests.evaluation.hybridrag_eval import HybridRAGEvaluator

    mock_pipeline = MagicMock(spec=HybridRAGBenchPipeline)

    with patch("tests.evaluation.hybridrag_eval.start_phoenix", return_value=None):
        evaluator = HybridRAGEvaluator(mock_pipeline)

    assert evaluator.tracer is None


def test_run_query_passes_tracer_to_pipeline():
    """_run_query passes self.tracer through to pipeline.query."""
    from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline
    from tests.evaluation.hybridrag_eval import HybridRAGEvaluator

    mock_pipeline = MagicMock(spec=HybridRAGBenchPipeline)
    mock_pipeline.query.return_value = {
        "answer": "ans", "contexts": [], "sources": [], "scores": [],
        "entities_found": [], "retrieval_type": "hybrid",
        "trace": {"answer_type": "answer"},
    }

    sentinel_tracer = object()

    with patch("tests.evaluation.hybridrag_eval.start_phoenix", return_value=sentinel_tracer):
        evaluator = HybridRAGEvaluator(mock_pipeline)

    evaluator._run_query("q?", "hybrid", 5)
    _, kwargs = mock_pipeline.query.call_args
    assert kwargs.get("tracer") is sentinel_tracer
