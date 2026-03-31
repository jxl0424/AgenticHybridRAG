"""
Tests for HybridRAGEvaluator._compute_metrics() type-based dispatch.
"""
import sys
from unittest.mock import MagicMock
import pytest


def _make_evaluator():
    """Import HybridRAGEvaluator with all heavy deps stubbed."""
    for mod in (
        "src.ingestion.hf_hybridrag_loader",
        "src.graph.cs_entity_extractor",
        "src.graph.cs_knowledge_graph",
        "src.retrieval.qdrant_storage",
        "src.retrieval.hybrid_retriever",
        "src.retrieval.graph_retriever",
        "src.retrieval.reranker",
        "src.generation.llm_client",
        "src.observability.tracer",
    ):
        sys.modules.setdefault(mod, MagicMock())

    from tests.evaluation.hybridrag_eval import HybridRAGEvaluator

    mock_pipeline = MagicMock()
    evaluator = HybridRAGEvaluator.__new__(HybridRAGEvaluator)
    evaluator.pipeline = mock_pipeline
    evaluator.tracer = None

    # Inject a controllable metrics mock
    metrics_mock = MagicMock()
    metrics_mock.normalize.return_value = "normalized prediction text"
    metrics_mock.calculate_exact_match.return_value = 0.0
    metrics_mock.calculate_token_f1.return_value = 0.0
    metrics_mock.calculate_answer_correctness.return_value = {
        "score": 0.8, "justification": "Correct."
    }
    metrics_mock.calculate_context_recall.return_value = 0.6
    metrics_mock.calculate_faithfulness.return_value = 0.7
    evaluator.metrics = metrics_mock
    return evaluator


RESPONSE = {"answer": "some prediction", "contexts": ["ctx1", "ctx2"], "trace": {}}


class TestDispatchSingleHop:
    def test_computes_em_and_token_f1(self):
        ev = _make_evaluator()
        ev.metrics.calculate_exact_match.return_value = 1.0
        ev.metrics.calculate_token_f1.return_value = 1.0
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="single_hop", domain="d", seed=1, run=1)
        assert item["metrics"]["exact_match"] == 1.0
        assert item["metrics"]["token_f1"] == 1.0

    def test_llm_judge_not_called(self):
        ev = _make_evaluator()
        ev._compute_metrics("q", "gt", "", RESPONSE,
                             question_type="single_hop", domain="d", seed=1, run=1)
        ev.metrics.calculate_answer_correctness.assert_not_called()

    def test_final_answer_correctness_source_is_token_f1(self):
        ev = _make_evaluator()
        ev.metrics.calculate_token_f1.return_value = 0.75
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="single_hop", domain="d", seed=1, run=1)
        assert item["metrics"]["final_answer_correctness"] == pytest.approx(0.75)
        assert item["metrics"]["final_answer_correctness_source"] == "token_f1"


class TestDispatchOpenEnded:
    def test_calls_llm_judge(self):
        ev = _make_evaluator()
        ev._compute_metrics("q", "gt", "", RESPONSE,
                             question_type="open_ended", domain="d", seed=1, run=1)
        ev.metrics.calculate_answer_correctness.assert_called_once()

    def test_em_and_token_f1_are_none(self):
        ev = _make_evaluator()
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="open_ended", domain="d", seed=1, run=1)
        assert item["metrics"]["exact_match"] is None
        assert item["metrics"]["token_f1"] is None

    def test_final_answer_correctness_source_is_llm(self):
        ev = _make_evaluator()
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="open_ended", domain="d", seed=1, run=1)
        assert item["metrics"]["final_answer_correctness_source"] == "llm"


class TestDispatchMultiHopDifficult:
    def test_escalates_to_llm_when_token_f1_zero_and_long_prediction(self):
        ev = _make_evaluator()
        ev.metrics.calculate_token_f1.return_value = 0.0
        # normalize returns 16 words -> triggers escalation (threshold > 15)
        ev.metrics.normalize.return_value = " ".join(["word"] * 16)
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="multi_hop_difficult", domain="d", seed=1, run=1)
        ev.metrics.calculate_answer_correctness.assert_called_once()
        assert item["metrics"]["final_answer_correctness_source"] == "llm"

    def test_no_escalation_when_token_f1_nonzero(self):
        ev = _make_evaluator()
        ev.metrics.calculate_token_f1.return_value = 0.5
        ev.metrics.normalize.return_value = " ".join(["word"] * 16)
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="multi_hop_difficult", domain="d", seed=1, run=1)
        ev.metrics.calculate_answer_correctness.assert_not_called()
        assert item["metrics"]["final_answer_correctness_source"] == "token_f1"

    def test_no_escalation_when_prediction_short(self):
        ev = _make_evaluator()
        ev.metrics.calculate_token_f1.return_value = 0.0
        # Only 5 words -> no escalation (threshold is > 15)
        ev.metrics.normalize.return_value = "five short words only here"
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="multi_hop_difficult", domain="d", seed=1, run=1)
        ev.metrics.calculate_answer_correctness.assert_not_called()

    def test_no_escalation_at_exact_boundary(self):
        """Exactly 15 words should NOT escalate (threshold is strictly > 15)."""
        ev = _make_evaluator()
        ev.metrics.calculate_token_f1.return_value = 0.0
        ev.metrics.normalize.return_value = " ".join(["word"] * 15)
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="multi_hop_difficult", domain="d", seed=1, run=1)
        ev.metrics.calculate_answer_correctness.assert_not_called()


class TestOutputSchema:
    def test_output_has_required_keys(self):
        ev = _make_evaluator()
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="single_hop", domain="arxiv_ai", seed=42, run=2)
        for key in ("question", "question_type", "domain", "prediction", "ground_truth",
                    "ground_truth_context", "seed", "run", "metrics", "num_contexts", "trace"):
            assert key in item, f"Missing key: {key}"

    def test_domain_and_seed_in_output(self):
        ev = _make_evaluator()
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="single_hop", domain="arxiv_cy", seed=77, run=3)
        assert item["domain"] == "arxiv_cy"
        assert item["seed"] == 77
        assert item["run"] == 3

    def test_prediction_renamed_from_answer(self):
        ev = _make_evaluator()
        item = ev._compute_metrics("q", "gt", "", RESPONSE,
                                   question_type="single_hop", domain="d", seed=1, run=1)
        assert item["prediction"] == "some prediction"
        assert "answer" not in item  # old key must be gone
