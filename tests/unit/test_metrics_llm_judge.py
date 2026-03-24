import pytest
from unittest.mock import MagicMock, patch
from tests.evaluation.metrics import RAGMetrics


def _make_metrics() -> RAGMetrics:
    """RAGMetrics with a mock LLM client (no patch started)."""
    return RAGMetrics(llm_client=MagicMock())


class TestAnswerCorrectnessStructuredOutput:
    def test_returns_dict_with_score_and_justification(self):
        m = _make_metrics()
        with patch.object(m, "_generate_with_retry", return_value="Score: 0.8\nJustification: Answer is mostly correct."):
            result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert isinstance(result, dict)
        assert "score" in result
        assert "justification" in result

    def test_score_parsed_correctly(self):
        m = _make_metrics()
        with patch.object(m, "_generate_with_retry", return_value="Score: 0.75\nJustification: Partially correct."):
            result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert result["score"] == pytest.approx(0.75)

    def test_score_clamped_above_1(self):
        m = _make_metrics()
        with patch.object(m, "_generate_with_retry", return_value="Score: 1.5\nJustification: Way too high."):
            result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert result["score"] == pytest.approx(1.0)

    def test_score_clamped_below_0(self):
        m = _make_metrics()
        with patch.object(m, "_generate_with_retry", return_value="Score: -0.2\nJustification: Negative."):
            result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert result["score"] == pytest.approx(0.0)

    def test_parse_failure_returns_none_score(self):
        m = _make_metrics()
        with patch.object(m, "_generate_with_retry", return_value="I think it is pretty good overall honestly"):
            result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert result["score"] is None

    def test_parse_failure_returns_raw_response_in_justification(self):
        raw = "I think it is pretty good overall"
        m = _make_metrics()
        with patch.object(m, "_generate_with_retry", return_value=raw):
            result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert result["justification"] is not None
        assert len(result["justification"]) > 0

    def test_counterfactual_prompt_contains_premise_clause(self):
        captured = {}
        m = RAGMetrics(llm_client=MagicMock())

        def _capture(prompt, **kwargs):
            captured["prompt"] = prompt
            return "Score: 0.8\nJustification: Correct."

        with patch.object(m, "_generate_with_retry", side_effect=_capture):
            m.calculate_answer_correctness("q", "gt", "pred", question_type="counterfactual")

        assert "premise is false" in captured["prompt"].lower()

    def test_structured_prompt_requires_score_format(self):
        captured = {}
        m = RAGMetrics(llm_client=MagicMock())

        def _capture(prompt, **kwargs):
            captured["prompt"] = prompt
            return "Score: 0.7\nJustification: OK."

        with patch.object(m, "_generate_with_retry", side_effect=_capture):
            m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")

        assert "Score:" in captured["prompt"]
        assert "Justification:" in captured["prompt"]

    def test_no_llm_returns_none_score(self):
        m = RAGMetrics(llm_client=None)
        result = m.calculate_answer_correctness("q", "gt", "pred", question_type="open_ended")
        assert result["score"] is None
        assert result["justification"] is None
