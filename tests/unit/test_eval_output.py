"""Tests for multi-run path helpers and summary aggregation."""
from unittest.mock import MagicMock
import pytest

from tests.evaluation.hybridrag_eval import (
    _derive_run_path,
    _derive_summary_path,
    validate_output_path,
    HybridRAGEvaluator,
)


class TestPathHelpers:
    def test_derive_run_path_run1(self):
        assert _derive_run_path("tests/results/eval.json", 1) == "tests/results/eval_run1.json"

    def test_derive_run_path_run3(self):
        assert _derive_run_path("tests/results/eval.json", 3) == "tests/results/eval_run3.json"

    def test_derive_summary_path(self):
        assert _derive_summary_path("tests/results/eval.json") == "tests/results/eval_summary.json"

    def test_validate_output_path_accepts_json(self):
        validate_output_path("my_eval.json")  # must not raise

    def test_validate_output_path_rejects_non_json(self):
        with pytest.raises(ValueError, match=".json"):
            validate_output_path("my_eval.txt")


@pytest.fixture
def ev():
    e = HybridRAGEvaluator.__new__(HybridRAGEvaluator)
    e.pipeline = MagicMock()
    e.tracer = None
    e.metrics = MagicMock()
    return e


class TestBuildSummary:
    def _make_items(self):
        """Six items: one per question type, with known metric values."""
        types = ["single_hop", "single_hop_w_conditions", "multi_hop",
                 "multi_hop_difficult", "open_ended", "counterfactual"]
        items = []
        for qt in types:
            items.append({
                "question_type": qt,
                "metrics": {
                    "final_answer_correctness": 0.8,
                    "exact_match": 1.0 if qt not in {"open_ended", "counterfactual"} else None,
                    "token_f1": 0.9 if qt not in {"open_ended", "counterfactual"} else None,
                    "answer_correctness_llm": 0.8 if qt in {"open_ended", "counterfactual"} else None,
                    "context_recall": 0.6,
                    "faithfulness": 0.7,
                },
            })
        return items

    def test_summary_has_required_top_level_keys(self, ev):
        summary = ev._build_summary(self._make_items(), {"seed": 42, "runs": 1,
                                                          "k_per_type": 5, "modes": ["hybrid"]})
        assert "config" in summary
        assert "per_metric" in summary
        assert "per_type" in summary

    def test_config_contains_domain_note(self, ev):
        summary = ev._build_summary([], {"seed": 0, "runs": 1, "k_per_type": 5, "modes": []})
        assert "domain" in summary["config"]["domain_note"].lower()

    def test_per_metric_contains_final_answer_correctness(self, ev):
        summary = ev._build_summary(self._make_items(), {"seed": 1, "runs": 1,
                                                          "k_per_type": 5, "modes": []})
        assert "final_answer_correctness" in summary["per_metric"]

    def test_per_metric_num_nulls_counts_nulls(self, ev):
        items = self._make_items()
        # answer_correctness_llm is None for 4 extractive types
        summary = ev._build_summary(items, {"seed": 1, "runs": 1, "k_per_type": 5, "modes": []})
        null_count = summary["per_metric"]["answer_correctness_llm"]["num_nulls"]
        assert null_count == 4

    def test_per_type_contains_all_six_types(self, ev):
        summary = ev._build_summary(self._make_items(), {"seed": 1, "runs": 1,
                                                          "k_per_type": 5, "modes": []})
        expected = {"single_hop", "single_hop_w_conditions", "multi_hop",
                    "multi_hop_difficult", "open_ended", "counterfactual"}
        assert set(summary["per_type"].keys()) == expected

    def test_per_type_count_reflects_non_null_final_ac(self, ev):
        summary = ev._build_summary(self._make_items(), {"seed": 1, "runs": 1,
                                                          "k_per_type": 5, "modes": []})
        for qt in summary["per_type"]:
            assert summary["per_type"][qt]["count"] == 1
