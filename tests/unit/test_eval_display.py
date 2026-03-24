"""Tests for print_type_breakdown() and print_multirun_summary()."""
from unittest.mock import MagicMock
import pytest

from tests.evaluation.hybridrag_eval import HybridRAGEvaluator


@pytest.fixture
def ev():
    e = HybridRAGEvaluator.__new__(HybridRAGEvaluator)
    e.pipeline = MagicMock()
    e.tracer = None
    e.metrics = MagicMock()
    return e


class TestPrintTypeBreakdown:
    def test_contains_extractive_section_header(self, ev, capsys):
        ev.print_type_breakdown({})
        assert "[Extractive: EM / Token F1]" in capsys.readouterr().out

    def test_contains_llm_judge_section_header(self, ev, capsys):
        ev.print_type_breakdown({})
        assert "[LLM Judge]" in capsys.readouterr().out

    def test_shows_na_for_missing_type(self, ev, capsys):
        ev.print_type_breakdown({})
        assert "N/A" in capsys.readouterr().out

    def test_shows_score_for_known_type(self, ev, capsys):
        ev.print_type_breakdown({"single_hop": {"final_answer_correctness": 0.85,
                                                  "context_recall": 0.60}})
        assert "0.8500" in capsys.readouterr().out


class TestPrintMultirunSummary:
    SUMMARY = {
        "config": {"seed": 42, "runs": 3, "k_per_type": 5,
                   "modes": ["hybrid"], "domain_note": "domain imbalance note"},
        "per_metric": {
            "final_answer_correctness": {
                "mean": 0.74, "std": 0.03, "median": 0.75,
                "min": 0.69, "max": 0.79, "num_nulls": 0,
            },
        },
        "per_type": {
            "single_hop": {
                "count": 15,
                "final_answer_correctness": {"mean": 0.81, "std": 0.03},
                "context_recall": {"mean": 0.55, "std": 0.05},
            },
        },
    }

    def test_contains_multi_run_summary_header(self, ev, capsys):
        ev.print_multirun_summary(self.SUMMARY)
        assert "Multi-Run Summary" in capsys.readouterr().out

    def test_shows_seed_and_runs(self, ev, capsys):
        ev.print_multirun_summary(self.SUMMARY)
        out = capsys.readouterr().out
        assert "42" in out
        assert "3" in out

    def test_shows_null_count_column(self, ev, capsys):
        ev.print_multirun_summary(self.SUMMARY)
        out = capsys.readouterr().out
        assert "null" in out.lower()

    def test_shows_per_type_mean(self, ev, capsys):
        ev.print_multirun_summary(self.SUMMARY)
        assert "0.8100" in capsys.readouterr().out

    def test_shows_domain_note_warning(self, ev, capsys):
        ev.print_multirun_summary(self.SUMMARY)
        out = capsys.readouterr().out
        assert "WARNING" in out or "domain" in out.lower()
