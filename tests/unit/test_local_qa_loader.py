"""
Tests for LocalParquetLoader.load_local_qa_pairs() and
HybridRAGEvaluator.load_qa_pairs(local_data_dir=...).
"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.local_parquet_loader import LocalParquetLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_qa_parquet(tmp_path: Path, domain: str, rows: list[dict]) -> None:
    """Write a qa.parquet file for a given domain under tmp_path."""
    out = tmp_path / "text_qa" / domain
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "qa.parquet", index=False)


@pytest.fixture
def qa_data_dir(tmp_path):
    """
    Minimal QA parquet dataset with two domains and mixed question types.
    """
    ai_rows = [
        {"domain": "arxiv_ai", "split": "test",  "question_id": 0,
         "question": "What is attention?",        "answer": "A mechanism.",
         "question_type": "single_hop"},
        {"domain": "arxiv_ai", "split": "test",  "question_id": 1,
         "question": "Why use transformers?",     "answer": "Parallelism.",
         "question_type": "multi_hop"},
        {"domain": "arxiv_ai", "split": "train", "question_id": 2,
         "question": "Train question?",           "answer": "Train answer.",
         "question_type": "single_hop"},
    ]
    cy_rows = [
        {"domain": "arxiv_cy", "split": "test",  "question_id": 0,
         "question": "What is a graph?",          "answer": "Nodes and edges.",
         "question_type": "single_hop"},
    ]
    _make_qa_parquet(tmp_path, "arxiv_ai", ai_rows)
    _make_qa_parquet(tmp_path, "arxiv_cy", cy_rows)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# LocalParquetLoader.load_local_qa_pairs()
# ---------------------------------------------------------------------------

class TestLoadLocalQAPairs:

    def test_returns_list_of_dicts(self, qa_data_dir):
        loader = LocalParquetLoader(qa_data_dir)
        pairs = loader.load_local_qa_pairs()
        assert isinstance(pairs, list)
        assert len(pairs) > 0

    def test_each_item_has_required_keys(self, qa_data_dir):
        loader = LocalParquetLoader(qa_data_dir)
        pairs = loader.load_local_qa_pairs()
        for p in pairs:
            assert "question" in p
            assert "ground_truth_answer" in p
            assert "ground_truth_context" in p
            assert "question_type" in p
            assert "domain" in p

    def test_ground_truth_context_is_empty_string(self, qa_data_dir):
        loader = LocalParquetLoader(qa_data_dir)
        pairs = loader.load_local_qa_pairs()
        for p in pairs:
            assert p["ground_truth_context"] == ""

    def test_filters_by_split_test_by_default(self, qa_data_dir):
        loader = LocalParquetLoader(qa_data_dir)
        pairs = loader.load_local_qa_pairs()
        # arxiv_ai has 2 test rows, arxiv_cy has 1 test row
        assert len(pairs) == 3
        for p in pairs:
            assert p["question"] != "Train question?"

    def test_filters_by_split_train(self, qa_data_dir):
        loader = LocalParquetLoader(qa_data_dir)
        pairs = loader.load_local_qa_pairs(split="train")
        assert len(pairs) == 1
        assert pairs[0]["question"] == "Train question?"

    def test_filters_to_specified_domains(self, qa_data_dir):
        loader = LocalParquetLoader(qa_data_dir)
        pairs = loader.load_local_qa_pairs(domains=["arxiv_ai"])
        assert all(p["domain"] == "arxiv_ai" for p in pairs)
        assert len(pairs) == 2  # two test rows in arxiv_ai

    def test_max_pairs_limits_results(self, qa_data_dir):
        loader = LocalParquetLoader(qa_data_dir)
        pairs = loader.load_local_qa_pairs(max_pairs=2)
        assert len(pairs) <= 2

    def test_skips_missing_domain_directory(self, qa_data_dir):
        """Requesting a domain with no file should not raise — just skip it."""
        loader = LocalParquetLoader(qa_data_dir)
        pairs = loader.load_local_qa_pairs(domains=["arxiv_qm"])  # not created
        assert pairs == []

    def test_question_and_answer_values_match_parquet(self, qa_data_dir):
        loader = LocalParquetLoader(qa_data_dir)
        pairs = loader.load_local_qa_pairs(domains=["arxiv_cy"])
        assert len(pairs) == 1
        assert pairs[0]["question"] == "What is a graph?"
        assert pairs[0]["ground_truth_answer"] == "Nodes and edges."
        assert pairs[0]["question_type"] == "single_hop"
        assert pairs[0]["domain"] == "arxiv_cy"


# ---------------------------------------------------------------------------
# HybridRAGEvaluator.load_qa_pairs(local_data_dir=...)
# ---------------------------------------------------------------------------

def _import_evaluator():
    """
    Import HybridRAGEvaluator while stubbing out the missing
    hf_hybridrag_loader module that hybridrag_pipeline.py depends on.
    """
    import sys
    from unittest.mock import MagicMock

    # Stub the missing HuggingFace loader before any import touches it
    hf_stub = MagicMock()
    hf_stub.HybridRAGLoader = MagicMock
    hf_stub.HybridRAGKGLoader = MagicMock
    hf_stub.embed_texts_with_model = MagicMock(return_value=[[0.0] * 768])

    sys.modules.setdefault("src.ingestion.hf_hybridrag_loader", hf_stub)

    # Also stub heavy pipeline deps so no DB connections are opened
    for mod in (
        "src.graph.cs_entity_extractor",
        "src.graph.cs_knowledge_graph",
        "src.graph.knowledge_graph",
        "src.retrieval.qdrant_storage",
        "src.retrieval.hybrid_retriever",
        "src.retrieval.graph_retriever",
        "src.retrieval.reranker",
        "src.generation.llm_client",
    ):
        sys.modules.setdefault(mod, MagicMock())

    from tests.evaluation.hybridrag_eval import HybridRAGEvaluator
    return HybridRAGEvaluator


class TestHybridRAGEvaluatorLocalLoad:

    def test_load_qa_pairs_with_local_data_dir(self, qa_data_dir):
        """When local_data_dir is given, QA pairs are read from local files."""
        from unittest.mock import MagicMock

        HybridRAGEvaluator = _import_evaluator()
        mock_pipeline = MagicMock()
        mock_pipeline.llm = MagicMock()
        evaluator = HybridRAGEvaluator(mock_pipeline)

        pairs = evaluator.load_qa_pairs(local_data_dir=qa_data_dir)
        assert len(pairs) > 0
        assert "question" in pairs[0]
        assert "ground_truth_answer" in pairs[0]

    def test_load_qa_pairs_local_does_not_call_hf_loader(self, qa_data_dir):
        """HuggingFace HybridRAGLoader is never instantiated when local_data_dir is set."""
        import sys
        from unittest.mock import MagicMock

        HybridRAGEvaluator = _import_evaluator()
        mock_pipeline = MagicMock()
        mock_pipeline.llm = MagicMock()
        evaluator = HybridRAGEvaluator(mock_pipeline)

        # HybridRAGLoader is imported inside load_qa_pairs via the hf stub;
        # replace its constructor with a sentinel that raises if called.
        hf_stub = sys.modules["src.ingestion.hf_hybridrag_loader"]
        original = hf_stub.HybridRAGLoader

        call_record = []

        class _GuardLoader:
            def __init__(self, *a, **kw):
                call_record.append(True)

        hf_stub.HybridRAGLoader = _GuardLoader
        try:
            evaluator.load_qa_pairs(local_data_dir=qa_data_dir)
        finally:
            hf_stub.HybridRAGLoader = original

        assert call_record == [], "HybridRAGLoader was instantiated but should not have been"

    def test_load_qa_pairs_local_max_pairs_forwarded(self, qa_data_dir):
        """max_pairs is forwarded to the local loader."""
        from unittest.mock import MagicMock

        HybridRAGEvaluator = _import_evaluator()
        mock_pipeline = MagicMock()
        mock_pipeline.llm = MagicMock()
        evaluator = HybridRAGEvaluator(mock_pipeline)

        pairs = evaluator.load_qa_pairs(local_data_dir=qa_data_dir, max_pairs=1)
        assert len(pairs) <= 1
