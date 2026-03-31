"""Shared pytest fixtures for the unit test suite."""
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True, scope="session")
def _stub_heavy_imports():
    """Register lightweight stubs for heavy optional dependencies.

    This prevents ImportError when the unit tests import modules that
    depend on Qdrant, HuggingFace datasets, Phoenix, etc.
    """
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
