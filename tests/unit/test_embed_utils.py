"""
Tests for embed_texts_with_model in src.utils.
"""
import sys
from unittest.mock import MagicMock, patch
import pytest


def test_embed_texts_with_model_importable():
    """embed_texts_with_model must be importable from src.utils."""
    from src.utils import embed_texts_with_model
    assert callable(embed_texts_with_model)


def test_embed_texts_with_model_returns_list_of_lists():
    """Returns one embedding vector per input text."""
    import numpy as np

    fake_embeddings = np.array([[0.1] * 768, [0.2] * 768])
    mock_model = MagicMock()
    mock_model.encode.return_value = fake_embeddings

    with patch("src.utils.SentenceTransformer", return_value=mock_model):
        from src.utils import embed_texts_with_model
        result = embed_texts_with_model(["hello", "world"], "some-model", batch_size=2)

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert len(result[0]) == 768


def test_cs_knowledge_graph_importable_without_knowledge_graph_module():
    """
    cs_knowledge_graph must be importable without src.graph.knowledge_graph.
    Confirms the module is standalone.
    """
    import importlib

    # Strip both modules from cache to force a fresh import
    sys.modules.pop("src.graph.knowledge_graph", None)
    sys.modules.pop("src.graph.cs_knowledge_graph", None)

    # Stub the Neo4j / storage dep so no live connection is attempted
    sys.modules.setdefault("src.storage.neo4j_client", MagicMock())
    sys.modules.setdefault("src.graph.cs_entity_extractor", MagicMock())

    mod = importlib.import_module("src.graph.cs_knowledge_graph")

    # The module must expose these names after the fix
    assert hasattr(mod, "GraphDocument")
    assert hasattr(mod, "GraphChunk")
    assert hasattr(mod, "GraphEntity")
    assert hasattr(mod, "CSKnowledgeGraph")


def test_hybridrag_pipeline_importable_without_hf_module():
    """
    hybridrag_pipeline must be importable without src.ingestion.hf_hybridrag_loader.
    Confirms the top-level import error is gone.
    """
    # Stub heavy deps so no live DB connections are attempted
    heavy = [
        "src.graph.cs_entity_extractor",
        "src.graph.cs_knowledge_graph",
        "src.graph.knowledge_graph",
        "src.retrieval.qdrant_storage",
        "src.retrieval.hybrid_retriever",
        "src.retrieval.graph_retriever",
        "src.retrieval.reranker",
        "src.generation.llm_client",
    ]
    for mod in heavy:
        sys.modules.setdefault(mod, MagicMock())

    # Ensure hf_hybridrag_loader is NOT in sys.modules so we confirm pipeline
    # no longer needs it at import time
    sys.modules.pop("src.ingestion.hf_hybridrag_loader", None)
    sys.modules.pop("src.pipelines.hybridrag_pipeline", None)

    import importlib
    # Should not raise ModuleNotFoundError
    importlib.import_module("src.pipelines.hybridrag_pipeline")
