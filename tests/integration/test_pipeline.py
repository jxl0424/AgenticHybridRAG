import pytest
from unittest.mock import MagicMock, patch
from src.pipelines.graph_rag_pipeline import GraphRAGPipeline

@patch('src.pipelines.graph_rag_pipeline.QdrantStorage')
@patch('src.pipelines.graph_rag_pipeline.KnowledgeGraph')
@patch('src.pipelines.graph_rag_pipeline.LLMClient')
@patch('src.pipelines.graph_rag_pipeline.embed_texts')
@patch('src.pipelines.graph_rag_pipeline.load_and_chunk_pdf')
def test_pipeline_query_flow(mock_load, mock_embed, mock_llm, mock_kg, mock_qdrant):
    # Setup mocks
    mock_load.return_value = ["Text chunk 1", "Text chunk 2"]
    mock_embed.return_value = [[0.1]*768, [0.2]*768]
    mock_llm.return_value.generate.return_value = "This is a generated answer."
    
    # Mock retrieval results
    mock_qdrant.return_value.search.return_value = {
        "contexts": ["Text chunk 1"],
        "sources": ["test.pdf"],
        "scores": [0.9]
    }
    
    # Create pipeline
    pipeline = GraphRAGPipeline()
    
    # Test query
    result = pipeline.query("What is the treatment for hypertension?")
    
    # Assertions
    assert result["answer"] == "This is a generated answer."
    assert result["num_contexts"] == 1
    assert "test.pdf" in result["sources"]
    
    # Verify LLM was called
    assert mock_llm.return_value.generate.called
