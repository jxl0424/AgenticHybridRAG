import uuid
from unittest.mock import MagicMock, patch

from src.pipelines.local_ingestion_pipeline import LocalIngestionPipeline
from src.ingestion.local_parquet_loader import NodeRecord, ChunkRecord, HYBRIDRAG_NS


def _make_pipeline():
    """Return a LocalIngestionPipeline with all IO dependencies mocked."""
    pipeline = LocalIngestionPipeline.__new__(LocalIngestionPipeline)
    pipeline.neo4j_batch_size = 500

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_driver.session.return_value = mock_session

    mock_qdrant = MagicMock()

    # Override lazy-init methods as instance attributes so tests control IO
    pipeline._neo4j = MagicMock(return_value=mock_driver)
    pipeline._q = MagicMock(return_value=mock_qdrant)
    pipeline.loader = MagicMock()

    return pipeline, mock_driver, mock_qdrant


@patch("src.pipelines.local_ingestion_pipeline.embed_texts_with_model")
def test_ingest_nodes_embeds_display_names_not_precomputed(mock_embed):
    """_ingest_nodes calls embed_texts_with_model with display_names, not rec.embedding."""
    fake_vec = [0.5] * 768
    mock_embed.return_value = [fake_vec]

    pipeline, _, mock_qdrant = _make_pipeline()

    # qdrant_id is NOT a constructor arg — computed in __post_init__
    rec = NodeRecord(
        node_id=1, domain="arxiv_ai", display_name="PARTITION FUNCTION",
        entity_type="Concept", embedding=[0.9] * 768,  # pre-computed — must NOT be used
    )
    pipeline.loader.iter_nodes.return_value = [rec]

    total = pipeline._ingest_nodes("arxiv_ai")

    mock_embed.assert_called_once_with(["PARTITION FUNCTION"], "allenai/specter2_base", batch_size=64)

    points = mock_qdrant.upsert.call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].vector == fake_vec      # SPECTER2 vector
    assert points[0].vector != [0.9] * 768  # NOT the pre-computed vector
    expected_id = str(uuid.uuid5(HYBRIDRAG_NS, "arxiv_ai:1"))
    assert points[0].id == expected_id
    assert total == 1


@patch("src.pipelines.local_ingestion_pipeline.embed_texts_with_model")
def test_ingest_nodes_skips_empty_display_name_for_qdrant(mock_embed):
    """Records with empty display_name are written to Neo4j but not embedded or upserted."""
    fake_vec = [0.5] * 768
    mock_embed.return_value = [fake_vec]  # only 1 embedding returned

    pipeline, _, mock_qdrant = _make_pipeline()

    rec_ok = NodeRecord(
        node_id=1, domain="arxiv_ai", display_name="NEURAL NETWORK",
        entity_type="Concept", embedding=[0.1] * 768,
    )
    rec_empty = NodeRecord(
        node_id=2, domain="arxiv_ai", display_name="",
        entity_type="Concept", embedding=[0.2] * 768,
    )
    pipeline.loader.iter_nodes.return_value = [rec_ok, rec_empty]

    total = pipeline._ingest_nodes("arxiv_ai")

    # Only "NEURAL NETWORK" embedded — empty display_name excluded
    mock_embed.assert_called_once_with(["NEURAL NETWORK"], "allenai/specter2_base", batch_size=64)
    points = mock_qdrant.upsert.call_args.kwargs["points"]
    assert len(points) == 1   # rec_empty NOT in Qdrant
    assert total == 2         # BOTH counted in Neo4j total


@patch("src.pipelines.local_ingestion_pipeline.embed_texts_with_model")
def test_ingest_chunks_embeds_paragraphs_not_precomputed(mock_embed):
    """_ingest_chunks embeds rec.paragraph with SPECTER2, not the pre-computed rec.embedding."""
    fake_vec = [0.7] * 768
    mock_embed.return_value = [fake_vec]

    pipeline, _, mock_qdrant = _make_pipeline()
    progress = MagicMock()

    # qdrant_id NOT a constructor arg — computed in __post_init__
    rec = ChunkRecord(
        edge_id=10, domain="arxiv_ai", src_id=1, dst_id=2,
        rel_type="HAS_CHUNK", paragraph="The partition function is central.",
        embedding=[0.9] * 768,  # pre-computed — must NOT be used
        paper_id="1912.13190",
    )
    pipeline.loader.iter_chunks.return_value = [rec]

    total = pipeline._ingest_chunks("arxiv_ai", include_src_names=False, progress=progress)

    mock_embed.assert_called_once_with(
        ["The partition function is central."], "allenai/specter2_base", batch_size=64
    )
    points = mock_qdrant.upsert.call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].vector == fake_vec       # SPECTER2 vector
    assert points[0].vector != [0.9] * 768   # NOT the pre-computed vector
    assert points[0].payload["paragraph"] == "The partition function is central."
    expected_id = str(uuid.uuid5(HYBRIDRAG_NS, "arxiv_ai:10"))
    assert points[0].id == expected_id
    assert total == 1


@patch("src.pipelines.local_ingestion_pipeline.embed_texts_with_model")
def test_ingest_chunks_neo4j_batch_uses_record_attributes(mock_embed):
    """_ingest_chunks builds neo4j_batch from ChunkRecord attributes, not p.payload dict lookups."""
    fake_vec = [0.3] * 768
    mock_embed.return_value = [fake_vec]

    pipeline, mock_driver, mock_qdrant = _make_pipeline()
    mock_session = mock_driver.session.return_value
    progress = MagicMock()

    rec = ChunkRecord(
        edge_id=20, domain="arxiv_ai", src_id=5, dst_id=6,
        rel_type="HAS_CHUNK", paragraph="Some text.",
        embedding=[0.1] * 768, paper_id="1234.5678",
    )
    pipeline.loader.iter_chunks.return_value = [rec]
    pipeline._ingest_chunks("arxiv_ai", include_src_names=False, progress=progress)

    # Find the session.run call that contains HAS_CHUNK
    has_chunk_calls = [
        c for c in mock_session.run.call_args_list
        if "HAS_CHUNK" in str(c.args[0])
    ]
    assert len(has_chunk_calls) == 1
    batch_arg = has_chunk_calls[0].args[1]["batch"]
    assert len(batch_arg) == 1
    assert batch_arg[0]["src_id"] == 5
    assert batch_arg[0]["dst_id"] == 6
    assert batch_arg[0]["edge_id"] == 20
    assert batch_arg[0]["domain"] == "arxiv_ai"
    assert batch_arg[0]["qdrant_id"] == str(uuid.uuid5(HYBRIDRAG_NS, "arxiv_ai:20"))
