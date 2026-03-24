# HybridRAG Retrieval Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-embed `arxiv_chunks` and `arxiv_nodes` with SPECTER2 at ingestion time, and replace the broken exact-name graph entity lookup with semantic vector search via `arxiv_nodes`.

**Architecture:** Three source files change. `local_ingestion_pipeline.py` stops using pre-computed dataset vectors and embeds with SPECTER2 at flush time. `cs_knowledge_graph.py` gains a new `get_chunk_refs_by_node_ids` method for node-ID-based HAS_CHUNK traversal. `graph_retriever.py` replaces the exact-name Neo4j lookup with SPECTER2 embedding of entity names followed by `arxiv_nodes` vector search. After code changes, re-ingest with `--reset` to rebuild all Qdrant collections.

**Tech Stack:** Python, Neo4j (bolt), Qdrant (`qdrant_client`), SPECTER2 via `sentence-transformers` (`allenai/specter2_base`), pytest + `unittest.mock`

---

## File Map

| File | Action | What changes |
|---|---|---|
| `src/graph/cs_knowledge_graph.py` | Modify | Add `get_chunk_refs_by_node_ids(node_ids, limit)` |
| `src/retrieval/graph_retriever.py` | Modify | Add `QdrantClient`/`embed_texts_with_model` imports, `NODES_PER_ENTITY`/`EMBEDDING_MODEL` constants, `qdrant_url` param in `__init__`, semantic lookup in `retrieve()` |
| `src/pipelines/local_ingestion_pipeline.py` | Modify | `_ingest_nodes`: remove outer `qdrant_points`, embed `display_name` with SPECTER2 in `_flush()`; `_ingest_chunks`: rename accumulator to `recs: list[ChunkRecord]`, embed `paragraph` with SPECTER2 in `_flush()` |
| `tests/unit/test_graph.py` | Modify | Add `test_get_chunk_refs_by_node_ids_*` tests |
| `tests/unit/test_retrieval.py` | Modify | Update `test_graph_retriever_uses_has_chunk_path`, add `test_graph_retriever_semantic_lookup` |
| `tests/unit/test_local_ingestion_pipeline.py` | Create | Unit tests for `_ingest_nodes` and `_ingest_chunks` SPECTER2 embedding behavior |

---

## Conftest note (read before touching test files)

`tests/unit/conftest.py` stubs heavy modules with `sys.modules.setdefault(mod, MagicMock())` inside a `session`-scoped autouse fixture. The `setdefault` call only replaces a module if it is NOT already in `sys.modules`. Pytest imports all test files during collection, **before** any session fixture runs. This means:

- Any module-level import in a test file (executed during collection) loads the REAL module into `sys.modules`.
- When the conftest fixture later runs `setdefault`, it finds the real module already present and does nothing.

**Rule:** Import any class you need to test at module level in the test file (not inside the test function). Classes imported at module level during collection will be the real implementations, not stubs.

---

## Task 1: Add `get_chunk_refs_by_node_ids` to `CSKnowledgeGraph`

**Files:**
- Modify: `tests/unit/test_graph.py`
- Modify: `src/graph/cs_knowledge_graph.py:242-258`

- [ ] **Step 1: Add module-level import to `tests/unit/test_graph.py`**

Add at the very top of the file (before the existing `@patch` test). This module-level import runs during pytest collection, before the conftest fixture can stub the module.

```python
from src.graph.cs_knowledge_graph import CSKnowledgeGraph as _RealCSKG
```

- [ ] **Step 2: Write the failing tests in `tests/unit/test_graph.py`**

Append:

```python
@patch("src.graph.cs_knowledge_graph.Neo4jClient")
def test_get_chunk_refs_by_node_ids_returns_qdrant_ids(mock_neo4j_cls):
    client = MagicMock()
    client.connect.return_value = None
    client.execute_read.return_value = [
        {"qdrant_id": "uuid-aaa"},
        {"qdrant_id": "uuid-bbb"},
        {"qdrant_id": None},  # must be filtered out
    ]

    kg = _RealCSKG(neo4j_client=client)
    result = kg.get_chunk_refs_by_node_ids([101, 202], limit=10)

    assert result == ["uuid-aaa", "uuid-bbb"]
    call_args = client.execute_read.call_args
    cypher = call_args.args[0]
    params = call_args.args[1]
    assert "node_id IN $node_ids" in cypher
    assert "-[r:HAS_CHUNK]-" in cypher       # undirected match (no arrow)
    assert "DISTINCT" in cypher               # deduplication
    assert params["node_ids"] == [101, 202]
    assert params["limit"] == 10


@patch("src.graph.cs_knowledge_graph.Neo4jClient")
def test_get_chunk_refs_by_node_ids_empty_input(mock_neo4j_cls):
    client = MagicMock()
    client.connect.return_value = None
    client.execute_read.return_value = []

    kg = _RealCSKG(neo4j_client=client)
    result = kg.get_chunk_refs_by_node_ids([], limit=10)

    assert result == []
```

- [ ] **Step 3: Run tests to confirm they fail**

```
pytest tests/unit/test_graph.py::test_get_chunk_refs_by_node_ids_returns_qdrant_ids tests/unit/test_graph.py::test_get_chunk_refs_by_node_ids_empty_input -v
```

Expected: `FAILED` with `AttributeError: 'CSKnowledgeGraph' object has no attribute 'get_chunk_refs_by_node_ids'`

- [ ] **Step 4: Add the method to `src/graph/cs_knowledge_graph.py`**

Insert after `get_chunk_refs_for_entity` (after the closing `return` on line 258):

```python
    def get_chunk_refs_by_node_ids(self, node_ids: list[int], limit: int = 20) -> list[str]:
        """
        Return qdrant_id strings from HAS_CHUNK edges connected to _Embeddable
        nodes whose node_id is in the given list. Matches both src and dst
        sides of the HAS_CHUNK relationship.
        """
        query = """
        MATCH (n:_Embeddable)-[r:HAS_CHUNK]-()
        WHERE n.node_id IN $node_ids
        RETURN DISTINCT r.qdrant_id AS qdrant_id
        LIMIT $limit
        """
        rows = self.client.execute_read(query, {"node_ids": node_ids, "limit": limit})
        return [r["qdrant_id"] for r in rows if r.get("qdrant_id")]
```

- [ ] **Step 5: Run tests to confirm they pass**

```
pytest tests/unit/test_graph.py -v
```

Expected: All tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/graph/cs_knowledge_graph.py tests/unit/test_graph.py
git commit -m "feat: add get_chunk_refs_by_node_ids to CSKnowledgeGraph"
```

---

## Task 2: Update `GraphRetriever` with semantic entity lookup

**Files:**
- Modify: `tests/unit/test_retrieval.py`
- Modify: `src/retrieval/graph_retriever.py`

`test_retrieval.py` already imports `GraphRetriever` at module level (line 4), so it uses the real class. The existing test `test_graph_retriever_uses_has_chunk_path` mocks `get_chunk_refs_for_entity` — the method being replaced. That test must be updated. A new test covers the semantic lookup path.

- [ ] **Step 1: Write the failing test in `tests/unit/test_retrieval.py`**

Append:

```python
@patch("src.retrieval.graph_retriever.QdrantClient")
@patch("src.retrieval.graph_retriever.embed_texts_with_model")
def test_graph_retriever_semantic_lookup(mock_embed, mock_qdrant_cls):
    """retrieve() embeds entity names with SPECTER2, searches arxiv_nodes, traverses HAS_CHUNK."""
    fake_vec = [0.1] * 768
    mock_embed.return_value = [fake_vec]

    mock_qdrant_inst = mock_qdrant_cls.return_value
    mock_point = MagicMock()
    mock_point.payload = {"node_id": 42}
    mock_qdrant_result = MagicMock()
    mock_qdrant_result.points = [mock_point]
    mock_qdrant_inst.query_points.return_value = mock_qdrant_result

    mock_kg = MagicMock()
    mock_kg.get_chunk_refs_by_node_ids.return_value = ["uuid-001", "uuid-002"]

    mock_extractor = MagicMock()
    mock_entity = MagicMock()
    mock_entity.text = "transformer"
    mock_extractor.extract_entities.return_value.entities = [mock_entity]

    mock_chunk_retriever = MagicMock(spec=ChunkRetriever)
    mock_chunk_retriever.fetch_by_ids.return_value = [
        RetrievedContext(
            text="Transformers use self-attention mechanisms.",
            source="1706.03762",
            score=1.0,
            collection="arxiv_chunks",
        )
    ]

    retriever = GraphRetriever(
        knowledge_graph=mock_kg,
        entity_extractor=mock_extractor,
        chunk_retriever=mock_chunk_retriever,
    )
    results = retriever.retrieve("What is the transformer model?", top_k=5)

    # Semantic lookup: embed name → search arxiv_nodes → get_chunk_refs_by_node_ids
    mock_embed.assert_called_once_with(["transformer"], "allenai/specter2_base", batch_size=1)
    mock_qdrant_inst.query_points.assert_called_once_with(
        "arxiv_nodes", query=fake_vec, with_payload=True, limit=3
    )
    mock_kg.get_chunk_refs_by_node_ids.assert_called_once_with([42], limit=10)
    mock_chunk_retriever.fetch_by_ids.assert_called_once_with(["uuid-001", "uuid-002"])
    assert len(results) == 1
    assert results[0].collection == "graph"
    # Old exact-name path must NOT be used
    mock_kg.get_chunk_refs_for_entity.assert_not_called()
```

- [ ] **Step 2: Update `test_graph_retriever_uses_has_chunk_path` in `tests/unit/test_retrieval.py`**

Replace the entire function (the one starting at line 33) with the version below. It now mocks the semantic path and asserts that `get_chunk_refs_for_entity` is not called:

```python
@patch("src.retrieval.graph_retriever.QdrantClient")
@patch("src.retrieval.graph_retriever.embed_texts_with_model")
def test_graph_retriever_uses_has_chunk_path(mock_embed, mock_qdrant_cls):
    """retrieve() resolves entity names to chunks via semantic arxiv_nodes search + HAS_CHUNK traversal."""
    fake_vec = [0.2] * 768
    mock_embed.return_value = [fake_vec]

    mock_qdrant_inst = mock_qdrant_cls.return_value
    mock_point = MagicMock()
    mock_point.payload = {"node_id": 99}
    mock_result = MagicMock()
    mock_result.points = [mock_point]
    mock_qdrant_inst.query_points.return_value = mock_result

    mock_kg = MagicMock()
    mock_kg.get_chunk_refs_by_node_ids.return_value = ["uuid-001", "uuid-002"]

    mock_extractor = MagicMock()
    mock_entity = MagicMock()
    mock_entity.text = "transformer"
    mock_extractor.extract_entities.return_value.entities = [mock_entity]

    mock_chunk_retriever = MagicMock(spec=ChunkRetriever)
    mock_chunk_retriever.fetch_by_ids.return_value = [
        RetrievedContext(
            text="Transformers use self-attention mechanisms extensively.",
            source="1706.03762",
            score=1.0,
            collection="arxiv_chunks",
        )
    ]

    retriever = GraphRetriever(
        knowledge_graph=mock_kg,
        entity_extractor=mock_extractor,
        chunk_retriever=mock_chunk_retriever,
    )
    results = retriever.retrieve("What is the transformer model?")

    assert len(results) == 1
    assert isinstance(results[0], RetrievedContext)
    assert results[0].collection == "graph"
    mock_chunk_retriever.fetch_by_ids.assert_called_once_with(["uuid-001", "uuid-002"])
    mock_kg.get_chunk_refs_for_entity.assert_not_called()
```

- [ ] **Step 3: Run tests to confirm both updated/new tests fail**

```
pytest tests/unit/test_retrieval.py -v
```

Expected: `test_graph_retriever_semantic_lookup` and `test_graph_retriever_uses_has_chunk_path` both `FAILED` (no `_qdrant` attribute yet, no `embed_texts_with_model` call)

- [ ] **Step 4: Update `src/retrieval/graph_retriever.py` — imports and constants**

Add at the top of the file, after the existing imports:

```python
from qdrant_client import QdrantClient
from src.utils import embed_texts_with_model
```

After `logger = get_logger(...)`, add:

```python
NODES_PER_ENTITY = 3
EMBEDDING_MODEL = "allenai/specter2_base"
```

- [ ] **Step 5: Update `GraphRetriever.__init__`**

Replace:

```python
    def __init__(
        self,
        knowledge_graph: Optional[CSKnowledgeGraph] = None,
        entity_extractor: Optional[CSEntityExtractor] = None,
        chunk_retriever=None,  # ChunkRetriever -- optional to avoid circular import
    ):
        self.kg = knowledge_graph or CSKnowledgeGraph()
        self.entity_extractor = entity_extractor or get_cs_entity_extractor()
        self.chunk_retriever = chunk_retriever
```

With:

```python
    def __init__(
        self,
        knowledge_graph: Optional[CSKnowledgeGraph] = None,
        entity_extractor: Optional[CSEntityExtractor] = None,
        chunk_retriever=None,  # ChunkRetriever -- optional to avoid circular import
        qdrant_url: str = "http://localhost:6333",
    ):
        self.kg = knowledge_graph or CSKnowledgeGraph()
        self.entity_extractor = entity_extractor or get_cs_entity_extractor()
        self.chunk_retriever = chunk_retriever
        self._qdrant = QdrantClient(url=qdrant_url, timeout=30)
```

- [ ] **Step 6: Replace the entity lookup loop in `GraphRetriever.retrieve()`**

Replace:

```python
        all_qdrant_ids: list[str] = []
        for name in entity_names:
            ids = self.kg.get_chunk_refs_for_entity(name, limit=top_k * 2)
            logger.debug("[GraphRetriever] entity=%r -> %d qdrant_ids: %s", name, len(ids), ids[:5])
            self._last_trace["qdrant_ids_per_entity"][name] = len(ids)
            all_qdrant_ids.extend(ids)
```

With:

```python
        all_qdrant_ids: list[str] = []
        for name in entity_names:
            name_emb = embed_texts_with_model([name], EMBEDDING_MODEL, batch_size=1)[0]
            results = self._qdrant.query_points(
                "arxiv_nodes", query=name_emb, with_payload=True, limit=NODES_PER_ENTITY
            )
            node_ids = [
                r.payload["node_id"]
                for r in results.points
                if r.payload and r.payload.get("node_id") is not None
            ]
            logger.debug("[GraphRetriever] entity=%r -> %d nodes", name, len(node_ids))
            self._last_trace["qdrant_ids_per_entity"][name] = len(node_ids)
            if node_ids:
                ids = self.kg.get_chunk_refs_by_node_ids(node_ids, limit=top_k * 2)
                all_qdrant_ids.extend(ids)
```

- [ ] **Step 7: Run tests to confirm they pass**

```
pytest tests/unit/test_retrieval.py -v
```

Expected: All tests `PASSED`

- [ ] **Step 8: Commit**

```bash
git add src/retrieval/graph_retriever.py tests/unit/test_retrieval.py
git commit -m "feat: replace exact-name graph lookup with SPECTER2 semantic entity search"
```

---

## Task 3: Update `_ingest_nodes` to embed with SPECTER2

**Files:**
- Create: `tests/unit/test_local_ingestion_pipeline.py`
- Modify: `src/pipelines/local_ingestion_pipeline.py:275-324`

**Background:**
- `local_ingestion_pipeline.py` is not in the conftest stub list — module-level imports work directly.
- `NodeRecord.qdrant_id` is `field(init=False)` — do NOT pass it to the constructor. It is computed in `__post_init__` as `uuid.uuid5(HYBRIDRAG_NS, f"{domain}:{node_id}")`.
- `_neo4j()` and `_q()` are class methods that return `self._driver` and `self._qdrant` respectively. Override them as instance-level mocks in `_make_pipeline` for full control.

- [ ] **Step 1: Create `tests/unit/test_local_ingestion_pipeline.py` with failing test**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/unit/test_local_ingestion_pipeline.py -v
```

Expected: Both tests `FAILED` — `embed_texts_with_model` not called (pipeline still uses `rec.embedding`)

- [ ] **Step 3: Replace `_ingest_nodes` in `src/pipelines/local_ingestion_pipeline.py`**

Replace the entire `_ingest_nodes` method (lines 275–324) with:

```python
    def _ingest_nodes(self, domain: str) -> int:
        from collections import defaultdict
        total = 0
        batch: list[NodeRecord] = []

        def _flush():
            nonlocal total
            by_type: dict[str, list[dict]] = defaultdict(list)
            for r in batch:
                by_type[r.entity_type].append({
                    "node_id": r.node_id,
                    "domain": r.domain,
                    "display_name": r.display_name,
                    "entity_type": r.entity_type,
                })
            with self._neo4j().session() as s:
                for entity_type, records in by_type.items():
                    s.run(
                        f"UNWIND $batch AS n "
                        f"CREATE (:{entity_type}:_Embeddable "
                        f"{{node_id: n.node_id, domain: n.domain, "
                        f"display_name: n.display_name, entity_type: n.entity_type}})",
                        {"batch": records},
                    )
            embeddable_recs = [r for r in batch if r.display_name]
            display_names = [r.display_name for r in embeddable_recs]
            if display_names:
                embeddings = embed_texts_with_model(display_names, EMBEDDING_MODEL, batch_size=64)
                qdrant_points = [
                    PointStruct(
                        id=str(r.qdrant_id),
                        vector=emb,
                        payload={
                            "node_id": r.node_id,
                            "domain": r.domain,
                            "display_name": r.display_name,
                            "entity_type": r.entity_type,
                        },
                    )
                    for r, emb in zip(embeddable_recs, embeddings)
                ]
                self._q().upsert(collection_name="arxiv_nodes", points=qdrant_points)
            total += len(batch)
            batch.clear()

        for rec in self.loader.iter_nodes(domain):
            batch.append(rec)
            if len(batch) >= self.neo4j_batch_size:
                _flush()

        if batch:
            _flush()

        return total
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/unit/test_local_ingestion_pipeline.py -v
```

Expected: Both tests `PASSED`

- [ ] **Step 5: Run the full unit test suite to confirm no regressions**

```
pytest tests/unit/ -v
```

Expected: All tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/pipelines/local_ingestion_pipeline.py tests/unit/test_local_ingestion_pipeline.py
git commit -m "feat: ingest_nodes embeds display_names with SPECTER2 instead of pre-computed vectors"
```

---

## Task 4: Update `_ingest_chunks` to embed with SPECTER2

**Files:**
- Modify: `tests/unit/test_local_ingestion_pipeline.py`
- Modify: `src/pipelines/local_ingestion_pipeline.py:395-473`

**Background:**
- `ChunkRecord.qdrant_id` is `field(init=False)` — do NOT pass it to the constructor. It is computed as `uuid.uuid5(HYBRIDRAG_NS, f"{domain}:{edge_id}")`.
- The accumulator renames from `points: list[PointStruct]` to `recs: list[ChunkRecord]`. Three lines inside `_flush()` also reference `points` by name and must be updated: `snapshot_ids`, `total += len(...)`, and `points.clear()`.
- The `neo4j_batch` is now built from `ChunkRecord` attributes (`rec.src_id`, `rec.dst_id`, `rec.edge_id`, `rec.domain`, `str(rec.qdrant_id)`) instead of `p.payload` dict lookups.

- [ ] **Step 1: Add failing tests to `tests/unit/test_local_ingestion_pipeline.py`**

Append:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/unit/test_local_ingestion_pipeline.py::test_ingest_chunks_embeds_paragraphs_not_precomputed tests/unit/test_local_ingestion_pipeline.py::test_ingest_chunks_neo4j_batch_uses_record_attributes -v
```

Expected: Both `FAILED`

- [ ] **Step 3: Replace `_ingest_chunks` in `src/pipelines/local_ingestion_pipeline.py`**

Replace the entire `_ingest_chunks` method (lines 395–473) with:

```python
    def _ingest_chunks(self, domain: str, include_src_names: bool, progress: _Progress) -> int:
        total = 0
        batch_index = 0
        recs: list[ChunkRecord] = []

        def _flush():
            nonlocal total, batch_index
            snapshot_ids = [str(rec.qdrant_id) for rec in recs]
            neo4j_batch = [
                {
                    "src_id": int(rec.src_id),
                    "dst_id": int(rec.dst_id),
                    "edge_id": int(rec.edge_id),
                    "domain": rec.domain,
                    "qdrant_id": str(rec.qdrant_id),
                }
                for rec in recs
                if rec.src_id is not None and rec.dst_id is not None
            ]

            qdrant_ok = False
            try:
                embeddings = embed_texts_with_model(
                    [r.paragraph for r in recs], EMBEDDING_MODEL, batch_size=64
                )
                points: list[PointStruct] = []
                for r, emb in zip(recs, embeddings):
                    payload = {
                        "edge_id": r.edge_id,
                        "src_id": r.src_id,
                        "dst_id": r.dst_id,
                        "rel_type": r.rel_type,
                        "paragraph": r.paragraph,
                        "domain": r.domain,
                        "paper_id": r.paper_id,
                    }
                    if r.src_name is not None:
                        payload["src_name"] = r.src_name
                    if r.dst_name is not None:
                        payload["dst_name"] = r.dst_name
                    points.append(PointStruct(id=str(r.qdrant_id), vector=emb, payload=payload))
                self._q().upsert(collection_name="arxiv_chunks", points=points)
                total += len(recs)
                qdrant_ok = True
            except Exception as e:
                progress.log_batch_failure(domain, "chunks", batch_index, str(e), snapshot_ids)
                logger.error(f"[{domain}] chunks batch {batch_index} Qdrant upsert failed: {e}")

            if qdrant_ok:
                try:
                    with self._neo4j().session() as s:
                        s.run(
                            """
                            UNWIND $batch AS c
                            MATCH (src:_Embeddable {node_id: c.src_id, domain: c.domain})
                            MATCH (dst:_Embeddable {node_id: c.dst_id, domain: c.domain})
                            MERGE (src)-[r:HAS_CHUNK {edge_id: c.edge_id, domain: c.domain}]->(dst)
                            SET r.qdrant_id = c.qdrant_id
                            """,
                            {"batch": neo4j_batch},
                        )
                except Exception as e:
                    progress.log_batch_failure(
                        domain, "chunks_has_chunk", batch_index, str(e), snapshot_ids
                    )
                    logger.error(
                        f"[{domain}] chunks batch {batch_index} HAS_CHUNK write failed: {e}"
                    )

            batch_index += 1
            recs.clear()

        for rec in self.loader.iter_chunks(domain, include_src_names=include_src_names):
            recs.append(rec)
            if len(recs) >= QDRANT_BATCH:
                _flush()

        if recs:
            _flush()

        return total
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/unit/test_local_ingestion_pipeline.py -v
```

Expected: All 4 tests `PASSED`

- [ ] **Step 5: Run the full unit suite to confirm no regressions**

```
pytest tests/unit/ -v
```

Expected: All tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/pipelines/local_ingestion_pipeline.py tests/unit/test_local_ingestion_pipeline.py
git commit -m "feat: ingest_chunks embeds paragraphs with SPECTER2 instead of pre-computed vectors"
```

---

## Task 5: Re-run ingestion with `--reset`

**Files:** None (operational step — no code changes)

**Background:** All three Qdrant collections and all Neo4j data are wiped by `--reset`. `arxiv_papers` is already correctly embedded with SPECTER2 but must be re-ingested because Neo4j is wiped. Ingestion embeds ~34,379 chunks and ~36,844 nodes with SPECTER2 — this takes significant time (~20–40 min on CPU).

- [ ] **Step 1: Verify both services are reachable**

```bash
python -c "from qdrant_client import QdrantClient; c = QdrantClient('http://localhost:6333'); print(c.get_collections())"
python -c "from neo4j import GraphDatabase; d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'Pokemon0424\$')); d.verify_connectivity(); print('neo4j ok'); d.close()"
```

Expected: No errors

- [ ] **Step 2: Run ingestion with `--reset`**

```bash
python ingest_local.py --reset
```

Expected: Progress logs per domain, ending with `Ingestion complete: {...}`. If it crashes mid-way, re-run WITHOUT `--reset` — the progress file will resume from the last completed phase within each domain.

- [ ] **Step 3: Verify collection sizes**

```bash
python -c "
from qdrant_client import QdrantClient
q = QdrantClient('http://localhost:6333')
for name in ['arxiv_chunks', 'arxiv_nodes', 'arxiv_papers']:
    info = q.get_collection(name)
    print(f'{name}: {info.points_count} points')
"
```

Expected (approximate):
```
arxiv_chunks: ~34379 points
arxiv_nodes: ~36844 points
arxiv_papers: ~6701 points
```

---

## Task 6: Run verification diagnostics

**Files:** None (diagnostic run — no code changes)

These 4 checks confirm the embedding alignment is fixed end-to-end. Run against the live services after Task 5 completes. Save this as a temporary script or run in a Python REPL.

- [ ] **Step 1: Vector search check — `arxiv_chunks`**

```python
from src.utils import embed_texts_with_model
from qdrant_client import QdrantClient

q = QdrantClient("http://localhost:6333")
query = "What concept was developed by McCaskill and is the basis for the software Contrafold?"
vec = embed_texts_with_model([query], "allenai/specter2_base", batch_size=1)[0]
results = q.query_points("arxiv_chunks", query=vec, with_payload=True, limit=10)
for r in results.points:
    print(r.score, r.payload.get("paper_id"), r.payload.get("paragraph", "")[:80])
```

**Pass condition:** Paper `1912.13190v3` appears in top-5 results, score > 0.5

- [ ] **Step 2: Entity semantic lookup check — `arxiv_nodes`**

```python
from src.utils import embed_texts_with_model
from qdrant_client import QdrantClient

q = QdrantClient("http://localhost:6333")
vec = embed_texts_with_model(["PARTITION FUNCTION"], "allenai/specter2_base", batch_size=1)[0]
results = q.query_points("arxiv_nodes", query=vec, with_payload=True, limit=5)
for r in results.points:
    print(r.score, r.payload.get("entity_type"), r.payload.get("display_name"))
```

**Pass condition:** At least one result with `entity_type=Concept` and a relevant `display_name`

- [ ] **Step 3: Graph retrieval end-to-end check**

```python
from src.graph.cs_knowledge_graph import CSKnowledgeGraph
from src.graph.cs_entity_extractor import get_cs_entity_extractor
from src.retrieval.chunk_retriever import ChunkRetriever
from src.retrieval.graph_retriever import GraphRetriever

chunk_retriever = ChunkRetriever()
graph = GraphRetriever(
    knowledge_graph=CSKnowledgeGraph(),
    entity_extractor=get_cs_entity_extractor(),
    chunk_retriever=chunk_retriever,
)
query = "What concept was developed by McCaskill and is the basis for the software Contrafold?"
results = graph.retrieve(query, top_k=5)
print("trace:", graph._last_trace)
print("fetched:", len(results))
for r in results:
    print(r.score, r.metadata.get("paper_id", ""), r.text[:80])
```

**Pass condition:** `fetched_count > 0` in trace, at least one result from paper `1912.13190v3`

- [ ] **Step 4: Full pipeline spot check (3 previously-failing questions)**

```python
from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline

pipeline = HybridRAGBenchPipeline()
questions = [
    "What concept was developed by McCaskill and is the basis for the software Contrafold?",
    "What are the main methodological differences between the approach proposed by Friston et al. and standard machine learning?",
    "What is the proposed solution for the Genesis environment in the paper?",
]
for q in questions:
    result = pipeline.query(q)
    print(f"Q: {q[:70]}")
    print(f"  answer_type={result.get('answer_type')} | reranker_top={result.get('reranker_scores', [None])[0]}")
    print()
```

**Pass condition:** All 3 questions return `answer_type = "answer"` (not `"refusal"`), reranker top score > 0

---

## Troubleshooting

**Ingestion OOM / CUDA error**: SPECTER2 runs on CPU by default. If GPU OOM occurs: `CUDA_VISIBLE_DEVICES="" python ingest_local.py --reset`

**Verification Step 1 fails (arxiv_chunks scores still low)**: Check that the `_ingest_chunks` change took effect by re-running the unit tests. Confirm the ingestion logs showed embedding calls (not just fast passthrough).

**Verification Step 3 fails (fetched_count = 0)**: Inspect `graph._last_trace["qdrant_ids_per_entity"]` — if node counts are all 0, `arxiv_nodes` may not have been re-ingested. Verify point count in Step 3 of Task 5.

**Verification Step 4 still shows refusals**: Check `pre_rerank_count` in the trace. If 0, vector search is still broken. If > 0, the issue is in the LLM prompt — a separate concern outside this plan's scope.

**Test `test_ingest_chunks_neo4j_batch_uses_record_attributes` fails**: If `has_chunk_calls` is empty, the session mock chain is wrong. Print `mock_session.run.call_args_list` to inspect all calls and check which session object the pipeline is actually using. Ensure `pipeline._neo4j` is mocked as an instance attribute (not just `pipeline._driver`).
