# HybridRAG Retrieval Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the HybridRAG query pipeline so it retrieves from the correct Qdrant collections with correct field mappings, connects graph traversal to text via `HAS_CHUNK` edges, and removes all medical-domain assumptions.

**Architecture:** Introduce a `RetrievedContext` dataclass as the single output contract for all retrievers. Two new per-collection retrievers (`ChunkRetriever`, `PaperRetriever`) replace the single `QdrantStorage` call. `GraphRetriever` injects `ChunkRetriever` to resolve `HAS_CHUNK` references. `HybridRetriever` fuses all three sources with per-source weighted RRF.

**Tech Stack:** Python 3.11+, Qdrant (`qdrant-client`), Neo4j (`neo4j` driver), pytest + `unittest.mock`, PyYAML

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/types.py` | Modify | Add `RetrievedContext` dataclass |
| `src/retrieval/chunk_retriever.py` | Create | Vector search on `arxiv_chunks`; `fetch_by_ids` for graph resolution |
| `src/retrieval/paper_retriever.py` | Create | Vector search on `arxiv_papers` |
| `src/graph/cs_knowledge_graph.py` | Modify | Add `get_chunk_refs_for_entity` for HAS_CHUNK traversal |
| `src/retrieval/graph_retriever.py` | Modify | Fix fallback, inject ChunkRetriever, return `list[RetrievedContext]` |
| `src/retrieval/hybrid_retriever.py` | Modify | 3-way RRF fusion, return `list[RetrievedContext]` |
| `src/pipelines/hybridrag_pipeline.py` | Modify | Wire new stack, deprecate `ingest()`, update `query()` |
| `src/pipelines/local_ingestion_pipeline.py` | Modify | Write HAS_CHUNK edges in `_ingest_chunks` |
| `scripts/migrate_has_chunk_edges.py` | Create | One-shot migration for already-ingested data |
| `config/defaults.yaml` | Modify | Replace 2-way with 3-way retrieval weights, remove stale keys |
| `tests/evaluation/metrics.py` | Modify | Strip medical-domain prompts |
| `tests/unit/test_retrieval.py` | Modify | Update tests to match new interfaces |
| `tests/unit/test_chunk_retriever.py` | Create | Unit tests for ChunkRetriever |
| `tests/unit/test_paper_retriever.py` | Create | Unit tests for PaperRetriever |

---

## Task 1: Domain Cleanup — `graph_retriever.py` and `metrics.py`

**Files:**
- Modify: `src/retrieval/graph_retriever.py`
- Modify: `tests/evaluation/metrics.py`

- [ ] **Step 1: Grep all medical references**

```bash
grep -rn "medical\|clinical\|patient\|Medicine\|physician" src/ tests/evaluation/ --include="*.py"
```

Note every file and line number.

- [ ] **Step 2: Fix comments and the inner `:Chunk` fallback in `graph_retriever.py`**

**Do NOT rewrite `_keyword_fallback` in this task** — its full return-type change to `list[RetrievedContext]` happens atomically in Task 6 when the whole file is rewritten. Doing it here would create an intermediate state where `retrieve()` returns a tuple from a `list[RetrievedContext]`-typed method.

Instead, only make these two surgical changes in Task 1:

1. In `_keyword_fallback`, replace the f-string Cypher body with a parameterised version that still returns `tuple[list, list]` (preserving the current interface):

```python
def _keyword_fallback(self, query: str, top_k: int) -> tuple[list, list]:
    """
    Fallback: search _Embeddable nodes by display_name keyword matching
    when no CS entities are detected in the query.
    Uses parameterised Cypher — never f-string interpolation.
    """
    import re
    stop_words = {
        "what", "were", "the", "results", "why", "how", "did", "does",
        "was", "are", "for", "and", "that", "this", "it", "at",
        "by", "on", "as", "from", "with", "not", "but", "its",
        "very", "just", "than", "then"
    }
    words = re.findall(r'\b[a-zA-Z]{2,}\b', query)
    keywords = [w.lower() for w in words if w.lower() not in stop_words]

    if not keywords:
        return [], []

    try:
        rows = self.kg.client.execute_read(
            """
            MATCH (n:_Embeddable)
            WHERE any(kw IN $keywords WHERE toLower(n.display_name) CONTAINS kw)
            RETURN n.display_name AS name, n.entity_type AS type
            LIMIT $lim
            """,
            {"keywords": keywords[:5], "lim": top_k},
        )
        names = [r["name"] for r in rows if r.get("name")]
        sources = ["neo4j/keyword"] * len(names)
        return names, sources
    except Exception:
        return [], []
```

2. Remove the inner `:Chunk` f-string fallback block inside `retrieve()` (lines ~109–120):
```python
if not chunks and not direct_contexts:
    try:
        text_rows = self.kg.client.execute_read(
            "MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS toLower($name) ...",
```
Delete this entire block. The `_keyword_fallback` on `_Embeddable` nodes is now the sole fallback path.

- [ ] **Step 3: Remove the inner `:Chunk` fallback inside `retrieve()`**

In `retrieve()`, find and delete the block (lines ~109–120) that reads:
```python
if not chunks and not direct_contexts:
    try:
        text_rows = self.kg.client.execute_read(
            "MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS toLower($name) ...",
```

Remove this entire `if not chunks and not direct_contexts:` block. The `_keyword_fallback` on `_Embeddable` nodes is now the sole fallback path.

- [ ] **Step 4: Update the comment on line ~69**

Change:
```python
# No medical entities found — fall back to keyword-based chunk search
```
To:
```python
# No CS entities found — fall back to keyword search on _Embeddable display names
```

- [ ] **Step 5: Fix medical prompts in `tests/evaluation/metrics.py`**

In `calculate_relevance` (lines ~59–73), replace:
```python
        Task: Rate the relevance of the answer to the medical query.

        Query: {query}
        Answer: {answer}

        A relevant answer directly addresses the query using professional medical knowledge.
```
With:
```python
        Task: Rate the relevance of the answer to the research query.

        Query: {query}
        Answer: {answer}

        A relevant answer directly addresses the query using scientific and technical knowledge.
```

- [ ] **Step 6: Fix any remaining medical references found in Step 1**

Update comments, docstrings, or variable names as noted.

- [ ] **Step 7: Run existing tests to confirm no regressions**

```bash
cd C:\Users\brend\Desktop\RAG && python -m pytest tests/unit/test_retrieval.py -v
```

Expected: tests pass or fail only for pre-existing reasons (the `GraphRetriever` test uses medical entity "Aspirin" — that's fine for now, it will be replaced in Task 6).

- [ ] **Step 8: Commit**

```bash
git add src/retrieval/graph_retriever.py tests/evaluation/metrics.py
git commit -m "fix: remove medical-domain logic from GraphRetriever and metrics"
```

---

## Task 2: Add `RetrievedContext` to `src/types.py`

**Files:**
- Modify: `src/types.py`

- [ ] **Step 1: Add `RetrievedContext` dataclass**

At the top of `src/types.py`, add the import and the dataclass alongside the existing Pydantic models:

```python
from dataclasses import dataclass, field as dc_field
```

Then add after the existing imports block:

```python
@dataclass
class RetrievedContext:
    """Single retrieved context item from any retrieval source."""
    text: str
    source: str        # arxiv_id, file path, or "graph"
    score: float
    collection: str    # "arxiv_chunks" | "arxiv_papers" | "graph"
    metadata: dict = dc_field(default_factory=dict)
    # metadata keys by collection:
    #   arxiv_chunks: edge_id, src_id, dst_id, rel_type, domain, paper_id
    #   arxiv_papers: arxiv_id, chunk_index, title, domain
    #   graph:        entities_found
```

- [ ] **Step 2: Verify import works**

```bash
cd C:\Users\brend\Desktop\RAG && python -c "from src.types import RetrievedContext; print(RetrievedContext(text='t', source='s', score=0.5, collection='arxiv_chunks'))"
```

Expected: prints the dataclass repr with no errors.

- [ ] **Step 3: Commit**

```bash
git add src/types.py
git commit -m "feat: add RetrievedContext dataclass to src/types"
```

---

## Task 3: Create `ChunkRetriever`

**Files:**
- Create: `src/retrieval/chunk_retriever.py`
- Create: `tests/unit/test_chunk_retriever.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_chunk_retriever.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.chunk_retriever import ChunkRetriever
from src.types import RetrievedContext


@patch("src.retrieval.chunk_retriever.QdrantClient")
def test_search_maps_paragraph_field(mock_client_cls):
    client = mock_client_cls.return_value
    client.collection_exists.return_value = True

    point = MagicMock()
    point.score = 0.85
    point.payload = {
        "paragraph": "Attention is all you need.",
        "source": "1706.03762",
        "edge_id": 42,
        "src_id": 1,
        "dst_id": 2,
        "rel_type": "CITES",
        "domain": "arxiv_ai",
        "paper_id": "1706.03762",
    }
    client.query_points.return_value.points = [point]

    retriever = ChunkRetriever()
    results = retriever.search(query_vector=[0.1] * 768, top_k=1)

    assert len(results) == 1
    assert isinstance(results[0], RetrievedContext)
    assert results[0].text == "Attention is all you need."
    assert results[0].collection == "arxiv_chunks"
    assert results[0].score == 0.85
    assert results[0].metadata["edge_id"] == 42


@patch("src.retrieval.chunk_retriever.QdrantClient")
def test_search_skips_short_text(mock_client_cls):
    client = mock_client_cls.return_value
    client.collection_exists.return_value = True

    point = MagicMock()
    point.score = 0.9
    point.payload = {"paragraph": "Too short.", "source": "x"}
    client.query_points.return_value.points = [point]

    retriever = ChunkRetriever()
    results = retriever.search(query_vector=[0.1] * 768, top_k=1)

    assert results == []


@patch("src.retrieval.chunk_retriever.QdrantClient")
def test_fetch_by_ids_returns_contexts(mock_client_cls):
    client = mock_client_cls.return_value
    client.collection_exists.return_value = True

    point = MagicMock()
    point.id = "abc-123"
    point.payload = {
        "paragraph": "A sufficiently long retrieved paragraph for the test.",
        "source": "paper1",
        "edge_id": 7,
        "src_id": 3,
        "dst_id": 4,
        "rel_type": "RELATED_TO",
        "domain": "arxiv_ai",
        "paper_id": "paper1",
    }
    client.retrieve.return_value = [point]

    retriever = ChunkRetriever()
    results = retriever.fetch_by_ids(["abc-123"])

    assert len(results) == 1
    assert results[0].text == "A sufficiently long retrieved paragraph for the test."
    assert results[0].collection == "arxiv_chunks"
    assert results[0].score == 1.0  # graph-resolved contexts score 1.0
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/unit/test_chunk_retriever.py -v
```

Expected: `ModuleNotFoundError` — `chunk_retriever` does not exist yet.

- [ ] **Step 3: Implement `ChunkRetriever`**

Create `src/retrieval/chunk_retriever.py`:

```python
"""
Retriever for the arxiv_chunks Qdrant collection.

Stores KG-edge paragraphs. Payload field: "paragraph".
"""
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from src.types import RetrievedContext

COLLECTION = "arxiv_chunks"
DIM = 768
MIN_TEXT_LEN = 50


class ChunkRetriever:
    """Vector search over arxiv_chunks; also resolves qdrant_id references for graph traversal."""

    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url, timeout=30)
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_score: float = 0.2,
    ) -> list[RetrievedContext]:
        """Search arxiv_chunks by vector similarity."""
        results = self.client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )
        out: list[RetrievedContext] = []
        for r in results.points:
            if r.score < min_score:
                continue
            payload = r.payload or {}
            text = payload.get("paragraph", "")
            if len(text) < MIN_TEXT_LEN:
                continue
            out.append(RetrievedContext(
                text=text,
                source=payload.get("source", payload.get("paper_id", "")),
                score=r.score,
                collection=COLLECTION,
                metadata={
                    "edge_id": payload.get("edge_id"),
                    "src_id": payload.get("src_id"),
                    "dst_id": payload.get("dst_id"),
                    "rel_type": payload.get("rel_type", ""),
                    "domain": payload.get("domain", ""),
                    "paper_id": payload.get("paper_id", ""),
                },
            ))
        return out

    def fetch_by_ids(self, qdrant_ids: list[str]) -> list[RetrievedContext]:
        """Fetch specific points by their UUID string IDs (used by GraphRetriever)."""
        if not qdrant_ids:
            return []
        points = self.client.retrieve(
            collection_name=COLLECTION,
            ids=qdrant_ids,
            with_payload=True,
        )
        out: list[RetrievedContext] = []
        for p in points:
            payload = p.payload or {}
            text = payload.get("paragraph", "")
            if len(text) < MIN_TEXT_LEN:
                continue
            out.append(RetrievedContext(
                text=text,
                source=payload.get("source", payload.get("paper_id", "")),
                score=1.0,  # graph-resolved: score reflects traversal relevance, not cosine
                collection=COLLECTION,
                metadata={
                    "edge_id": payload.get("edge_id"),
                    "src_id": payload.get("src_id"),
                    "dst_id": payload.get("dst_id"),
                    "rel_type": payload.get("rel_type", ""),
                    "domain": payload.get("domain", ""),
                    "paper_id": payload.get("paper_id", ""),
                },
            ))
        return out
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/unit/test_chunk_retriever.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/chunk_retriever.py tests/unit/test_chunk_retriever.py
git commit -m "feat: add ChunkRetriever for arxiv_chunks collection"
```

---

## Task 4: Create `PaperRetriever`

**Files:**
- Create: `src/retrieval/paper_retriever.py`
- Create: `tests/unit/test_paper_retriever.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_paper_retriever.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.paper_retriever import PaperRetriever
from src.types import RetrievedContext


@patch("src.retrieval.paper_retriever.QdrantClient")
def test_search_maps_chunk_text_field(mock_client_cls):
    client = mock_client_cls.return_value
    client.collection_exists.return_value = True

    point = MagicMock()
    point.score = 0.78
    point.payload = {
        "chunk_text": "Transformers use self-attention to process sequences in parallel.",
        "arxiv_id": "1706.03762",
        "chunk_index": 3,
        "title": "Attention Is All You Need",
        "domain": "arxiv_ai",
    }
    client.query_points.return_value.points = [point]

    retriever = PaperRetriever()
    results = retriever.search(query_vector=[0.1] * 768, top_k=1)

    assert len(results) == 1
    assert isinstance(results[0], RetrievedContext)
    assert results[0].text == "Transformers use self-attention to process sequences in parallel."
    assert results[0].collection == "arxiv_papers"
    assert results[0].score == 0.78
    assert results[0].metadata["arxiv_id"] == "1706.03762"
    assert results[0].metadata["chunk_index"] == 3


@patch("src.retrieval.paper_retriever.QdrantClient")
def test_search_filters_below_min_score(mock_client_cls):
    client = mock_client_cls.return_value
    client.collection_exists.return_value = True

    point = MagicMock()
    point.score = 0.05
    point.payload = {"chunk_text": "A sufficiently long text that passes length check but fails score.", "arxiv_id": "x"}
    client.query_points.return_value.points = [point]

    retriever = PaperRetriever()
    results = retriever.search(query_vector=[0.1] * 768, top_k=1, min_score=0.2)

    assert results == []
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/unit/test_paper_retriever.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `PaperRetriever`**

Create `src/retrieval/paper_retriever.py`:

```python
"""
Retriever for the arxiv_papers Qdrant collection.

Stores full paper text chunks. Payload field: "chunk_text".
"""
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from src.types import RetrievedContext

COLLECTION = "arxiv_papers"
DIM = 768
MIN_TEXT_LEN = 50


class PaperRetriever:
    """Vector search over arxiv_papers (full paper text chunks)."""

    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url, timeout=30)
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_score: float = 0.2,
    ) -> list[RetrievedContext]:
        """Search arxiv_papers by vector similarity."""
        results = self.client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )
        out: list[RetrievedContext] = []
        for r in results.points:
            if r.score < min_score:
                continue
            payload = r.payload or {}
            text = payload.get("chunk_text", "")
            if len(text) < MIN_TEXT_LEN:
                continue
            out.append(RetrievedContext(
                text=text,
                source=payload.get("arxiv_id", ""),
                score=r.score,
                collection=COLLECTION,
                metadata={
                    "arxiv_id": payload.get("arxiv_id", ""),
                    "chunk_index": payload.get("chunk_index"),
                    "title": payload.get("title", ""),
                    "domain": payload.get("domain", ""),
                },
            ))
        return out
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/unit/test_paper_retriever.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/paper_retriever.py tests/unit/test_paper_retriever.py
git commit -m "feat: add PaperRetriever for arxiv_papers collection"
```

---

## Task 5: Add `get_chunk_refs_for_entity` to `CSKnowledgeGraph`

**Files:**
- Modify: `src/graph/cs_knowledge_graph.py`

- [ ] **Step 1: Write a failing test for `get_chunk_refs_for_entity`**

Add to `tests/unit/test_graph.py` (or create it if absent):

```python
from unittest.mock import MagicMock, patch
from src.graph.cs_knowledge_graph import CSKnowledgeGraph

@patch("src.graph.cs_knowledge_graph.Neo4jClient")
def test_get_chunk_refs_for_entity_returns_qdrant_ids(mock_neo4j_cls):
    client = mock_neo4j_cls.return_value
    client.connect.return_value = None
    client.execute_read.return_value = [
        {"qdrant_id": "uuid-aaa"},
        {"qdrant_id": "uuid-bbb"},
        {"qdrant_id": None},  # should be filtered out
    ]

    kg = CSKnowledgeGraph(neo4j_client=client)
    result = kg.get_chunk_refs_for_entity("transformer")

    assert result == ["uuid-aaa", "uuid-bbb"]
    call_args = client.execute_read.call_args
    assert "$name" in call_args[0][0]
    assert call_args[0][1]["name"] == "transformer"
```

Run to confirm it fails: `python -m pytest tests/unit/test_graph.py::test_get_chunk_refs_for_entity_returns_qdrant_ids -v`

- [ ] **Step 2: Add the method**

After the existing `get_entity_context` method, add:

```python
def get_chunk_refs_for_entity(self, display_name: str, limit: int = 20) -> list[str]:
    """
    Return qdrant_id strings from HAS_CHUNK edges connected to _Embeddable
    nodes whose display_name contains the given name (case-insensitive).

    HAS_CHUNK edges connect _Embeddable src to _Embeddable dst and carry
    a qdrant_id property (UUID string) pointing to the arxiv_chunks collection.
    """
    query = """
    MATCH (src:_Embeddable)-[r:HAS_CHUNK]->(dst:_Embeddable)
    WHERE toLower(src.display_name) CONTAINS toLower($name)
       OR toLower(dst.display_name) CONTAINS toLower($name)
    RETURN r.qdrant_id AS qdrant_id
    LIMIT $limit
    """
    rows = self.client.execute_read(query, {"name": display_name, "limit": limit})
    return [r["qdrant_id"] for r in rows if r.get("qdrant_id")]
```

- [ ] **Step 3: Run test to confirm it passes**

```bash
python -m pytest tests/unit/test_graph.py::test_get_chunk_refs_for_entity_returns_qdrant_ids -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/graph/cs_knowledge_graph.py tests/unit/test_graph.py
git commit -m "feat: add get_chunk_refs_for_entity to CSKnowledgeGraph for HAS_CHUNK traversal"
```

---

## Task 6: Update `GraphRetriever`

**Files:**
- Modify: `src/retrieval/graph_retriever.py`
- Modify: `tests/unit/test_retrieval.py`

- [ ] **Step 1: Write failing tests for the new `GraphRetriever` interface**

Replace the `test_graph_retrieval` test in `tests/unit/test_retrieval.py` with:

```python
from src.retrieval.chunk_retriever import ChunkRetriever
from src.types import RetrievedContext

def test_graph_retriever_uses_has_chunk_path():
    """GraphRetriever finds entities, gets qdrant_ids via HAS_CHUNK, resolves via ChunkRetriever."""
    mock_kg = MagicMock()
    mock_kg.get_chunk_refs_for_entity.return_value = ["uuid-001", "uuid-002"]
    mock_kg.get_related_entities.return_value = []

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


def test_graph_retriever_keyword_fallback_no_entities():
    """When no entities are found, _keyword_fallback uses parameterised Cypher on _Embeddable nodes."""
    mock_kg = MagicMock()
    mock_kg.client.execute_read.return_value = [
        {"name": "attention mechanism", "type": "ALGORITHM"}
    ]

    mock_extractor = MagicMock()
    mock_extractor.extract_entities.return_value.entities = []

    mock_chunk_retriever = MagicMock(spec=ChunkRetriever)
    mock_chunk_retriever.fetch_by_ids.return_value = []

    retriever = GraphRetriever(
        knowledge_graph=mock_kg,
        entity_extractor=mock_extractor,
        chunk_retriever=mock_chunk_retriever,
    )
    results = retriever.retrieve("explain attention in neural networks", top_k=3)

    # Verify parameterised Cypher — must NOT use f-string injection
    call_args = mock_kg.client.execute_read.call_args
    cypher_str = call_args[0][0]
    params_dict = call_args[0][1]
    assert "$keywords" in cypher_str, "Cypher must use $keywords param, not f-string injection"
    assert isinstance(params_dict.get("keywords"), list), "keywords must be a list param"
    assert isinstance(results, list)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/unit/test_retrieval.py::test_graph_retriever_uses_has_chunk_path tests/unit/test_retrieval.py::test_graph_retriever_keyword_fallback_no_entities -v
```

Expected: FAIL — old `GraphRetriever` interface.

- [ ] **Step 3: Rewrite `GraphRetriever`**

Replace the full content of `src/retrieval/graph_retriever.py` with:

```python
"""
Graph-based retrieval using Neo4j knowledge graph.

Traverses HAS_CHUNK edges from _Embeddable entity nodes to resolve
qdrant_id references, then fetches text from arxiv_chunks via ChunkRetriever.
"""
import re
from typing import Optional
from dataclasses import dataclass

from src.graph.cs_knowledge_graph import CSKnowledgeGraph
from src.graph.cs_entity_extractor import CSEntityExtractor, get_cs_entity_extractor
from src.types import RetrievedContext


# Kept for backward-compat imports elsewhere; will be cleaned up separately
@dataclass
class GraphRetrievalResult:
    contexts: list[str]
    sources: list[str]
    entities_found: list[str]
    graph_paths: list[dict]


class GraphRetriever:
    """
    Retriever that uses the CS/arXiv knowledge graph for entity-based search.

    Flow:
        1. Extract CS entities from query via CSEntityExtractor
        2. For each entity, traverse HAS_CHUNK edges in Neo4j to get qdrant_ids
        3. Resolve qdrant_ids to text via ChunkRetriever.fetch_by_ids()
        4. Return list[RetrievedContext] with collection="graph"

    Fallback (no entities found):
        Keyword-search _Embeddable.display_name; returns entity name strings
        as minimal context (no Qdrant lookup needed for the fallback path).
    """

    _STOP_WORDS = {
        "what", "were", "the", "results", "why", "how", "did", "does",
        "was", "are", "for", "and", "that", "this", "it", "at",
        "by", "on", "as", "from", "with", "not", "but", "its",
        "very", "just", "than", "then",
    }

    def __init__(
        self,
        knowledge_graph: Optional[CSKnowledgeGraph] = None,
        entity_extractor: Optional[CSEntityExtractor] = None,
        chunk_retriever=None,  # ChunkRetriever — optional to avoid circular import
    ):
        self.kg = knowledge_graph or CSKnowledgeGraph()
        self.entity_extractor = entity_extractor or get_cs_entity_extractor()
        self.chunk_retriever = chunk_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        entity_types: list[str] | None = None,  # kept as no-op for backward compat
    ) -> list[RetrievedContext]:
        """
        Retrieve relevant contexts using graph-based search.

        Returns list[RetrievedContext] with collection="graph".
        """
        extraction = self.entity_extractor.extract_entities(query)
        entity_names = list({e.text for e in extraction.entities})

        if not entity_names:
            # No CS entities found — fall back to keyword search on _Embeddable display names
            return self._keyword_fallback(query, top_k)

        all_qdrant_ids: list[str] = []
        for name in entity_names:
            ids = self.kg.get_chunk_refs_for_entity(name, limit=top_k * 2)
            all_qdrant_ids.extend(ids)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_ids = [i for i in all_qdrant_ids if not (i in seen or seen.add(i))]

        if not unique_ids or self.chunk_retriever is None:
            return self._keyword_fallback(query, top_k)

        contexts = self.chunk_retriever.fetch_by_ids(unique_ids[:top_k * 2])
        # Tag as graph-sourced. fetch_by_ids() returns fresh objects each call — mutation is safe.
        for ctx in contexts:
            ctx.collection = "graph"
            ctx.metadata["entities_found"] = entity_names

        return contexts[:top_k]

    def retrieve_by_entity(
        self,
        entity_name: str,
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        """Retrieve contexts for a specific entity via HAS_CHUNK traversal."""
        ids = self.kg.get_chunk_refs_for_entity(entity_name, limit=top_k * 2)
        if not ids or self.chunk_retriever is None:
            return []
        contexts = self.chunk_retriever.fetch_by_ids(ids[:top_k])
        for ctx in contexts:
            ctx.collection = "graph"
            ctx.metadata["entities_found"] = [entity_name]
        return contexts

    def _keyword_fallback(self, query: str, top_k: int) -> list[RetrievedContext]:
        """
        Fallback: search _Embeddable nodes by display_name keyword matching
        when no CS entities are detected in the query.
        Uses parameterised Cypher — never f-string interpolation.
        """
        words = re.findall(r'\b[a-zA-Z]{2,}\b', query)
        keywords = [w.lower() for w in words if w.lower() not in self._STOP_WORDS]

        if not keywords:
            return []

        try:
            rows = self.kg.client.execute_read(
                """
                MATCH (n:_Embeddable)
                WHERE any(kw IN $keywords WHERE toLower(n.display_name) CONTAINS kw)
                RETURN n.display_name AS name, n.entity_type AS type
                LIMIT $lim
                """,
                {"keywords": keywords[:5], "lim": top_k},
            )
            return [
                RetrievedContext(
                    text=r["name"],
                    source="neo4j/keyword",
                    score=0.3,
                    collection="graph",
                    metadata={"entity_type": r.get("type", "")},
                )
                for r in rows
                if r.get("name")
            ]
        except Exception:
            return []
```

- [ ] **Step 4: Run the new tests**

```bash
python -m pytest tests/unit/test_retrieval.py::test_graph_retriever_uses_has_chunk_path tests/unit/test_retrieval.py::test_graph_retriever_keyword_fallback_no_entities -v
```

Expected: both PASS.

- [ ] **Step 5: Remove the now-stale `test_graph_retrieval` test**

In `tests/unit/test_retrieval.py`, delete the old `test_graph_retrieval` function (the one that patches `KnowledgeGraph` and checks for "Aspirin") since it tested the old medical-domain interface.

- [ ] **Step 6: Run full unit suite**

```bash
python -m pytest tests/unit/ -v
```

Expected: all remaining tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/retrieval/graph_retriever.py src/graph/cs_knowledge_graph.py tests/unit/test_retrieval.py
git commit -m "feat: rewrite GraphRetriever to use HAS_CHUNK traversal and return RetrievedContext"
```

---

## Task 7: Update `HybridRetriever` and config

**Files:**
- Modify: `src/retrieval/hybrid_retriever.py`
- Modify: `config/defaults.yaml`
- Modify: `tests/unit/test_retrieval.py`

- [ ] **Step 1: Update `config/defaults.yaml`**

Replace the `hybrid_retrieval` block:
```yaml
hybrid_retrieval:
  vector_weight: 0.6
  graph_weight: 0.4
  top_k: 5
  min_score: 0.2
```
With:
```yaml
hybrid_retrieval:
  chunk_weight: 0.5
  paper_weight: 0.3
  graph_weight: 0.2
  top_k: 5
  min_score: 0.2
```

Also remove `hybridrag.qdrant_collection: "arxiv"` from the `hybridrag` block — leave the rest of `hybridrag` intact. This removal must happen in the same commit as the pipeline constructor change (Task 8) to stay atomic.

- [ ] **Step 2: Write failing test for new `HybridRetriever`**

Add to `tests/unit/test_retrieval.py`:

```python
from src.retrieval.chunk_retriever import ChunkRetriever
from src.retrieval.paper_retriever import PaperRetriever
from src.types import RetrievedContext

def test_hybrid_retriever_three_way_fusion():
    """HybridRetriever fuses chunk, paper, and graph results with per-source RRF weights."""
    mock_chunk = MagicMock(spec=ChunkRetriever)
    mock_chunk.search.return_value = [
        RetrievedContext(text="Chunk context A", source="s1", score=0.9, collection="arxiv_chunks"),
    ]

    mock_paper = MagicMock(spec=PaperRetriever)
    mock_paper.search.return_value = [
        RetrievedContext(text="Paper context B", source="s2", score=0.8, collection="arxiv_papers"),
    ]

    mock_graph = MagicMock()
    mock_graph.retrieve.return_value = [
        RetrievedContext(text="Graph context C", source="graph", score=1.0, collection="graph"),
    ]

    hybrid = HybridRetriever(
        chunk_retriever=mock_chunk,
        paper_retriever=mock_paper,
        graph_retriever=mock_graph,
        chunk_weight=0.5,
        paper_weight=0.3,
        graph_weight=0.2,
        config_path="nonexistent.yaml",  # explicit args must win over config file
    )
    results = hybrid.retrieve(query="test query", query_embedding=[0.1] * 768, top_k=3)

    texts = [r.text for r in results]
    assert "Chunk context A" in texts
    assert "Paper context B" in texts
    assert "Graph context C" in texts
    # All results are RetrievedContext
    assert all(isinstance(r, RetrievedContext) for r in results)


def test_hybrid_retriever_deduplicates_identical_text():
    """Identical text (including whitespace variants) from chunk and graph paths is deduplicated."""
    base_text = "Self-attention computes queries, keys, and values from the same input sequence."
    # One version has leading whitespace — dedup key normalises this
    padded_text = "  " + base_text + "  "

    mock_chunk = MagicMock(spec=ChunkRetriever)
    mock_chunk.search.return_value = [
        RetrievedContext(text=base_text, source="s1", score=0.9, collection="arxiv_chunks"),
    ]

    mock_paper = MagicMock(spec=PaperRetriever)
    mock_paper.search.return_value = []

    mock_graph = MagicMock()
    mock_graph.retrieve.return_value = [
        RetrievedContext(text=padded_text, source="graph", score=1.0, collection="graph"),
    ]

    hybrid = HybridRetriever(
        chunk_retriever=mock_chunk,
        paper_retriever=mock_paper,
        graph_retriever=mock_graph,
        config_path="nonexistent.yaml",  # prevent config file from overriding test weights
    )
    results = hybrid.retrieve(query="self-attention", query_embedding=[0.1] * 768, top_k=5)

    # Deduplicated — normalised key collapses both into one result
    matching = [r for r in results if r.text.strip() == base_text]
    assert len(matching) == 1, "Whitespace-variant duplicates must be collapsed to one result"
```

- [ ] **Step 3: Run to confirm tests fail**

```bash
python -m pytest tests/unit/test_retrieval.py::test_hybrid_retriever_three_way_fusion tests/unit/test_retrieval.py::test_hybrid_retriever_deduplicates_identical_text -v
```

Expected: FAIL — old `HybridRetriever` signature.

- [ ] **Step 4: Rewrite `HybridRetriever`**

Replace the full content of `src/retrieval/hybrid_retriever.py`:

```python
"""
Hybrid retriever combining arxiv_chunks, arxiv_papers, and Neo4j graph search.

Uses Reciprocal Rank Fusion (RRF) with per-source weights.
"""
from typing import Optional
import yaml

from src.types import RetrievedContext

RRF_K = 60


class HybridRetriever:
    """
    Fuses results from ChunkRetriever, PaperRetriever, and GraphRetriever.

    RRF score per context = sum over sources of:
        source_weight * (1 / (RRF_K + rank + 1))

    Deduplication key: normalised text (strip + lowercase).
    Identical text from multiple sources boosts score rather than duplicating.
    """

    _SENTINEL = object()

    def __init__(
        self,
        chunk_retriever=None,
        paper_retriever=None,
        graph_retriever=None,
        chunk_weight=_SENTINEL,
        paper_weight=_SENTINEL,
        graph_weight=_SENTINEL,
        config_path: str = "config/defaults.yaml",
    ):
        # Load defaults from config; explicit constructor args override config values.
        cfg_chunk, cfg_paper, cfg_graph = 0.5, 0.3, 0.2
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f).get("hybrid_retrieval", {})
                cfg_chunk = cfg.get("chunk_weight", cfg_chunk)
                cfg_paper = cfg.get("paper_weight", cfg_paper)
                cfg_graph = cfg.get("graph_weight", cfg_graph)
        except Exception:
            pass

        self.chunk_weight = chunk_weight if chunk_weight is not self._SENTINEL else cfg_chunk
        self.paper_weight = paper_weight if paper_weight is not self._SENTINEL else cfg_paper
        self.graph_weight = graph_weight if graph_weight is not self._SENTINEL else cfg_graph

        self.chunk_retriever = chunk_retriever
        self.paper_retriever = paper_retriever
        self.graph_retriever = graph_retriever

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.2,
        use_graph: bool = True,
    ) -> list[RetrievedContext]:
        """
        Retrieve from all sources and fuse with weighted RRF.

        Args:
            query: Query text (for graph retriever)
            query_embedding: Pre-computed query vector (for vector retrievers)
            top_k: Number of final results
            min_score: Minimum score for vector retrievers
            use_graph: If False, skip graph retrieval (vector-only mode)
        """
        fetch_k = top_k * 3

        chunk_results: list[RetrievedContext] = []
        if self.chunk_retriever:
            chunk_results = self.chunk_retriever.search(
                query_vector=query_embedding, top_k=fetch_k, min_score=min_score
            )

        paper_results: list[RetrievedContext] = []
        if self.paper_retriever:
            paper_results = self.paper_retriever.search(
                query_vector=query_embedding, top_k=fetch_k, min_score=min_score
            )

        graph_results: list[RetrievedContext] = []
        if use_graph and self.graph_retriever:
            graph_results = self.graph_retriever.retrieve(query=query, top_k=fetch_k)

        return self._fuse(
            chunk_results, paper_results, graph_results, top_k=top_k
        )

    def _fuse(
        self,
        chunk_results: list[RetrievedContext],
        paper_results: list[RetrievedContext],
        graph_results: list[RetrievedContext],
        top_k: int,
    ) -> list[RetrievedContext]:
        """Weighted RRF fusion with text-based deduplication."""
        # dedup_key -> (RetrievedContext, rrf_score)
        seen: dict[str, tuple[RetrievedContext, float]] = {}

        def _key(text: str) -> str:
            return text.strip().lower()

        def _add(results: list[RetrievedContext], weight: float):
            for rank, ctx in enumerate(results):
                k = _key(ctx.text)
                rrf = weight * (1.0 / (RRF_K + rank + 1))
                if k in seen:
                    existing_ctx, existing_score = seen[k]
                    seen[k] = (existing_ctx, existing_score + rrf)
                else:
                    seen[k] = (ctx, rrf)

        _add(chunk_results, self.chunk_weight)
        _add(paper_results, self.paper_weight)
        _add(graph_results, self.graph_weight)

        sorted_items = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        out: list[RetrievedContext] = []
        for ctx, score in sorted_items[:top_k]:
            ctx.score = round(score, 6)
            out.append(ctx)
        return out
```

- [ ] **Step 5: Run new tests**

```bash
python -m pytest tests/unit/test_retrieval.py::test_hybrid_retriever_three_way_fusion tests/unit/test_retrieval.py::test_hybrid_retriever_deduplicates_identical_text -v
```

Expected: both PASS.

- [ ] **Step 6: Remove the old `test_hybrid_fusion` test**

Delete `test_hybrid_fusion` from `tests/unit/test_retrieval.py` — it tested the old 2-way interface with raw score math that no longer applies.

- [ ] **Step 7: Run full unit suite**

```bash
python -m pytest tests/unit/ -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/retrieval/hybrid_retriever.py config/defaults.yaml tests/unit/test_retrieval.py
git commit -m "feat: rewrite HybridRetriever with 3-way RRF fusion returning RetrievedContext"
```

---

## Task 8: Update `HybridRAGBenchPipeline`

**Files:**
- Modify: `src/pipelines/hybridrag_pipeline.py`

- [ ] **Step 1: Wire the new retriever stack and deprecate `ingest()`**

In `hybridrag_pipeline.py`, make the following changes:

**Imports** — replace the `QdrantStorage` import with the new retrievers:
```python
from src.retrieval.chunk_retriever import ChunkRetriever
from src.retrieval.paper_retriever import PaperRetriever
```
Remove: `from src.retrieval.qdrant_storage import QdrantStorage`

**`__init__`** — replace the `self.qdrant` + `self.hybrid_retriever` construction:

Remove:
```python
self.qdrant = QdrantStorage(
    collection=self.qdrant_collection,
    dim=self.embedding_dim,
)
```

Add:
```python
self.chunk_retriever = ChunkRetriever()
self.paper_retriever = PaperRetriever()
```

Replace the `HybridRetriever` construction:
```python
self.hybrid_retriever = HybridRetriever(
    chunk_retriever=self.chunk_retriever,
    paper_retriever=self.paper_retriever,
    graph_retriever=self.graph_retriever,
)
```

Remove these now-unused config reads from `__init__`:
```python
self.qdrant_collection: str = hcfg.get("qdrant_collection", "arxiv")
```

**`ingest()` method** — replace the entire body with:
```python
def ingest(self, *args, **kwargs):
    raise NotImplementedError(
        "HybridRAGBenchPipeline.ingest() is deprecated. "
        "Use ingest_local.py (LocalIngestionPipeline) instead."
    )
```

**`query()` method** — update to use the new retriever stack:

```python
def query(self, question: str, top_k: int = 5, use_hybrid: bool = True) -> dict:
    query_embedding = embed_texts_with_model(
        [question], self.embedding_model, batch_size=1
    )[0]

    contexts_raw = self.hybrid_retriever.retrieve(
        query=question,
        query_embedding=query_embedding,
        top_k=top_k * 3,
        min_score=self.min_score,
        use_graph=use_hybrid,
    )

    context_texts = [c.text for c in contexts_raw]
    context_sources = [c.source for c in contexts_raw]
    context_scores = [c.score for c in contexts_raw]
    entities_found = list({
        e
        for c in contexts_raw
        for e in c.metadata.get("entities_found", [])
    })

    # Rerank
    if self.reranker and context_texts:
        reranked = self.reranker.rerank(
            query=question,
            documents=context_texts,
            sources=context_sources,
            scores=context_scores,
            top_k=top_k,
        )
        context_texts = reranked.get("contexts", context_texts[:top_k])
        context_sources = reranked.get("sources", context_sources[:top_k])
        context_scores = reranked.get("rerank_scores", context_scores[:top_k])
    else:
        context_texts = context_texts[:top_k]
        context_sources = context_sources[:top_k]
        context_scores = context_scores[:top_k]

    messages = build_messages(question, context_texts)
    answer = self.llm.generate(messages)

    return {
        "question": question,
        "answer": answer,
        "contexts": context_texts,
        "sources": context_sources,
        "scores": context_scores,
        "entities_found": entities_found,
        "retrieval_type": "hybrid" if use_hybrid else "vector",
    }
```

- [ ] **Step 2: Verify pipeline imports cleanly**

```bash
python -c "from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline; print('OK')"
```

Expected: `OK` (no connection errors at import time — connections are made lazily).

- [ ] **Step 3: Run integration smoke check (skip if live services unavailable)**

If Neo4j and Qdrant are running:
```bash
python -c "
from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline
p = HybridRAGBenchPipeline()
print('Pipeline initialized OK')
"
```

**Note on dead code:** `_flush_batch` and `_ProgressTracker` in `hybridrag_pipeline.py` are now dead since `ingest()` is deprecated. Leave them in place for this change — removing them is a separate cleanup task and does not affect correctness.

- [ ] **Step 4: Commit**

```bash
git add src/pipelines/hybridrag_pipeline.py config/defaults.yaml
git commit -m "feat: wire ChunkRetriever+PaperRetriever into pipeline, deprecate ingest()"
```

---

## Task 9: Add `HAS_CHUNK` Writes to `LocalIngestionPipeline`

**Files:**
- Modify: `src/pipelines/local_ingestion_pipeline.py`

- [ ] **Step 1: Add the Neo4j write to `_ingest_chunks`**

In `_ingest_chunks`, after the `self._q().upsert(...)` call inside `_flush`, add the Neo4j HAS_CHUNK write:

```python
def _flush():
    nonlocal total, batch_index
    # Snapshot IDs and Neo4j params before any mutation of points list
    snapshot_ids = [str(p.id) for p in points]
    neo4j_batch = [
        {
            "src_id": int(p.payload.get("src_id", 0)),
            "dst_id": int(p.payload.get("dst_id", 0)),
            "edge_id": int(p.payload.get("edge_id", 0)),
            "domain": p.payload.get("domain", ""),
            "qdrant_id": str(p.id),  # UUID string — must be str()
        }
        for p in points
        if p.payload.get("src_id") is not None and p.payload.get("dst_id") is not None
    ]

    qdrant_ok = False
    try:
        self._q().upsert(collection_name="arxiv_chunks", points=points)
        total += len(points)
        qdrant_ok = True
    except Exception as e:
        progress.log_batch_failure(domain, "chunks", batch_index, str(e), snapshot_ids)
        logger.error(f"[{domain}] chunks batch {batch_index} Qdrant upsert failed: {e}")

    if qdrant_ok:
        # Write HAS_CHUNK edges — separate try/except so Qdrant success is not lost
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
            progress.log_batch_failure(domain, "chunks_has_chunk", batch_index, str(e), snapshot_ids)
            logger.error(f"[{domain}] chunks batch {batch_index} HAS_CHUNK write failed: {e}")

    # Always advance batch_index and clear — exactly once per _flush() call
    batch_index += 1
    points.clear()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "from src.pipelines.local_ingestion_pipeline import LocalIngestionPipeline; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/pipelines/local_ingestion_pipeline.py
git commit -m "feat: write HAS_CHUNK edges in _ingest_chunks for graph-to-text linkage"
```

---

## Task 10: Migration Script for Existing Data

**Files:**
- Create: `scripts/migrate_has_chunk_edges.py`

- [ ] **Step 1: Create the scripts directory and migration script**

```bash
mkdir -p scripts
```

Create `scripts/migrate_has_chunk_edges.py`:

```python
"""
One-shot migration: create HAS_CHUNK edges in Neo4j for already-ingested data.

Reads parquet files from data/hybridrag/ (same source as LocalIngestionPipeline),
recomputes qdrant_id = uuid5(HYBRIDRAG_NS, f"{domain}:{edge_id}"), and writes
MERGE-idempotent HAS_CHUNK edges between _Embeddable nodes.

Safe to re-run — uses MERGE + unconditional SET so partial runs are healed.

Usage:
    python scripts/migrate_has_chunk_edges.py
    python scripts/migrate_has_chunk_edges.py --data-dir data/hybridrag --batch-size 500
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.local_parquet_loader import LocalParquetLoader, DOMAINS, HYBRIDRAG_NS
from src.utils import get_logger

logger = get_logger(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pokemon0424$"
BATCH_SIZE = 500

CYPHER = """
UNWIND $batch AS c
MATCH (src:_Embeddable {node_id: c.src_id, domain: c.domain})
MATCH (dst:_Embeddable {node_id: c.dst_id, domain: c.domain})
MERGE (src)-[r:HAS_CHUNK {edge_id: c.edge_id, domain: c.domain}]->(dst)
SET r.qdrant_id = c.qdrant_id
"""


def migrate(data_dir: str, batch_size: int, domains: list[str]):
    from neo4j import GraphDatabase

    loader = LocalParquetLoader(data_dir)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    total = 0
    errors = 0

    try:
        for domain in domains:
            logger.info(f"[{domain}] Starting HAS_CHUNK migration...")
            batch = []

            for rec in loader.iter_chunks(domain):
                # Use rec.qdrant_id directly — ChunkRecord.__post_init__ computes it via
                # uuid5(HYBRIDRAG_NS, f"{domain}:{edge_id}"), same formula as LocalIngestionPipeline
                batch.append({
                    "src_id": rec.src_id,
                    "dst_id": rec.dst_id,
                    "edge_id": rec.edge_id,
                    "domain": domain,
                    "qdrant_id": str(rec.qdrant_id),  # always str — rec.qdrant_id is uuid.UUID
                })

                if len(batch) >= batch_size:
                    try:
                        with driver.session() as s:
                            s.run(CYPHER, {"batch": batch})
                        total += len(batch)
                        logger.info(f"[{domain}] {total} edges written...")
                    except Exception as e:
                        errors += len(batch)
                        logger.error(f"[{domain}] Batch failed: {e}")
                    batch = []

            if batch:
                try:
                    with driver.session() as s:
                        s.run(CYPHER, {"batch": batch})
                    total += len(batch)
                except Exception as e:
                    errors += len(batch)
                    logger.error(f"[{domain}] Final batch failed: {e}")

            logger.info(f"[{domain}] Done.")

    finally:
        driver.close()

    print(f"\nMigration complete: {total} edges written, {errors} errors.")
    if errors:
        print("Re-run to retry failed batches — MERGE is idempotent.")


def main():
    parser = argparse.ArgumentParser(description="Migrate HAS_CHUNK edges into Neo4j")
    parser.add_argument("--data-dir", default="data/hybridrag")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--domains", nargs="+", default=DOMAINS,
        choices=DOMAINS,
    )
    args = parser.parse_args()
    migrate(args.data_dir, args.batch_size, args.domains)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script imports cleanly**

```bash
python -c "import scripts.migrate_has_chunk_edges; print('OK')"
```

Or:
```bash
python scripts/migrate_has_chunk_edges.py --help
```

Expected: help text prints, no import errors.

- [ ] **Step 3: Run the migration (requires live Neo4j + data)**

```bash
python scripts/migrate_has_chunk_edges.py
```

Expected output: per-domain progress lines followed by `Migration complete: N edges written, 0 errors.`

- [ ] **Step 4: Verify edges were created**

```bash
python -c "
from neo4j import GraphDatabase
d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'Pokemon0424\$'))
with d.session() as s:
    n = s.run('MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) AS n').single()['n']
    print(f'HAS_CHUNK edges: {n}')
d.close()
"
```

Expected: non-zero count matching the number of chunk records in the parquet files.

- [ ] **Step 5: Commit**

```bash
git add scripts/migrate_has_chunk_edges.py
git commit -m "feat: add HAS_CHUNK migration script for existing ingested data"
```

---

## Task 11: Final Integration Check

- [ ] **Step 1: Run full unit suite**

```bash
python -m pytest tests/unit/ -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run a quick end-to-end query (requires live services)**

```bash
python query_hybridrag.py
```

Or:
```bash
python -c "
from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline
p = HybridRAGBenchPipeline()
r = p.query('What is the attention mechanism in transformers?', top_k=3, use_hybrid=True)
print('Answer:', r['answer'][:200])
print('Num contexts:', len(r['contexts']))
print('Retrieval type:', r['retrieval_type'])
"
```

Expected: answer printed, `num_contexts > 0`, `retrieval_type = hybrid`.

- [ ] **Step 3: Run eval with 10 questions to compare before/after**

```bash
python tests/evaluation/hybridrag_eval.py --max-pairs 10 --modes vector hybrid
```

Expected:
- `context_recall` and `faithfulness` should be higher than the baseline (0.23/0.11)
- `answer_correctness` comparable or improved
- No N/A metrics (N/A still expected for hit_rate/mrr/ndcg since no ground truth context in dataset)

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "fix: complete hybridrag retrieval fixes — typed layer, HAS_CHUNK edges, domain cleanup"
```
