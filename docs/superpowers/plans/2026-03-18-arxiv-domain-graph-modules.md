# Arxiv Domain Graph Modules Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `query_hybridrag.py` crash by creating the two missing CS/AI domain graph modules and removing stale medical imports from the live retrieval files.

**Architecture:** Two new files implement CS-domain entity extraction and knowledge graph access, reusing `ExtractionResult`/`ExtractedEntity` dataclasses from `entity_extractor.py`. The retriever files are patched to remove hard dependencies on medical-specific classes. `CSKnowledgeGraph` query methods target the `:_Embeddable` label and `display_name` property that `local_ingestion_pipeline.py` actually writes — not `:CSEntity` — so graph traversal works against real data.

**Tech Stack:** Python, Neo4j (via `Neo4jClient`), existing `ExtractionResult`/`ExtractedEntity` dataclasses.

**Schema note:** `local_ingestion_pipeline.py` (used by `ingest_local.py`) writes nodes with label `:{entity_type}:_Embeddable` and property `display_name`. `CSKnowledgeGraph.ingest_extraction_results_batch()` (used by the HuggingFace ingestion path) writes `:CSEntity {name}`. The query methods target `_Embeddable`/`display_name` so they work for locally-ingested data. The `ingest_extraction_results_batch()` method keeps writing `:CSEntity` for the HuggingFace path.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/graph/cs_entity_extractor.py` | Rule-based CS/AI entity extraction; `CSEntityExtractor` + `get_cs_entity_extractor()` |
| Create | `src/graph/cs_knowledge_graph.py` | Neo4j graph access for CS domain; query methods use `_Embeddable` label |
| Modify | `src/retrieval/graph_retriever.py` | Remove `MedicalEntityExtractor`/`KnowledgeGraph` imports; use `Any`; fix `get_graph_retriever()` |
| Modify | `src/retrieval/hybrid_retriever.py` | Remove unused `get_entity_extractor` import; fix `_get_graph_retriever()` dead fallback |
| Modify | `tests/unit/test_retrieval.py` | Fix broken `@patch` target + `extract` → `extract_entities` method name |
| Test | `tests/unit/test_cs_graph.py` | Unit tests for both new modules (no Neo4j needed — mock the client) |

---

## Task 1: Create `cs_entity_extractor.py`

**Files:**
- Create: `src/graph/cs_entity_extractor.py`
- Test: `tests/unit/test_cs_graph.py`

### Interfaces required

`hybridrag_pipeline.py` calls:
```python
extraction = self.entity_extractor.extract(chunk.paragraph, chunk.description, chunk_id_str)
# extraction.entities -> list[ExtractedEntity]
# extraction.relationships -> list[tuple]
```

`graph_retriever.py` calls:
```python
query_entities = self.entity_extractor.extract_entities(query)
# query_entities.entities -> list[ExtractedEntity]  (.text, .entity_type, .confidence)
```

`ExtractionResult` signature (from `entity_extractor.py:353`):
```python
@dataclass
class ExtractionResult:
    chunk_id: str
    text: str                                   # required — must be passed
    entities: list[ExtractedEntity] = ...
    relationships: list[tuple[str, str, str]] = ...
```

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_cs_graph.py`:

```python
"""Unit tests for CS-domain graph modules."""
import pytest
from unittest.mock import MagicMock, patch


# ── CSEntityExtractor ──────────────────────────────────────────────────────

def test_cs_extractor_extract_entities_returns_result():
    from src.graph.cs_entity_extractor import CSEntityExtractor
    extractor = CSEntityExtractor()
    result = extractor.extract_entities("attention mechanism in transformer models")
    assert hasattr(result, "entities")
    assert hasattr(result, "relationships")


def test_cs_extractor_extract_entities_finds_known_term():
    from src.graph.cs_entity_extractor import CSEntityExtractor
    extractor = CSEntityExtractor()
    result = extractor.extract_entities("BERT is a transformer model for NLP tasks")
    names = [e.text.lower() for e in result.entities]
    assert any("bert" in n or "transformer" in n for n in names)


def test_cs_extractor_extract_three_arg_signature():
    """hybridrag_pipeline calls extract(paragraph, description, chunk_id)."""
    from src.graph.cs_entity_extractor import CSEntityExtractor
    extractor = CSEntityExtractor()
    result = extractor.extract("BERT uses attention", "a transformer model", "chunk-001")
    assert hasattr(result, "entities")


def test_cs_extractor_empty_text_returns_empty():
    from src.graph.cs_entity_extractor import CSEntityExtractor
    extractor = CSEntityExtractor()
    result = extractor.extract_entities("")
    assert result.entities == []


def test_get_cs_entity_extractor_returns_singleton():
    from src.graph.cs_entity_extractor import get_cs_entity_extractor
    a = get_cs_entity_extractor()
    b = get_cs_entity_extractor()
    assert a is b
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_cs_graph.py -v -k "extractor"
```
Expected: `ModuleNotFoundError: No module named 'src.graph.cs_entity_extractor'`

- [ ] **Step 3: Implement `cs_entity_extractor.py`**

Create `src/graph/cs_entity_extractor.py`:

```python
"""
Rule-based CS/AI entity extractor for the arXiv HybridRAG-Bench domain.

Designed to match the interface expected by:
  - hybridrag_pipeline.py: extract(paragraph, description, chunk_id) -> ExtractionResult
  - graph_retriever.py:    extract_entities(text) -> ExtractionResult

ExtractionResult requires both chunk_id and text positional args — see entity_extractor.py:353.
"""
import re
from typing import Optional
from src.graph.entity_extractor import ExtractionResult, ExtractedEntity
from src.utils import get_logger

logger = get_logger(__name__)

# ── Entity type definitions ────────────────────────────────────────────────

CS_ENTITY_TYPES = {
    "ALGORITHM":  "Algorithms and computational methods (attention, backprop, etc.)",
    "MODEL":      "ML/AI model architectures (BERT, GPT, ResNet, etc.)",
    "DATASET":    "Benchmark datasets (ImageNet, COCO, SQuAD, etc.)",
    "TASK":       "ML/NLP/CV tasks (classification, translation, summarisation, etc.)",
    "METRIC":     "Evaluation metrics (accuracy, F1, BLEU, ROUGE, perplexity, etc.)",
    "FRAMEWORK":  "Software frameworks and libraries (PyTorch, TensorFlow, JAX, etc.)",
    "CONCEPT":    "General CS/AI concepts (neural network, gradient descent, etc.)",
}

# ── Keyword dictionaries ───────────────────────────────────────────────────

_MODELS = {
    "bert", "gpt", "gpt-2", "gpt-3", "gpt-4", "t5", "roberta", "albert",
    "xlnet", "electra", "deberta", "llama", "mistral", "gemma",
    "resnet", "vgg", "efficientnet", "vit", "clip", "dall-e",
    "transformer", "lstm", "gru", "rnn", "cnn", "gan", "vae",
    "word2vec", "glove", "fasttext", "stable diffusion",
}

_ALGORITHMS = {
    "attention", "self-attention", "multi-head attention", "cross-attention",
    "backpropagation", "gradient descent", "adam", "sgd", "adagrad", "rmsprop",
    "dropout", "batch normalisation", "layer normalisation", "softmax",
    "beam search", "greedy decoding", "nucleus sampling",
    "k-means", "svm", "random forest", "xgboost", "lightgbm",
    "reinforcement learning", "q-learning", "ppo", "dqn",
    "contrastive learning", "fine-tuning", "transfer learning",
    "few-shot learning", "zero-shot", "in-context learning",
    "prompt tuning", "lora", "qlora",
    "retrieval augmented generation", "rag",
}

_DATASETS = {
    "imagenet", "coco", "cifar", "mnist", "squad", "glue", "superglue",
    "wikitext", "openwebtext", "common crawl", "bookcorpus", "c4",
    "ms marco", "natural questions", "triviaqa", "hotpotqa",
    "penn treebank", "conll", "ontonotes",
}

_TASKS = {
    "classification", "regression", "segmentation", "object detection",
    "named entity recognition", "ner", "pos tagging",
    "machine translation", "summarisation", "summarization",
    "question answering", "reading comprehension",
    "text generation", "language modelling", "language modeling",
    "sentiment analysis", "relation extraction", "coreference resolution",
    "image captioning", "visual question answering", "vqa",
    "speech recognition", "information retrieval", "semantic search",
}

_METRICS = {
    "accuracy", "precision", "recall", "f1", "f1 score",
    "bleu", "rouge", "meteor",
    "perplexity",
    "map", "ndcg", "mrr",
    "auc", "roc",
    "exact match",
}

_FRAMEWORKS = {
    "pytorch", "tensorflow", "keras", "jax", "flax",
    "huggingface", "scikit-learn", "sklearn",
    "cuda", "triton", "onnx", "vllm", "ollama",
}

_CONCEPTS = {
    "neural network", "deep learning", "machine learning",
    "natural language processing", "nlp",
    "computer vision", "multimodal",
    "knowledge graph", "graph neural network", "gnn",
    "embedding", "representation learning",
    "encoder", "decoder", "encoder-decoder",
    "tokenisation", "tokenization", "subword", "bpe",
    "overfitting", "regularisation", "regularization",
    "pretraining", "inference", "quantisation", "quantization",
    "pruning", "distillation",
}

# Map each keyword to its entity type
_KEYWORD_TO_TYPE: dict[str, str] = {}
for _kw in _MODELS:      _KEYWORD_TO_TYPE[_kw] = "MODEL"
for _kw in _ALGORITHMS:  _KEYWORD_TO_TYPE[_kw] = "ALGORITHM"
for _kw in _DATASETS:    _KEYWORD_TO_TYPE[_kw] = "DATASET"
for _kw in _TASKS:       _KEYWORD_TO_TYPE[_kw] = "TASK"
for _kw in _METRICS:     _KEYWORD_TO_TYPE[_kw] = "METRIC"
for _kw in _FRAMEWORKS:  _KEYWORD_TO_TYPE[_kw] = "FRAMEWORK"
for _kw in _CONCEPTS:    _KEYWORD_TO_TYPE[_kw] = "CONCEPT"

# Sort by length descending so longer phrases match before substrings
_SORTED_KEYWORDS = sorted(_KEYWORD_TO_TYPE.keys(), key=len, reverse=True)


def _extract_from_text(text: str, chunk_id: str = "") -> ExtractionResult:
    """
    Scan *text* for CS keyword matches and return an ExtractionResult.
    Case-insensitive. Longer phrases take priority (sorted above).

    NOTE: ExtractionResult requires both chunk_id and text — see entity_extractor.py:353.
    """
    # text is stored on the result for the ExtractionResult.text field
    result = ExtractionResult(chunk_id=chunk_id, text=text)
    if not text or not text.strip():
        return result

    lowered = text.lower()
    seen: set[str] = set()

    for keyword in _SORTED_KEYWORDS:
        if keyword in lowered and keyword not in seen:
            pattern = r'(?<![a-z0-9])' + re.escape(keyword) + r'(?![a-z0-9])'
            if re.search(pattern, lowered):
                seen.add(keyword)
                start = lowered.find(keyword)
                result.entities.append(
                    ExtractedEntity(
                        text=keyword,
                        entity_type=_KEYWORD_TO_TYPE[keyword],
                        start_pos=start,
                        end_pos=start + len(keyword),
                        confidence=1.0,
                    )
                )

    return result


class CSEntityExtractor:
    """
    Rule-based CS/AI entity extractor.

    Matches against curated keyword dictionaries for algorithms, models,
    datasets, tasks, metrics, frameworks, and general CS concepts.
    """

    def extract_entities(self, text: str) -> ExtractionResult:
        """
        Extract CS entities from a single text string.
        Called by graph_retriever.py at query time.
        """
        return _extract_from_text(text)

    def extract(
        self, paragraph: str, description: str = "", chunk_id: str = ""
    ) -> ExtractionResult:
        """
        Extract CS entities from a paragraph + optional description.
        Called by hybridrag_pipeline.py during ingestion.
        """
        combined = f"{paragraph} {description}".strip()
        return _extract_from_text(combined, chunk_id)


# ── Singleton factory ──────────────────────────────────────────────────────

_default_extractor: Optional[CSEntityExtractor] = None


def get_cs_entity_extractor() -> CSEntityExtractor:
    """Return (or create) the module-level CSEntityExtractor singleton."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = CSEntityExtractor()
    return _default_extractor
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_cs_graph.py -v -k "extractor"
```
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/graph/cs_entity_extractor.py tests/unit/test_cs_graph.py
git commit -m "feat: add CSEntityExtractor for arXiv CS/AI domain"
```

---

## Task 2: Create `cs_knowledge_graph.py`

**Files:**
- Create: `src/graph/cs_knowledge_graph.py`
- Test: `tests/unit/test_cs_graph.py` (append)

### Schema reality check

`local_ingestion_pipeline.py` writes Neo4j nodes as:
```cypher
CREATE (:{entity_type}:_Embeddable {node_id: ..., domain: ..., display_name: ..., entity_type: ...})
```

So the **entity name property is `display_name`** (not `name`) and the **shared label is `_Embeddable`** (not `CSEntity`). All query methods below match `_Embeddable` / `display_name` so they work against real locally-ingested data. The `ingest_extraction_results_batch()` method still writes `:CSEntity {name}` for the HuggingFace ingestion path — this is a known two-schema situation.

### Interfaces required

From `hybridrag_pipeline.py`:
```python
cs_kg.initialize_schema()
cs_kg.add_document(GraphDocument(...))
cs_kg.ingest_extraction_results_batch(list[tuple[GraphChunk, ExtractionResult]])
cs_kg.client.execute_write(cypher, params)   # direct client access for native KG import
```

From `graph_retriever.py`:
```python
kg.get_related_entities(entity_name, depth=2)   # -> list[dict]
kg.get_chunks_for_entity(entity_name)            # -> list[dict]
kg.get_entity_context(entity_name, max_chunks)   # -> list[str]
kg.get_entities_by_type(entity_type)             # -> list[dict]
kg.get_stats()                                   # -> dict
kg.client.execute_read(cypher, params)           # keyword fallback in graph_retriever
```

- [ ] **Step 6: Write the failing tests** (append to `tests/unit/test_cs_graph.py`)

```python
# ── CSKnowledgeGraph ───────────────────────────────────────────────────────

def _make_mock_client():
    """Return a MagicMock that looks like Neo4jClient."""
    client = MagicMock()
    client.execute_write = MagicMock(return_value=None)
    client.execute_read = MagicMock(return_value=[])
    client.execute_query = MagicMock(return_value=None)
    client.connect = MagicMock(return_value=None)
    return client


def test_cs_kg_initialize_schema_calls_client():
    from src.graph.cs_knowledge_graph import CSKnowledgeGraph
    client = _make_mock_client()
    kg = CSKnowledgeGraph(neo4j_client=client)
    kg.initialize_schema()
    assert client.execute_query.called


def test_cs_kg_add_document():
    from src.graph.cs_knowledge_graph import CSKnowledgeGraph
    from src.graph.knowledge_graph import GraphDocument
    client = _make_mock_client()
    kg = CSKnowledgeGraph(neo4j_client=client)
    doc = GraphDocument(id="doc-1", title="Test Paper", source="arxiv", metadata={})
    kg.add_document(doc)
    assert client.execute_write.called
    cypher_arg = client.execute_write.call_args[0][0]
    assert "Document" in cypher_arg


def test_cs_kg_ingest_batch_writes_csentity():
    from src.graph.cs_knowledge_graph import CSKnowledgeGraph
    from src.graph.knowledge_graph import GraphChunk
    from src.graph.entity_extractor import ExtractionResult, ExtractedEntity
    client = _make_mock_client()
    kg = CSKnowledgeGraph(neo4j_client=client)

    chunk = GraphChunk(id="c-1", text="BERT is a transformer", document_id="doc-1",
                       chunk_index=0, metadata={})
    entity = ExtractedEntity(text="bert", entity_type="MODEL",
                             start_pos=0, end_pos=4, confidence=1.0)
    extraction = ExtractionResult(chunk_id="c-1", text="BERT is a transformer",
                                  entities=[entity], relationships=[])
    kg.ingest_extraction_results_batch([(chunk, extraction)])

    all_cyphers = " ".join(
        str(call[0][0]) for call in client.execute_write.call_args_list
    )
    assert "CSEntity" in all_cyphers


def test_cs_kg_exposes_client():
    from src.graph.cs_knowledge_graph import CSKnowledgeGraph
    client = _make_mock_client()
    kg = CSKnowledgeGraph(neo4j_client=client)
    assert kg.client is client


def test_cs_kg_get_entity_context_chunk_path():
    """When Chunk nodes exist, returns their text."""
    from src.graph.cs_knowledge_graph import CSKnowledgeGraph
    client = _make_mock_client()
    client.execute_read.return_value = [{"text": "some chunk context"}]
    kg = CSKnowledgeGraph(neo4j_client=client)
    result = kg.get_entity_context("bert", max_chunks=3)
    assert result == ["some chunk context"]


def test_cs_kg_get_entity_context_fallback_to_neighbours():
    """When no Chunk nodes exist (local ingest), falls back to entity-neighbour text."""
    from src.graph.cs_knowledge_graph import CSKnowledgeGraph
    client = _make_mock_client()
    # First call (chunk lookup) returns empty; second call (neighbour graph) returns data
    client.execute_read.side_effect = [
        [],   # chunk lookup — empty (no Chunk nodes in local ingest)
        [{"src": "bert", "rel": "USES", "dst": "attention"}],
    ]
    kg = CSKnowledgeGraph(neo4j_client=client)
    result = kg.get_entity_context("bert", max_chunks=3)
    assert len(result) == 1
    assert "bert" in result[0].lower()
    assert "attention" in result[0].lower()


def test_cs_kg_get_related_entities_queries_embeddable():
    from src.graph.cs_knowledge_graph import CSKnowledgeGraph
    client = _make_mock_client()
    client.execute_read.return_value = []
    kg = CSKnowledgeGraph(neo4j_client=client)
    kg.get_related_entities("bert", depth=2)
    cypher = client.execute_read.call_args[0][0]
    assert "_Embeddable" in cypher
```

- [ ] **Step 7: Run tests to verify they fail**

```
pytest tests/unit/test_cs_graph.py -v -k "cs_kg"
```
Expected: `ModuleNotFoundError: No module named 'src.graph.cs_knowledge_graph'`

- [ ] **Step 8: Implement `cs_knowledge_graph.py`**

Create `src/graph/cs_knowledge_graph.py`:

```python
"""
CS/AI domain knowledge graph for the arXiv HybridRAG-Bench system.

Query methods target the :_Embeddable label and display_name property
that local_ingestion_pipeline.py writes into Neo4j.

ingest_extraction_results_batch() writes :CSEntity nodes (for the HuggingFace
ingestion path in hybridrag_pipeline.ingest()).
"""
import json
from typing import Optional
from src.storage.neo4j_client import Neo4jClient
from src.graph.knowledge_graph import GraphDocument, GraphChunk
from src.graph.entity_extractor import ExtractionResult
from src.utils import get_logger

logger = get_logger(__name__)


class CSKnowledgeGraph:
    """
    Knowledge graph access layer for the CS/AI arXiv domain.

    Reads query against :_Embeddable nodes (written by local_ingestion_pipeline.py).
    Writes via ingest_extraction_results_batch() use :CSEntity (HuggingFace path).
    """

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.client = neo4j_client or Neo4jClient()
        self.client.connect()

    # ── Schema ──────────────────────────────────────────────────────────────

    def initialize_schema(self) -> None:
        """Create indexes and constraints."""
        stmts = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:CSEntity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (e:CSEntity) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:CSEntity) ON (e.entity_type)",
            "CREATE INDEX IF NOT EXISTS FOR (e:_Embeddable) ON (e.display_name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:_Embeddable) ON (e.entity_type)",
        ]
        for stmt in stmts:
            try:
                self.client.execute_query(stmt)
            except Exception as exc:
                logger.debug(f"Schema note: {exc}")

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _clean_metadata(self, metadata: Optional[dict]) -> str:
        if not metadata:
            return '{}'
        try:
            import numpy as np

            class _Enc(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer,)):   return int(obj)
                    if isinstance(obj, (np.floating,)):  return float(obj)
                    if isinstance(obj, np.ndarray):      return obj.tolist()
                    return super().default(obj)

            return json.dumps(metadata, cls=_Enc)
        except Exception:
            return json.dumps({k: str(v) for k, v in metadata.items()})

    # ── Document ingestion ───────────────────────────────────────────────────

    def add_document(self, document: GraphDocument) -> None:
        self.client.execute_write(
            """
            MERGE (d:Document {id: $id})
            SET d.title    = $title,
                d.source   = $source,
                d.metadata = $metadata
            """,
            {
                "id": document.id,
                "title": document.title,
                "source": document.source,
                "metadata": self._clean_metadata(document.metadata),
            },
        )

    def add_chunk(self, chunk: GraphChunk) -> None:
        self.client.execute_write(
            """
            MERGE (c:Chunk {id: $id})
            SET c.text        = $text,
                c.document_id = $document_id,
                c.chunk_index = $chunk_index,
                c.metadata    = $metadata
            """,
            {
                "id": chunk.id,
                "text": chunk.text,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "metadata": self._clean_metadata(chunk.metadata),
            },
        )
        self.client.execute_write(
            """
            MATCH (d:Document {id: $doc_id})
            MATCH (c:Chunk    {id: $chunk_id})
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            {"doc_id": chunk.document_id, "chunk_id": chunk.id},
        )

    # ── Batch ingestion (HuggingFace path) ──────────────────────────────────

    def ingest_extraction_results_batch(
        self,
        batch_items: list[tuple[GraphChunk, ExtractionResult]],
    ) -> None:
        """
        Store a batch of (GraphChunk, ExtractionResult) pairs into Neo4j.
        Writes :CSEntity nodes and :HAS_ENTITY edges for the HuggingFace path.
        """
        if not batch_items:
            return

        # 1. Chunks
        chunk_data = [
            {
                "id": item[0].id,
                "text": item[0].text,
                "document_id": item[0].document_id,
                "chunk_index": item[0].chunk_index,
                "metadata": self._clean_metadata(item[0].metadata),
            }
            for item in batch_items
        ]
        self.client.execute_write(
            """
            UNWIND $batch AS c
            MERGE (n:Chunk {id: c.id})
            SET n.text        = c.text,
                n.document_id = c.document_id,
                n.chunk_index = c.chunk_index,
                n.metadata    = c.metadata
            """,
            {"batch": chunk_data},
        )
        self.client.execute_write(
            """
            UNWIND $batch AS c
            MATCH (d:Document {id: c.document_id})
            MATCH (n:Chunk    {id: c.id})
            MERGE (d)-[:HAS_CHUNK]->(n)
            """,
            {"batch": chunk_data},
        )

        # 2. CSEntity nodes + chunk links (grouped by entity_type for dynamic labels)
        entities_by_type: dict[str, list[dict]] = {}
        all_links: list[dict] = []

        for chunk, result in batch_items:
            for ent in result.entities:
                etype = (
                    "".join(c for c in ent.entity_type if c.isalnum() or c == "_")
                    or "Unknown"
                )
                entities_by_type.setdefault(etype, []).append(
                    {"name": ent.text, "type": etype, "confidence": ent.confidence}
                )
                all_links.append({"chunk_id": chunk.id, "entity_name": ent.text})

        for etype, batch in entities_by_type.items():
            self.client.execute_write(
                f"""
                UNWIND $batch AS ent
                MERGE (e:CSEntity {{name: ent.name}})
                SET e:{etype},
                    e.entity_type = ent.type,
                    e.confidence  = ent.confidence
                """,
                {"batch": batch},
            )

        if all_links:
            self.client.execute_write(
                """
                UNWIND $batch AS lnk
                MATCH (c:Chunk    {id:   lnk.chunk_id})
                MATCH (e:CSEntity {name: lnk.entity_name})
                MERGE (c)-[:HAS_ENTITY]->(e)
                """,
                {"batch": all_links},
            )

        # 3. Entity-entity relationships
        rels_by_type: dict[str, list[dict]] = {}
        for _, result in batch_items:
            for src, rel_type, tgt in result.relationships:
                clean = (
                    "".join(c for c in rel_type if c.isalnum() or c == "_").upper()
                    or "RELATED_TO"
                )
                rels_by_type.setdefault(clean, []).append(
                    {"source": src, "target": tgt}
                )

        for rtype, batch in rels_by_type.items():
            self.client.execute_write(
                f"""
                UNWIND $batch AS r
                MATCH (e1:CSEntity {{name: r.source}})
                MATCH (e2:CSEntity {{name: r.target}})
                MERGE (e1)-[:{rtype}]->(e2)
                """,
                {"batch": batch},
            )

    # ── Query methods (called by GraphRetriever) ────────────────────────────
    # These target :_Embeddable nodes written by local_ingestion_pipeline.py.
    # Property: display_name (not name) — matching what the loader writes.

    def get_related_entities(self, entity_name: str, depth: int = 2) -> list[dict]:
        """Traverse the _Embeddable entity graph up to *depth* hops."""
        query = (
            "MATCH path = (e:_Embeddable)-[*1..%d]-(related:_Embeddable) "
            "WHERE toLower(e.display_name) = toLower($name) AND e <> related "
            "RETURN path, length(path) as distance "
            "ORDER BY distance"
        ) % depth
        return self.client.execute_read(query, {"name": entity_name})

    def get_chunks_for_entity(self, entity_name: str) -> list[dict]:
        """Return Chunk nodes linked to a CSEntity (HuggingFace ingestion path)."""
        return self.client.execute_read(
            """
            MATCH (c:Chunk)-[:HAS_ENTITY]->(e:CSEntity)
            WHERE toLower(e.name) = toLower($name)
            RETURN c
            ORDER BY c.chunk_index
            """,
            {"name": entity_name},
        )

    def get_entity_context(
        self, entity_name: str, max_chunks: int = 5
    ) -> list[str]:
        """
        Return text context for an entity.

        Primary path: Chunk text via HAS_ENTITY edges (HuggingFace ingestion).
        Fallback: synthesise context from entity-neighbour pairs in the
                  _Embeddable graph (local_ingestion_pipeline data).
        """
        rows = self.client.execute_read(
            """
            MATCH (c:Chunk)-[:HAS_ENTITY]->(e:CSEntity)
            WHERE toLower(e.name) = toLower($name)
            RETURN c.text as text
            ORDER BY c.chunk_index
            LIMIT $lim
            """,
            {"name": entity_name, "lim": max_chunks},
        )
        texts = [r["text"] for r in rows if r.get("text")]

        if not texts:
            rows = self.client.execute_read(
                """
                MATCH (e:_Embeddable)-[r]-(nb:_Embeddable)
                WHERE toLower(e.display_name) = toLower($name)
                RETURN e.display_name as src, type(r) as rel, nb.display_name as dst
                LIMIT $lim
                """,
                {"name": entity_name, "lim": max_chunks},
            )
            texts = [
                f"{r['src']} {r['rel'].lower().replace('_', ' ')} {r['dst']}"
                for r in rows
                if r.get("src") and r.get("dst")
            ]

        return texts

    def get_entities_by_type(self, entity_type: str) -> list[dict]:
        return self.client.execute_read(
            """
            MATCH (e:_Embeddable {entity_type: $entity_type})
            RETURN e
            """,
            {"entity_type": entity_type},
        )

    def get_stats(self) -> dict:
        nodes = self.client.execute_read(
            "MATCH (n) RETURN labels(n)[0] as node_type, count(n) as count"
        )
        rels = self.client.execute_read(
            "MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count"
        )
        return {
            "nodes": {r["node_type"]: r["count"] for r in nodes},
            "relationships": {r["rel_type"]: r["count"] for r in rels},
        }

    def close(self) -> None:
        self.client.close()
```

- [ ] **Step 9: Run tests to verify they pass**

```
pytest tests/unit/test_cs_graph.py -v -k "cs_kg"
```
Expected: all 7 tests PASS

- [ ] **Step 10: Commit**

```bash
git add src/graph/cs_knowledge_graph.py tests/unit/test_cs_graph.py
git commit -m "feat: add CSKnowledgeGraph targeting _Embeddable nodes for arXiv domain"
```

---

## Task 3: Fix `graph_retriever.py` imports

**Files:**
- Modify: `src/retrieval/graph_retriever.py`

Changes:
1. Remove `from src.graph.knowledge_graph import KnowledgeGraph` (line 7)
2. Remove `from src.graph.entity_extractor import MedicalEntityExtractor, get_entity_extractor` (line 8)
3. Replace `Optional[KnowledgeGraph]` / `Optional[MedicalEntityExtractor]` type hints with `Optional[Any]`
4. Replace `knowledge_graph or KnowledgeGraph()` / `entity_extractor or get_entity_extractor()` defaults with explicit `ValueError` raises — `hybridrag_pipeline.py` always injects both; default construction would silently load the wrong domain objects
5. Simplify `get_graph_retriever()` — remove the broken singleton pattern (requires arguments now, so a module-level singleton cached on first call with arbitrary args is wrong); just construct and return

- [ ] **Step 11: Check existing test baseline**

```
pytest tests/unit/test_retrieval.py -v
```
Note results. The `test_graph_retrieval` test uses `@patch('src.retrieval.graph_retriever.KnowledgeGraph')` (line 28) which will fail after the import is removed. Fix this in Step 14.

- [ ] **Step 12: Edit `graph_retriever.py` — imports and `__init__`**

Remove lines 7-8 entirely:
```python
# DELETE:
from src.graph.knowledge_graph import KnowledgeGraph
from src.graph.entity_extractor import MedicalEntityExtractor, get_entity_extractor
```

Replace the `__init__` method:
```python
# OLD:
def __init__(
    self,
    knowledge_graph: Optional[KnowledgeGraph] = None,
    entity_extractor: Optional[MedicalEntityExtractor] = None
):
    self.kg = knowledge_graph or KnowledgeGraph()
    self.entity_extractor = entity_extractor or get_entity_extractor()

# NEW:
def __init__(
    self,
    knowledge_graph: Optional[Any] = None,
    entity_extractor: Optional[Any] = None,
):
    if knowledge_graph is None:
        raise ValueError("GraphRetriever requires a knowledge_graph instance")
    if entity_extractor is None:
        raise ValueError("GraphRetriever requires an entity_extractor instance")
    self.kg = knowledge_graph
    self.entity_extractor = entity_extractor
```

Replace `get_graph_retriever()` at the bottom of the file:
```python
# OLD (broken singleton with required args):
_default_retriever: Optional[GraphRetriever] = None

def get_graph_retriever(
    knowledge_graph: Optional[KnowledgeGraph] = None,
    entity_extractor: Optional[MedicalEntityExtractor] = None
) -> GraphRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = GraphRetriever(knowledge_graph, entity_extractor)
    return _default_retriever

# NEW (simple factory — no singleton, args now required):
def get_graph_retriever(
    knowledge_graph: Any,
    entity_extractor: Any,
) -> GraphRetriever:
    return GraphRetriever(knowledge_graph, entity_extractor)
```

- [ ] **Step 13: Fix `test_retrieval.py` — broken patch and wrong method name**

In `tests/unit/test_retrieval.py`, the `test_graph_retrieval` test has two issues after this change:

**Issue A** — `@patch('src.retrieval.graph_retriever.KnowledgeGraph')` at line 28: the import is removed so the patch target no longer exists.

**Issue B** — `mock_extractor.extract.return_value.entities` at line 42: `graph_retriever.py:58` calls `self.entity_extractor.extract_entities(query)`, not `.extract()`. The mock should configure `.extract_entities.return_value`.

Replace the `test_graph_retrieval` function:
```python
# REMOVE the @patch decorator — KnowledgeGraph is no longer imported in graph_retriever
def test_graph_retrieval():
    from src.retrieval.graph_retriever import GraphRetriever

    # Mock KG
    kg_instance = MagicMock()
    kg_instance.get_chunks_for_entity.return_value = [
        {"c": {"text": "Chunk 1", "metadata": {"source": "src1"}}}
    ]
    kg_instance.get_related_entities.return_value = []
    kg_instance.get_entity_context.return_value = ["Chunk 1"]

    # Mock extractor — must use extract_entities (not extract)
    mock_extractor = MagicMock()
    mock_entity = MagicMock()
    mock_entity.text = "transformer"
    mock_extractor.extract_entities.return_value.entities = [mock_entity]

    retriever = GraphRetriever(knowledge_graph=kg_instance, entity_extractor=mock_extractor)
    results = retriever.retrieve("What is a transformer?")

    assert "Chunk 1" in results.contexts
    assert "transformer" in results.entities_found
```

- [ ] **Step 14: Run all retrieval tests**

```
pytest tests/unit/test_retrieval.py tests/unit/test_cs_graph.py -v
```
Expected: all pass

- [ ] **Step 15: Commit**

```bash
git add src/retrieval/graph_retriever.py tests/unit/test_retrieval.py
git commit -m "fix: remove medical domain imports from GraphRetriever; require injected dependencies"
```

---

## Task 4: Fix `hybrid_retriever.py`

**Files:**
- Modify: `src/retrieval/hybrid_retriever.py`

Two changes:
1. Remove `from src.graph.entity_extractor import get_entity_extractor` (line 10) — unused import
2. Remove `self.entity_extractor = get_entity_extractor()` (line 71) — field is never read anywhere in the class
3. Fix `_get_graph_retriever()` (lines 79-83): currently calls `GraphRetriever()` with no args, which now raises `ValueError`. `hybridrag_pipeline.py` always passes `graph_retriever=` to `HybridRetriever.__init__`, so the lazy fallback is never needed; raise clearly instead of silently constructing the wrong thing.

- [ ] **Step 16: Edit `hybrid_retriever.py`**

Remove line 10:
```python
# DELETE:
from src.graph.entity_extractor import get_entity_extractor
```

Remove line 71:
```python
# DELETE:
self.entity_extractor = get_entity_extractor()
```

Replace `_get_graph_retriever()`:
```python
# OLD:
def _get_graph_retriever(self) -> GraphRetriever:
    if self.graph_retriever is None:
        self.graph_retriever = GraphRetriever()
    return self.graph_retriever

# NEW:
def _get_graph_retriever(self) -> GraphRetriever:
    if self.graph_retriever is None:
        raise ValueError("HybridRetriever requires a graph_retriever instance")
    return self.graph_retriever
```

- [ ] **Step 17: Verify no stale references**

```
grep -n "entity_extractor\|get_entity_extractor" src/retrieval/hybrid_retriever.py
```
Expected: no output

- [ ] **Step 18: Run all unit tests**

```
pytest tests/unit/ -v
```
Expected: all pass

- [ ] **Step 19: Commit**

```bash
git add src/retrieval/hybrid_retriever.py
git commit -m "fix: remove unused medical entity extractor from HybridRetriever"
```

---

## Task 5: Smoke-test the full query path

- [ ] **Step 20: Verify the import chain resolves**

```
python -c "from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline; print('imports OK')"
```
Expected: `imports OK`

- [ ] **Step 21: Run a live query (requires Neo4j + Qdrant running)**

```
python query_hybridrag.py "what is the attention mechanism in transformers" --vector-only
```
`--vector-only` skips graph retrieval so it works even if Neo4j is empty.
Expected: answer printed without crash.

- [ ] **Step 22: Final commit if any fixups were needed**

```bash
git add -p
git commit -m "fix: resolve remaining import issues from smoke test"
```

---

## Rollback Notes

- `graph_retriever.py` and `hybrid_retriever.py` changes are removals only — no logic changes.
- The two new files are purely additive.
- Breaking the old medical pipeline is expected and acceptable (it is already dead code).
