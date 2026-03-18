# HybridRAG Retrieval Fixes Design

**Date:** 2026-03-18
**Status:** Approved for implementation

## Context

Post-evaluation diagnostics identified three root causes behind poor evaluation scores:

1. The query pipeline targets the `"arxiv"` Qdrant collection (stale/wrong) instead of the locally-ingested `arxiv_chunks` and `arxiv_papers` collections. `QdrantStorage.search` also reads `payload["text"]` while local ingestion stores `payload["paragraph"]`.
2. Neo4j `_Embeddable` nodes have no links to their source text chunks, so graph traversal reaches entity topology but cannot resolve actual text.
3. Residual medical-domain logic in `GraphRetriever` and `RAGMetrics` causes incorrect fallback behaviour and misleading LLM prompts for a CS/arXiv research domain.

---

## Goals

- Point the query pipeline at the correct Qdrant collections with correct field mappings
- Enable graph traversal to reach text via `HAS_CHUNK` references
- Remove all medical-domain assumptions from retrieval and evaluation logic
- Preserve collection type information through fusion and reranking

---

## Design

### 1. Domain Cleanup (Priority 1)

**`src/retrieval/graph_retriever.py`**

- Replace `_keyword_fallback` body: currently searches `:Chunk` nodes (which do not exist in the local ingestion schema). New implementation searches `_Embeddable` nodes by `display_name` keyword matching:

```cypher
MATCH (n:_Embeddable)
WHERE any(kw IN $keywords WHERE toLower(n.display_name) CONTAINS kw)
RETURN n.display_name AS name, n.entity_type AS type
LIMIT $lim
```

The keywords list must be passed as a Cypher parameter (`$keywords`), never via f-string interpolation. The current implementation uses f-string keyword injection (a Cypher injection vector); the replacement must not replicate this.

- Update all comments: "medical" → "CS/arXiv research", "No medical entities found" → "No CS entities found"
- No changes to entity extraction logic — `CSEntityExtractor` is already domain-correct

**`tests/evaluation/metrics.py`**

- `calculate_relevance` prompt: replace "medical query" with "research query" and "professional medical knowledge" with "scientific and technical knowledge"
- Grep full codebase for remaining `medical` / `clinical` / `patient` references and update to domain-neutral language

---

### 2. Typed Retrieval Layer

#### 2a. `RetrievedContext` dataclass — `src/types.py`

Add alongside existing Pydantic models:

```python
from dataclasses import dataclass, field

@dataclass
class RetrievedContext:
    text: str
    source: str          # arxiv_id, file path, or "graph"
    score: float
    collection: str      # "arxiv_chunks" | "arxiv_papers" | "graph"
    metadata: dict = field(default_factory=dict)
    # arxiv_chunks: edge_id, src_id, dst_id, rel_type, domain
    # arxiv_papers: arxiv_id, chunk_index, title, domain
    # graph:        entities_found, traversal_depth
```

#### 2b. `ChunkRetriever` — `src/retrieval/chunk_retriever.py`

- Wraps `QdrantStorage(collection="arxiv_chunks", dim=768)`
- `search(query_vector, top_k, min_score) -> list[RetrievedContext]`
- Maps `payload["paragraph"]` → `text`
- Metadata: `edge_id`, `src_id`, `dst_id`, `rel_type`, `domain`, `paper_id`
- `collection = "arxiv_chunks"`

Also exposes:
- `fetch_by_ids(qdrant_ids: list[str]) -> list[RetrievedContext]` — used by `GraphRetriever` to resolve `HAS_CHUNK` references

Note: `qdrant_id` values are stored as UUID strings (e.g. `"a1b2c3d4-..."`) both on Neo4j `HAS_CHUNK` edges and as Qdrant point IDs. All code that computes or passes `qdrant_id` must call `str(uuid_obj)` explicitly before writing to Neo4j or Qdrant to prevent silent type mismatch.

#### 2c. `PaperRetriever` — `src/retrieval/paper_retriever.py`

- Wraps `QdrantStorage(collection="arxiv_papers", dim=768)`
- `search(query_vector, top_k, min_score) -> list[RetrievedContext]`
- Maps `payload["chunk_text"]` → `text`
- Metadata: `arxiv_id`, `chunk_index`, `title`, `domain`
- `collection = "arxiv_papers"`

#### 2d. `GraphRetriever` updates — `src/retrieval/graph_retriever.py`

- Constructor gains an optional `chunk_retriever: ChunkRetriever` parameter
- `retrieve()` return type changes from `GraphRetrievalResult` to `list[RetrievedContext]`
- After graph traversal collects `qdrant_id` values from `HAS_CHUNK` edges, calls `chunk_retriever.fetch_by_ids(qdrant_ids)` to resolve text
- Falls back to `_keyword_fallback` (now searching `_Embeddable.display_name`) when no entities found
- All returned contexts have `collection = "graph"`
- `retrieve_by_entity()` currently calls `self.kg.get_chunks_for_entity()` which looks for `(:Chunk)-[:HAS_ENTITY]->` nodes that do not exist in the local ingestion schema. This method must be re-implemented to traverse `HAS_CHUNK` edges and resolve via `chunk_retriever.fetch_by_ids()`, or removed if unused by the pipeline.
- The inner `:Chunk` fallback inside `retrieve()` (lines ~109–120: `MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS...`) must be removed — `:Chunk` nodes do not exist in the local schema. The updated `_keyword_fallback` on `_Embeddable.display_name` is the sole fallback path.

#### 2e. `HybridRetriever` updates — `src/retrieval/hybrid_retriever.py`

- Constructor replaces single `qdrant_storage` with `chunk_retriever: ChunkRetriever` and `paper_retriever: PaperRetriever`; retains `graph_retriever: GraphRetriever`
- `retrieve()` calls all three retrievers, collects `list[RetrievedContext]` from each
- `_fuse_results` applies per-source weighted RRF:

```
rrf_score += collection_weight * (1 / (RRF_K + rank + 1))
```

Config weights (new keys in `config/defaults.yaml` under `hybrid_retrieval`):
```yaml
chunk_weight: 0.5
paper_weight: 0.3
graph_weight: 0.2
```

- Deduplication key: normalised `text` (strip whitespace, lowercase) — exact duplicates from different collection paths (e.g. graph traversal + direct vector search hitting the same `arxiv_chunks` point) boost score rather than producing duplicate results
- `arxiv_chunks` and `arxiv_papers` store structurally different text sources (`_paragraph` from KG edges vs. raw paper markdown chunks) and will rarely produce exact string matches. Cross-collection semantic overlap is handled by the reranker downstream, not by dedup.
- Returns `list[RetrievedContext]` preserving `collection` field throughout

#### 2f. `HybridRAGBenchPipeline` updates — `src/pipelines/hybridrag_pipeline.py`

- Replace `QdrantStorage` + `HybridRetriever` init with `ChunkRetriever`, `PaperRetriever`, updated `GraphRetriever`, updated `HybridRetriever`
- `query()` passes `list[RetrievedContext]` to reranker; extracts `.text` for LLM context window
- Vector-only mode (`use_hybrid=False`): runs `ChunkRetriever` + `PaperRetriever` only, fuses results with RRF (no graph)
- Remove the now-unused direct `qdrant.search()` call

---

### 3. `HAS_CHUNK` Edges

#### 3a. Migration script — `scripts/migrate_has_chunk_edges.py`

Standalone script for already-ingested data. Reads parquet files (same source as ingestion), recomputes `qdrant_id = uuid5(HYBRIDRAG_NS, f"{domain}:{edge_id}")`, and writes edges:

```cypher
UNWIND $batch AS c
MATCH (src:_Embeddable {node_id: c.src_id, domain: c.domain})
MATCH (dst:_Embeddable {node_id: c.dst_id, domain: c.domain})
MERGE (src)-[r:HAS_CHUNK {edge_id: c.edge_id, domain: c.domain}]->(dst)
SET r.qdrant_id = c.qdrant_id
```

Uses `MERGE` not `CREATE` — idempotent, safe to re-run. Uses unconditional `SET` (not `ON CREATE SET`) to ensure `qdrant_id` is always populated even if the edge was previously created without it in a partial run. `c.qdrant_id` must be the pre-stringified UUID value (`str(uuid_obj)`).

#### 3b. `LocalIngestionPipeline._ingest_chunks` — `src/pipelines/local_ingestion_pipeline.py`

After each Qdrant upsert batch, write `HAS_CHUNK` edges for the same batch using the same `MERGE` Cypher above. This ensures future ingestion runs are always consistent.

The Neo4j write must be wrapped in its own `try/except` with `progress.log_batch_failure` logging — a Qdrant upsert success followed by a Neo4j failure must not silently mark the chunks phase as done with missing edges. The migration script provides the safety net for fixing any edges missed in a partial run.

---

### 4. Config Changes — `config/defaults.yaml`

```yaml
hybrid_retrieval:
  chunk_weight: 0.5
  paper_weight: 0.3
  graph_weight: 0.2
  top_k: 5
  min_score: 0.2
```

Remove the now-redundant `vector_weight` / `graph_weight` keys (replaced by three-way weights). Also remove `hybridrag.qdrant_collection` (`"arxiv"`) — this key is no longer used since collection names are now hardcoded in `ChunkRetriever` and `PaperRetriever`. These config removals must be made atomically with the pipeline and retriever constructor changes — old keys must not linger as silent fallback defaults.

---

## File Changelist

| File | Change |
|------|--------|
| `src/types.py` | Add `RetrievedContext` dataclass |
| `src/retrieval/chunk_retriever.py` | New — `ChunkRetriever` |
| `src/retrieval/paper_retriever.py` | New — `PaperRetriever` |
| `src/retrieval/graph_retriever.py` | Fix fallback, domain cleanup, return `list[RetrievedContext]`, inject `ChunkRetriever` |
| `src/retrieval/hybrid_retriever.py` | Use all three retrievers, per-source RRF weights, return `list[RetrievedContext]` |
| `src/pipelines/hybridrag_pipeline.py` | Wire new retriever stack, update `query()` |
| `src/pipelines/local_ingestion_pipeline.py` | Add `HAS_CHUNK` edge writes in `_ingest_chunks` |
| `scripts/migrate_has_chunk_edges.py` | New — one-shot migration for existing data |
| `tests/evaluation/metrics.py` | Strip medical domain from prompts |
| `config/defaults.yaml` | Replace 2-way with 3-way retrieval weights |

---

## Duplication Handling

| Source | Risk | Mitigation |
|--------|------|------------|
| `arxiv_chunks` + `arxiv_papers` semantic overlap | Different text, overlapping content | Reranker sorts by relevance; collections have different structural sources so exact dedup rarely fires |
| Graph + vector returning same chunk | `HAS_CHUNK` traversal and `ChunkRetriever` hit same Qdrant point | RRF dedup by normalised text — identical strings boost score instead of duplicating |
| `HAS_CHUNK` migration re-run | Duplicate Neo4j edges | `MERGE` + unconditional `SET` — idempotent |
| Partial migration leaving `qdrant_id` unset | `GraphRetriever` silently returns no text | Unconditional `SET` overwrites on re-run |

---

## `HybridRAGBenchPipeline.ingest()` — Deprecated

All ingestion is performed exclusively via `ingest_local.py` → `LocalIngestionPipeline`, which uses `uuid5(HYBRIDRAG_NS, f"{domain}:{edge_id}")` as Qdrant point IDs. `HybridRAGBenchPipeline.ingest()` uses `edge_id` directly as the point ID (int64), producing a different ID scheme. The two paths are incompatible.

`HybridRAGBenchPipeline.ingest()` is deprecated and must not be called. Its method body should be removed or replaced with a `raise NotImplementedError` guard in this change to prevent accidental use. The `query()` path is retained and updated.

---

## Out of Scope

- Replacing the LLM model (`qwen2.5:7b-instruct`) — separate concern
- Adding `ground_truth_context` to the eval harness — the dataset does not provide it; retrieval metrics (hit_rate, mrr, ndcg) require a separate annotation effort
