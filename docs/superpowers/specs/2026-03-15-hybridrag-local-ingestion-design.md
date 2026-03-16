# HybridRAG Local Ingestion Pipeline — Design Spec
**Date:** 2026-03-15
**Status:** Approved

---

## 1. Overview

Replace the existing HuggingFace-streaming ingestion with a local parquet-based pipeline that performs a clean load of the HybridRAG-Bench dataset into Neo4j and three Qdrant collections. The pipeline supports domain selection, resume, and an explicit `--reset` flag for clean ingestion.

---

## 2. Source Data

All data lives under `data/hybridrag/`. Three domain subsets: `arxiv_ai`, `arxiv_cy`, `arxiv_qm`.

```
data/hybridrag/
  kg/{domain}/
    nodes.parquet           # entity nodes with pre-computed 768-dim SPECTER2 embeddings
    edges.parquet           # typed relationships (edge_id, src_id, dst_id, rel_type, properties_json)
    edge_properties.parquet # long-format (edge_id, key, value): core keys are
                            #   _paragraph, _embedding, _description, _ref; + optional extra props
    node_properties.parquet # long-format (node_id, key, value) — present but NOT ingested
    schema.json             # entity label list + relationship type list
    constraints.cypher      # Neo4j constraint DDL (reference only)
    indexes.cypher          # Neo4j index DDL (reference only)
  text_qa/{domain}/
    papers.parquet          # 100-108 papers per domain: arxiv_id, title, md_text, metadata
    qa.parquet              # benchmark QA pairs (evaluation only, NOT ingested)
```

**Totals across all 3 domains:** ~50K nodes (`_Embeddable` only), ~41K edges, 314 papers.

**Node label structure:** The `nodes.parquet` `primary_label` column always contains one of
`_Embeddable`, `_RelationSchema`, or `_EntitySchema` — not the specific entity type.
The actual entity type (e.g. `Concept`, `Dataset`, `Method`) is `labels[1]` — the second
element of the `labels` array column. Only `_Embeddable` rows are ingested.

---

## 3. Target Stores

### 3.1 Neo4j

| Node/Relationship | Label(s) / Type | Key Properties |
|---|---|---|
| Entity node | `_Embeddable` + `labels[1]` (e.g. `:_Embeddable:Concept`) | `node_id` (int), `domain` (str), `display_name` (str), `entity_type` (str, = `labels[1]`) |
| Relationship | `{sanitised rel_type}` (alnum + underscore, uppercased) | `edge_id` (int), `domain` (str) |
| Paper node | `:Paper` | `arxiv_id` (str), `title` (str), `domain` (str), `categories` (str), `published` (str) |

**Sanitise `rel_type`:** `"".join(c for c in rel_type if c.isalnum() or c == "_").upper()` — same
logic as existing `ingest_native_kg`. Store the sanitised value on the relationship AND in the
`arxiv_chunks` Qdrant payload so they always match exactly.

**Neo4j node properties written from `properties_json`:** extract only `name` (as `display_name`).
Do NOT write `_paragraph`, `_description`, `_ref`, or `_embedding` as node properties.

**Indexes (created before ingestion begins):**
- Compound index on `(:_Embeddable {node_id, domain})` — enables O(1) lookup from Qdrant node hits
- Relationship property index on `edge_id` per type — created **after** edges are loaded, by
  enumerating distinct `rel_type` values from `edges.parquet` at that point in the sequence
- Index on `(:Paper {arxiv_id})` — used for paper node lookups

**Write strategy:** `CREATE` (not `MERGE`) — the `--reset` wipe guarantees no duplicates.

**Dynamic label creation:** Cypher does not support variable label names in `CREATE` patterns.
Use Python f-string interpolation on the sanitised label string (same pattern as the existing
`ingest_native_kg` method in `hybridrag_pipeline.py`, line 354):
```python
session.run(f"UNWIND $batch AS n CREATE (:{entity_type}:_Embeddable {{...}})", batch=batch)
```
`entity_type` is sourced from `labels[1]` and sanitised (alnum + underscore) before injection.

### 3.2 Qdrant Collections

All three collections: **768-dim vectors**, **Cosine distance**.

**uuid5 namespace constant (fixed, must be committed to source):**
```python
import uuid
HYBRIDRAG_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
```
This constant is used for all point ID generation. It must never change after first ingestion
(changing it would generate different IDs for existing records, breaking upsert idempotency).

#### `arxiv_nodes`
- **Source:** `kg/{domain}/nodes.parquet`, rows where `primary_label == "_Embeddable"` only
- **Vector:** `json.loads(properties_json)["_embedding"]` (pre-computed, no re-embedding needed)
- **Point ID:** `uuid.uuid5(HYBRIDRAG_NS, f"{domain}:{node_id}")`
- **Payload:** `node_id` (int), `domain` (str), `display_name` (str), `entity_type` (str, from `labels[1]`)
- **Payload indexes:** `domain`, `entity_type`

#### `arxiv_chunks`
- **Source:** `edge_properties.parquet` pivoted by `edge_id` (core keys: `_paragraph`, `_embedding`,
  `_description`, `_ref`) **joined with** `edges.parquet` on `edge_id` to obtain `src_id`, `dst_id`,
  `rel_type` (sanitised)
- **Vector:** `_embedding` value (pre-computed, no re-embedding needed)
- **Point ID:** `uuid.uuid5(HYBRIDRAG_NS, f"{domain}:{edge_id}")`
- **Payload:**
  - `edge_id` (int)
  - `src_id` (int), `dst_id` (int)
  - `src_name` (str, optional — `display_name` of source node), `dst_name` (str, optional)
  - `rel_type` (str, sanitised — matches Neo4j exactly)
  - `paragraph` (str — the `_paragraph` value)
  - `domain` (str)
  - `paper_id` (str — from `_ref` JSON key `"id"`; local parquet files use `"id"` exclusively)
- **Payload indexes:** `domain`, `rel_type`, `src_id`, `dst_id`

#### `arxiv_papers`
- **Source:** `text_qa/{domain}/papers.parquet`, `md_text` field
- **Chunking:** split `md_text` on `\n\n`, merge short paragraphs forward until `min_tokens=128`
  is reached, hard cap at `max_tokens=512`. Chunk index assigned in stable document order.
- **Vector:** SPECTER2 (`allenai/specter2_base`) embedding of each chunk, computed at ingest time,
  batch size 64
- **Point ID:** `uuid.uuid5(HYBRIDRAG_NS, f"{arxiv_id}:{chunk_index}")`
- **Payload:** `arxiv_id` (str), `chunk_index` (int), `chunk_text` (str), `domain` (str), `title` (str)
- **Payload indexes:** `domain`, `arxiv_id`

---

## 4. New Files

```
src/ingestion/local_parquet_loader.py      # reads parquet files, yields typed records
src/pipelines/local_ingestion_pipeline.py  # orchestrates clear + load
ingest_local.py                            # CLI entry point
```

No existing files are modified.

---

## 5. Component Responsibilities

### `LocalParquetLoader`
- Reads all parquet files from local `data/hybridrag/` paths
- Validates file existence and required columns before any yields
- Pivots `edge_properties.parquet` by `edge_id`, then joins with `edges.parquet` on `edge_id`
  to attach `src_id`, `dst_id`, `rel_type` to each chunk record
- Builds an in-memory `node_map: dict[int, str]` of `{node_id: display_name}` per domain,
  used to populate optional `src_name`/`dst_name` in chunk payloads
- Has zero DB knowledge — pure data access layer

### `LocalIngestionPipeline`
- Wires `LocalParquetLoader` → Neo4j → Qdrant
- Handles: clear, index creation, batch writes, progress tracking, resume
- Batching: Neo4j writes 500 records/batch; Qdrant upserts 256 points/batch
- `--batch-size N` controls the Neo4j write batch size only; Qdrant batch size is fixed at 256
- Failure policy:
  - Node/edge batch failure → abort the entire domain phase, mark it failed in progress file
  - Paper/chunk batch failure → log structured record, skip batch, continue; phase still marked `true` on completion
- Paper phase: SPECTER2 model loaded once and cached via `embed_texts_with_model`

### `ingest_local.py` (CLI)
```
python ingest_local.py [--reset] [--domains arxiv_ai arxiv_cy arxiv_qm]
                       [--src-names] [--batch-size 500]
```
- `--reset`: wipe Neo4j (`MATCH (n) DETACH DELETE n`) + drop and recreate all 3 Qdrant
  collections + delete progress file. Gated explicitly — no accidental data loss.
  **Warning:** `DETACH DELETE` removes ALL nodes and relationships in the database, including
  any data from other pipelines (e.g. CSEntity nodes from `HybridRAGBenchPipeline`). This is
  intentional for a clean ingestion run. Do not use `--reset` if other pipeline data must be preserved.
- `--domains`: subset of domains to process (default: all three)
- `--src-names`: populate optional `src_name`/`dst_name` in `arxiv_chunks` payload
- `--batch-size`: Neo4j write batch size (default 500)

---

## 6. Ingestion Sequence

```
startup
  1. Validate all expected parquet files exist + required columns present
       Expected files per domain: nodes, edges, edge_properties (kg/); papers (text_qa/)
       node_properties.parquet is present in source but is not validated or ingested
  2. Ping Neo4j + Qdrant — fail immediately if either is unreachable
  3. Dry-run import check for SPECTER2 model (import sentence_transformers)
  4. If --reset:
       - Neo4j: MATCH (n) DETACH DELETE n
       - Qdrant: drop arxiv_nodes, arxiv_chunks, arxiv_papers; recreate with 768-dim Cosine
       - Delete data/hybridrag/.local_ingest_progress.json
  5. Create Neo4j indexes (if not exist):
       - (:_Embeddable {node_id, domain})
       - (:Paper {arxiv_id})
       Note: relationship edge_id index is created after phase 2 (edges), once rel_types are known
  6. Create Qdrant payload indexes on all three collections

per domain (in order: arxiv_ai, arxiv_cy, arxiv_qm), skip if already complete in progress file

  phase 1 — nodes + papers:
    - READ nodes.parquet; filter to primary_label == "_Embeddable"
    - For each batch: CREATE (:_Embeddable:{labels[1]} {node_id, domain, display_name, entity_type})
    - UPSERT arxiv_nodes Qdrant (uuid5 point IDs, pre-computed embeddings)
    - READ papers.parquet; for each paper:
        CREATE (:Paper {arxiv_id, title, domain, categories, published})
    - On any batch failure: abort domain, mark phase failed in progress file

  phase 2 — edges:
    - READ edges.parquet; sanitise rel_type
    - For each batch grouped by rel_type:
        MATCH (src:_Embeddable {node_id: $src_id, domain: $domain})
        MATCH (dst:_Embeddable {node_id: $dst_id, domain: $domain})
        CREATE (src)-[:{rel_type} {edge_id: $edge_id, domain: $domain}]->(dst)
    - On any batch failure: abort domain, mark phase failed in progress file

  post-phase 2 — relationship indexes:
    - Enumerate distinct rel_types from edges.parquet
    - CREATE INDEX FOR ()-[r:{rel_type}]-() ON (r.edge_id) for each type

  phase 3 — chunks:
    - READ edge_properties.parquet; pivot by edge_id; join with edges.parquet on edge_id
    - If --src-names: look up src_name / dst_name from in-memory node_map
    - UPSERT arxiv_chunks Qdrant (uuid5 point IDs, pre-computed embeddings)
    - On batch failure: log structured entry, skip batch, continue

  phase 4 — papers (embedding):
    - READ papers.parquet; chunk md_text (min 128 tokens, max 512 tokens)
    - Embed with SPECTER2 in batches of 64
    - UPSERT arxiv_papers Qdrant (uuid5 point IDs)
    - On batch failure: log structured entry, skip batch, continue

completion
  - Mark all completed phases true in progress file
  - Print summary: nodes, edges, chunks, paper-chunks written; batches skipped
```

---

## 7. Progress File

Location: `data/hybridrag/.local_ingest_progress.json`
(Distinct from the existing `HybridRAGBenchPipeline`'s `.ingest_progress.json`.)

```json
{
  "arxiv_ai":  {"nodes": true,  "edges": true,  "chunks": true,  "papers": false},
  "arxiv_cy":  {"nodes": false, "edges": false, "chunks": false, "papers": false},
  "arxiv_qm":  {"nodes": false, "edges": false, "chunks": false, "papers": false},
  "failed_batches": [
    {
      "domain": "arxiv_ai",
      "phase": "papers",
      "batch_index": 3,
      "error": "CUDA OOM",
      "timestamp": "2026-03-15T10:23:00Z",
      "record_ids": ["1111.3735v1", "2210.02769v1"]
    }
  ]
}
```

**Resume semantics:** A phase is marked `true` upon completion of the full iteration over its
source data, regardless of skipped batches. Entries in `failed_batches` are not retried on
resume — they require a `--reset` run to re-attempt. Phases marked `true` are skipped on resume.

---

## 8. Startup Validation

Before any writes, the pipeline checks:

| Check | Failure action |
|---|---|
| 12 expected parquet files exist (`kg/{domain}/(nodes, edges, edge_properties).parquet` × 3 + `text_qa/{domain}/papers.parquet` × 3) | Halt with per-file list |
| Required columns present in each file | Halt with per-file column diff |
| Neo4j reachable (connection ping) | Halt immediately |
| Qdrant reachable (connection ping) | Halt immediately |
| `sentence_transformers` importable | Halt with install hint |

Note: `node_properties.parquet` is present in the source data but is intentionally not validated
or ingested.

---

## 9. Retrieval Integration (query-time contract)

This spec defines what is stored; retrieval code updates are out of scope but must respect this
contract.

**Graph lookup from `arxiv_nodes` hit (direct O(1) via compound index):**
```cypher
MATCH (n:_Embeddable {node_id: $node_id, domain: $domain})
RETURN n
```

**Graph lookup from `arxiv_chunks` hit (direct 1-hop, no variable-length traversal):**
```cypher
MATCH (src:_Embeddable {node_id: $src_id, domain: $domain})-[r]->(dst:_Embeddable {node_id: $dst_id, domain: $domain})
WHERE r.edge_id = $edge_id
RETURN src, r, dst
```
Neighbourhood expansion (2-hop) is a separate optional step after anchoring — never part of
the initial lookup.

**Vector similarity scores:** Qdrant scores propagate through the retrieval context object to the
reranker/RRF fusion step. Scores are never discarded.

---

## 10. Out of Scope

- Updates to `HybridRetriever` / `GraphRetriever` to use 3 collections (separate task)
- QA evaluation pipeline using `qa.parquet` (separate task)
- Additional entity extraction from `md_text` to enrich the KG (future improvement)
- Overlap/stride chunking for `arxiv_papers` (can be added later)
- Ingesting `node_properties.parquet` or `constraints.cypher` / `indexes.cypher` directly
