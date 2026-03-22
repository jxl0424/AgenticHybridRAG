# HybridRAG Observability — Retrieval Quality per Query

**Date:** 2026-03-22
**Scope:** Eval runs only (`tests/evaluation/hybridrag_eval.py`)
**Phase 1:** Retrieval quality per query
**Phase 2 (deferred):** Answer quality over time

---

## Goal

Surface per-query retrieval quality in a local web UI during evaluation runs so that underperforming queries can be inspected stage-by-stage: which retriever fired, what scores came back, how many results the reranker dropped, and what entities were extracted.

---

## Approach

Arize Phoenix (local, free, open source) running on `localhost:6006`. Each eval query produces one OpenTelemetry trace with child spans per pipeline stage. Phoenix renders the span tree, timing breakdowns, and score distributions without any cloud dependency.

---

## Architecture

### New file

**`src/observability/tracer.py`**

Two responsibilities:
1. `start_phoenix()` — launches the Phoenix server (idempotent; reuses existing instance) and configures the OTel exporter to send to it. Returns a configured `opentelemetry.trace.Tracer`.
2. `pipeline_span(tracer, name, attrs)` — a `@contextmanager` that opens a named OTel span, sets attributes from `attrs`, and closes it. Returns a no-op context if `tracer` is `None`, so calling code in `query()` is always safe.

### Modified files

**`src/pipelines/hybridrag_pipeline.py`** — `query()` only

`time.perf_counter()` is added around each stage call. After each call the existing `trace` dict data plus timing is written into a child span via `pipeline_span`. The individual retriever classes (`ChunkRetriever`, `PaperRetriever`, `GraphRetriever`, `HybridRetriever`) are not modified.

`query()` gains an optional `tracer` parameter (default `None`). When `None`, all `pipeline_span` calls are no-ops — interactive use via `query_hybridrag.py` is unaffected.

**`tests/evaluation/hybridrag_eval.py`** — eval loop only

`start_phoenix()` is called once at the top of the eval run. Each QA pair is wrapped in a root span. After `pipeline.query()` returns, retrieval eval metrics (hit rate, MRR, NDCG, context precision) are attached to the root span as attributes. This lets Phoenix's table view sort all queries by metric.

---

## Span tree

```
query  [root]
├── embed_query
├── chunk_retriever
├── paper_retriever
├── graph_retriever
│   ├── entity_extraction
│   └── chunk_fetch
├── rrf_fuse
├── rerank
└── llm_generate
```

---

## Span attributes

| Span | Attributes |
|------|-----------|
| `query` | `query.text`, `retrieval.mode`, `retrieval.top_k`, `answer.type`, `answer.text`, `eval.hit_rate`, `eval.mrr`, `eval.ndcg`, `eval.context_precision` |
| `embed_query` | `embedding.model`, `latency_ms` |
| `chunk_retriever` | `chunk.raw_count`, `chunk.top_score`, `latency_ms` |
| `paper_retriever` | `paper.raw_count`, `paper.top_score`, `latency_ms` |
| `graph_retriever` | `graph.entity_count`, `graph.total_qdrant_ids`, `graph.fetched_count`, `latency_ms` |
| `entity_extraction` | `entities_extracted` (JSON-encoded list) |
| `chunk_fetch` | `ids_requested`, `ids_resolved` |
| `rrf_fuse` | `fused.count`, `fused.source_breakdown` (JSON), `top_rrf_scores` (JSON) |
| `rerank` | `pre_count`, `post_count`, `dropped`, `top_scores` (JSON), `latency_ms` |
| `llm_generate` | `llm.model`, `latency_ms` |

---

## Trace persistence

Phoenix stores traces in a local SQLite database across runs. Eval runs from different dates are queryable by time range in the UI. No data leaves the machine.

---

## No-op guarantee

When no tracer is active (e.g. `query_hybridrag.py` interactive use), all `pipeline_span` calls are no-ops. Zero overhead, zero behaviour change outside eval runs.

---

## New dependencies

```
arize-phoenix
opentelemetry-sdk
opentelemetry-exporter-otlp-proto-grpc
```

---

## Out of scope (Phase 2)

Answer quality over time (faithfulness, answer correctness trends) is explicitly deferred. The span structure above leaves room for `eval.faithfulness` and `eval.answer_correctness` to be added to the root span in a later phase without changing the instrumentation architecture.

---

## Files changed summary

| File | Change |
|------|--------|
| `src/observability/__init__.py` | New (empty) |
| `src/observability/tracer.py` | New — Phoenix setup + span helper |
| `src/pipelines/hybridrag_pipeline.py` | `query()` gains optional `tracer` param + timing + child spans |
| `tests/evaluation/hybridrag_eval.py` | `start_phoenix()` at startup + root span per query + eval metric attrs |
