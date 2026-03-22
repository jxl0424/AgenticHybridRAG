# HybridRAG Observability — Retrieval Quality per Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument the HybridRAG eval harness with Arize Phoenix so every evaluation query produces a browsable span tree showing per-stage retrieval quality at `http://localhost:6006`.

**Architecture:** A new `src/observability/tracer.py` module owns Phoenix startup and a `pipeline_span` context manager. `HybridRAGBenchPipeline.query()` gains an optional `tracer` param and wraps each pipeline stage with timing spans. The eval harness starts Phoenix once, creates a root span per query, and attaches eval metrics to it after scoring. When `tracer=None` (interactive use), all spans are no-ops.

**Tech Stack:** `arize-phoenix[otel]`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-grpc`, `grpcio`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/observability/__init__.py` | Create | Empty package marker |
| `src/observability/tracer.py` | Create | `_NoOpSpan`, `pipeline_span`, `start_phoenix` |
| `src/pipelines/hybridrag_pipeline.py` | Modify | Add `tracer=None` to `query()`, wrap each stage with spans |
| `tests/evaluation/hybridrag_eval.py` | Modify | Start Phoenix in `__init__`, root span per query in `run()`, pass `tracer` to `_run_query` |
| `tests/unit/test_tracer.py` | Create | Unit tests for `_NoOpSpan` and `pipeline_span(None, ...)` |

---

## Task 1: Create `src/observability/` module

**Files:**
- Create: `src/observability/__init__.py`
- Create: `src/observability/tracer.py`
- Create: `tests/unit/test_tracer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tracer.py`:

```python
"""Unit tests for src/observability/tracer.py — no Phoenix server required."""
from src.observability.tracer import _NoOpSpan, pipeline_span, start_phoenix


def test_noop_span_set_attribute_does_not_raise():
    s = _NoOpSpan()
    s.set_attribute("str_key", "value")
    s.set_attribute("int_key", 42)
    s.set_attribute("float_key", 3.14)


def test_pipeline_span_yields_noop_when_tracer_is_none():
    with pipeline_span(None, "test_span") as span:
        assert isinstance(span, _NoOpSpan)
        span.set_attribute("x", 1)  # must not raise


def test_pipeline_span_noop_exits_cleanly():
    entered = False
    with pipeline_span(None, "test_span") as span:
        entered = True
    assert entered


def test_start_phoenix_returns_none_when_phoenix_not_importable(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "phoenix", None)
    # Re-importing after patching sys.modules triggers ImportError inside start_phoenix
    result = start_phoenix()
    assert result is None
```

- [ ] **Step 2: Verify tests fail**

```
pytest tests/unit/test_tracer.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.observability'`

- [ ] **Step 3: Create the package and implement `tracer.py`**

Create `src/observability/__init__.py` (empty):

```python
```

Create `src/observability/tracer.py`:

```python
"""
Arize Phoenix tracing utilities for the HybridRAG pipeline.

Public API:
    start_phoenix() -> tracer or None
    pipeline_span(tracer, name) -> context manager yielding a span or _NoOpSpan
"""
import json
import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger("rag.observability")


class _NoOpSpan:
    """Returned by pipeline_span when tracing is disabled. All methods are no-ops."""

    def set_attribute(self, key: str, value) -> None:  # noqa: ANN001
        pass


@contextmanager
def pipeline_span(tracer, name: str):
    """
    Open a named OTel span as a context manager.

    Yields _NoOpSpan (safe to call .set_attribute on) when tracer is None.
    When a tracer is active the yielded span is a live opentelemetry.trace.Span;
    it becomes a child of whatever span is currently active in the OTel context.
    """
    if tracer is None:
        yield _NoOpSpan()
        return
    with tracer.start_as_current_span(name) as span:
        yield span


def start_phoenix() -> Optional[object]:
    """
    Launch Arize Phoenix and configure the OTel OTLP/gRPC exporter.

    - Idempotent: if Phoenix is already running on localhost:6006 it is reused.
    - Returns a configured opentelemetry.trace.Tracer on success.
    - Returns None and logs a warning on any failure — never raises.
      The eval run continues without tracing when None is returned.
    """
    try:
        import phoenix as px
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        px.launch_app()

        exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        tracer = trace.get_tracer("hybridrag")
        logger.info("Phoenix running at http://localhost:6006")
        return tracer

    except Exception as exc:
        logger.warning("Phoenix could not start: %s — tracing disabled for this run", exc)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_tracer.py -v
```

Expected: 4 PASSED

Note: `test_start_phoenix_returns_none_when_phoenix_not_importable` will pass because `start_phoenix` catches all exceptions including `ImportError` caused by `sys.modules` patching.

- [ ] **Step 5: Commit**

```bash
git add src/observability/__init__.py src/observability/tracer.py tests/unit/test_tracer.py
git commit -m "feat: add observability module with Phoenix tracer and no-op span helper"
```

---

## Task 2: Instrument `HybridRAGBenchPipeline.query()`

**Files:**
- Modify: `src/pipelines/hybridrag_pipeline.py` — `query()` method only (lines 255–365)

**What changes:**
- Add `import time` to the module-level imports (line 14 area)
- Add `tracer=None` parameter to `query()`
- Wrap each pipeline stage with `pipeline_span` + timing
- Split the rerank block into `rerank_crossencoder` and `rerank_threshold` spans
- The function's return value and all existing logic are preserved exactly

- [ ] **Step 1: Write a regression test**

Add to `tests/unit/test_ingestion.py` or create `tests/unit/test_query_tracer_noop.py`:

```python
"""Verify query() returns the same shape dict when tracer=None (no-op path)."""
from unittest.mock import MagicMock, patch
import pytest


def _make_mock_context(text="ctx", source="arxiv_chunks", score=0.5):
    from src.types import RetrievedContext
    return RetrievedContext(text=text, source=source, score=score,
                            collection="arxiv_chunks", metadata={})


def test_query_tracer_none_returns_expected_keys():
    """query(tracer=None) must return dict with all standard keys — no functional change."""
    from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline

    pipeline = MagicMock(spec=HybridRAGBenchPipeline)

    # Simulate the actual query() call by calling the real method on a fully-mocked instance
    ctx = _make_mock_context()
    pipeline.hybrid_retriever.retrieve.return_value = [ctx]
    pipeline.graph_retriever._last_trace = {
        "entities_extracted": [], "qdrant_ids_per_entity": {},
        "total_qdrant_ids": 0, "fetched_count": 0,
    }
    pipeline.hybrid_retriever._last_trace = {
        "chunk_count_raw": 1, "paper_count_raw": 0, "graph_count_raw": 0,
        "fused_count": 1, "fused_source_breakdown": {"arxiv_chunks": 1},
        "top_rrf_scores": [0.5],
    }
    pipeline.hybrid_retriever.chunk_weight = 0.5
    pipeline.hybrid_retriever.paper_weight = 0.3
    pipeline.hybrid_retriever.graph_weight = 0.2
    pipeline.embedding_model = "allenai/specter2_base"
    pipeline.min_score = 0.05
    pipeline.reranker = None
    pipeline.llm.generate.return_value = "Test answer."
    pipeline.llm.model = "qwen2.5:7b-instruct"

    with patch("src.pipelines.hybridrag_pipeline.embed_texts_with_model",
               return_value=[[0.1] * 768]):
        with patch("src.pipelines.hybridrag_pipeline.build_messages", return_value=[]):
            result = HybridRAGBenchPipeline.query(pipeline, "test question", tracer=None)

    assert set(result.keys()) == {"question", "answer", "contexts", "sources",
                                   "scores", "entities_found", "retrieval_type", "trace"}
```

- [ ] **Step 2: Run test to verify it fails with TypeError**

```
pytest tests/unit/test_query_tracer_noop.py -v
```

Expected: FAIL with `TypeError: query() got an unexpected keyword argument 'tracer'` — the `tracer` parameter does not exist yet. This confirms the test exercises the new code path and will act as a green gate after Step 4.

- [ ] **Step 3: Add `import time` to hybridrag_pipeline.py**

In `src/pipelines/hybridrag_pipeline.py`, find the imports block (around line 13):

```python
import json
import os
import yaml
```

Add `import time` after `import json`:

```python
import json
import os
import time
import yaml
```

- [ ] **Step 4: Replace `query()` with the instrumented version**

Replace the entire `query()` method (lines 255–365 in `hybridrag_pipeline.py`) with:

```python
def query(self, question: str, top_k: int = 5, use_hybrid: bool = True, tracer=None) -> dict:
    """
    Query the CS/AI RAG system.

    Args:
        question: Natural language question
        top_k: Number of final contexts to return
        use_hybrid: If True, use vector + graph hybrid; otherwise vector only
        tracer: Optional OTel tracer from start_phoenix(). When None all span
                calls are no-ops and behaviour is identical to the untraced path.

    Returns:
        Dict with answer, contexts, sources, entities_found, retrieval_type, trace
    """
    from src.observability.tracer import pipeline_span

    # --- Embed ---
    with pipeline_span(tracer, "embed_query") as span:
        t0 = time.perf_counter()
        query_embedding = embed_texts_with_model(
            [question], self.embedding_model, batch_size=1
        )[0]
        span.set_attribute("embedding.model", self.embedding_model)
        span.set_attribute("embedding.vector_dim", len(query_embedding))
        span.set_attribute("latency_ms", round((time.perf_counter() - t0) * 1000, 2))

    # --- Retrieve (all three sources, timed as a unit) ---
    t0 = time.perf_counter()
    contexts_raw = self.hybrid_retriever.retrieve(
        query=question,
        query_embedding=query_embedding,
        top_k=top_k * 3,
        min_score=self.min_score,
        use_graph=use_hybrid,
    )
    retrieve_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Single read — used for both span attributes below and the trace dict at the end.
    # Do NOT add a second getattr call later; this prevents a stale-read trip hazard.
    graph_trace = getattr(self.graph_retriever, "_last_trace", {})
    hybrid_trace = getattr(self.hybrid_retriever, "_last_trace", {})

    # Sub-spans from trace data — attribute-only, timing is the shared retrieve_ms
    with pipeline_span(tracer, "chunk_retriever") as span:
        span.set_attribute("chunk.raw_count", hybrid_trace.get("chunk_count_raw", 0))
        span.set_attribute("latency_ms", retrieve_ms)

    with pipeline_span(tracer, "paper_retriever") as span:
        span.set_attribute("paper.raw_count", hybrid_trace.get("paper_count_raw", 0))
        span.set_attribute("latency_ms", retrieve_ms)

    with pipeline_span(tracer, "graph_retriever") as span:
        entities = graph_trace.get("entities_extracted", [])
        span.set_attribute("graph.entity_count", len(entities))
        span.set_attribute("graph.total_qdrant_ids", graph_trace.get("total_qdrant_ids", 0))
        span.set_attribute("graph.fetched_count", graph_trace.get("fetched_count", 0))
        span.set_attribute("latency_ms", retrieve_ms)

        with pipeline_span(tracer, "entity_extraction") as s:
            s.set_attribute("entities_extracted", json.dumps(
                graph_trace.get("entities_extracted", [])
            ))

        with pipeline_span(tracer, "chunk_fetch") as s:
            s.set_attribute("ids_requested", graph_trace.get("total_qdrant_ids", 0))
            s.set_attribute("ids_resolved", graph_trace.get("fetched_count", 0))

    with pipeline_span(tracer, "rrf_fuse") as span:
        span.set_attribute("fused.count", hybrid_trace.get("fused_count", 0))
        span.set_attribute("fused.source_breakdown", json.dumps(
            hybrid_trace.get("fused_source_breakdown", {})
        ))
        span.set_attribute("top_rrf_scores", json.dumps(
            hybrid_trace.get("top_rrf_scores", [])
        ))
        span.set_attribute("rrf.weights", json.dumps({
            "chunk_weight": self.hybrid_retriever.chunk_weight,
            "paper_weight": self.hybrid_retriever.paper_weight,
            "graph_weight": self.hybrid_retriever.graph_weight,
        }))

    context_texts = [c.text for c in contexts_raw]
    context_sources = [c.source for c in contexts_raw]
    context_scores = [c.score for c in contexts_raw]
    entities_found = list({
        e
        for c in contexts_raw
        for e in c.metadata.get("entities_found", [])
    })

    # Snapshot before reranking
    pre_rerank_count = len(context_texts)
    pre_rerank_source_counts: dict[str, int] = {}
    for c in contexts_raw:
        pre_rerank_source_counts[c.collection] = (
            pre_rerank_source_counts.get(c.collection, 0) + 1
        )

    # --- Rerank (two stages: cross-encoder + relative threshold) ---
    if self.reranker and context_texts:
        with pipeline_span(tracer, "rerank_crossencoder") as span:
            t0 = time.perf_counter()
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
            span.set_attribute("pre_count", pre_rerank_count)
            span.set_attribute("post_count", len(context_texts))
            span.set_attribute("latency_ms", round((time.perf_counter() - t0) * 1000, 2))

        pre_threshold_count = len(context_texts)
        top_score = max(context_scores) if context_scores else 0
        keep = [i for i, s in enumerate(context_scores) if s >= top_score - 8]

        with pipeline_span(tracer, "rerank_threshold") as span:
            span.set_attribute("pre_count", pre_threshold_count)
            span.set_attribute("threshold_dropped", pre_threshold_count - len(keep))
            span.set_attribute("post_count", len(keep))

        if keep:
            context_texts = [context_texts[i] for i in keep]
            context_sources = [context_sources[i] for i in keep]
            context_scores = [context_scores[i] for i in keep]
    else:
        context_texts = context_texts[:top_k]
        context_sources = context_sources[:top_k]
        context_scores = context_scores[:top_k]

    # --- Generate ---
    messages = build_messages(question, context_texts)

    with pipeline_span(tracer, "llm_generate") as span:
        t0 = time.perf_counter()
        answer = self.llm.generate(messages)
        span.set_attribute("llm.model", self.llm.model)
        span.set_attribute("latency_ms", round((time.perf_counter() - t0) * 1000, 2))

    _refusal_phrases = [
        "i don't have enough information",
        "i do not have enough information",
        "i don't know",
        "cannot answer",
        "no information",
    ]
    answer_type = (
        "refusal" if any(p in answer.lower() for p in _refusal_phrases) else "answer"
    )

    trace = {
        "entities_extracted": graph_trace.get("entities_extracted", []),
        "qdrant_ids_per_entity": graph_trace.get("qdrant_ids_per_entity", {}),
        "graph_qdrant_ids_total": graph_trace.get("total_qdrant_ids", 0),
        "graph_fetched_count": graph_trace.get("fetched_count", 0),
        "raw_counts": {
            "chunk": hybrid_trace.get("chunk_count_raw", 0),
            "paper": hybrid_trace.get("paper_count_raw", 0),
            "graph": hybrid_trace.get("graph_count_raw", 0),
        },
        "pre_rerank_count": pre_rerank_count,
        "pre_rerank_source_breakdown": pre_rerank_source_counts,
        "post_rerank_count": len(context_texts),
        "dropped_by_reranker": pre_rerank_count - len(context_texts),
        "reranker_scores": [round(s, 3) for s in context_scores],
        "answer_type": answer_type,
        "top3_contexts": [
            {"source": s, "rerank_score": round(sc, 3), "text": t[:150]}
            for t, s, sc in zip(
                context_texts[:3], context_sources[:3], context_scores[:3]
            )
        ],
    }

    return {
        "question": question,
        "answer": answer,
        "contexts": context_texts,
        "sources": context_sources,
        "scores": context_scores,
        "entities_found": entities_found,
        "retrieval_type": "hybrid" if use_hybrid else "vector",
        "trace": trace,
    }
```

- [ ] **Step 5: Run regression test to verify it still passes**

```
pytest tests/unit/test_query_tracer_noop.py -v
```

Expected: PASS

- [ ] **Step 6: Run full unit test suite to catch regressions**

```
pytest tests/unit/ -v
```

Expected: all previously passing tests still PASS

- [ ] **Step 7: Commit**

```bash
git add src/pipelines/hybridrag_pipeline.py tests/unit/test_query_tracer_noop.py
git commit -m "feat: instrument query() with pipeline spans and per-stage timing"
```

---

## Task 3: Add Phoenix startup and root spans to the eval harness

**Files:**
- Modify: `tests/evaluation/hybridrag_eval.py`
  - `__init__` (line 37–40): add `self.tracer = start_phoenix()`
  - `_run_query` (line 125–131): pass `tracer=self.tracer` to `pipeline.query()`
  - `run()` loop (line 96–119): wrap each mode query in a root `query` span; attach eval metrics to it

- [ ] **Step 1: Write a test for the tracer-integration path**

Create `tests/unit/test_eval_tracer_integration.py`:

```python
"""Verify HybridRAGEvaluator sets up self.tracer without crashing."""
from unittest.mock import MagicMock, patch


def test_evaluator_tracer_is_none_when_phoenix_unavailable():
    """When Phoenix is unavailable, self.tracer is None and no exception is raised."""
    from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline
    from tests.evaluation.hybridrag_eval import HybridRAGEvaluator

    mock_pipeline = MagicMock(spec=HybridRAGBenchPipeline)

    with patch("tests.evaluation.hybridrag_eval.start_phoenix", return_value=None):
        evaluator = HybridRAGEvaluator(mock_pipeline)

    assert evaluator.tracer is None


def test_run_query_passes_tracer_to_pipeline():
    """_run_query passes self.tracer through to pipeline.query."""
    from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline
    from tests.evaluation.hybridrag_eval import HybridRAGEvaluator

    mock_pipeline = MagicMock(spec=HybridRAGBenchPipeline)
    mock_pipeline.query.return_value = {
        "answer": "ans", "contexts": [], "sources": [], "scores": [],
        "entities_found": [], "retrieval_type": "hybrid",
        "trace": {"answer_type": "answer"},
    }

    sentinel_tracer = object()

    with patch("tests.evaluation.hybridrag_eval.start_phoenix", return_value=sentinel_tracer):
        evaluator = HybridRAGEvaluator(mock_pipeline)

    evaluator._run_query("q?", "hybrid", 5)
    _, kwargs = mock_pipeline.query.call_args
    assert kwargs.get("tracer") is sentinel_tracer
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_eval_tracer_integration.py -v
```

Expected: `ImportError` or `AttributeError` — `start_phoenix` not imported in eval harness yet.

- [ ] **Step 3: Modify `hybridrag_eval.py`**

**3a — Add import at top of file** (after existing imports, around line 26):

```python
from src.observability.tracer import start_phoenix, pipeline_span
```

**3b — Add `self.tracer` to `__init__`** (lines 37–40). Replace:

```python
    def __init__(self, pipeline: HybridRAGBenchPipeline):
        self.pipeline = pipeline
        judge_llm = LLMClient(model="llama3.2:latest")
        self.metrics = RAGMetrics(llm_client=judge_llm)
```

With:

```python
    def __init__(self, pipeline: HybridRAGBenchPipeline):
        self.pipeline = pipeline
        judge_llm = LLMClient(model="llama3.2:latest")
        self.metrics = RAGMetrics(llm_client=judge_llm)
        self.tracer = start_phoenix()
```

**3c — Pass tracer in `_run_query`** (lines 125–131). Replace:

```python
    def _run_query(self, question: str, mode: str, top_k: int) -> dict:
        if mode == "hybrid":
            return self.pipeline.query(question, top_k=top_k, use_hybrid=True)
        elif mode == "vector":
            return self.pipeline.query(question, top_k=top_k, use_hybrid=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")
```

With:

```python
    def _run_query(self, question: str, mode: str, top_k: int) -> dict:
        if mode == "hybrid":
            return self.pipeline.query(
                question, top_k=top_k, use_hybrid=True, tracer=self.tracer
            )
        elif mode == "vector":
            return self.pipeline.query(
                question, top_k=top_k, use_hybrid=False, tracer=self.tracer
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
```

**3d — Wrap each query in a root span in `run()`** (lines 96–119). Replace the inner loop body:

```python
            for mode in modes:
                try:
                    response = self._run_query(question, mode, top_k)
                    item = self._compute_metrics(question, gt_answer, gt_context, response)
                    item["mode"] = mode
                    results[mode].append(item)
                except Exception as e:
                    print(f"  [{mode}] ERROR: {e}")
                    results[mode].append({
                        "question": question,
                        "mode": mode,
                        "error": str(e),
                        "metrics": {},
                    })
```

With:

```python
            for mode in modes:
                try:
                    with pipeline_span(self.tracer, "query") as root_span:
                        root_span.set_attribute("query.text", question)
                        root_span.set_attribute("retrieval.mode", mode)
                        root_span.set_attribute("retrieval.use_hybrid", mode == "hybrid")
                        root_span.set_attribute("retrieval.top_k", top_k)

                        response = self._run_query(question, mode, top_k)
                        item = self._compute_metrics(question, gt_answer, gt_context, response)
                        item["mode"] = mode

                        m = item.get("metrics", {})
                        for metric_key in ("hit_rate", "mrr", "context_precision", "ndcg_at_5"):
                            val = m.get(metric_key)
                            if isinstance(val, float):
                                root_span.set_attribute(f"eval.{metric_key}", round(val, 4))
                        root_span.set_attribute(
                            "answer.type",
                            response.get("trace", {}).get("answer_type", ""),
                        )
                        root_span.set_attribute(
                            "answer.text", response.get("answer", "")[:500]
                        )

                        results[mode].append(item)
                except Exception as e:
                    print(f"  [{mode}] ERROR: {e}")
                    results[mode].append({
                        "question": question,
                        "mode": mode,
                        "error": str(e),
                        "metrics": {},
                    })
```

- [ ] **Step 4: Run integration tests**

```
pytest tests/unit/test_eval_tracer_integration.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Run full unit test suite**

```
pytest tests/unit/ -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/evaluation/hybridrag_eval.py tests/unit/test_eval_tracer_integration.py
git commit -m "feat: add Phoenix startup and per-query root spans to eval harness"
```

---

## Task 4: Install dependencies and smoke test

- [ ] **Step 1: Install dependencies**

```bash
pip install "arize-phoenix[otel]" opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc grpcio
```

- [ ] **Step 2: Run the full unit test suite**

```
pytest tests/unit/ -v
```

Expected: all tests PASS

- [ ] **Step 3: Smoke test — run 2 eval pairs with Phoenix live**

With Qdrant and Neo4j running:

```bash
python tests/evaluation/hybridrag_eval.py --max-pairs 2 --modes hybrid
```

Expected output includes:
```
Phoenix running at http://localhost:6006
[1/2] <question text>
[2/2] <question text>
Results saved to tests/results/hybridrag_eval.json
```

Then open `http://localhost:6006` in a browser. You should see 2 traces, each with a span tree matching:
```
query
├── embed_query
├── chunk_retriever
├── paper_retriever
├── graph_retriever
│   ├── entity_extraction
│   └── chunk_fetch
├── rrf_fuse
├── rerank_crossencoder
├── rerank_threshold
└── llm_generate
```

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete HybridRAG observability Phase 1 — retrieval quality per query in Phoenix"
```

---

## Notes for implementer

**Why sub-spans have `latency_ms = retrieve_ms` (approximate):** The three retrievers are called sequentially inside `HybridRetriever.retrieve()`. Since we don't instrument the retriever classes directly, we report the total wall time of the whole retrieve call on each sub-span. The value is approximate but still useful for cross-run comparison.

**`chunk.top_score` / `paper.top_score` not included:** `HybridRetriever._last_trace` does not store per-source top scores (only counts). Including them would require modifying the retriever classes, which is out of scope. The RRF scores in the `rrf_fuse` span serve the same diagnostic purpose.

**Phoenix OTLP port:** Phoenix listens for OTel spans on gRPC port `4317` (not the UI port `6006`). The exporter in `start_phoenix()` targets `localhost:4317`.
