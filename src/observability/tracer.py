"""
Arize Phoenix tracing utilities for the HybridRAG pipeline.

Public API:
    start_phoenix() -> tracer or None
    pipeline_span(tracer, name) -> context manager yielding a span or _NoOpSpan
"""
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("rag.observability")


class _NoOpSpan:
    """Returned by pipeline_span when tracing is disabled. All methods are no-ops."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any, description: str = "") -> None:
        pass

    def record_exception(self, exception: BaseException, **kwargs: Any) -> None:
        pass

    def add_event(self, name: str, **kwargs: Any) -> None:
        pass


@contextmanager
def pipeline_span(tracer: Optional[Any], name: str):
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


def start_phoenix() -> Optional[Any]:
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

        storage_path = Path("data/phoenix")
        storage_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PHOENIX_WORKING_DIR", str(storage_path.resolve()))
        px.launch_app(use_temp_dir=False)

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
