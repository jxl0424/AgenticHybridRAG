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


def test_start_phoenix_returns_none_when_phoenix_not_importable(monkeypatch):
    import unittest.mock as mock
    import sys
    fake_phoenix = mock.MagicMock()
    fake_phoenix.launch_app.side_effect = RuntimeError("simulated failure")
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    result = start_phoenix()
    assert result is None
