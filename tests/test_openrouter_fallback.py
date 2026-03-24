"""Tests for LLMClient multi-provider fallback."""
import logging
import yaml
from src.generation.llm_client import LLMClient, ProviderConfig, build_providers_from_config


def _ollama_provider() -> ProviderConfig:
    return ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        models=["qwen2.5:7b-instruct"],
    )


def _bad_provider() -> ProviderConfig:
    return ProviderConfig(
        base_url="http://127.0.0.1:9999/v1",  # nothing listening here
        api_key="fake",
        models=["fake-model"],
    )


def test_fallback_to_next_provider(caplog):
    """Bad first provider triggers fallback to working Ollama provider."""
    # The 'rag' logger has propagate=False, so caplog's root handler never
    # receives records from it. Attach caplog's handler directly to 'rag'.
    rag_logger = logging.getLogger("rag")
    rag_logger.addHandler(caplog.handler)
    try:
        caplog.handler.setLevel(logging.WARNING)
        client = LLMClient(providers=[_bad_provider(), _ollama_provider()])
        answer = client.generate([{"role": "user", "content": "Say: hello"}])
    finally:
        rag_logger.removeHandler(caplog.handler)

    assert answer != "", "Expected a non-empty answer from Ollama fallback"
    assert any("failed" in r.message.lower() for r in caplog.records), (
        "Expected a warning log about the failed provider"
    )


def test_all_providers_fail_returns_empty():
    """When every provider fails, generate() returns '' without raising."""
    client = LLMClient(providers=[_bad_provider(), _bad_provider()])
    result = client.generate([{"role": "user", "content": "hello"}])
    assert result == ""


def test_openrouter_skipped_when_no_api_key():
    """build_providers_from_config() returns only the Ollama provider when key is absent."""
    cfg = yaml.safe_load(open("config/defaults.yaml"))
    # Pass an env dict with no OPENROUTER_API_KEY
    providers = build_providers_from_config(cfg, env={})
    assert len(providers) == 1, f"Expected 1 provider (Ollama), got {len(providers)}"
    assert "localhost" in providers[0].base_url or "11434" in providers[0].base_url, (
        f"Expected Ollama provider, got base_url={providers[0].base_url}"
    )
