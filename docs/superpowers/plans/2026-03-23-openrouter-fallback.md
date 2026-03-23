# OpenRouter Free-Model Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an OpenRouter free-model provider (qwen/qwq-32b:free → deepseek/deepseek-chat-v3-0324:free) as the primary LLM backend with automatic fallback to local Ollama when OpenRouter fails.

**Architecture:** `LLMClient` gains a `ProviderConfig` dataclass and a provider loop in both `generate()` and `chat_completion()`. A helper `build_providers_from_config()` reads the new `openrouter` block from `defaults.yaml` and builds the ordered provider list. `HybridRAGBenchPipeline` calls this helper and passes the result to `LLMClient(providers=[...])`.

**Tech Stack:** Python 3.11, `openai>=1.3.0` (already installed), `python-dotenv>=1.0.0` (already installed), `pytest` with `monkeypatch`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/generation/llm_client.py` | Modify | Add `ProviderConfig` dataclass, `build_providers_from_config()` helper, provider loop in `generate()` and `chat_completion()` |
| `config/defaults.yaml` | Modify | Add `openrouter` config block |
| `.env.example` | Create | Document `OPENROUTER_API_KEY` for new contributors |
| `src/pipelines/hybridrag_pipeline.py` | Modify | Call `build_providers_from_config()`, pass result to `LLMClient`, fix span attribute |
| `tests/test_openrouter_fallback.py` | Create | Three focused unit tests |

---

## Task 1: `ProviderConfig` + `LLMClient` provider loop

**Files:**
- Modify: `src/generation/llm_client.py`
- Create: `tests/test_openrouter_fallback.py`

### Background

The current `LLMClient` has a single `self.client = OpenAI(...)` created at init time. We replace this with a `providers` list. Each `generate()` / `chat_completion()` call iterates the list and returns on first success.

The `print()` calls in the current code must be replaced with `logger.warning()` (the project uses `src.utils.get_logger`).

- [ ] **Step 1: Write the two failing tests**

Create `tests/test_openrouter_fallback.py` with this content:

```python
"""Tests for LLMClient multi-provider fallback."""
import logging
import pytest
from src.generation.llm_client import LLMClient, ProviderConfig


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
    client = LLMClient(providers=[_bad_provider(), _ollama_provider()])
    with caplog.at_level(logging.WARNING, logger="src.generation.llm_client"):
        answer = client.generate([{"role": "user", "content": "Say: hello"}])
    assert answer != "", "Expected a non-empty answer from Ollama fallback"
    assert any("failed" in r.message.lower() for r in caplog.records), (
        "Expected a warning log about the failed provider"
    )


def test_all_providers_fail_returns_empty():
    """When every provider fails, generate() returns '' without raising."""
    client = LLMClient(providers=[_bad_provider(), _bad_provider()])
    result = client.generate([{"role": "user", "content": "hello"}])
    assert result == ""
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd C:/Users/brend/Desktop/RAG
ragproject/Scripts/python.exe -m pytest tests/test_openrouter_fallback.py::test_fallback_to_next_provider tests/test_openrouter_fallback.py::test_all_providers_fail_returns_empty -v
```

Expected: both tests **FAIL** — `ProviderConfig` does not exist yet.

- [ ] **Step 3: Implement `ProviderConfig` and the updated `LLMClient`**

Replace `src/generation/llm_client.py` entirely with:

```python
"""
LLM client wrapper for interacting with language models.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProviderConfig:
    base_url: str
    api_key: str
    models: list[str]
    extra_headers: dict = field(default_factory=dict)


def build_providers_from_config(cfg: dict, env: dict | None = None) -> list[ProviderConfig]:
    """
    Build an ordered provider list from the loaded config dict.

    Reads the 'openrouter' and 'llm' blocks. If OpenRouter is enabled and its
    API key env var is set, OpenRouter is prepended as the first provider.
    Ollama (from the 'llm' block) is always appended as the final fallback.

    Args:
        cfg:  The full parsed YAML config dict (from defaults.yaml).
        env:  Optional env-var override dict for testing (defaults to os.environ).
    """
    if env is None:
        env = os.environ

    providers: list[ProviderConfig] = []

    or_cfg = cfg.get("openrouter", {})
    if or_cfg.get("enabled", False):
        key_var = or_cfg.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = env.get(key_var, "")
        if api_key:
            extra_headers: dict[str, str] = {}
            if or_cfg.get("site_url"):
                extra_headers["HTTP-Referer"] = or_cfg["site_url"]
            if or_cfg.get("app_name"):
                extra_headers["X-Title"] = or_cfg["app_name"]
            providers.append(ProviderConfig(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                models=or_cfg.get("models", ["qwen/qwq-32b:free"]),
                extra_headers=extra_headers,
            ))
        else:
            logger.warning(
                "OpenRouter enabled in config but %s is not set — skipping OpenRouter", key_var
            )

    llm_cfg = cfg.get("llm", {})
    providers.append(ProviderConfig(
        base_url=llm_cfg.get("base_url", "http://localhost:11434/v1"),
        api_key="ollama",
        models=[llm_cfg.get("model", "qwen2.5:7b-instruct")],
    ))

    return providers


class LLMClient:
    """Wrapper for LLM inference via OpenAI-compatible APIs with provider fallback."""

    def __init__(
        self,
        providers: Optional[list[ProviderConfig]] = None,
        # Legacy single-provider params — kept for backward compatibility.
        # If providers is None or empty, these are used to build one ProviderConfig.
        api_key: Optional[str] = "ollama",
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b-instruct",
    ):
        if providers:
            self.providers = providers
        else:
            self.providers = [ProviderConfig(
                base_url=base_url,
                api_key=api_key or "ollama",
                models=[model],
            )]

        self._active_model: str = self.providers[0].models[0]
        self.total_tokens_used = 0
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0

    def reset_usage(self):
        """Reset token usage counters."""
        self.total_tokens_used = 0
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0

    def _accumulate_usage(self, usage: Any) -> None:
        if usage:
            self.prompt_tokens_used += getattr(usage, "prompt_tokens", 0)
            self.completion_tokens_used += getattr(usage, "completion_tokens", 0)
            self.total_tokens_used += getattr(usage, "total_tokens", 0)

    def generate(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0) -> str:
        """Generate a response, trying each provider in order until one succeeds."""
        for provider in self.providers:
            try:
                client = OpenAI(
                    api_key=provider.api_key,
                    base_url=provider.base_url,
                    default_headers=provider.extra_headers,
                )
                extra_body = {"models": provider.models} if len(provider.models) > 1 else {}
                response = client.chat.completions.create(
                    model=provider.models[0],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=extra_body,
                )
                self._active_model = response.model
                self._accumulate_usage(getattr(response, "usage", None))
                if not response.choices:
                    return ""
                content = response.choices[0].message.content
                logger.warning("LLM answered via %s (model: %s)", provider.base_url, self._active_model)
                return content.strip() if content else ""
            except Exception as e:
                logger.warning(
                    "Provider %s failed (%s): %s — trying next provider",
                    provider.base_url, type(e).__name__, e,
                )
        return ""

    def chat_completion(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Any] = "auto",
        max_tokens: int = 2048,
        temperature: float = 0,
    ) -> Any:
        """
        Generate a full chat completion response, supporting tools.
        Returns the raw message object (which includes .content and .tool_calls).
        Falls back across providers. When tools are passed, extra_body model routing
        is omitted (tool-calling behaviour across routed models is inconsistent).
        """
        for provider in self.providers:
            try:
                client = OpenAI(
                    api_key=provider.api_key,
                    base_url=provider.base_url,
                    default_headers=provider.extra_headers,
                )
                kwargs: dict[str, Any] = {
                    "model": provider.models[0],
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                # Multi-model routing only when no tools (tool-calling across routed
                # models is inconsistent on OpenRouter)
                if not tools and len(provider.models) > 1:
                    kwargs["extra_body"] = {"models": provider.models}
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice

                response = client.chat.completions.create(**kwargs)
                self._active_model = response.model
                self._accumulate_usage(getattr(response, "usage", None))
                return response.choices[0].message if response.choices else None
            except Exception as e:
                logger.warning(
                    "Provider %s failed (%s): %s — trying next provider",
                    provider.base_url, type(e).__name__, e,
                )
        return None
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
ragproject/Scripts/python.exe -m pytest tests/test_openrouter_fallback.py::test_fallback_to_next_provider tests/test_openrouter_fallback.py::test_all_providers_fail_returns_empty -v
```

Expected output:
```
PASSED tests/test_openrouter_fallback.py::test_fallback_to_next_provider
PASSED tests/test_openrouter_fallback.py::test_all_providers_fail_returns_empty
```

> **Verify before proceeding:** Both tests must be green. `test_fallback_to_next_provider` requires Ollama running locally with `qwen2.5:7b-instruct` loaded. If Ollama is not running, start it first: `ollama serve`.

- [ ] **Step 5: Commit**

```bash
git add src/generation/llm_client.py tests/test_openrouter_fallback.py
git commit -m "feat: add ProviderConfig and multi-provider fallback to LLMClient"
```

---

## Task 2: Config and `.env.example`

**Files:**
- Modify: `config/defaults.yaml`
- Create: `.env.example`

- [ ] **Step 1: Add `openrouter` block to `config/defaults.yaml`**

Add the following block after the `# LLM settings` block (between `llm:` and `# Hybrid retrieval settings`):

```yaml
# OpenRouter free-model fallback (answer generation)
# Set OPENROUTER_API_KEY in .env to enable. Falls back to local Ollama if unavailable.
openrouter:
  enabled: true
  api_key_env: "OPENROUTER_API_KEY"
  models:
    - "qwen/qwq-32b:free"
    - "deepseek/deepseek-chat-v3-0324:free"
  site_url: ""     # optional — sent as HTTP-Referer (OpenRouter best practice)
  app_name: ""     # optional — sent as X-Title header
```

The `defaults.yaml` `llm` block stays unchanged — it remains the Ollama fallback config.

- [ ] **Step 2: Create `.env.example`**

Create `.env.example` in the project root (this file is committed to git — `.gitignore` only ignores `.env` and `.env.local`):

```
# Copy this file to .env and fill in your values.
# .env is gitignored; .env.example is committed.

# OpenRouter API key — get one free at https://openrouter.ai
# Leave blank to skip OpenRouter and use local Ollama only.
OPENROUTER_API_KEY=your-openrouter-key-here
```

- [ ] **Step 3: Verify the YAML parses cleanly**

```bash
ragproject/Scripts/python.exe -c "import yaml; cfg = yaml.safe_load(open('config/defaults.yaml')); print(cfg.get('openrouter'))"
```

Expected output:
```
{'enabled': True, 'api_key_env': 'OPENROUTER_API_KEY', 'models': ['qwen/qwq-32b:free', 'deepseek/deepseek-chat-v3-0324:free'], 'site_url': '', 'app_name': ''}
```

- [ ] **Step 4: Commit**

```bash
git add config/defaults.yaml .env.example
git commit -m "feat: add openrouter config block and .env.example"
```

---

## Task 3: Wire pipeline + fix span attribute

**Files:**
- Modify: `src/pipelines/hybridrag_pipeline.py`
- Modify: `tests/test_openrouter_fallback.py`

### Background

Three changes to `hybridrag_pipeline.py`:
1. Add `build_providers_from_config` to the existing `LLMClient` import.
2. In `__init__`, replace the direct `LLMClient(base_url=..., model=...)` call with `LLMClient(providers=build_providers_from_config(cfg))`.
3. In `query()` at line 397, change `self.llm.model` → `self.llm._active_model` so the span reports the model that actually answered.

Test 3 tests `build_providers_from_config()` directly (a pure function) rather than constructing a full `HybridRAGBenchPipeline`. The spec suggested constructing the pipeline, but that requires Qdrant and Neo4j to be running — testing the helper function directly achieves the same coverage goal (verifying that the provider list is built correctly from config + env) without service dependencies.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_openrouter_fallback.py`:

```python
import yaml
from src.generation.llm_client import build_providers_from_config


def test_openrouter_skipped_when_no_api_key():
    """build_providers_from_config() returns only the Ollama provider when key is absent."""
    cfg = yaml.safe_load(open("config/defaults.yaml"))
    # Pass an env dict with no OPENROUTER_API_KEY
    providers = build_providers_from_config(cfg, env={})
    assert len(providers) == 1, f"Expected 1 provider (Ollama), got {len(providers)}"
    assert "localhost" in providers[0].base_url or "11434" in providers[0].base_url, (
        f"Expected Ollama provider, got base_url={providers[0].base_url}"
    )
```

- [ ] **Step 2: Run test — it should already pass**

`build_providers_from_config` was implemented in Task 1 and already handles the "no key" case. Run it now to confirm:

```bash
ragproject/Scripts/python.exe -m pytest tests/test_openrouter_fallback.py::test_openrouter_skipped_when_no_api_key -v
```

Expected: **PASS**. If it passes, proceed directly to Step 3 (pipeline wiring). If it unexpectedly fails, read the error before continuing.

- [ ] **Step 3: Update `HybridRAGBenchPipeline.__init__`**

In `src/pipelines/hybridrag_pipeline.py`, make three changes:

**Change 1** — add the import at the top of the file (alongside the existing `LLMClient` import):

```python
from src.generation.llm_client import LLMClient, build_providers_from_config
```

**Change 2** — replace the `LLMClient` construction block (currently lines 124–128):

```python
        # LLM
        self.llm = LLMClient(
            base_url=llm_cfg.get("base_url", "http://localhost:11434/v1"),
            model=llm_cfg.get("model", "qwen2.5:7b-instruct"),
        )
```

with:

```python
        # LLM — OpenRouter first, Ollama fallback (see config/defaults.yaml openrouter block)
        self.llm = LLMClient(providers=build_providers_from_config(cfg))
```

Note: `cfg` is the full parsed YAML dict (already available in `__init__` from `yaml.safe_load(f)`).

**Change 3** — in `query()`, find the `llm_generate` span (around line 397):

```python
            span.set_attribute("llm.model", self.llm.model)
```

Replace with:

```python
            span.set_attribute("llm.model", self.llm._active_model)
```

- [ ] **Step 4: Run all three tests**

```bash
ragproject/Scripts/python.exe -m pytest tests/test_openrouter_fallback.py -v
```

Expected:
```
PASSED tests/test_openrouter_fallback.py::test_fallback_to_next_provider
PASSED tests/test_openrouter_fallback.py::test_all_providers_fail_returns_empty
PASSED tests/test_openrouter_fallback.py::test_openrouter_skipped_when_no_api_key
```

> **Verify before proceeding:** All three tests must be green.

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/hybridrag_pipeline.py tests/test_openrouter_fallback.py
git commit -m "feat: wire OpenRouter provider chain into HybridRAGBenchPipeline"
```

---

## Task 4: Set up API key and manual end-to-end verification

**Files:** `.env` (local only, gitignored)

### Background

Get a free OpenRouter API key at https://openrouter.ai (no credit card required for free models). Then verify the full pipeline uses OpenRouter by running a query and reading the logs.

- [ ] **Step 1: Create `.env` with your OpenRouter API key**

Create `.env` in the project root (gitignored):

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

- [ ] **Step 2: Confirm the key is loaded**

```bash
ragproject/Scripts/python.exe -c "
from dotenv import load_dotenv; load_dotenv()
import os; key = os.environ.get('OPENROUTER_API_KEY', '')
print('Key set:', bool(key), '| Length:', len(key))
"
```

Expected: `Key set: True | Length: <some number>`

- [ ] **Step 3: Confirm the provider list includes OpenRouter**

```bash
ragproject/Scripts/python.exe -c "
from dotenv import load_dotenv; load_dotenv()
import yaml
from src.generation.llm_client import build_providers_from_config
cfg = yaml.safe_load(open('config/defaults.yaml'))
providers = build_providers_from_config(cfg)
for p in providers:
    print(p.base_url, '->', p.models)
"
```

Expected:
```
https://openrouter.ai/api/v1 -> ['qwen/qwq-32b:free', 'deepseek/deepseek-chat-v3-0324:free']
http://localhost:11434/v1 -> ['qwen2.5:7b-instruct']
```

- [ ] **Step 4: Run a query and verify OpenRouter responds**

Make sure Qdrant and Neo4j are running, then:

```bash
ragproject/Scripts/python.exe -c "
from dotenv import load_dotenv; load_dotenv()
import logging; logging.basicConfig(level=logging.WARNING)
from src.pipelines.hybridrag_pipeline import HybridRAGBenchPipeline
pipeline = HybridRAGBenchPipeline()
result = pipeline.query('What is the attention mechanism in transformers?')
print('Model used:', pipeline.llm._active_model)
print('Answer snippet:', result['answer'][:200])
"
```

Expected: logs show `LLM answered via https://openrouter.ai/api/v1 (model: qwen/qwq-32b:free)` (or whichever model OpenRouter routed to). `_active_model` reflects the actual model name returned in the response.

> **This is the key verification gate.** If `_active_model` shows an OpenRouter model ID, the integration is working end-to-end.

- [ ] **Step 5: Verify Ollama fallback works**

Temporarily set a bad key and confirm fallback:

```bash
ragproject/Scripts/python.exe -c "
import logging; logging.basicConfig(level=logging.WARNING)
import os; os.environ['OPENROUTER_API_KEY'] = 'bad-key'
import yaml
from src.generation.llm_client import LLMClient, build_providers_from_config
cfg = yaml.safe_load(open('config/defaults.yaml'))
client = LLMClient(providers=build_providers_from_config(cfg))
answer = client.generate([{'role': 'user', 'content': 'Say: hello'}])
print('Active model:', client._active_model)
print('Answer:', answer)
"
```

Expected: warning log about OpenRouter failure, then `_active_model` shows `qwen2.5:7b-instruct` (Ollama), answer is non-empty.

No commit needed for this task — it's verification only.

---

## Done

All tasks complete when:
- [ ] All 3 tests in `tests/test_openrouter_fallback.py` pass
- [ ] Manual end-to-end query shows OpenRouter model in `_active_model`
- [ ] Ollama fallback confirmed working with bad key
