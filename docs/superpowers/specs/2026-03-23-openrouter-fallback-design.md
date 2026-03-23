# OpenRouter Free-Model Fallback — Design Spec

**Date:** 2026-03-23
**Branch:** feat/hybridrag-observability
**Status:** Approved

---

## Goal

Integrate OpenRouter free models as the primary LLM backend for answer generation in the HybridRAG pipeline, with automatic fallback to local Ollama when OpenRouter fails. Each step must be verified working before the next begins.

---

## Context

- `LLMClient` (`src/generation/llm_client.py`) already uses the OpenAI Python SDK with configurable `base_url`, `api_key`, and `model`. OpenRouter exposes an OpenAI-compatible API, making it a drop-in swap.
- `HybridRAGBenchPipeline` constructs `LLMClient` from `config/defaults.yaml` (`llm` block).
- The dead `relation_llm` config block has already been removed from `defaults.yaml`.

---

## Architecture

### Provider Chain

```
generate(messages)
    └─► Provider 1: OpenRouter
            models: ["qwen/qwq-32b:free", "deepseek/deepseek-chat-v3-0324:free"]
            OpenRouter retries model 2 internally if model 1 is rate-limited
            → On success: return answer
            → On any exception: log warning, try next provider
    └─► Provider 2: Ollama (local fallback)
            model: qwen2.5:7b-instruct @ http://localhost:11434/v1
            → On success: return answer
            → On failure: return ""
```

### Model Choice Rationale

- **Primary: `qwen/qwq-32b:free`** — 32B reasoning model, strong on CS/technical questions, fast enough for interactive use.
- **Secondary: `deepseek/deepseek-chat-v3-0324:free`** — DeepSeek V3, excellent general quality, different underlying provider so it won't fail for the same reason as QwQ.
- Both are free tier on OpenRouter (no billing required).

---

## Components

### 1. `ProviderConfig` dataclass (`src/generation/llm_client.py`)

```python
@dataclass
class ProviderConfig:
    base_url: str
    api_key: str
    models: list[str]          # OpenRouter: tried in order server-side; Ollama: first model used
    extra_headers: dict = field(default_factory=dict)
```

### 2. Updated `LLMClient`

- Accepts `providers: list[ProviderConfig]` as primary constructor input.
- Legacy single-provider params (`api_key`, `base_url`, `model`) preserved for backward compatibility — if `providers` is empty, one `ProviderConfig` is auto-built from them.
- `generate()` and `chat_completion()` loop over providers, catching all exceptions, returning on first success.
- For OpenRouter providers with multiple models, passes `extra_body={"models": [...]}` so OpenRouter handles model-level retry server-side.
- Sets `self._active_model` on each successful call so the observability span can report the actual model used.

### 3. `config/defaults.yaml` — new `openrouter` block

```yaml
openrouter:
  enabled: true
  api_key_env: "OPENROUTER_API_KEY"
  models:
    - "qwen/qwq-32b:free"
    - "deepseek/deepseek-chat-v3-0324:free"
  site_url: ""      # optional HTTP-Referer header (OpenRouter best practice)
  app_name: ""      # optional X-Title header
```

### 4. `HybridRAGBenchPipeline.__init__` update

Reads the `openrouter` block from config. If `enabled: true` and the env var is set, prepends an OpenRouter `ProviderConfig` to the list before the Ollama `ProviderConfig`. Constructs `LLMClient(providers=[...])`.

### 5. `.env` file (gitignored)

```
OPENROUTER_API_KEY=sk-or-...
```

---

## Error Handling

| Situation | Behavior |
|---|---|
| OpenRouter returns valid response | Use it; log which model answered |
| OpenRouter raises any exception | `logger.warning(...)`, move to Ollama |
| Ollama also fails | Return `""` (same as today) |
| `OPENROUTER_API_KEY` not set | Skip OpenRouter silently, use Ollama |
| `openrouter.enabled: false` | Skip OpenRouter, use Ollama |

---

## Observability

- The `llm_generate` span's `llm.model` attribute is updated to `self._active_model` (the model that actually answered).
- No new spans needed.

---

## Testing

File: `tests/test_openrouter_fallback.py`

1. **`test_fallback_to_next_provider`** — bad first provider URL, working Ollama second; assert non-empty answer and warning logged.
2. **`test_all_providers_fail_returns_empty`** — all bad URLs; assert `""` returned, no exception raised.
3. **`test_openrouter_skipped_when_no_api_key`** — `OPENROUTER_API_KEY` unset; assert pipeline builds with only Ollama provider.

No live OpenRouter calls in automated tests. Real integration verified manually by running a query and inspecting logs.

---

## Implementation Order (incremental — verify each before proceeding)

1. Add `ProviderConfig` dataclass and update `LLMClient` — verify with a direct unit test.
2. Add `openrouter` block to `defaults.yaml` and `.env.example`.
3. Update `HybridRAGBenchPipeline` to build the provider list from config — verify pipeline constructs without error.
4. Manual end-to-end test: run a query, confirm OpenRouter responds, check `llm.model` in logs.
5. Write `tests/test_openrouter_fallback.py` — verify all three tests pass.

---

## Files Changed

| File | Change |
|---|---|
| `src/generation/llm_client.py` | Add `ProviderConfig`, update `LLMClient` |
| `config/defaults.yaml` | Add `openrouter` block |
| `.env.example` | Add `OPENROUTER_API_KEY=` placeholder |
| `src/pipelines/hybridrag_pipeline.py` | Read openrouter config, build provider list |
| `tests/test_openrouter_fallback.py` | New test file |
| `requirements.txt` | No change needed (`openai` SDK already present) |
