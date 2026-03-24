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
- `LLMClient` tracks token usage in `total_tokens_used`, `prompt_tokens_used`, `completion_tokens_used` — these must be preserved and incremented only on successful calls.

---

## Architecture

### Provider Chain

```
generate(messages)
    └─► Provider 1: OpenRouter
            models: ["qwen/qwq-32b:free", "deepseek/deepseek-chat-v3-0324:free"]
            OpenRouter retries model 2 internally if model 1 is rate-limited
            → On success: set _active_model = response.model, accumulate tokens, return answer
            → On any exception: log warning, try next provider
    └─► Provider 2: Ollama (local fallback)
            model: qwen2.5:7b-instruct @ http://localhost:11434/v1
            → On success: set _active_model = response.model, accumulate tokens, return answer
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
- `self._active_model: str` is initialized to `providers[0].models[0]` and updated to `response.model` after every successful API call. This reflects the model OpenRouter actually used (which may differ from the requested model when server-side fallback triggers). The `llm_generate` span reads `self.llm._active_model`, not `self.llm.model`.
- Token usage counters (`total_tokens_used`, `prompt_tokens_used`, `completion_tokens_used`) are retained unchanged. They are incremented only inside the success path, after a valid response is received — never in an exception handler.
- `reset_usage()` is retained unchanged.

#### `generate()` provider loop

```python
for provider in self.providers:
    try:
        response = OpenAI(
            base_url=provider.base_url,
            api_key=provider.api_key,
            default_headers=provider.extra_headers,
        ).chat.completions.create(
            model=provider.models[0],
            extra_body={"models": provider.models} if len(provider.models) > 1 else {},
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Success path — update state before returning
        self._active_model = response.model
        # accumulate token usage here
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Provider {provider.base_url} failed ({type(e).__name__}): {e}")
        continue
return ""
```

#### `chat_completion()` provider loop

`chat_completion()` is also wrapped in the same provider loop with identical fallback semantics. **Exception:** when `tools` are passed, the `extra_body={"models": [...]}` multi-model routing is **omitted** — only `provider.models[0]` is used, because tool-calling behavior across routed models is inconsistent. The provider loop still applies (OpenRouter → Ollama fallback still works at the provider level).

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

The `llm_generate` span in `query()` changes from `self.llm.model` to `self.llm._active_model` so it reports the model that actually answered.

### 5. `.env` file (gitignored) and `.env.example` (committed)

`.env` (gitignored, not committed):
```
OPENROUTER_API_KEY=sk-or-...
```

`.env.example` (new file, committed to git, **not** in `.gitignore`):
```
OPENROUTER_API_KEY=your-openrouter-key-here
```

---

## Error Handling

| Situation | Behavior |
|---|---|
| OpenRouter returns valid response | Set `_active_model = response.model`; log model name; return answer |
| OpenRouter raises any exception | `logger.warning(...)`, move to Ollama |
| Ollama also fails | Return `""` (same as today) |
| `OPENROUTER_API_KEY` not set | Skip OpenRouter silently, use Ollama |
| `openrouter.enabled: false` | Skip OpenRouter, use Ollama |

---

## Observability

- The `llm_generate` span's `llm.model` attribute is set from `self.llm._active_model` (updated to `response.model` after each successful call — reflects the actual model OpenRouter routed to).
- No new spans needed.

---

## Testing

File: `tests/test_openrouter_fallback.py`

All tests construct `LLMClient` or `HybridRAGBenchPipeline` directly — **do not use `get_pipeline()`**. The module-level singleton caches the provider list at first construction and will produce wrong results if the env var state changes between tests.

1. **`test_fallback_to_next_provider`** — bad first provider URL, working Ollama second; assert non-empty answer and warning logged.
2. **`test_all_providers_fail_returns_empty`** — all bad URLs; assert `""` returned, no exception raised.
3. **`test_openrouter_skipped_when_no_api_key`** — `OPENROUTER_API_KEY` unset (use `monkeypatch.delenv`); construct a fresh `HybridRAGBenchPipeline`; assert `len(pipeline.llm.providers) == 1` and it points to Ollama.

No live OpenRouter calls in automated tests. Real integration verified manually by running a query and inspecting logs.

---

## Implementation Order (incremental — verify each before proceeding)

1. Add `ProviderConfig` dataclass and update `LLMClient` (`generate()` + `chat_completion()`) — verify with tests 1 and 2.
2. Add `openrouter` block to `defaults.yaml` and create `.env.example`.
3. Update `HybridRAGBenchPipeline` (provider list construction + span attribute fix) — verify pipeline constructs without error and test 3 passes.
4. Manual end-to-end test: run a query, confirm OpenRouter responds, check `llm.model` in logs shows actual OpenRouter model.
5. All three tests in `tests/test_openrouter_fallback.py` must pass before marking done.

---

## Files Changed

| File | Change |
|---|---|
| `src/generation/llm_client.py` | Add `ProviderConfig` dataclass; update `LLMClient` with provider loop for both `generate()` and `chat_completion()` |
| `config/defaults.yaml` | Add `openrouter` block |
| `.env.example` | New file (committed, not gitignored) — `OPENROUTER_API_KEY` placeholder |
| `src/pipelines/hybridrag_pipeline.py` | Build provider list from config; change span attribute from `self.llm.model` to `self.llm._active_model` |
| `tests/test_openrouter_fallback.py` | New test file (3 tests, no singleton use) |
| `requirements.txt` | No change needed (`openai` SDK already present) |
