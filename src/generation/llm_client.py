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
                "OpenRouter enabled in config but %s is not set — skipping OpenRouter",
                key_var,
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
            _providers = providers
        else:
            _providers = [ProviderConfig(
                base_url=base_url,
                api_key=api_key or "ollama",
                models=[model],
            )]

        self._providers: list[tuple[ProviderConfig, OpenAI]] = [
            (p, OpenAI(api_key=p.api_key, base_url=p.base_url, default_headers=p.extra_headers))
            for p in _providers
        ]

        self._active_model: str = self._providers[0][0].models[0]
        self.total_tokens_used = 0
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0

    @property
    def providers(self) -> list[ProviderConfig]:
        return [p for p, _ in self._providers]

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
        for provider, client in self._providers:
            try:
                extra_body = {"models": provider.models} if len(provider.models) > 1 else {}
                response = client.chat.completions.create(
                    model=provider.models[0],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=extra_body,
                )
                if not response.choices:
                    logger.warning(
                        "Provider %s returned empty choices — trying next provider", provider.base_url
                    )
                    continue
                self._active_model = response.model
                self._accumulate_usage(getattr(response, "usage", None))
                content = response.choices[0].message.content
                logger.info("LLM answered via %s (model: %s)", provider.base_url, self._active_model)
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
        for provider, client in self._providers:
            try:
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
                if not response.choices:
                    logger.warning(
                        "Provider %s returned empty choices — trying next provider", provider.base_url
                    )
                    continue
                self._active_model = response.model
                self._accumulate_usage(getattr(response, "usage", None))
                logger.info("LLM answered via %s (model: %s)", provider.base_url, self._active_model)
                return response.choices[0].message
            except Exception as e:
                logger.warning(
                    "Provider %s failed (%s): %s — trying next provider",
                    provider.base_url, type(e).__name__, e,
                )
        return None
