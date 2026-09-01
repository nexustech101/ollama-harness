"""Provider layer: which model service the harness talks to.

The harness and the API never construct provider SDKs themselves; every chat
model comes from this module. Switching services is configuration (a flag, an
env var), not code.

Built-in providers:
- ollama      -> ChatOllama   (local Ollama server)
- openai      -> ChatOpenAI   (OpenAI, or any OpenAI-compatible endpoint:
                              vLLM, LM Studio, Together, Groq, ...)
- openrouter  -> ChatOpenAI   (OpenRouter's OpenAI-compatible API)

To add a provider: extend the tables below and `Provider.build`, then give it
a `--provider` choice in main.py. Nothing else needs to know it exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

PROVIDERS = ("ollama", "openai", "openrouter")

# Per-provider defaults, overridable by --base-url / --model / env vars.
DEFAULT_BASE_URLS = {
    "ollama": "http://127.0.0.1:11434",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
DEFAULT_MODELS = {
    "ollama": "qwen3.8-coder:latest",
    "openai": "gpt-4o-mini",
    "openrouter": "deepseek/deepseek-chat",
}
# Provider-specific key env vars, consulted before the generic HARNESS_API_KEY.
_API_KEY_ENVS = {
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def api_key_from_env(provider: str) -> Optional[str]:
    """Best available API key for a provider: its own env var, then HARNESS_API_KEY."""
    for var in _API_KEY_ENVS.get(provider, ()) + ("HARNESS_API_KEY",):
        value = os.environ.get(var)
        if value:
            return value
    return None


@dataclass
class Provider:
    """Everything needed to build one chat model, resolved from flags/env."""

    name: str = "ollama"
    model: str = ""
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    num_ctx: int = 32_768
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        return self.base_url or DEFAULT_BASE_URLS[self.name]

    def build(self) -> BaseChatModel:
        """Construct the provider's chat model, tool-ready (bind_tools upstream)."""
        if self.name == "ollama":
            from langchain_ollama import ChatOllama  # deferred: clearer import errors

            return ChatOllama(
                model=self.model,
                base_url=self.endpoint,
                temperature=self.temperature,
                num_ctx=self.num_ctx,
            )
        # openai and openrouter are both OpenAI-compatible.
        from langchain_openai import ChatOpenAI  # deferred for the same reason

        return ChatOpenAI(
            model=self.model,
            base_url=self.endpoint,
            api_key=self.api_key or "not-needed",  # local compatible servers accept any key
            temperature=self.temperature,
            **self.extra,
        )