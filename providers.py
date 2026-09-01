"""Provider layer: which model service the harness talks to.

Providers resolve, most specific first:
1. Command-line flags       (--provider, --model, --base-url, --api-key)
2. A YAML config file       (providers.yaml) mapping named providers to their
                            base_url / api_key / model options. It is read only
                            when --provider names a provider.
3. Environment variables    (.env / HARNESS_*, OLLAMA_MODEL, OLLAMA_HOST, ...)
4. Per-provider defaults    (tables below)

The default provider is always ollama. Selecting any other provider (openai,
openrouter, or a named entry from providers.yaml) requires --provider; without
it the harness never looks at the config file and uses ollama's .env settings.

The harness speaks three wire protocols: `ollama`, `openai` (and any
OpenAI-compatible endpoint), `openrouter`. A config entry that uses a built-in
name inherits that protocol; any other name defaults to `openai` unless the
entry sets `kind`.

Config file locations, first match wins: --config, $HARNESS_CONFIG,
./providers.yaml, ~/.config/harness/providers.yaml, %APPDATA%/harness/providers.yaml.

    # providers.yaml — consulted only when --provider NAME is used
    providers:
      openrouter:
        base_url: https://openrouter.ai/api/v1
        api_key: sk-or-v1-xxxx
        model: openai/gpt-5.5
      groq:
        base_url: https://api.groq.com/openai/v1
        api_key: gsk_xxxx
        model: llama-3.3-70b-versatile
      local:
        kind: ollama                       # optional; defaults to openai
        base_url: http://127.0.0.1:11434
        model: qwen3
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml  # PyYAML (declared dependency; also shipped by langchain-core)
from langchain_core.language_models.chat_models import BaseChatModel

PROVIDERS = ("ollama", "openai", "openrouter")
DEFAULT_NUM_CTX = 200_000  # single source of truth for the default context window

# Per-provider defaults, overridable by flags / config / env vars.
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


def _first(*values: Any) -> Any:
    """First non-empty value."""
    for value in values:
        if value is not None and value != "":
            return value
    return None


# ---------------------------------------------------------------------------
# YAML config file
# ---------------------------------------------------------------------------

def find_config(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate the providers YAML: --config, $HARNESS_CONFIG, cwd, user dirs."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get("HARNESS_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path.cwd() / "providers.yaml")
    candidates.append(Path.cwd() / "providers.yml")
    if os.name == "nt" and os.environ.get("APPDATA"):
        candidates.append(Path(os.environ["APPDATA"]) / "harness" / "providers.yaml")
    candidates.append(Path.home() / ".config" / "harness" / "providers.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path) -> dict[str, Any]:
    """Parse providers.yaml -> {"providers": {name: opts}}.

    A top-level `default:` key is accepted for compatibility but ignored: the
    default provider is always ollama; the config is only consulted when
    --provider names one of its entries.
    """
    if not path.is_file():
        raise FileNotFoundError(f"provider config not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: invalid YAML: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of provider settings, "
                         f"got {type(data).__name__}")
    providers = data.get("providers") or {}
    if not isinstance(providers, dict):
        raise ValueError(f"{path}: 'providers' must be a mapping of name -> options")
    cleaned: dict[str, dict[str, Any]] = {}
    for name, opts in providers.items():
        if not isinstance(opts, dict):
            raise ValueError(f"{path}: provider {name!r} must be a mapping "
                             f"of options (got {type(opts).__name__})")
        cleaned[str(name)] = {str(k): v for k, v in opts.items()}
    return {"providers": cleaned}


def available_providers(config: Optional[dict[str, Any]] = None) -> str:
    """Human-readable name list for error messages."""
    names = list(PROVIDERS)
    if config:
        names += [n for n in config.get("providers", {}) if n not in names]
    return ", ".join(names)


def provider_names(config: Optional[dict[str, Any]] = None) -> list[str]:
    """All known provider names: built-ins first, then config entries."""
    names = list(PROVIDERS)
    if config:
        names += [n for n in config.get("providers", {}) if n not in names]
    return names


def build_provider_entry(*, kind: Optional[str] = None, base_url: str,
                         api_key: Optional[str] = None,
                         model: Optional[str] = None,
                         num_ctx: Optional[int] = None) -> dict[str, Any]:
    """Build a provider entry dict for the YAML config (None values dropped).

    ``kind`` may be omitted entirely — a name matching a built-in inherits that
    protocol; any other name defaults to the openai protocol.
    """
    entry: dict[str, Any] = {}
    if kind:
        entry["kind"] = kind
    if base_url:
        entry["base_url"] = base_url
    if api_key:
        entry["api_key"] = api_key
    if model:
        entry["model"] = model
    if num_ctx:
        entry["num_ctx"] = num_ctx
    return entry


# ---------------------------------------------------------------------------
# Resolved provider
# ---------------------------------------------------------------------------

@dataclass
class Provider:
    """Everything needed to build one chat model."""

    name: str = "ollama"        # wire protocol family: one of PROVIDERS
    model: str = ""
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    num_ctx: int = DEFAULT_NUM_CTX
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    label: str = ""             # config alias or display name, when name is a family

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


def resolve_provider(
        name: Optional[str] = None,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        num_ctx: Optional[int] = None,
        temperature: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
) -> Provider:
    """Resolve one named provider into a concrete Provider.

    Precedence per field: flag > config entry > env var > built-in default.
    """
    config = config or {}
    entries = config.get("providers", {}) if isinstance(config.get("providers", {}), dict) else {}

    # Provider name: only the explicit argument selects a provider; the
    # default is always ollama (config/env never switch the default).
    wanted = _first(name) or "ollama"
    entry = entries.get(str(wanted))
    if entry is None and wanted not in PROVIDERS:
        raise ValueError(
            f"unknown provider {wanted!r}. Known providers: {available_providers(config)}")

    # Wire protocol family: built-in names keep their protocol; custom names
    # default to openai unless the entry sets `kind`.
    family = entry.get("kind") if entry else None
    family = _first(family, wanted if wanted in PROVIDERS else None) or "openai"
    if family not in PROVIDERS:
        raise ValueError(
            f"provider {wanted!r}: kind {family!r} is not a supported protocol "
            f"({', '.join(PROVIDERS)}); set kind or use a built-in name")

    # Fall backs for the ollama-protocol family keep the legacy env names.
    legacy_model_env = os.environ.get("OLLAMA_MODEL") if family == "ollama" else None
    legacy_base_env = os.environ.get("OLLAMA_HOST") if family == "ollama" else None

    # Precedence per field: flag > config entry > env var > built-in default.
    resolved_model = _first(model, entry.get("model") if entry else None,
                            os.environ.get("HARNESS_MODEL"), legacy_model_env,
                            DEFAULT_MODELS[family]) or DEFAULT_MODELS[family]
    resolved_base = _first(base_url, entry.get("base_url") if entry else None,
                           os.environ.get("HARNESS_BASE_URL"), legacy_base_env,
                           DEFAULT_BASE_URLS[family])
    resolved_key = _first(api_key, entry.get("api_key") if entry else None,
                          api_key_from_env(family))
    extra = dict(entry.get("extra") or {}) if entry else {}

    return Provider(
        name=family,
        model=resolved_model,
        base_url=resolved_base,
        api_key=resolved_key,
        num_ctx=_first(num_ctx, entry.get("num_ctx") if entry else None) or DEFAULT_NUM_CTX,
        temperature=temperature if temperature is not None else 0.0,
        extra=extra,
        label=wanted if entry is not None else "",
    )