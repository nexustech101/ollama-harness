"""Provider configuration tests: defaults, env resolution, endpoint selection."""

from __future__ import annotations

import providers


def test_provider_defaults_are_complete():
    assert set(providers.PROVIDERS) == {"ollama", "openai", "openrouter"}
    for name in providers.PROVIDERS:
        assert providers.DEFAULT_BASE_URLS[name].startswith(("http://", "https://"))
        assert providers.DEFAULT_MODELS[name]


def test_api_key_from_env_prefers_provider_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "rk-123")
    monkeypatch.setenv("HARNESS_API_KEY", "hk-456")
    assert providers.api_key_from_env("openrouter") == "rk-123"
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert providers.api_key_from_env("openrouter") == "hk-456"
    monkeypatch.delenv("HARNESS_API_KEY")
    assert providers.api_key_from_env("openrouter") is None


def test_ollama_needs_no_key(monkeypatch):
    monkeypatch.delenv("HARNESS_API_KEY", raising=False)
    assert providers.api_key_from_env("ollama") is None


def test_endpoint_prefers_explicit_base_url():
    assert providers.Provider("openai", "m", base_url="http://local:8000/v1").endpoint == "http://local:8000/v1"
    assert providers.Provider("openai", "m").endpoint == providers.DEFAULT_BASE_URLS["openai"]
    assert providers.Provider("openrouter", "m").endpoint == providers.DEFAULT_BASE_URLS["openrouter"]


def test_build_ollama_returns_bound_model():
    model = providers.Provider("ollama", "qwen3", num_ctx=8192).build()
    assert model.model == "qwen3"
    assert model.num_ctx == 8192