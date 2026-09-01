"""Provider config file tests: YAML parsing, named providers, precedence."""

from __future__ import annotations

import pytest

import providers

# NOTE: `default:` is accepted for compatibility but ignored — the default
# provider is always ollama, and the config is only read when --provider is used.
SAMPLE = """
default: openrouter
providers:
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key: sk-or-v1-xxxx
    model: openai/gpt-5.5
  groq:
    base_url: https://api.groq.com/openai/v1
    api_key: gsk-123
    model: llama-3.3-70b-versatile
    num_ctx: 16000
  local:
    kind: ollama
    base_url: http://127.0.0.1:11434
    model: qwen3
"""


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "providers.yaml"
    p.write_text(SAMPLE, encoding="utf-8")
    return providers.load_config(p)


def test_load_parses_named_providers(cfg):
    assert sorted(cfg["providers"]) == ["groq", "local", "openrouter"]
    assert "default" not in cfg  # accepted but ignored
    assert cfg["providers"]["openrouter"]["base_url"] == "https://openrouter.ai/api/v1"


def test_custom_name_defaults_to_openai_protocol(cfg):
    prov = providers.resolve_provider("groq", config=cfg)
    assert prov.name == "openai"           # wire protocol family
    assert prov.label == "groq"            # alias kept for display
    assert prov.model == "llama-3.3-70b-versatile"
    assert prov.api_key == "gsk-123"
    assert prov.endpoint == "https://api.groq.com/openai/v1"
    assert prov.num_ctx == 16000


def test_builtin_name_inherits_its_protocol(cfg):
    prov = providers.resolve_provider("openrouter", config=cfg)
    assert prov.name == "openrouter"
    assert prov.model == "openai/gpt-5.5"  # config default model
    assert prov.api_key == "sk-or-v1-xxxx"
    assert prov.endpoint == "https://openrouter.ai/api/v1"


def test_config_entry_can_switch_protocol(cfg):
    prov = providers.resolve_provider("local", config=cfg)
    assert prov.name == "ollama"
    assert prov.model == "qwen3"
    assert prov.endpoint == "http://127.0.0.1:11434"


def test_flag_overrides_config(monkeypatch, cfg):
    prov = providers.resolve_provider("openrouter", model="gpt-5", config=cfg)
    assert prov.model == "gpt-5"           # --model beats config model
    prov2 = providers.resolve_provider("openrouter", api_key="sk-flag", config=cfg)
    assert prov2.api_key == "sk-flag"      # --api-key beats config api_key


def test_no_provider_means_ollama_even_with_config_default(monkeypatch, cfg):
    monkeypatch.delenv("HARNESS_PROVIDER", raising=False)
    monkeypatch.delenv("HARNESS_MODEL", raising=False)
    monkeypatch.delenv("HARNESS_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    prov = providers.resolve_provider(None, config=cfg)  # config has default: openrouter
    assert prov.name == "ollama"           # the config default does NOT select it
    assert prov.label == ""
    assert prov.endpoint == providers.DEFAULT_BASE_URLS["ollama"]


def test_env_never_selects_a_provider(monkeypatch, cfg):
    monkeypatch.delenv("HARNESS_PROVIDER", raising=False)
    monkeypatch.setenv("HARNESS_MODEL", "gpt-x")
    monkeypatch.setenv("HARNESS_BASE_URL", "http://env-endpoint:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama-env")
    prov = providers.resolve_provider(None, config=cfg)
    assert prov.name == "ollama"           # only --provider switches providers
    assert prov.model == "gpt-x"           # .env model config still applies to ollama
    assert prov.endpoint == "http://env-endpoint:11434"  # .env endpoint applies too
    monkeypatch.delenv("HARNESS_MODEL")
    monkeypatch.delenv("HARNESS_BASE_URL")
    monkeypatch.delenv("OLLAMA_MODEL")


def test_ollama_default_uses_env_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1")
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.local:11434")
    prov = providers.resolve_provider(None)
    assert prov.name == "ollama"
    assert prov.model == "llama3.1"        # .env ollama config is used
    assert prov.endpoint == "http://ollama.local:11434"


def test_unknown_provider_raises_with_list(cfg):
    with pytest.raises(ValueError, match="unknown provider 'nope'.*Known"):
        providers.resolve_provider("nope", config=cfg)


def test_invalid_kind_raises(tmp_path):
    p = tmp_path / "providers.yaml"
    p.write_text("providers:\n  x:\n    kind: anthropic\n", encoding="utf-8")
    cfg = providers.load_config(p)
    with pytest.raises(ValueError, match="not a supported protocol"):
        providers.resolve_provider("x", config=cfg)


def test_find_config_prefers_explicit(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("providers: {}\n", encoding="utf-8")
    (cwd / "providers.yaml").write_text("providers: {}\n", encoding="utf-8")
    assert providers.find_config(str(explicit)) == explicit
    assert providers.find_config() == cwd / "providers.yaml"  # cwd beats user dirs


def test_bad_yaml_raises(tmp_path):
    p = tmp_path / "providers.yaml"
    p.write_text("providers: [not, a, mapping]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'providers' must be a mapping"):
        providers.load_config(p)


def test_build_provider_entry_omits_kind_for_builtin():
    entry = providers.build_provider_entry(
        base_url="https://openrouter.ai/api/v1", api_key="sk-x", model="m")
    assert entry == {"base_url": "https://openrouter.ai/api/v1",
                     "api_key": "sk-x", "model": "m"}  # no kind: inherited
    entry2 = providers.build_provider_entry(
        kind="openai", base_url="https://x/v1", model="m")
    assert entry2 == {"kind": "openai", "base_url": "https://x/v1", "model": "m"}


def test_add_provider_to_config_merges(tmp_path):
    import main as m

    cfg = tmp_path / "providers.yaml"
    m.add_provider_to_config(cfg, "openrouter",
                             {"base_url": "https://openrouter.ai/api/v1",
                              "api_key": "sk-a", "model": "openai/gpt-5.5"})
    m.add_provider_to_config(cfg, "groq",
                             {"kind": "openai", "base_url": "https://api.groq.com/openai/v1",
                              "api_key": "gsk-b", "model": "llama-3.3-70b-versatile"})
    parsed = providers.load_config(cfg)
    assert sorted(parsed["providers"]) == ["groq", "openrouter"]
    assert parsed["providers"]["openrouter"]["model"] == "openai/gpt-5.5"
    assert parsed["providers"]["groq"]["kind"] == "openai"


def test_init_provider_fully_flagged_writes_file(tmp_path):
    import argparse

    import main as m

    target = tmp_path / "providers.yaml"
    args = argparse.Namespace(
        config=str(target), name="openrouter",
        base_url="https://openrouter.ai/api/v1", model="openai/gpt-5.5",
        kind=None, api_key="sk-or-v1-xxxx")
    m.run_init_provider(args)
    parsed = providers.load_config(target)
    entry = parsed["providers"]["openrouter"]
    assert entry["base_url"] == "https://openrouter.ai/api/v1"
    assert entry["api_key"] == "sk-or-v1-xxxx"
    assert "kind" not in entry             # openrouter inherits its protocol