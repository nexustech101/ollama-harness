"""Provider config file tests: YAML parsing, named providers, precedence."""

from __future__ import annotations

import os

import pytest

import providers

# NOTE: `default:` inside a provider entry is the default model (the harness
# uses it unless --model is given). The top-level `default:` key is accepted
# for compatibility but ignored — the default provider is always ollama, and
# the config is only read when --provider is used.
SAMPLE = """
default: openrouter
providers:
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key: sk-or-v1-xxxx
    models: [openai/gpt-5.5, deepseek/deepseek-chat]
    default: openai/gpt-5.5
  groq:
    base_url: https://api.groq.com/openai/v1
    api_key: gsk-123
    default: llama-3.3-70b-versatile
    num_ctx: 16000
  local:
    kind: ollama
    base_url: http://127.0.0.1:11434
    default: qwen3
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
    prov = providers.resolve_provider("openrouter", model="deepseek/deepseek-chat",
                                      config=cfg)
    assert prov.model == "deepseek/deepseek-chat"  # --model beats the default field
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


def test_find_config_ignores_current_directory(tmp_path, monkeypatch):
    """The config belongs to the harness, not the cwd: a providers.yaml left
    in the directory where you run harness is NOT picked up."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    if os.name == "nt":
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    else:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (cwd / "providers.yaml").write_text("providers: {}\n", encoding="utf-8")
    assert providers.find_config() is None


def test_find_config_uses_harness_config_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("HARNESS_CONFIG", raising=False)
    if os.name == "nt":
        monkeypatch.setenv("APPDATA", str(tmp_path))
        expected = tmp_path / "harness" / "providers.yaml"
    else:
        monkeypatch.setenv("HOME", str(tmp_path))
        expected = tmp_path / ".config" / "harness" / "providers.yaml"
    assert providers.find_config() is None          # nothing there yet
    expected.parent.mkdir(parents=True)
    expected.write_text("providers: {}\n", encoding="utf-8")
    assert providers.find_config() == expected      # found in the harness dir


def test_bad_yaml_raises(tmp_path):
    p = tmp_path / "providers.yaml"
    p.write_text("providers: [not, a, mapping]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'providers' must be a mapping"):
        providers.load_config(p)


def test_build_provider_entry_omits_kind_for_builtin():
    entry = providers.build_provider_entry(
        base_url="https://openrouter.ai/api/v1", api_key="sk-x",
        default_model="m")
    assert entry == {"base_url": "https://openrouter.ai/api/v1",
                     "api_key": "sk-x", "default": "m"}  # no kind: inherited
    entry2 = providers.build_provider_entry(
        kind="openai", base_url="https://x/v1", default_model="m")
    assert entry2 == {"kind": "openai", "base_url": "https://x/v1", "default": "m"}


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
        kind=None, api_key="sk-or-v1-xxxx", models=None)
    m.run_init_provider(args)
    parsed = providers.load_config(target)
    entry = parsed["providers"]["openrouter"]
    assert entry["base_url"] == "https://openrouter.ai/api/v1"
    assert entry["api_key"] == "sk-or-v1-xxxx"
    assert entry["default"] == "openai/gpt-5.5"   # --model writes the default field
    assert "model" not in entry
    assert "kind" not in entry             # openrouter inherits its protocol


def test_models_list_default_is_first(tmp_path):
    p = tmp_path / "providers.yaml"
    p.write_text(
        "providers:\n"
        "  openrouter:\n"
        "    base_url: https://openrouter.ai/api/v1\n"
        "    models: [openai/gpt-5.5, deepseek/deepseek-chat]\n",
        encoding="utf-8")
    cfg = providers.load_config(p)
    prov = providers.resolve_provider("openrouter", config=cfg)
    assert prov.models == ("openai/gpt-5.5", "deepseek/deepseek-chat")
    assert prov.model == "openai/gpt-5.5"      # first entry is the default


def test_model_flag_must_be_in_models_list(tmp_path):
    p = tmp_path / "providers.yaml"
    p.write_text(
        "providers:\n"
        "  openrouter:\n"
        "    base_url: https://openrouter.ai/api/v1\n"
        "    models: [openai/gpt-5.5, deepseek/deepseek-chat]\n",
        encoding="utf-8")
    cfg = providers.load_config(p)
    pick = providers.resolve_provider("openrouter", model="deepseek/deepseek-chat",
                                      config=cfg)
    assert pick.model == "deepseek/deepseek-chat"
    with pytest.raises(ValueError, match="not among its configured models"):
        providers.resolve_provider("openrouter", model="not-a-model", config=cfg)


def test_default_field_beats_first_of_models(tmp_path):
    p = tmp_path / "providers.yaml"
    p.write_text(
        "providers:\n"
        "  openrouter:\n"
        "    base_url: https://openrouter.ai/api/v1\n"
        "    default: deepseek/deepseek-chat\n"
        "    models: [openai/gpt-5.5, deepseek/deepseek-chat]\n",
        encoding="utf-8")
    cfg = providers.load_config(p)
    prov = providers.resolve_provider("openrouter", config=cfg)
    assert prov.model == "deepseek/deepseek-chat"   # `default:` wins over first


def test_legacy_model_key_is_an_alias_for_default(tmp_path):
    p = tmp_path / "providers.yaml"
    p.write_text(
        "providers:\n"
        "  openrouter:\n"
        "    base_url: https://openrouter.ai/api/v1\n"
        "    model: deepseek/deepseek-chat\n",
        encoding="utf-8")
    cfg = providers.load_config(p)
    assert providers.resolve_provider("openrouter", config=cfg).model == \
        "deepseek/deepseek-chat"


def test_malformed_models_list_raises(tmp_path):
    p = tmp_path / "providers.yaml"
    p.write_text(
        "providers:\n"
        "  bad:\n"
        "    base_url: https://x/v1\n"
        "    models: not-a-list\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="'models' must be a list"):
        providers.load_config(p)


def test_harness_set_model_enforces_curated_list(tmp_path):
    from harness import Harness

    h = Harness(model="a", workspace=tmp_path, base_url="http://127.0.0.1:1",
                models=("a", "b"))
    assert h.set_model("b") is None
    assert h.model == "b"
    assert "not among" in h.set_model("zzz")   # returns an error string, not raise


def test_api_v1_models_lists_curated(tmp_path):
    from api import ServerConfig, create_app

    app = create_app(ServerConfig(
        model="a", base_url="http://x", workspace=tmp_path,
        models=("a", "b", "c")))
    client = pytest.importorskip("fastapi.testclient").TestClient(app)
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["a", "b", "c"]


def test_add_model_appends_to_models_list(tmp_path):
    import argparse

    import main as m

    target = tmp_path / "providers.yaml"
    target.write_text(
        "providers:\n"
        "  openrouter:\n"
        "    base_url: https://openrouter.ai/api/v1\n"
        "    models: [openai/gpt-5.5, deepseek/deepseek-chat]\n",
        encoding="utf-8")
    args = argparse.Namespace(provider="openrouter", config=str(target),
                              add="deepseek/deepseek-v4-flash-0731")
    m.run_add_model(args)
    assert providers.load_config(target)["providers"]["openrouter"]["models"] == [
        "openai/gpt-5.5", "deepseek/deepseek-chat", "deepseek/deepseek-v4-flash-0731"]


def test_add_model_converts_single_default_to_list(tmp_path):
    import argparse

    import main as m

    target = tmp_path / "providers.yaml"
    target.write_text(
        "providers:\n"
        "  openrouter:\n"
        "    base_url: https://openrouter.ai/api/v1\n"
        "    default: openai/gpt-5.5\n",
        encoding="utf-8")
    args = argparse.Namespace(provider="openrouter", config=str(target),
                              add="deepseek/deepseek-v4-flash-0731")
    m.run_add_model(args)
    entry = providers.load_config(target)["providers"]["openrouter"]
    assert entry["models"] == ["openai/gpt-5.5", "deepseek/deepseek-v4-flash-0731"]
    assert entry["default"] == "openai/gpt-5.5"   # default field preserved


def test_add_model_creates_entry_for_new_provider(tmp_path):
    import argparse

    import main as m

    target = tmp_path / "providers.yaml"
    target.write_text("providers:\n", encoding="utf-8")
    args = argparse.Namespace(provider="openrouter", config=str(target),
                              add="qwen3.8-coder")
    m.run_add_model(args)
    assert providers.load_config(target)["providers"]["openrouter"]["models"] == ["qwen3.8-coder"]


def test_add_model_duplicate_is_noop(tmp_path, capsys):
    import argparse

    import main as m

    target = tmp_path / "providers.yaml"
    target.write_text(
        "providers:\n"
        "  openrouter:\n"
        "    models: [openai/gpt-5.5]\n",
        encoding="utf-8")
    args = argparse.Namespace(provider="openrouter", config=str(target),
                              add="openai/gpt-5.5")
    m.run_add_model(args)
    assert "already" in capsys.readouterr().out
    assert providers.load_config(target)["providers"]["openrouter"]["models"] == ["openai/gpt-5.5"]


def test_add_model_without_provider_errors(tmp_path):
    import argparse

    import main as m

    args = argparse.Namespace(provider=None, config=str(tmp_path / "p.yaml"),
                              add="x")
    with pytest.raises(SystemExit):
        m.run_add_model(args)


def test_set_default_replaces_the_default_field(tmp_path):
    import argparse

    import main as m

    target = tmp_path / "providers.yaml"
    target.write_text(
        "providers:\n"
        "  openrouter:\n"
        "    base_url: https://openrouter.ai/api/v1\n"
        "    model: openai/gpt-5.5\n",
        encoding="utf-8")
    args = argparse.Namespace(provider="openrouter", config=str(target),
                              set_default="deepseek/deepseek-v4-flash-0731")
    m.run_set_default_model(args)
    entry = providers.load_config(target)["providers"]["openrouter"]
    assert entry["default"] == "deepseek/deepseek-v4-flash-0731"  # replaced
    assert "model" not in entry           # legacy alias superseded


def test_set_default_leaves_models_list_untouched(tmp_path):
    import argparse

    import main as m

    target = tmp_path / "providers.yaml"
    target.write_text(
        "providers:\n"
        "  openrouter:\n"
        "    base_url: https://openrouter.ai/api/v1\n"
        "    default: openai/gpt-5.5\n"
        "    models: [openai/gpt-5.5, deepseek/deepseek-chat]\n",
        encoding="utf-8")
    args = argparse.Namespace(provider="openrouter", config=str(target),
                              set_default="deepseek/deepseek-chat")
    m.run_set_default_model(args)
    entry = providers.load_config(target)["providers"]["openrouter"]
    assert entry["default"] == "deepseek/deepseek-chat"   # default field replaced
    assert entry["models"] == ["openai/gpt-5.5", "deepseek/deepseek-chat"]  # untouched, no reorder


def test_set_default_creates_entry_for_new_provider(tmp_path):
    import argparse

    import main as m

    target = tmp_path / "providers.yaml"
    target.write_text("providers:\n", encoding="utf-8")
    args = argparse.Namespace(provider="openrouter", config=str(target),
                              set_default="my-model")
    m.run_set_default_model(args)
    entry = providers.load_config(target)["providers"]["openrouter"]
    assert entry["default"] == "my-model"


def test_set_default_without_provider_errors(tmp_path):
    import argparse

    import main as m

    args = argparse.Namespace(provider=None, config=str(tmp_path / "p.yaml"),
                              set_default="x")
    with pytest.raises(SystemExit):
        m.run_set_default_model(args)


def test_default_field_used_unless_model_flag(tmp_path):
    p = tmp_path / "providers.yaml"
    p.write_text(
        "providers:\n"
        "  openrouter:\n"
        "    base_url: https://openrouter.ai/api/v1\n"
        "    default: deepseek/deepseek-v4-flash-0731\n"
        "    models: [openai/gpt-5.5, deepseek/deepseek-v4-flash-0731]\n",
        encoding="utf-8")
    cfg = providers.load_config(p)
    assert providers.resolve_provider("openrouter", config=cfg).model == \
        "deepseek/deepseek-v4-flash-0731"           # default field, no flag
    assert providers.resolve_provider("openrouter", model="openai/gpt-5.5",
                                      config=cfg).model == "openai/gpt-5.5"  # flag wins