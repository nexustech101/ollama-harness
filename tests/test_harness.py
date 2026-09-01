"""Agent-loop unit tests: tool dispatch, permissions, ownership, trimming.

These exercise the product-facing behaviour of `Harness` directly, without a
model backend: ``call_tool`` is the same code the loop runs for every tool
call, so gating, error paths and ownership enforcement are covered here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

@pytest.fixture
def harness(tmp_path: Path):
    from harness import Harness

    return Harness(model="m", workspace=tmp_path, base_url="http://127.0.0.1:1",
                   api_key="k", stream=False)


def _invoke(harness, name: str, **kwargs) -> str:
    """Run a tool through Harness.call_tool (permission-gated, ownership-checked)."""
    return harness.call_tool(name, kwargs)


# ---------------------------------------------------------------------------
# call_tool: dispatch, unknown tools, tool errors
# ---------------------------------------------------------------------------

def test_call_tool_dispatches_to_registered_tool(harness):
    out = harness.call_tool("file_info", {"path": "."})
    assert out.startswith("directory .")


def test_call_tool_reports_unknown_tool(harness, tmp_path: Path):
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    out = harness.call_tool("no_such_tool", {"path": "a.txt"})
    assert out.startswith("Error: unknown tool 'no_such_tool'")
    assert "read_file" in out          # lists the available tools


# ---------------------------------------------------------------------------
# Permissions (approve): ask / deny / allow
# ---------------------------------------------------------------------------

def test_deny_mode_blocks_destructive_tools(harness):
    harness.mode = "deny"
    out = _invoke(harness, "write_file", path="x.txt", content="hi")
    assert "read-only" in out
    assert not (harness.workspace / "x.txt").exists()


def test_deny_mode_allows_read_tools(harness, tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    harness.mode = "deny"
    out = _invoke(harness, "read_file", path="a.txt")
    assert "hello" in out


def test_ask_mode_denies_without_tty(harness):
    """ask mode is unusable headless: a destructive tool is refused because
    no terminal is attached."""
    harness.mode = "ask"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("harness.sys.stdin", type("FakeStdin", (), {"isatty": lambda self: False})())
        out = _invoke(harness, "write_file", path="x.txt", content="hi")
    assert "no terminal" in out


def test_allow_mode_runs_destructive_tools(harness):
    harness.mode = "allow"
    out = _invoke(harness, "write_file", path="y.txt", content="data")
    assert out.startswith("Wrote")
    assert (harness.workspace / "y.txt").read_text(encoding="utf-8") == "data"


# ---------------------------------------------------------------------------
# Sub-agent file ownership
# ---------------------------------------------------------------------------

def test_ownership_blocks_outside_files(harness, tmp_path: Path):
    (tmp_path / "mine.txt").write_text("mine", encoding="utf-8")
    (tmp_path / "theirs.txt").write_text("theirs", encoding="utf-8")
    harness.mode = "allow"
    harness.owned = {"mine.txt"}
    out = _invoke(harness, "write_file", path="theirs.txt", content="x")
    assert "belongs to another agent" in out
    assert (tmp_path / "theirs.txt").read_text(encoding="utf-8") == "theirs"


def test_ownership_allows_owned_files(harness, tmp_path: Path):
    harness.mode = "allow"
    harness.owned = {"mine.txt"}
    _invoke(harness, "write_file", path="mine.txt", content="x")
    assert (tmp_path / "mine.txt").read_text(encoding="utf-8") == "x"


def test_apply_patch_targets_are_ownership_checked(harness):
    harness.mode = "allow"
    harness.owned = {"a.txt"}
    patch = "--- a/b.txt\n+++ b/b.txt\n@@ -1 +1 @@\n-old\n+new\n"
    out = _invoke(harness, "apply_patch", patch=patch, fuzz=0)
    assert "belongs to another agent" in out


def test_apply_patch_owned_target_applies(harness, tmp_path: Path):
    (tmp_path / "a.txt").write_text("old\n", encoding="utf-8")
    harness.mode = "allow"
    harness.owned = {"a.txt"}
    patch = "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n"
    out = _invoke(harness, "apply_patch", patch=patch, fuzz=0)
    assert "applied 1/1 hunk(s)" in out
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "new"


# ---------------------------------------------------------------------------
# System prompt and message trimming
# ---------------------------------------------------------------------------

def test_system_prompt_includes_environment(harness):
    prompt = harness.system_prompt()
    assert "Workspace root" in prompt
    assert str(harness.workspace.resolve()) in prompt


def test_trim_drops_orphan_tool_results(harness):
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    harness.messages = [SystemMessage("sys")]
    for i in range(30):
        harness.messages.append(HumanMessage(f"h{i}"))
        harness.messages.append(ToolMessage(content="r", tool_call_id=f"t{i}"))
    harness.trim(keep=10)
    assert len(harness.messages) == 11          # system + 10 tail messages
    assert not isinstance(harness.messages[1], ToolMessage)


def test_trim_is_noop_under_limit(harness):
    from langchain_core.messages import HumanMessage, SystemMessage

    harness.messages = [SystemMessage("s"), HumanMessage("h")]
    harness.trim(keep=10)
    assert len(harness.messages) == 2


# ---------------------------------------------------------------------------
# write_targets: which files a tool call would modify
# ---------------------------------------------------------------------------

def test_write_targets_known_tools(harness):
    assert harness.write_targets("write_file", {"path": "x.py"}) == ["x.py"]
    assert harness.write_targets("git_restore", {"path": "y.py"}) == ["y.py"]
    assert harness.write_targets("read_file", {"path": "z.py"}) == []
    patch = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n--- a/b.py\n+++ b/b.py\n"
    targets = harness.write_targets("apply_patch", {"patch": patch})
    assert targets == ["a.py", "b.py"]


# ---------------------------------------------------------------------------
# stats / usage plumbing
# ---------------------------------------------------------------------------

def test_usage_from_message_counts():
    from harness import usage_from

    class Msg:
        usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        response_metadata = {}

    assert usage_from(Msg()) == {"input": 10, "output": 5, "eval_seconds": 0.0}


def test_usage_from_ollama_meta_fallback():
    from harness import usage_from

    class Msg:
        usage_metadata = None
        response_metadata = {"prompt_eval_count": 7, "eval_count": 3,
                             "eval_duration": 2_000_000_000}

    assert usage_from(Msg()) == {"input": 7, "output": 3, "eval_seconds": 2.0}


# ---------------------------------------------------------------------------
# formatting helpers used by the console and the API
# ---------------------------------------------------------------------------

def test_fmt_args_elides_long_values():
    from harness import fmt_args

    out = fmt_args({"path": "x.py", "content": "a" * 100})
    assert "content=a" in out and "…" in out


def test_summarize_error_multiline():
    from harness import summarize

    out = summarize("Error: boom\nTraceback ...\nMore", limit=120)
    assert "Error: boom" in out and "Traceback" in out


def test_normalize_base_url_rewrites_bind_addresses():
    from harness import normalize_base_url

    assert normalize_base_url("0.0.0.0:11434") == "http://127.0.0.1:11434"
    assert normalize_base_url("[::]:11434") == "http://127.0.0.1:11434"
    assert normalize_base_url("http://localhost:8000/") == "http://localhost:8000"


# ---------------------------------------------------------------------------
# spawn_agent validation (no model backend needed for the guards)
# ---------------------------------------------------------------------------

def test_delegate_requires_files(harness):
    assert "list the files" in harness.delegate("task", [])


def test_delegate_refuses_overlapping_ownership(harness):
    harness.claimed = {"a.txt": 1}
    out = harness.delegate("task", ["a.txt"])
    assert "already owned by sub-agent 1" in out