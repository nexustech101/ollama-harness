"""Workspace tool tests: path sandboxing, singleton args, read ranges, patches."""

from __future__ import annotations

import pytest

from tools import (
    ApplyPatchTool,
    ReadFileTool,
    RunTestsTool,
    WriteFileTool,
    describe_environment,
    find_project_python,
)


def test_write_refuses_escape(tmp_path):
    tool = WriteFileTool(workspace_root=tmp_path)
    with pytest.raises(ValueError, match="escapes the workspace"):
        tool.invoke({"path": "../escape.txt", "content": "x"})
    assert not (tmp_path.parent / "escape.txt").exists()


def test_singleton_list_args_are_unwrapped(tmp_path):
    tool = WriteFileTool(workspace_root=tmp_path)
    out = tool.invoke({"path": ["a.txt"], "content": "hi"})  # model wrapped scalar in a list
    assert (tmp_path / "a.txt").exists()
    assert out.startswith("Wrote")


def test_read_file_range_streams_only_requested_lines(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("".join(f"line {i}\n" for i in range(1, 1001)))
    tool = ReadFileTool(workspace_root=tmp_path)
    out = tool.invoke({"path": "big.txt", "start_line": 3, "end_line": 5})
    assert "line 3" in out and "line 4" in out and "line 5" in out
    assert "line 6" not in out
    assert out.count("\n") <= 30  # lazy: far fewer lines than the file holds


def test_read_file_whole_caps_at_max_lines(tmp_path):
    f = tmp_path / "many.txt"
    f.write_text("".join(f"line {i}\n" for i in range(1, 501)))
    tool = ReadFileTool(workspace_root=tmp_path)
    out = tool.invoke({"path": "many.txt"})
    assert "line 300" in out
    assert "line 301" not in out
    assert "showing first 300 of 500" in out


def test_read_file_refuses_giant_whole_read(tmp_path):
    f = tmp_path / "huge.bin"
    f.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    tool = ReadFileTool(workspace_root=tmp_path)
    out = tool.invoke({"path": "huge.bin"})
    assert "too large" in out


def test_read_file_out_of_range_reports_clearly(tmp_path):
    f = tmp_path / "small.txt"
    f.write_text("one\ntwo\n")
    tool = ReadFileTool(workspace_root=tmp_path)
    assert "no lines in range" in tool.invoke({"path": "small.txt", "start_line": 5})


def test_apply_patch_replaces_lines(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("line1\nline2\n", encoding="utf-8")
    tool = ApplyPatchTool(workspace_root=tmp_path)
    patch = "--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n-line1\n-line2\n+LINE1\n+LINE2\n"
    out = tool.invoke({"patch": patch, "fuzz": 0})
    assert "applied 1/1 hunk(s)" in out
    assert f.read_text(encoding="utf-8") == "LINE1\nLINE2"


def test_apply_patch_multiple_hunks(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = ApplyPatchTool(workspace_root=tmp_path)
    patch = "--- a/a.txt\n+++ b/a.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+2\n three\n@@ -3,1 +3,1 @@\n-three\n+3\n"
    out = tool.invoke({"patch": patch, "fuzz": 0})
    assert "applied 2/2 hunk(s)" in out
    assert f.read_text(encoding="utf-8") == "one\n2\n3"


def test_apply_patch_failed_hunk_reports_partial(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one\n", encoding="utf-8")
    tool = ApplyPatchTool(workspace_root=tmp_path)
    patch = "--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n-one\n-missing\n+1\n+2\n"
    out = tool.invoke({"patch": patch, "fuzz": 0})
    assert "cannot apply" in out
    assert "no files were changed" in out
    assert f.read_text(encoding="utf-8") == "one\n"


def test_apply_patch_multiple_files(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello\n", encoding="utf-8")
    tool = ApplyPatchTool(workspace_root=tmp_path)
    patch = (
        "--- a/a.py\n+++ b/a.py\n"
        "@@ -1,3 +1,3 @@\n def f():\n-    return 1\n+    return 2\n"
        "--- a/b.txt\n+++ b/b.txt\n"
        "@@ -1 +1 @@\n-hello\n+goodbye\n"
    )
    out = tool.invoke({"patch": patch, "fuzz": 0})
    assert "applied 2/2 hunk(s)" in out
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "def f():\n    return 2"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "goodbye"


# ---------------------------------------------------------------------------
# Project interpreter: tools that run the project's code must use the project
# venv, not the harness's own interpreter (report item #6: run_tests -> 0 tests
# because the harness python lacked the project's dependencies).
# ---------------------------------------------------------------------------


def test_find_project_python_prefers_dot_venv(tmp_path):
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv" / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    assert find_project_python(tmp_path) == str(tmp_path / ".venv" / "Scripts" / "python.exe")


def test_find_project_python_falls_back_to_venv(tmp_path):
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "venv" / "bin" / "python").write_text("", encoding="utf-8")
    assert find_project_python(tmp_path) == str(tmp_path / "venv" / "bin" / "python")


def test_find_project_python_falls_back_to_harness(tmp_path):
    import sys

    assert find_project_python(tmp_path) == sys.executable


def test_run_tests_uses_project_python(tmp_path, monkeypatch):
    """run_tests must invoke the project venv's python, not the harness's."""
    import sys

    import tools

    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
    venv_py = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_py.write_text("", encoding="utf-8")

    captured: list = []

    def fake_run_proc(cmd, cwd, timeout):
        captured.append(cmd)
        return "exit_code: 0\nstdout:\n3 passed in 0.1s"

    monkeypatch.setattr(tools, "run_proc", fake_run_proc)
    tool = RunTestsTool(workspace_root=tmp_path)
    out = tool._run()
    assert captured and captured[0][0] == str(venv_py)
    assert captured[0][1:3] == ["-m", "pytest"]
    assert out.startswith("Tests: 3 passed")
    # sanity: the harness interpreter is a different path from the project venv
    assert str(venv_py) != sys.executable


def test_run_tests_summarize_surfaces_raw_output_when_no_result(tmp_path):
    """A collection error / missing interpreter must not be masked as 0 tests."""
    tool = RunTestsTool(workspace_root=tmp_path)
    raw = (
        "exit_code: 2\nstderr:\nERROR collecting tests/test_x.py\nModuleNotFoundError: No module named 'langchain_core'"
    )
    out = tool._summarize(raw)
    assert "did not report a result" in out
    assert "ModuleNotFoundError" in out
    assert "Tests: 0 passed" not in out


def test_run_tests_summarize_still_counts_results(tmp_path):
    tool = RunTestsTool(workspace_root=tmp_path)
    raw = "exit_code: 1\nstdout:\nFAILED tests/a.py::t1\n1 failed, 2 passed in 0.2s"
    out = tool._summarize(raw)
    assert "Tests: 2 passed, 1 failed" in out
    assert "tests/a.py::t1" in out


def test_apply_patch_no_hunks_is_diagnostic(tmp_path):
    """File headers without @@ hunk headers must explain what is missing."""
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    tool = ApplyPatchTool(workspace_root=tmp_path)
    patch = "--- a/a.txt\n+++ b/a.txt\n-old\n+new\n"  # no @@ header
    out = tool.invoke({"patch": patch, "fuzz": 0})
    assert "no hunks" in out
    assert "@@" in out  # tells the model the hunk header is required
    assert "Each change needs" in out


def test_describe_environment_mentions_project_venv(tmp_path):
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv" / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    env = describe_environment(tmp_path)
    assert "project venv" in env
    assert str(tmp_path / ".venv" / "Scripts" / "python.exe") in env
