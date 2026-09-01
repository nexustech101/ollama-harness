"""Workspace tools for the agent: files, search, shell, tests, git.

Every tool is a LangChain BaseTool bound to a workspace root; paths that escape
that root are refused. Nothing here knows about the model or the console.
"""

from __future__ import annotations

import fnmatch
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

MAX_OUTPUT = 20_000        # cap on a single tool's returned text
MAX_READ_LINES = 300       # lines read_file returns when given no explicit range
MAX_WHOLE_READ = 4_000_000 # read_file refuses whole-file reads above this many bytes
IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".idea",
}
# tools that mutate the workspace or the machine; gated by the harness
DESTRUCTIVE = {"write_file", "apply_patch", "run_command", "run_tests",
               "git_restore", "install_dependency"}


def describe_environment(workspace: Path) -> str:
    """Facts the model cannot guess: which OS, which shell run_command actually
    uses, where temp files go, whether git is present. Cheap — no subprocesses."""
    windows = os.name == "nt"
    if windows:
        shell = os.environ.get("COMSPEC", r"C:\Windows\system32\cmd.exe")
        shell_note = " (cmd.exe, NOT PowerShell)" if "cmd" in shell.lower() else ""
    else:
        shell = os.environ.get("SHELL", "/bin/sh")
        shell_note = ""
    git = shutil.which("git")
    lines = [
        f"- OS: {platform.platform()} ({platform.system()})",
        f"- run_command runs through: {shell}{shell_note}",
        f"- Shell syntax: {'cmd.exe — no ls, grep, cat, /tmp, or POSIX quoting' if windows else 'POSIX sh'}",
        f"- Python: {sys.executable} ({platform.python_version()}); "
        "invoke it as `python -m pytest`, never bare `pytest`",
        f"- Temp directory: {tempfile.gettempdir()}" + (" — /tmp does not exist" if windows else ""),
        f"- Path separator: {os.sep!r}; tool arguments accept forward slashes and "
        "must stay relative to the workspace root",
        f"- git: {'available' if git else 'NOT installed — git tools will fail'}; "
        f"the workspace {'is' if (workspace / '.git').is_dir() else 'is NOT'} a git repository"
        + ("" if (workspace / ".git").is_dir() else " (apply_patch and git tools will not work)"),
        f"- Workspace root: {workspace}",
    ]
    return "Environment:\n" + "\n".join(lines)


def truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [truncated {len(text) - limit} chars]"



def run_proc(cmd: list[str] | str, cwd: Path, timeout: int) -> str:
    kwargs: dict[str, Any] = dict(
        shell=isinstance(cmd, str),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",      # stable decoding regardless of console code page
        errors="replace",
        timeout=timeout,
    )
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        p = subprocess.run(cmd, **kwargs)
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {timeout}s"
    except FileNotFoundError as e:
        return f"Error: command not found: {e}"
    out = [f"exit_code: {p.returncode}"]
    if p.stdout.strip():
        out.append(f"stdout:\n{p.stdout.rstrip()}")
    if p.stderr.strip():
        out.append(f"stderr:\n{p.stderr.rstrip()}")
    return truncate("\n".join(out))



class ToolArgs(BaseModel):
    """Base for every tool schema. Small models routinely wrap a scalar in a
    one-item list (test_path=["tests/x.py"]); accept that instead of erroring."""

    @model_validator(mode="before")
    @classmethod
    def _unwrap_singletons(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return {
            key: (value[0] if isinstance(value, list) and len(value) == 1
                  and isinstance(value[0], (str, int, float, bool)) else value)
            for key, value in data.items()
        }


class WorkspaceTool(BaseTool):
    """Base: declares workspace_root as a real pydantic field and sandboxes paths."""

    workspace_root: Path = Field(default_factory=Path.cwd)

    def resolve(self, path: Optional[str]) -> Path:
        root = self.workspace_root.resolve()
        target = (root / (path or ".")).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"path escapes the workspace: {path!r}")
        return target

    def rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.workspace_root.resolve()).as_posix() or "."
        except ValueError:
            return str(p)

    def walk(self, root: Path) -> Iterator[Path]:
        if root.is_file():
            yield root
            return
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
            for name in filenames:
                yield Path(dirpath) / name


class ReadFileArgs(ToolArgs):
    path: str = Field(description="File path relative to the workspace root.")
    start_line: Optional[int] = Field(None, description="1-based first line to return.")
    end_line: Optional[int] = Field(None, description="1-based last line to return (inclusive).")


class ReadFileTool(WorkspaceTool):
    name: str = "read_file"
    description: str = "Read a text file from the workspace. Returns numbered lines."
    args_schema: Type[BaseModel] = ReadFileArgs

    def _run(self, path: str, start_line: Optional[int] = None,
             end_line: Optional[int] = None) -> str:
        f = self.resolve(path)
        if not f.is_file():
            return f"Error: not a file: {path}"
        try:
            size = f.stat().st_size
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                if start_line is None and end_line is None:
                    # whole-file read, but never blow up the context on a giant file
                    if size > MAX_WHOLE_READ:
                        return (f"{self.rel(f)}: {size:,} bytes — too large to read "
                                "whole. Pass start_line/end_line for a range.")
                    lines = fh.readlines()
                    total = len(lines)
                    end = min(total, MAX_READ_LINES)
                    note = ""
                    if total > MAX_READ_LINES:
                        end = MAX_READ_LINES
                        note = (f"\n[showing first {MAX_READ_LINES} of {total} lines; "
                                "re-read with start_line/end_line for the rest]")
                    body = "\n".join(
                        f"{i:>5}  {lines[i - 1].rstrip()}" for i in range(1, end + 1))
                    return truncate(f"{self.rel(f)} (lines 1-{end} of {total})\n{body}{note}")

                # explicit range: stream only the requested lines
                first = max(1, start_line or 1)
                rows: list[str] = []
                for i, line in enumerate(fh, start=1):
                    if i < first:
                        continue
                    if end_line is not None and i > end_line:
                        break
                    rows.append(f"{i:>5}  {line.rstrip()}")
                    if end_line is None and len(rows) >= MAX_READ_LINES:
                        rows.append("... [re-read with end_line for the rest]")
                        break
                if not rows:
                    where = f"{first}" + (f"-{end_line}" if end_line else "+")
                    return f"{path}: no lines in range {where}"
                shown = f"{first}-{first + len(rows) - 1}"
                if rows and rows[-1].startswith("... ["):
                    shown = f"{first}-{first + len(rows) - 2}+"
                return truncate(f"{self.rel(f)} (lines {shown})\n" + "\n".join(rows))
        except OSError as e:
            return f"Error reading {path}: {e}"


class ListDirArgs(ToolArgs):
    path: str = Field(".", description="Directory relative to the workspace root.")
    recursive: bool = Field(False, description="Recurse into subdirectories.")


class ListDirectoryTool(WorkspaceTool):
    name: str = "list_directory"
    description: str = "List files and directories in the workspace."
    args_schema: Type[BaseModel] = ListDirArgs

    def _run(self, path: str = ".", recursive: bool = False) -> str:
        d = self.resolve(path)
        if not d.is_dir():
            return f"Error: not a directory: {path}"
        try:
            if recursive:
                entries = sorted(self.walk(d), key=lambda p: self.rel(p))
                rows = [f"  {self.rel(p)}" for p in entries]
            else:
                entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                rows = [f"  {p.name}/" if p.is_dir() else f"  {p.name}" for p in entries]
        except OSError as e:
            return f"Error listing {path}: {e}"
        return truncate("\n".join(rows)) if rows else "(empty)"


class SearchArgs(ToolArgs):
    query: str = Field(description="Regular expression to search for.")
    path: Optional[str] = Field(None, description="Directory or file to search under.")
    file_pattern: Optional[str] = Field(None, description="Glob filter on file names, e.g. '*.py'.")
    max_results: int = Field(50, description="Maximum matching lines to return.")


class SearchFilesTool(WorkspaceTool):
    name: str = "search_files"
    description: str = "Search workspace files for a regex; returns path:line and the matching line."
    args_schema: Type[BaseModel] = SearchArgs

    def _run(self, query: str, path: Optional[str] = None,
             file_pattern: Optional[str] = None, max_results: int = 50) -> str:
        try:
            pattern = re.compile(query)
        except re.error as e:
            return f"Error: invalid regex {query!r}: {e}"
        root = self.resolve(path)
        hits: list[str] = []
        for f in self.walk(root):
            if file_pattern and not fnmatch.fnmatch(f.name, file_pattern):
                continue
            try:
                if f.stat().st_size > 2_000_000:
                    continue
                text = f.read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeDecodeError):
                continue  # binary or unreadable
            for lineno, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    hits.append(f"{self.rel(f)}:{lineno}: {line.strip()[:200]}")
                    if len(hits) >= max_results:
                        return truncate("\n".join(hits) + f"\n[stopped at {max_results} matches]")
        return truncate("\n".join(hits)) if hits else "No matches found."


class FindArgs(ToolArgs):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py' or 'test_*.py'.")
    path: Optional[str] = Field(None, description="Directory to search under.")


class FindFilesTool(WorkspaceTool):
    name: str = "find_files"
    description: str = "Find files matching a glob pattern."
    args_schema: Type[BaseModel] = FindArgs

    def _run(self, pattern: str, path: Optional[str] = None) -> str:
        root = self.resolve(path)
        if not root.is_dir():
            return f"Error: not a directory: {path}"
        found = [
            self.rel(p) for p in sorted(root.rglob(pattern))
            if p.is_file() and not (set(p.parts) & IGNORED_DIRS)
        ]
        return truncate("\n".join(found)) if found else "No files matched."


class WriteArgs(ToolArgs):
    path: str = Field(description="File path relative to the workspace root.")
    content: str = Field(description="Full new contents of the file.")


class WriteFileTool(WorkspaceTool):
    name: str = "write_file"
    description: str = "Write complete contents to a file, creating parent directories as needed."
    args_schema: Type[BaseModel] = WriteArgs

    def _run(self, path: str, content: str) -> str:
        f = self.resolve(path)
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content, encoding="utf-8", newline="\n")
        except OSError as e:
            return f"Error writing {path}: {e}"
        return f"Wrote {len(content)} chars ({content.count(chr(10)) + 1} lines) to {self.rel(f)}"


class PatchArgs(ToolArgs):
    patch: str = Field(description="A unified diff, with a/ and b/ prefixes, applied at the workspace root.")


class ApplyPatchTool(WorkspaceTool):
    name: str = "apply_patch"
    description: str = (
        "Apply a unified diff to workspace files via `git apply`. "
        "Paths must be relative to the workspace root."
    )
    args_schema: Type[BaseModel] = PatchArgs

    def _run(self, patch: str) -> str:
        if not patch.endswith("\n"):
            patch += "\n"
        base = ["git", "apply", "--whitespace=nowarn"]
        try:
            check = subprocess.run(base + ["--check", "-"], cwd=str(self.workspace_root),
                                   input=patch, capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return f"Error: cannot run git apply: {e}"
        if check.returncode != 0:
            return f"Patch does not apply cleanly:\n{check.stderr.strip()}"
        applied = subprocess.run(base + ["-"], cwd=str(self.workspace_root),
                                 input=patch, capture_output=True, text=True, timeout=30)
        if applied.returncode != 0:
            return f"Error applying patch:\n{applied.stderr.strip()}"
        files = re.findall(r"^\+\+\+ b/(.+)$", patch, flags=re.M)
        return "Patch applied to: " + (", ".join(files) if files else "(unknown files)")


class CommandArgs(ToolArgs):
    command: str = Field(description="Shell command to execute.")
    cwd: Optional[str] = Field(None, description="Directory relative to the workspace root.")
    timeout: int = Field(60, description="Timeout in seconds.")


class RunCommandTool(WorkspaceTool):
    name: str = "run_command"
    description: str = "Execute a shell command in the workspace; returns exit code, stdout, stderr."
    args_schema: Type[BaseModel] = CommandArgs

    def _run(self, command: str, cwd: Optional[str] = None, timeout: int = 60) -> str:
        return run_proc(command, self.resolve(cwd), timeout)


class TestArgs(ToolArgs):
    test_path: Optional[str] = Field(None, description="File or directory of tests to run.")
    filter: Optional[str] = Field(None, description="pytest -k expression.")
    timeout: int = Field(300, description="Timeout in seconds.")


class RunTestsTool(WorkspaceTool):
    name: str = "run_tests"
    description: str = "Run the pytest suite and return the results."
    args_schema: Type[BaseModel] = TestArgs

    def _run(self, test_path: Optional[str] = None, filter: Optional[str] = None,
             timeout: int = 300) -> str:
        cmd = [sys.executable, "-m", "pytest", "-q"]
        if test_path:
            cmd.append(str(self.resolve(test_path)))
        if filter:
            cmd += ["-k", filter]
        return run_proc(cmd, self.workspace_root, timeout)


class GitDiffArgs(ToolArgs):
    path: Optional[str] = Field(None, description="Limit the diff to this path.")
    staged: bool = Field(False, description="Show staged changes instead of unstaged.")


class GitDiffTool(WorkspaceTool):
    name: str = "git_diff"
    description: str = "Show the current git diff."
    args_schema: Type[BaseModel] = GitDiffArgs

    def _run(self, path: Optional[str] = None, staged: bool = False) -> str:
        cmd = ["git", "--no-pager", "diff"] + (["--cached"] if staged else [])
        if path:
            cmd += ["--", str(self.resolve(path))]
        return run_proc(cmd, self.workspace_root, 30)


class NoArgs(ToolArgs):
    pass


class GitStatusTool(WorkspaceTool):
    name: str = "git_status"
    description: str = "Show the git working tree status."
    args_schema: Type[BaseModel] = NoArgs

    def _run(self) -> str:
        return run_proc(["git", "status", "--short", "--branch"], self.workspace_root, 30)


class GitLogArgs(ToolArgs):
    limit: int = Field(10, description="Number of commits to show.")
    path: Optional[str] = Field(None, description="Limit history to this path.")


class GitLogTool(WorkspaceTool):
    name: str = "git_log"
    description: str = "Show recent git commit history."
    args_schema: Type[BaseModel] = GitLogArgs

    def _run(self, limit: int = 10, path: Optional[str] = None) -> str:
        cmd = ["git", "--no-pager", "log", f"--max-count={limit}", "--oneline"]
        if path:
            cmd += ["--", str(self.resolve(path))]
        return run_proc(cmd, self.workspace_root, 30)


class GitRestoreArgs(ToolArgs):
    path: str = Field(description="File to restore from HEAD.")


class GitRestoreTool(WorkspaceTool):
    name: str = "git_restore"
    description: str = "Discard local changes to a file by restoring it from HEAD."
    args_schema: Type[BaseModel] = GitRestoreArgs

    def _run(self, path: str) -> str:
        target = self.resolve(path)
        out = run_proc(["git", "checkout", "HEAD", "--", str(target)], self.workspace_root, 30)
        return f"Restored {self.rel(target)}\n{out}"


class InspectEnvironmentTool(WorkspaceTool):
    name: str = "inspect_environment"
    description: str = "Report OS, Python, git and package-manager versions, and the workspace root."
    args_schema: Type[BaseModel] = NoArgs

    def _run(self) -> str:
        def version(cmd: list[str]) -> str:
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                return (p.stdout or p.stderr).strip().splitlines()[0] if p.returncode == 0 else "not found"
            except (OSError, subprocess.TimeoutExpired, IndexError):
                return "not found"

        return "\n".join([
            f"os: {platform.platform()}",
            f"python: {sys.version.split()[0]} ({sys.executable})",
            f"git: {version(['git', '--version'])}",
            f"uv: {version(['uv', '--version'])}",
            f"pytest: {version([sys.executable, '-m', 'pytest', '--version'])}",
            f"workspace: {self.workspace_root.resolve()}",
        ])


class InstallArgs(ToolArgs):
    package: str = Field(description="Package specifier, e.g. 'requests' or 'httpx>=0.27'.")


class InstallDependencyTool(WorkspaceTool):
    name: str = "install_dependency"
    description: str = "Install a Python package into the current environment (uv if available, else pip)."
    args_schema: Type[BaseModel] = InstallArgs

    def _run(self, package: str) -> str:
        if shutil.which("uv"):
            cmd = ["uv", "pip", "install", package]
        else:
            cmd = [sys.executable, "-m", "pip", "install", package]
        return run_proc(cmd, self.workspace_root, 300)


class FileInfoArgs(ToolArgs):
    path: str = Field(description="File or directory to inspect.")


class FileInfoTool(WorkspaceTool):
    name: str = "file_info"
    description: str = "Return size, type and modification time for a file or directory."
    args_schema: Type[BaseModel] = FileInfoArgs

    def _run(self, path: str) -> str:
        p = self.resolve(path)
        if not p.exists():
            return f"Error: does not exist: {path}"
        st = p.stat()
        mtime = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
        if p.is_dir():
            return f"directory {self.rel(p)}\nentries: {len(list(p.iterdir()))}\nmodified: {mtime}"
        return f"file {self.rel(p)}\nbytes: {st.st_size}\nmodified: {mtime}"


def create_tools(workspace_root: Path) -> list[BaseTool]:
    kinds = [
        ReadFileTool, ListDirectoryTool, SearchFilesTool, FindFilesTool,
        WriteFileTool, ApplyPatchTool, RunCommandTool, RunTestsTool,
        GitDiffTool, GitStatusTool, GitLogTool, GitRestoreTool,
        InspectEnvironmentTool, InstallDependencyTool, FileInfoTool,
    ]
    return [k(workspace_root=workspace_root) for k in kinds]


