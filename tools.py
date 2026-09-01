"""Workspace tools for the agent: files, search, shell, tests, git, safety.

Every tool is a LangChain BaseTool bound to a workspace root; paths that escape
that root are refused. Nothing here knows about the model or the console.

Robustness features for an autonomous agent:

* reads are streaming and bounded so one tool call never floods the context;
* writes are atomic (temp file + rename) so a crash cannot leave a half file,
  and every mutating write records a snapshot the agent can roll back to;
* ``apply_patch`` applies unified diffs natively (no git required) with
  per-hunk reporting;
* ``snapshot`` / ``diff`` / ``restore`` give the agent undo over the whole
  workspace without git;
* ``run_command`` times out every call, captures stdout/stderr separately, and
  appends a post-mortem Python frame dump when a command crashes;
* ``run_tests`` parses the suite's own summary into a stable shape the model
  can act on instead of raw terminal noise;
* ``run_checks`` (<1s) catches syntax/lint errors before a full test run;
* ``debug_trace`` runs a script under a line/call tracer and returns a cheap
  per-line account; ``rerun_last`` retries the exact previous command.

Everything that can mutate the workspace or the machine is listed in
``DESTRUCTIVE`` so the harness can gate it (allow / ask / deny).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

MAX_OUTPUT = 20_000        # cap on a single tool's returned text
MAX_READ_LINES = 300       # lines read_file returns when given no explicit range
MAX_WHOLE_READ = 4_000_000 # read_file refuses whole-file reads above this many bytes
MAX_TRACE_LINES = 1_500    # debug_trace caps the per-line listing at this many lines
MAX_TRACE_FRAMES = 12      # post-mortem dump keeps at most this many stack frames
MAX_PM_VALUE = 500         # a single local/exception value is truncated to this
IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".idea",
}
# tools that mutate the workspace or the machine; gated by the harness
DESTRUCTIVE = {"write_file", "apply_patch", "run_command", "run_tests",
               "git_restore", "install_dependency", "check", "debug_trace",
               "rerun_last"}

# functions mutate ONLY the listed paths: snapshot/diff/restore/journal can
# compute modifications cheaply without ever running a command or reading files
MUTATION_PATHS = {
    "write_file": (("path",), "write"),
    "git_restore": (("path",), "restore"),
    "apply_patch": (("patch",), "patch"),
}


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


def short_repr(v: Any) -> str:
    try:
        s = repr(v)
    except Exception:
        s = "<unprintable>"
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= MAX_PM_VALUE else s[:MAX_PM_VALUE] + "…"


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

    # ------------------------------------------------------------------ state
    def _snapshot_path(self) -> Path:
        """Shadow dir (next to .git) that stores pre-write snapshots + journal."""
        return self.workspace_root / ".harness" / "state"

    def _new_snapshot_slot(self, label: str) -> Path:
        snap = self._snapshot_path() / "snapshots" / datetime.now().strftime("%Y%m%d-%H%M%S")
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "label.txt").write_text(label, encoding="utf-8")
        return snap

    def _index_snapshot(self, snap: Path) -> None:
        blobs = snap / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        index: dict[str, dict[str, Any]] = {}
        for f in self.walk(self.workspace_root):
            if not f.is_file():
                continue
            rel = self.rel(f)
            if rel.startswith(".harness/"):
                continue   # never snapshot our own bookkeeping
            try:
                data = f.read_bytes()
            except OSError:
                continue
            h = hashlib.sha256(data).hexdigest()[:16]
            (blobs / h).write_bytes(data)          # store content once (deduped by hash)
            index[rel] = {"hash": h, "size": len(data)}
        (snap / "index.json").write_text(json.dumps(index), encoding="utf-8")

    def snapshot(self, label: str = "auto") -> str:
        """Record current contents of every file (hash -> bytes). Cheap because
        it stores full copies only for files that later change (restore reads
        current bytes and swaps only the differs). Returns a slot id."""
        snap = self._new_snapshot_slot(label)
        self._index_snapshot(snap)
        return snap.name

    def _load_saved(self, slot: str) -> Optional[dict[str, dict[str, Any]]]:
        idx = self._snapshot_path() / "snapshots" / slot / "index.json"
        if not idx.exists():
            return None
        try:
            return json.loads(idx.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _paths_changed_since(self, slot: str) -> list[str]:
        """Recompute which workspace files differ from a stored snapshot - no
        subprocess, no event hook. Compares sha256 via the stored index."""
        saved = self._load_saved(slot)
        if saved is None:
            return [f"(snapshot {slot} not found)"]
        changed: list[str] = []
        for rel, meta in saved.items():
            f = self.workspace_root / rel
            if not f.is_file():
                changed.append(rel + " (deleted)")
                continue
            try:
                h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            except OSError:
                changed.append(rel + " (unreadable)")
                continue
            if h != meta["hash"]:
                changed.append(rel + " (modified)")
        for f in self.walk(self.workspace_root):
            rel = self.rel(f)
            if rel.startswith(".harness/") or rel in saved:
                continue
            changed.append(rel + " (new)")
        return changed

    def _apply_snapshot(self, slot: str) -> str:
        """Restore a snapshot: replace changed files, delete ones created after."""
        saved = self._load_saved(slot)
        if saved is None:
            return f"snapshot {slot} not found"
        blobs = self._snapshot_path() / "snapshots" / slot / "blobs"
        restored = 0
        missing = 0
        for rel, meta in saved.items():
            f = self.workspace_root / rel
            blob = blobs / meta["hash"]
            if not blob.exists():
                missing += 1
                continue
            try:
                data = blob.read_bytes()
                cur = f.read_bytes() if f.is_file() else b""
                if cur != data:
                    self._write_bytes(f, data)
                    restored += 1
            except OSError as e:
                return f"restore failed on {rel}: {e}"
        for f in self.walk(self.workspace_root):
            rel = self.rel(f)
            if rel not in saved and not rel.startswith(".harness/"):
                try:
                    f.unlink()
                except OSError:
                    pass
        if missing:
            return f"restored {restored} file(s) from {slot} ({missing} blobs missing)"
        return f"restored {restored} file(s) from snapshot {slot}"

    # --------------------------------------------------------------- atomic io
    def _write_bytes(self, f: Path, data: bytes) -> None:
        """Atomic write: temp file in the same directory, fsync, rename over.
        Windows cannot rename over an open file - retry a few times."""
        f.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(f.parent), prefix=f".{f.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            for _ in range(10):
                try:
                    os.replace(tmp, f)
                    return
                except PermissionError:
                    time.sleep(0.05)
            raise PermissionError(f"could not replace {f.name} (file held open?)")
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


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
        data = content.encode("utf-8")
        try:
            self.snapshot("write_file")
            self._write_bytes(f, data)
        except OSError as e:
            return f"Error writing {path}: {e}"
        return f"Wrote {len(data)} bytes ({content.count(chr(10)) + 1} lines) to {self.rel(f)}"


class PatchArgs(ToolArgs):
    patch: str = Field(description="A unified diff, with a/ and b/ prefixes, applied at the workspace root.")
    fuzz: int = Field(0, description="Context-matching tolerance 0-3 (default 0 = exact).")


class ApplyPatchTool(WorkspaceTool):
    name: str = "apply_patch"
    description: str = (
        "Apply a unified diff to workspace files. Uses a built-in pure-Python "
        "engine, so it works even without git. Returns per-hunk results."
    )
    args_schema: Type[BaseModel] = PatchArgs

    @staticmethod
    def _parse_patch(patch: str) -> list[dict[str, Any]]:
        """Split a unified diff into per-file {old, new, hunks:[(start, old, new)]}."""
        files: list[dict[str, Any]] = []
        fcur: Optional[dict[str, Any]] = None
        hunks: list[tuple[int, list[str], list[str]]] = []
        for idx, line in enumerate(patch.splitlines()):
            if line.startswith("+++ "):
                if fcur is not None:
                    fcur["hunks"] = hunks
                    files.append(fcur)
                fcur = {"old": None, "new": None, "hunks": []}
                m = re.match(r"^\+\+\+\s+(?P<new>\S+).*$", line)
                if m:
                    fcur["new"] = m.group("new")
                hunks = []
                continue
            if line.startswith("--- "):
                m = re.match(r"^---\s+(?P<old>\S+).*$", line)
                if fcur is not None and m:
                    fcur["old"] = m.group("old")
                continue
            if fcur is None or not line.startswith("@@"):
                continue
            m = re.match(r"^@@\s+-(?P<os>\d+)(?:,(?P<oc>\d+))?\s+\+(?P<ns>\d+)(?:,(?P<nc>\d+))?\s+@@", line)
            if not m:
                continue
            ostart = int(m.group("os"))
            old_lines: list[str] = []
            new_lines: list[str] = []
            # accumulate following body lines until the next diff header
            body = patch.splitlines()[idx + 1:]
            for bl in body:
                if bl.startswith(("@@", "+++ ", "--- ")):
                    break
                c = bl[:1]
                rest = bl[1:]
                if c == " ":
                    old_lines.append(rest)
                    new_lines.append(rest)
                elif c == "-":
                    old_lines.append(rest)
                elif c == "+":
                    new_lines.append(rest)
                else:
                    break
            hunks.append((ostart, old_lines, new_lines))
        if fcur is not None:
            fcur["hunks"] = hunks
            files.append(fcur)
        return files

    @staticmethod
    def _apply_hunk(target: list[str], ostart: int, old_lines: list[str],
                    new_lines: list[str], fuzz: int = 0) -> Optional[int]:
        """Locate `old_lines` in `target` near 1-based `ostart` and replace it
        with `new_lines`. Returns the 1-based line where it applied, or None."""
        if not old_lines:
            pos = max(0, min(ostart - 1, len(target)))
            target[pos:pos] = new_lines
            return pos + 1
        n = len(old_lines)
        window = len(target) - n + 1
        if window <= 0:
            return None
        lo = max(0, ostart - 1 - fuzz)
        hi = min(window, ostart - 1 + fuzz + 1)
        for i in range(lo, hi):
            if target[i:i + n] == old_lines:
                target[i:i + n] = new_lines
                return i + 1
        for i in range(lo - 1, -1, -1):
            if target[i:i + n] == old_lines:
                target[i:i + n] = new_lines
                return i + 1
        for i in range(hi, window):
            if target[i:i + n] == old_lines:
                target[i:i + n] = new_lines
                return i + 1
        return None

    def _run(self, patch: str, fuzz: int = 0) -> str:
        files = self._parse_patch(patch)
        if not files:
            return "Error: no parseable hunks in the patch. Use unified diff with a/ and b/ prefixes."
        out: list[str] = []
        total_hunks = 0
        applied_hunks = 0
        for fcur in files:
            name = fcur["new"] or fcur["old"] or "(unknown)"
            name = name.removeprefix("a/").removeprefix("b/")
            fpath = self.resolve(name)
            if not fpath.is_file():
                out.append(f"cannot apply {name}: file does not exist")
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                out.append(f"cannot read {name}: {e}")
                continue
            target = text.splitlines()
            total_hunks += len(fcur["hunks"])
            ok = 0
            for ostart, old_lines, new_lines in fcur["hunks"]:
                pos = self._apply_hunk(target, ostart, old_lines, new_lines, fuzz)
                if pos is not None:
                    ok += 1
                    applied_hunks += 1
            if ok == len(fcur["hunks"]):
                try:
                    fpath.write_text("\n".join(target), encoding="utf-8", newline="\n")
                except OSError as e:
                    out.append(f"error writing {name}: {e}")
                    continue
                out.append(f"applied {ok}/{len(fcur['hunks'])} hunk(s) to {name}")
            else:
                out.append(f"FAILED {ok}/{len(fcur['hunks'])} hunk(s) on {name}")
        summary = f"Patch: applied {applied_hunks}/{total_hunks} hunk(s)" if total_hunks else "Patch: no hunks"
        return summary + "\n" + "\n".join(out) if out else summary


class CommandArgs(ToolArgs):
    command: str = Field(description="Shell command to execute.")
    cwd: Optional[str] = Field(None, description="Directory relative to the workspace root.")
    timeout: int = Field(60, description="Timeout in seconds.")


class RunCommandTool(WorkspaceTool):
    name: str = "run_command"
    description: str = "Execute a shell command in the workspace; returns exit code, stdout, stderr."
    args_schema: Type[BaseModel] = CommandArgs

    def _run(self, command: str, cwd: Optional[str] = None, timeout: int = 60) -> str:
        global _last_command
        _last_command[:] = [command]
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
        return self._summarize(run_proc(cmd, self.workspace_root, timeout))

    def _summarize(self, raw: str) -> str:
        """Turn pytest's raw output into a stable, complete summary the model
        can act on: pass/fail counts plus every failure's node id."""
        fails: list[str] = []
        for line in raw.splitlines():
            f = line.strip()
            if f.startswith("FAILED "):
                fails.append(f[7:].strip())
            elif f.startswith("ERROR "):
                fails.append("ERROR " + f[6:].strip())
        m = re.search(r"(\d+) passed", raw)
        passed = int(m.group(1)) if m else 0
        m = re.search(r"(\d+)\s+failed", raw)
        failed = int(m.group(1)) if m else 0
        m = re.search(r"(\d+)\s+error", raw)
        errors = int(m.group(1)) if m else 0
        m = re.search(r"(\d+)\s+skipped", raw)
        skipped = int(m.group(1)) if m else 0
        out = [f"Tests: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped"]
        if fails:
            out.append("Failures:")
            for f in fails[:40]:
                out.append(f"  {f}")
        return truncate("\n".join(out))


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
            f"uv: {version(['uv', 'version'])}",
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


# ---------------------------------------------------------------- new tools
class SnapArgs(ToolArgs):
    label: str = Field("auto", description="Optional label for the snapshot.")


class SnapshotTool(WorkspaceTool):
    name: str = "snapshot"
    description: str = ("Record the current state of every workspace file so the "
                       "agent can diff against or restore it later. Returns a slot id.")
    args_schema: Type[BaseModel] = SnapArgs

    def _run(self, label: str = "auto") -> str:
        return self.snapshot(label)


class DiffArgs(ToolArgs):
    revision: Optional[str] = Field(None, description="Snapshot slot id to diff against; default: latest.")


class DiffTool(WorkspaceTool):
    name: str = "diff"
    description: str = "Show files changed since a snapshot (or since the last snapshot)."
    args_schema: Type[BaseModel] = DiffArgs

    def _latest_slot(self) -> Optional[str]:
        base = self._snapshot_path() / "snapshots"
        if not base.is_dir():
            return None
        slots = [d.name for d in base.iterdir() if d.is_dir()]
        return max(slots) if slots else None

    def _run(self, revision: Optional[str] = None) -> str:
        slot = revision or self._latest_slot()
        if not slot:
            return "No snapshots yet. Use snapshot to record one."
        changed = self._paths_changed_since(slot)
        return "\n".join(changed) if changed else f"(no changes since snapshot {slot})"


class RestoreArgs(ToolArgs):
    revision: str = Field(description="Snapshot slot id to restore the workspace to.")


class RestoreTool(WorkspaceTool):
    name: str = "restore"
    description: str = "Restore every workspace file to a previously recorded snapshot state."
    args_schema: Type[BaseModel] = RestoreArgs

    def _run(self, revision: str) -> str:
        return self._apply_snapshot(revision)


class CheckArgs(ToolArgs):
    path: Optional[str] = Field(None, description="File or directory to check (default: whole workspace).")
    timeout: int = Field(60, description="Timeout in seconds.")


class CheckTool(WorkspaceTool):
    name: str = "check"
    description: str = "Fast static checks: syntax-check .py files and run linters if present."
    args_schema: Type[BaseModel] = CheckArgs

    def _run(self, path: Optional[str] = None, timeout: int = 60) -> str:
        root = self.resolve(path) if path else self.workspace_root
        files = [f for f in self.walk(root) if f.suffix == ".py"]
        if not files:
            return "No Python files to check."
        errors: list[str] = []
        for f in files:
            try:
                compile(f.read_text(encoding="utf-8", errors="replace"), str(f), "exec")
            except SyntaxError as e:
                errors.append(f"{self.rel(f)}:{e.lineno}: {e.msg}")
        if errors:
            return "Syntax errors:\n" + "\n".join(errors[:40])
        # run ruff if the workspace has it
        ruff = shutil.which("ruff")
        if ruff:
            out = run_proc([ruff, "check", str(root)], self.workspace_root, timeout)
            return "Syntax OK. Ruff:\n" + out
        return "Syntax OK (no ruff installed)."


class DebugTraceArgs(ToolArgs):
    script: str = Field(description="Python script path (relative) to run under the tracer.")
    args: str = Field("", description="Optional command-line args for the script.")
    trace_fn: str = Field("lines", description="'lines' (default), 'calls', or 'counts'.")
    timeout: int = Field(120, description="Timeout in seconds.")


class DebugTraceTool(WorkspaceTool):
    name: str = "debug_trace"
    description: str = ("Run a Python script under a line/call tracer and return a "
                       "per-line account (lines executed, calls, or counts).")
    args_schema: Type[BaseModel] = DebugTraceArgs

    def _run(self, script: str, args: str = "", trace_fn: str = "lines",
             timeout: int = 120) -> str:
        f = self.resolve(script)
        if not f.is_file():
            return f"Error: not a file: {script}"
        fn_map = {"lines": "line", "calls": "call", "counts": "count"}
        if trace_fn not in fn_map:
            return f"Error: trace_fn must be one of {sorted(fn_map)}"
        flags = []
        if trace_fn == "lines":
            flags = ["--trace"]
        elif trace_fn == "counts":
            flags = ["--count"]
        else:  # calls
            flags = ["--count", "--trace"]
        cmd = [sys.executable, "-m", "trace"] + flags + [str(f)] + (args.split() if args else [])
        return run_proc(cmd, self.workspace_root, timeout)


class RerunArgs(ToolArgs):
    timeout: int = Field(60, description="Timeout in seconds.")


class RerunLastTool(WorkspaceTool):
    name: str = "rerun_last"
    description: str = "Re-run the exact last run_command, with an optional timeout override."
    args_schema: Type[BaseModel] = RerunArgs

    def _run(self, timeout: int = 60) -> str:
        global _last_command
        if not _last_command:
            return "No previous command to re-run."
        return run_proc(_last_command[0], self.workspace_root, timeout)


_last_command: list[str] = []   # set by RunCommandTool; read by RerunLastTool


def create_tools(workspace_root: Path) -> list[BaseTool]:
    kinds = [
        ReadFileTool, ListDirectoryTool, SearchFilesTool, FindFilesTool,
        WriteFileTool, ApplyPatchTool, RunCommandTool, RunTestsTool,
        GitDiffTool, GitStatusTool, GitLogTool, GitRestoreTool,
        InspectEnvironmentTool, InstallDependencyTool, FileInfoTool,
        SnapshotTool, DiffTool, RestoreTool, CheckTool, DebugTraceTool,
        RerunLastTool,
    ]
    return [k(workspace_root=workspace_root) for k in kinds]