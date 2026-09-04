"""Central workspace policy for tools and the run layer.

This module replaces the old ``.harness/state`` snapshot design with an
explicit, bounded undo journal and a declarative sandbox policy. The actual tool
call sites still live in ``tools.py``; this keeps sandbox decisions and run
journal book-keeping testable independently of LangChain.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class RollbackMode(StrEnum):
    NONE = "none"
    GIT = "git"
    MEMORY = "memory"
    EXTERNAL = "external"


@dataclass(frozen=True)
class WorkspacePolicy:
    """Declarative sandbox boundary consumed by tools and approval code."""

    workspace_root: Path
    readable_paths: tuple[Path, ...] = ()
    writable_paths: tuple[Path, ...] = ()
    command_allow: tuple[str, ...] = ("*",)
    command_deny: tuple[str, ...] = ()
    network: bool = True
    max_output: int = 20_000
    allow_symlinks: bool = False
    ignored_dirs: frozenset[str] = frozenset(
        {
            ".git",
            ".hg",
            ".svn",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "dist",
            "build",
            ".idea",
        }
    )

    @property
    def root(self) -> Path:
        return self.workspace_root.expanduser().resolve()

    def in_root(self, path: Path) -> bool:
        """True when `path` is the root or a descendant."""
        resolved = path.expanduser().resolve()
        return resolved == self.root or self.root in resolved.parents

    def resolve_existing(self, path: Path) -> Path:
        """Resolve an existing (or about-to-exist) path with symlink checks.

        Lexical containment is always required. When symlinks are disallowed the
        resolved target must also stay inside the workspace.
        """
        target = path.expanduser()
        if not self.in_root(target):
            self._raise(path)
        if not self.allow_symlinks:
            existing = target.resolve()
            if existing.exists() and not self.in_root(existing):
                self._raise(path)
        return target

    def validate_write(self, path: Path) -> Path:
        """Return a validated target path or raise ``ValueError``."""
        return self.resolve_existing(path)

    def is_readable(self, path: Path) -> bool:
        """True if the path is inside the root or explicitly allowed."""
        if self.in_root(path):
            return True
        return any(path.resolve() == allowed.expanduser().resolve() for allowed in self.readable_paths)

    def is_writable(self, path: Path) -> bool:
        """True if the path is inside the root or explicitly allowed."""
        if not self.writable_paths:
            return self.in_root(path)
        if self.in_root(path):
            return True
        return any(path.resolve() == allowed.expanduser().resolve() for allowed in self.writable_paths)

    @staticmethod
    def _raise(path: Path) -> None:
        raise ValueError(f"path escapes the workspace: {str(path)!r}")


@dataclass
class InMemoryJournal:
    """Process-local undo journal for non-git workspaces.

    One original byte string per changed path, bounded by count and bytes. It is
    intentionally not written under the workspace (so no ``.harness/`` directory
    is created) and is discarded when the run ends.
    """

    max_entries: int = 256
    max_bytes: int = 16 * 1024 * 1024
    entries: dict[str, bytes] = field(default_factory=dict)
    _bytes: int = 0

    def enabled(self) -> bool:
        return self.max_entries > 0 and self.max_bytes > 0

    def write(self, rel: str, original: bytes) -> bool:
        if not self.enabled() or len(self.entries) >= self.max_entries:
            return False
        if self._bytes + len(original) > self.max_bytes:
            return False
        if rel not in self.entries:
            self.entries[rel] = original
            self._bytes += len(original)
        return True

    def contents(self, rel: str) -> bytes | None:
        return self.entries.get(rel)

    def clear(self) -> None:
        self.entries.clear()
        self._bytes = 0

    def snapshot_text(self) -> str:
        return f"run journal: {len(self.entries)} path(s), {self._bytes} byte(s) (memory-only, discarded at run end)"


@dataclass
class ToolResult:
    """Structured result from a single tool invocation."""

    text: str
    ok: bool = True
    code: str = "ok"
    truncated: bool = False
    args: dict[str, Any] = field(default_factory=dict)
    started_ms: int | None = None
    finished_ms: int | None = None
    exit_status: int | None = None
    changed_paths: tuple[str, ...] = ()
    approval: str | None = None

    @property
    def is_error(self) -> bool:
        return not self.ok


@dataclass
class CommandSpec:
    """Validated command execution request."""

    argv: list[str]
    cwd: Path
    timeout: int


def make_policy(workspace_root: Path, **_: Any) -> WorkspacePolicy:
    """Create a conservative default policy for a workspace."""
    return WorkspacePolicy(workspace_root=workspace_root)


def changed_paths_for(_name: str, args: dict[str, Any]) -> tuple[str, ...]:
    """Best-effort changed-path extraction for tools that know their targets."""
    if "path" in args and isinstance(args["path"], str):
        return (args["path"].replace("\\", "/"),)
    return ()


def fingerprint_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


__all__ = [
    "CommandSpec",
    "InMemoryJournal",
    "RollbackMode",
    "ToolResult",
    "WorkspacePolicy",
    "changed_paths_for",
    "fingerprint_bytes",
    "make_policy",
]
