"""Tests for the workspace policy and in-memory undo journal."""

from __future__ import annotations

from workspace.policy import InMemoryJournal, WorkspacePolicy


def test_workspace_policy_accepts_child(tmp_path):
    policy = WorkspacePolicy(tmp_path)
    target = tmp_path / "a.py"
    assert policy.validate_write(target) == target.resolve()


def test_workspace_policy_rejects_outside_path(tmp_path):
    policy = WorkspacePolicy(tmp_path)
    try:
        policy.validate_write(tmp_path.parent / "escape.py")
    except ValueError as exc:
        assert "escapes the workspace" in str(exc)
    else:
        raise AssertionError("outside write did not raise")


def test_journal_is_bounded(tmp_path):
    journal = InMemoryJournal(max_entries=2, max_bytes=100)
    assert journal.write("a.txt", b"x" * 10) is True
    assert journal.write("b.txt", b"y" * 10) is True
    assert journal.write("c.txt", b"z" * 10) is False
    assert len(journal.entries) == 2
    journal.clear()
    assert len(journal.entries) == 0
