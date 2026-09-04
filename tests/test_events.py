"""Test the new event schema and run-result collector without a model backend."""

from __future__ import annotations

from domain.events import (
    VERSION,
    ErrorEvent,
    RunCancelledEvent,
    StepEvent,
    final_error_code,
)
from domain.results import RunResult


def test_version_is_canonical():
    assert VERSION == "0.3.0"


def test_base_event_lines_have_metadata():
    event = StepEvent(2)
    line = event.line(run_id="run-1", sequence=3, step=2, status="running")
    assert line["type"] == "step"
    assert line["step"] == 2
    assert line["run_id"] == "run-1"
    assert line["sequence"] == 3


def test_error_event_classifies_codes():
    assert ErrorEvent("interrupted").payload["code"] == "cancelled"
    assert final_error_code("tool calls failed in a row") == "tool_failure_limit"
    assert final_error_code("unknown tool 'x'") == "invalid_tool_call"


def test_cancelled_event_type():
    event = RunCancelledEvent()
    assert event.type == "run_cancelled"
    assert event.payload["code"] == "cancelled"


def test_run_result_defaults():
    result = RunResult(run_id="r")
    assert result.ok
    assert result.tool_calls == 0
    result.status = "failed"
    assert not result.ok
