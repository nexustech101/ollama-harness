"""Typed event schema and run-lifecycle primitives.

These models are the internal protocol for every harness run. `harness.py`,
`main.py`, and `api.py` render/serialize them; no OpenAI-specific field names
belong here.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

VERSION = "0.3.0"
RUN_KIND = "harness"

EventType = Literal[
    "run_started",
    "step",
    "token",
    "usage",
    "message",
    "tool",
    "result",
    "done",
    "error",
    "run_cancelled",
    "run_limit_reached",
    "approval_required",
]


def _now() -> float:
    return time.time()


def _sequence() -> str:
    return uuid.uuid4().hex


def final_error_code(text: str) -> str:
    """Classify a terminal error string into a stable machine-readable code."""
    lowered = text.lower()
    if "interrupted" in lowered:
        return "cancelled"
    if "step limit" in lowered or "safety ceiling" in lowered:
        return "run_limit_reached"
    if "tool calls failed" in lowered or "failed in a row" in lowered:
        return "tool_failure_limit"
    if "permission" in lowered or "read-only" in lowered or "needs confirmation" in lowered:
        return "permission_denied"
    if "unknown tool" in lowered:
        return "invalid_tool_call"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    return "error"


class BaseEvent:
    """Common envelope for every event emitted by a run."""

    type: str
    payload: dict[str, Any]

    def __init__(self, event_type: EventType, payload: dict[str, Any] | None = None) -> None:
        self.type = event_type
        self.payload = dict(payload or {})

    def line(
        self,
        *,
        run_id: str,
        sequence: int,
        step: int,
        status: str,
        elapsed_ms: int | None = None,
    ) -> dict[str, Any]:
        """Assemble an API-ready event object.

        The domain model is small and explicit; this envelope is the only place that adds transport metadata.
        """
        return {
            "id": f"evt-{_sequence()}",
            "object": "harness.event",
            "created": _now(),
            "type": self.type,
            "run_id": run_id,
            "sequence": sequence,
            "step": step,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "payload": self.payload,
            **self.payload,
        }


class RunStartedEvent(BaseEvent):
    def __init__(self, run_id: str, status: str = "running", **payload: Any) -> None:
        super().__init__("run_started", {"run_id": run_id, "status": status, **payload})


class StepEvent(BaseEvent):
    def __init__(self, step: int) -> None:
        super().__init__("step", {"step": step})


class TokenEvent(BaseEvent):
    def __init__(self, text: str) -> None:
        super().__init__("token", {"text": text})


class UsageEvent(BaseEvent):
    def __init__(
        self, *, input: int, output: int, eval_seconds: float, seconds: float, chunks: int, context: int, limit: int
    ) -> None:
        super().__init__(
            "usage",
            {
                "input": input,
                "output": output,
                "eval_seconds": eval_seconds,
                "seconds": seconds,
                "chunks": chunks,
                "context": context,
                "limit": limit,
            },
        )


class ToolEvent(BaseEvent):
    def __init__(self, name: str, args: dict[str, Any]) -> None:
        super().__init__("tool", {"name": name, "args": args})


class ResultEvent(BaseEvent):
    def __init__(self, name: str, result: str, summary: str) -> None:
        super().__init__("result", {"name": name, "result": result, "summary": summary})


class ErrorEvent(BaseEvent):
    def __init__(self, text: str, code: str | None = None) -> None:
        super().__init__("error", {"text": text, "code": code or final_error_code(text)})


class RunLimitReachedEvent(ErrorEvent):
    def __init__(self, text: str) -> None:
        super().__init__(text, "run_limit_reached")
        self.type = "run_limit_reached"


class RunCancelledEvent(ErrorEvent):
    def __init__(self, text: str = "run cancelled") -> None:
        super().__init__(text, "cancelled")
        self.type = "run_cancelled"


class ApprovalRequiredEvent(BaseEvent):
    def __init__(self, approval_id: str, name: str, args: dict[str, Any]) -> None:
        super().__init__(
            "approval_required",
            {
                "approval_id": approval_id,
                "name": name,
                "args": args,
            },
        )


__all__ = [
    "ApprovalRequiredEvent",
    "BaseEvent",
    "ErrorEvent",
    "EventType",
    "ResultEvent",
    "RunCancelledEvent",
    "RunLimitReachedEvent",
    "RunStartedEvent",
    "StepEvent",
    "TokenEvent",
    "ToolEvent",
    "UsageEvent",
    "VERSION",
    "final_error_code",
]
