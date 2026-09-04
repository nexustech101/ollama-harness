"""Structured results and run status for native API callers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RunStatus = str  # kept as a string to allow provider/queue extensions


@dataclass
class RunResult:
    """Terminal result of one agent run."""

    run_id: str
    status: str = "completed"
    text: str = ""
    error: str | None = None
    error_code: str | None = None
    steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    model_seconds: float = 0.0
    context_used: int = 0
    context_limit: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    elapsed_ms: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "completed"


__all__ = ["RunResult", "RunStatus"]
