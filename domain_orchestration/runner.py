"""Self-contained run collector built on ``Harness.run_events``.

The legacy loop yields dictionaries. This class converts that stream into the
canonical domain result without changing the loop, which keeps CLI and OpenAI
compatibility intact while native API callers get a structured contract.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from domain.events import VERSION, ErrorEvent
from domain.results import RunResult
from harness import Harness


class RunCollector:
    """Drain one run's events into flags and a :class:`RunResult`."""

    def __init__(self, run_id: str, agent: Harness) -> None:
        self.run_id = run_id
        self.agent = agent
        self.result = RunResult(run_id=run_id)
        self.version = VERSION
        self._events: list[dict[str, Any]] = []
        self._started = time.monotonic()

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def run(self, query: str) -> RunResult:
        for event in self.agent.run_events(query):
            self.consume(event)
        self.finish()
        return self.result

    def consume(self, event: dict[str, Any]) -> dict[str, Any]:
        self._events.append(event)
        kind = event["type"]
        if kind == "step":
            self.result.steps = max(self.result.steps, int(event["step"]))
        elif kind == "usage":
            self.result.tokens_in += int(event.get("input") or 0)
            self.result.tokens_out += int(event.get("output") or 0)
            self.result.model_seconds += float(event.get("seconds") or 0)
            self.result.context_used = int(event.get("context") or 0)
            self.result.context_limit = int(event.get("limit") or 0)
        elif kind == "tool":
            self.result.tool_calls += 1
        elif kind == "result":
            if str(event.get("result") or "").startswith("Error"):
                self.result.tool_failures += 1
        elif kind == "done":
            self.result.text = event.get("text", "")
        elif kind == "error":
            self.result.error = event.get("text", "")
            self.result.error_code = event.get("code") or ErrorEvent(self.result.error).payload["code"]
        return event

    def finish(self) -> None:
        self.result.elapsed_ms = int((time.monotonic() - self._started) * 1000)
        stats = self.agent.stats
        self.result.steps = max(self.result.steps, stats.get("steps", 0))
        self.result.tokens_in = max(self.result.tokens_in, stats.get("input", 0))
        self.result.tokens_out = max(self.result.tokens_out, stats.get("output", 0))
        self.result.model_seconds = max(self.result.model_seconds, stats.get("seconds", 0.0))
        if self.result.text:
            self.result.status = "completed"
        elif self.result.error is not None:
            self.result.status = "failed"
        else:
            self.result.status = "completed"
        self.result.metrics = {
            "events": len(self._events),
            "tool_calls": self.result.tool_calls,
            "tool_failures": self.result.tool_failures,
        }


def run_to_result(run_id: str, agent: Harness, query: str) -> RunResult:
    return RunCollector(run_id, agent).run(query)


def iter_run(agent: Harness, query: str) -> Iterator[dict[str, Any]]:
    """Compatibility helper: yield the loop events unchanged."""
    yield from agent.run_events(query)


__all__ = ["RunCollector", "run_to_result"]
