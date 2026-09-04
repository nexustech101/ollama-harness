"""The native (non-OpenAI) API surface.

Kept intentionally small in this iteration: health/readiness remains under the
OpenAI app, while `/api/v1` reads are backed by the same settings values. It
provides the baseline versioned contract for later run/approval/artifact
endpoints without duplicating the existing server wiring.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request

from domain.events import VERSION


def create_native_router(settings: Any) -> APIRouter:
    """Return the future native router prefix (``/api/v1``)."""
    router = APIRouter(prefix="/api/v1", tags=["native"])

    @router.get("/health")
    def native_health(request: Request) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "ollama-harness",
            "version": VERSION,
            "provider": getattr(settings, "provider", None),
            "model": getattr(settings, "model", None),
            "timestamp": int(time.time()),
        }

    return router


__all__ = ["create_native_router"]
