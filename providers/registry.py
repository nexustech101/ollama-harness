"""Provider registry for the flat ``providers.py`` implementation.

This package-level view keeps client code independent of the flat module's
import location while the actual resolution remains in ``providers.py`` (stable
external entry point). New provider adapters can be registered here without
moving existing CLI code.
"""

from __future__ import annotations

from typing import Any

_PROTOCOLS: dict[str, str] = {
    "ollama": "ollama",
    "openai": "openai",
    "openrouter": "openai",
}


def protocol_for(provider: str) -> str:
    return _PROTOCOLS.get(provider, "openai")


def register_protocol(name: str, protocol: str) -> None:
    _PROTOCOLS[name] = protocol


def known_protocols() -> tuple[str, ...]:
    return tuple(sorted(set(_PROTOCOLS.values())))


def convert_legacy_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Narrow provider payloads to known adapter fields before construction."""
    allowed = {"model", "base_url", "api_key", "temperature", "num_ctx", "extra"}
    return {key: value for key, value in payload.items() if key in allowed}


__all__ = [
    "convert_legacy_model_payload",
    "known_protocols",
    "protocol_for",
    "register_protocol",
]
