"""Provider adapter package.

The existing flat module (``providers.py``) remains the canonical
implementation and public entry point for CLI/tests. Loading it here keeps
``import providers`` and ``from providers import ...`` working even though this
package shares the name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_FLAT = Path(__file__).resolve().parent.parent / "providers.py"


def _load_flat() -> None:
    spec = importlib.util.spec_from_file_location("_providers_flat", _FLAT)
    if spec is None or spec.loader is None:  # pragma: no cover - should not happen
        raise ImportError(f"could not load provider implementation from {_FLAT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_providers_flat"] = module
    spec.loader.exec_module(module)
    _published = [
        "DEFAULT_BASE_URLS",
        "DEFAULT_MODELS",
        "DEFAULT_NUM_CTX",
        "PROVIDERS",
        "Provider",
        "api_key_from_env",
        "available_providers",
        "build_provider_entry",
        "default_config_path",
        "find_config",
        "load_config",
        "provider_names",
        "resolve_provider",
    ]
    for name in _published:
        if hasattr(module, name):
            globals()[name] = getattr(module, name)


_load_flat()

__all__ = [
    "DEFAULT_BASE_URLS",
    "DEFAULT_MODELS",
    "DEFAULT_NUM_CTX",
    "PROVIDERS",
    "Provider",
    "api_key_from_env",
    "available_providers",
    "build_provider_entry",
    "default_config_path",
    "find_config",
    "load_config",
    "provider_names",
    "resolve_provider",
]
