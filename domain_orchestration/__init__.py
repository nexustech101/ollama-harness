"""Bridge the old flat modules to the new domain and workspace packages.

This exists so existing imports (`from harness import Harness`, `from tools
import ...`, etc.) keep working while the real implementation remains in the
flat files.
"""

from domain.events import VERSION

__all__ = ["VERSION"]
