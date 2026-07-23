"""Resolve symbols from the package root for test patch compatibility."""

from __future__ import annotations

import sys
from typing import Any


def attr(name: str) -> Any:
    """Return ``mistral_converter.<name>`` (supports ``patch.object(mistral_converter, ...)``)."""
    return getattr(sys.modules[__package__], name)
