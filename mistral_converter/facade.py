"""Resolve symbols from the package root for test patch compatibility."""

from __future__ import annotations

import sys
from typing import Any


def attr(name: str) -> Any:
    """Return ``mistral_converter.<name>`` (supports ``patch.object(mistral_converter, ...)``)."""
    package_name = __package__
    if package_name is None:  # pragma: no cover - package imports always set this
        raise RuntimeError("mistral_converter package context is unavailable")
    return getattr(sys.modules[package_name], name)
