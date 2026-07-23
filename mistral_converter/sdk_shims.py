"""Mistral SDK imports and test-compat shims."""

from typing import Any
from urllib.parse import urlparse

try:
    from mistralai.client import Mistral
    from mistralai.client.utils import retries
except ImportError:  # pragma: no cover
    try:
        from mistralai import Mistral  # type: ignore[no-redef]
        from mistralai.utils import retries  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        import logging as _logging

        _logging.getLogger("document_converter").warning(
            "mistralai package not available. Install with: pip install mistralai"
        )
        Mistral = None
        retries = None

try:
    from mistralai.client.models import DocumentURLChunk, ImageURLChunk
except ImportError:  # pragma: no cover
    try:
        from mistralai import (  # type: ignore[no-redef]
            DocumentURLChunk,
            ImageURLChunk,
        )
    except ImportError:  # pragma: no cover
        DocumentURLChunk = None
        ImageURLChunk = None

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

FileChunk = None  # type: ignore[misc, assignment]
models = None  # type: ignore[misc, assignment]

try:
    from mistralai.extra import response_format_from_pydantic_model
except ImportError:  # pragma: no cover
    response_format_from_pydantic_model = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None
