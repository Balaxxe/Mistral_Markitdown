#!/usr/bin/env python3
"""One-off helper to split mistral_converter.py into a package (run from repo root)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "mistral_converter.py"
PKG = ROOT / "mistral_converter"

MODULE_ASSIGNMENTS: dict[str, list[str]] = {
    "client": [
        "_http_client_exceptions",
        "_client_lock",
        "_client_instance",
        "get_mistral_client",
        "reset_mistral_client",
        "get_retry_config",
    ],
    "session": [
        "_session_pages_processed",
        "_session_pages_inflight",
        "_session_pages_warned",
        "_session_pages_lock",
        "_estimate_session_pages_for_ocr",
        "_reserve_session_pages",
        "_commit_session_pages",
        "_release_session_pages_reservation",
        "_is_page_limit_reached",
        "_ocr_session_page_delta",
        "reset_session_page_counter",
    ],
    "upload": [
        "_UPLOAD_REGISTRY_LOCK",
        "_UPLOAD_REGISTRY_FILENAME",
        "_upload_registry_path",
        "_load_upload_registry",
        "_save_upload_registry",
        "_register_uploaded_file",
        "_unregister_uploaded_file",
        "_parse_registry_created_at",
        "cleanup_uploaded_files",
        "_cleanup_registry_scoped",
        "_cleanup_files_by_purpose",
        "_delete_ocr_file_ids",
        "_upload_file_for_ocr_pair",
        "upload_file_for_ocr",
        "_cleanup_temp_files",
    ],
    "images": [
        "optimize_image",
        "preprocess_image",
        "save_extracted_images",
    ],
    "schemas_fmt": [
        "_extract_model_json_schema",
        "_wrap_response_format",
        "_resolve_document_schema_type",
        "get_bbox_annotation_format",
        "get_document_annotation_format",
        "_filename_has_keyword",
        "classify_document_type",
        "_ocr_shared_optional_params",
        "build_ocr_process_kwargs",
        "_get_mistralai_package_version",
        "_document_annotation_prompt_sha256",
        "build_mistral_ocr_cache_contract_metadata",
        "mistral_ocr_cache_contract_matches",
    ],
    "url_validation": [
        "_SIGNED_URL_SPECIFIC_HINTS",
        "_SIGNED_URL_FETCH_HINTS",
        "_PERMANENT_AUTH_HINTS",
        "is_signed_url_expiry_error",
        "_dns_executor",
        "_is_forbidden_address",
        "_validate_ip_str",
        "_resolve_and_validate_dns",
        "_validate_document_url",
        "validate_https_document_url",
    ],
    "ocr": [
        "_validate_file_for_ocr",
        "_prepare_ocr_document",
        "process_with_ocr",
        "_extract_page_text",
        "_parse_page_object",
        "_parse_pages_response",
        "_parse_single_text_response",
        "_parse_dict_response",
        "_extract_structured_outputs",
        "_extract_response_metadata",
        "_parse_ocr_response",
        "_is_weak_page",
        "assess_ocr_quality",
        "_detect_weak_pages",
        "_run_weak_page_improvements",
        "improve_weak_pages",
        "_process_ocr_result_pipeline",
        "convert_with_mistral_ocr",
        "_save_structured_outputs",
        "_create_markdown_output",
    ],
    "qna": [
        "_DEFAULT_QNA_SYSTEM_PROMPT",
        "_build_qna_messages",
        "_prepare_qna_call",
        "query_document",
        "query_document_stream",
        "query_document_file",
    ],
    "batch": [
        "_prepare_batch_entries",
        "create_batch_ocr_file",
        "submit_batch_ocr_job",
        "get_batch_job_status",
        "download_batch_results",
        "list_batch_jobs",
    ],
}

MODULE_HEADERS: dict[str, str] = {
    "client": '''"""Mistral SDK client singleton and retry configuration."""

import threading
from typing import Any, Dict, Optional, Tuple

import config
import utils

from .sdk_shims import Mistral, retries

logger = utils.logger
''',
    "session": '''"""Session page budget globals and helpers."""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import utils

logger = utils.logger
''',
    "upload": '''"""Mistral Files API upload helpers and local upload registry."""

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
import utils

from .client import Mistral
from .images import optimize_image, preprocess_image

logger = utils.logger
''',
    "images": '''"""Image optimization, preprocessing, and OCR image extraction."""

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import utils
from .sdk_shims import Image

logger = utils.logger
''',
    "schemas_fmt": '''"""Annotation format helpers, document classification, and OCR kwargs builders."""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import schemas
import utils

from .client import get_mistral_client, get_retry_config
from .sdk_shims import response_format_from_pydantic_model

logger = utils.logger
''',
    "url_validation": '''"""SSRF-safe document URL validation and signed-URL error classification."""

import ipaddress
import socket
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as DnsTimeoutError
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

import config
import utils

logger = utils.logger
''',
    "ocr": '''"""Mistral OCR processing, parsing, quality assessment, and conversion pipeline."""

import html
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import config
import utils

from .client import Mistral, get_mistral_client, get_retry_config
from .schemas_fmt import (
    build_mistral_ocr_cache_contract_metadata,
    build_ocr_process_kwargs,
    mistral_ocr_cache_contract_matches,
)
from .sdk_shims import DocumentURLChunk, ImageURLChunk
from .session import (
    _commit_session_pages,
    _estimate_session_pages_for_ocr,
    _ocr_session_page_delta,
    _release_session_pages_reservation,
    _reserve_session_pages,
)
from .upload import upload_file_for_ocr

logger = utils.logger
''',
    "qna": '''"""Document QnA via Mistral chat.complete."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
import utils

from .client import get_mistral_client, get_retry_config
from .upload import upload_file_for_ocr
from .url_validation import _validate_document_url

logger = utils.logger


def _http_client_exceptions() -> Tuple[type, ...]:
    from .client import _http_client_exceptions as _fn

    return _fn()
''',
    "batch": '''"""Mistral Batch OCR job helpers."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
import utils

from .client import Mistral, get_mistral_client, get_retry_config, _http_client_exceptions
from .schemas_fmt import build_ocr_process_kwargs
from .sdk_shims import httpx
from .upload import (
    _delete_ocr_file_ids,
    _register_uploaded_file,
    _unregister_uploaded_file,
    _upload_file_for_ocr_pair,
)

logger = utils.logger
''',
}

SDK_SHIMS = '''"""Mistral SDK imports and test-compat shims."""

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
'''

INIT = '''"""
Enhanced Document Converter - Mistral AI Integration Package

Facade module preserving ``import mistral_converter`` for callers and tests.
"""

from .batch import (
    create_batch_ocr_file,
    download_batch_results,
    get_batch_job_status,
    list_batch_jobs,
    submit_batch_ocr_job,
)
from .client import (
    _client_instance,
    get_mistral_client,
    get_retry_config,
    reset_mistral_client,
)
from .images import optimize_image, preprocess_image, save_extracted_images
from .ocr import (
    _create_markdown_output,
    _detect_weak_pages,
    _extract_page_text,
    _extract_response_metadata,
    _extract_structured_outputs,
    _is_weak_page,
    _parse_dict_response,
    _parse_ocr_response,
    _parse_page_object,
    _parse_single_text_response,
    _process_ocr_result_pipeline,
    _run_weak_page_improvements,
    _save_structured_outputs,
    _validate_file_for_ocr,
    assess_ocr_quality,
    convert_with_mistral_ocr,
    improve_weak_pages,
    process_with_ocr,
)
from .qna import (
    _build_qna_messages,
    query_document,
    query_document_file,
    query_document_stream,
)
from .schemas_fmt import (
    _extract_model_json_schema,
    _ocr_shared_optional_params,
    _wrap_response_format,
    build_mistral_ocr_cache_contract_metadata,
    build_ocr_process_kwargs,
    classify_document_type,
    get_bbox_annotation_format,
    get_document_annotation_format,
    mistral_ocr_cache_contract_matches,
)
from .sdk_shims import (  # noqa: F401 - re-exported for tests
    DocumentURLChunk,
    FileChunk,
    Image,
    ImageURLChunk,
    Mistral,
    httpx,
    models,
    response_format_from_pydantic_model,
    retries,
    urlparse,
)
from .session import (
    _commit_session_pages,
    _estimate_session_pages_for_ocr,
    _is_page_limit_reached,
    _ocr_session_page_delta,
    _release_session_pages_reservation,
    _reserve_session_pages,
    _session_pages_processed,
    _session_pages_warned,
    reset_session_page_counter,
)
from .upload import (
    _cleanup_temp_files,
    _load_upload_registry,
    _register_uploaded_file,
    _save_upload_registry,
    _unregister_uploaded_file,
    _upload_file_for_ocr_pair,
    cleanup_uploaded_files,
    upload_file_for_ocr,
)
from .url_validation import (
    _is_forbidden_address,
    _validate_document_url,
    is_signed_url_expiry_error,
    validate_https_document_url,
)

__all__ = [
    "get_mistral_client",
    "reset_mistral_client",
    "reset_session_page_counter",
    "get_retry_config",
    "get_bbox_annotation_format",
    "get_document_annotation_format",
    "optimize_image",
    "preprocess_image",
    "cleanup_uploaded_files",
    "upload_file_for_ocr",
    "process_with_ocr",
    "assess_ocr_quality",
    "improve_weak_pages",
    "save_extracted_images",
    "convert_with_mistral_ocr",
    "query_document",
    "query_document_stream",
    "query_document_file",
    "create_batch_ocr_file",
    "submit_batch_ocr_job",
    "get_batch_job_status",
    "download_batch_results",
    "list_batch_jobs",
    "validate_https_document_url",
]
'''


def _find_blocks(source: str) -> dict[str, tuple[int, int]]:
    """Map top-level name -> (start_line, end_line) 1-based inclusive."""
    lines = source.splitlines(keepends=True)
    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines, start=1):
        if line.startswith((" ", "\t")):
            continue
        stripped = line.lstrip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            name = stripped.split("(")[0].replace("async def ", "").replace("def ", "").strip()
            starts.append((i, name))
        elif stripped.startswith("class "):
            name = stripped.split("(")[0].split(":")[0].replace("class ", "").strip()
            starts.append((i, name))
        elif re.match(r"^[A-Za-z_][A-Za-z0-9_]*(?:\s*:.*?)?\s*=", stripped):
            name = re.split(r"\s*[:=]", stripped, maxsplit=1)[0].strip()
            starts.append((i, name))

    blocks: dict[str, tuple[int, int]] = {}
    for idx, (start, name) in enumerate(starts):
        end = starts[idx + 1][0] - 1 if idx + 1 < len(starts) else len(lines)
        blocks[name] = (start, end)
    return blocks


def main() -> None:
    source = SRC.read_text(encoding="utf-8")
    blocks = _find_blocks(source)
    lines = source.splitlines(keepends=True)

    PKG.mkdir(exist_ok=True)
    (PKG / "sdk_shims.py").write_text(SDK_SHIMS, encoding="utf-8")
    (PKG / "__init__.py").write_text(INIT, encoding="utf-8")

    assigned: set[str] = set()
    for mod, names in MODULE_ASSIGNMENTS.items():
        parts = [MODULE_HEADERS[mod]]
        for name in names:
            if name not in blocks:
                raise SystemExit(f"Missing block {name!r} in source")
            start, end = blocks[name]
            parts.append("".join(lines[start - 1 : end]))
            assigned.add(name)
        (PKG / f"{mod}.py").write_text("".join(parts), encoding="utf-8")

    all_assigned = {n for names in MODULE_ASSIGNMENTS.values() for n in names}
    missing = all_assigned - set(blocks)
    if missing:
        raise SystemExit(f"Blocks not found in source: {missing}")

    print(f"Wrote package under {PKG}")


if __name__ == "__main__":
    main()
