"""
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
import importlib

from . import sdk_shims as _sdk_shims

# Reload shims whenever this package is reloaded so import-fallback tests that
# block mistralai/PIL and call importlib.reload(mistral_converter) see None.
_sdk_shims = importlib.reload(_sdk_shims)
DocumentURLChunk = _sdk_shims.DocumentURLChunk
FileChunk = _sdk_shims.FileChunk
Image = _sdk_shims.Image
ImageURLChunk = _sdk_shims.ImageURLChunk
Mistral = _sdk_shims.Mistral
httpx = _sdk_shims.httpx
models = _sdk_shims.models
response_format_from_pydantic_model = _sdk_shims.response_format_from_pydantic_model
retries = _sdk_shims.retries
urlparse = _sdk_shims.urlparse
from .session import (
    _commit_session_pages,
    _estimate_session_pages_for_ocr,
    _is_page_limit_reached,
    _ocr_session_page_delta,
    _release_session_pages_reservation,
    _reserve_session_pages,
    reset_session_page_counter,
)
from .upload import (
    _cleanup_temp_files,
    _delete_ocr_file_ids,
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

# Mutable module state must be read through __getattr__ so callers see live values
# (ints/bools rebound in session.py / client.py are not updated by a one-time import).
_DYNAMIC_ATTRS = {
    "_session_pages_processed": ("session", "_session_pages_processed"),
    "_session_pages_inflight": ("session", "_session_pages_inflight"),
    "_session_pages_warned": ("session", "_session_pages_warned"),
    "_client_instance": ("client", "_client_instance"),
}


def __getattr__(name: str):
    target = _DYNAMIC_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    from importlib import import_module

    return getattr(import_module(f".{mod_name}", __name__), attr_name)
