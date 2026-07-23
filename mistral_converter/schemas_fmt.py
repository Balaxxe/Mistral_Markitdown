"""Annotation format helpers, document classification, and OCR kwargs builders."""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import schemas
import utils

from .facade import attr

logger = utils.logger


def _extract_model_json_schema(pydantic_model: Any) -> Optional[Dict[str, Any]]:
    """Return JSON schema dict from a Pydantic model across v1/v2 APIs."""
    if hasattr(pydantic_model, "model_json_schema"):
        return pydantic_model.model_json_schema()
    if hasattr(pydantic_model, "schema"):
        return pydantic_model.schema()
    return None


def _wrap_response_format(raw_schema: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Wrap a raw JSON schema dict in the ResponseFormat envelope the OCR API expects.

    The OCR endpoint requires ``bbox_annotation_format`` and
    ``document_annotation_format`` to be ``ResponseFormat`` objects of the form::

        {
            "type": "json_schema",
            "json_schema": {
                "schema": { ... },
                "name": "...",
                "strict": true
            }
        }

    Args:
        raw_schema: The JSON schema dict (e.g. from Pydantic ``model_json_schema()``).
        name: A short identifier for the schema (e.g. ``"bbox_annotation"``).

    Returns:
        Wrapped ResponseFormat dict ready for the OCR API.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "schema": raw_schema,
            "name": name,
            "strict": True,
        },
    }


def _resolve_document_schema_type(doc_type: str = "auto") -> str:
    """Resolve ``auto`` / nested auto to a concrete document schema key."""
    if doc_type == "auto":
        nested = config.MISTRAL_DOCUMENT_SCHEMA_TYPE
        if nested == "auto":
            return "generic"
        return nested
    return doc_type


def get_bbox_annotation_format() -> Optional[Dict[str, Any]]:
    """
    Get ResponseFormat for bounding box annotation.

    The OCR API expects a ResponseFormat envelope wrapping the raw JSON schema::

        {"type": "json_schema", "json_schema": {"schema": ..., "name": ..., "strict": true}}

    Returns:
        ResponseFormat dict for bbox annotation, or None if disabled
    """
    if not config.MISTRAL_ENABLE_STRUCTURED_OUTPUT or not config.MISTRAL_ENABLE_BBOX_ANNOTATION:
        return None

    # Try SDK helper with Pydantic model (preferred - handles schema extraction automatically)
    pydantic_model = schemas.get_bbox_pydantic_model("structured")
    response_format_from_pydantic_model = attr("response_format_from_pydantic_model")
    if pydantic_model is not None and response_format_from_pydantic_model is not None:
        try:
            fmt = response_format_from_pydantic_model(pydantic_model)
            logger.debug("Using SDK response_format_from_pydantic_model for bbox annotation")
            return dict(fmt)  # type: ignore[arg-type]
        except Exception as e:
            logger.debug("SDK helper failed for bbox annotation: %s, falling back...", e)

    # Fallback: manual JSON schema extraction from Pydantic model
    if pydantic_model is not None:
        try:
            json_schema = _extract_model_json_schema(pydantic_model)
            if json_schema:
                logger.debug("Using Pydantic-derived JSON schema for bbox annotation")
                return _wrap_response_format(json_schema, "bbox_annotation")
        except Exception as e:
            logger.debug(
                "Could not get JSON schema from Pydantic model: %s, falling back to predefined schema",
                e,
            )

    # Fallback to predefined JSON schema from schemas.py
    bbox_schema = schemas.get_bbox_schema("structured")
    raw = bbox_schema.get("schema")
    if raw:
        return _wrap_response_format(raw, "bbox_annotation")
    return None


def get_document_annotation_format(doc_type: str = "auto") -> Optional[Dict[str, Any]]:
    """
    Get ResponseFormat for document-level annotation.

    The OCR API expects a ResponseFormat envelope wrapping the raw JSON schema::

        {"type": "json_schema", "json_schema": {"schema": ..., "name": ..., "strict": true}}

    Args:
        doc_type: Document type (invoice, financial_statement, form, generic, auto).
                  When set to "auto", the value of MISTRAL_DOCUMENT_SCHEMA_TYPE from
                  the environment is used.  If that is also "auto", "generic" is used
                  as the final fallback.  True content-based auto-detection is not yet
                  implemented; set the schema type explicitly for best results.

    Returns:
        ResponseFormat dict for document annotation, or None if disabled
    """
    if not config.MISTRAL_ENABLE_STRUCTURED_OUTPUT or not config.MISTRAL_ENABLE_DOCUMENT_ANNOTATION:
        return None

    doc_type = _resolve_document_schema_type(doc_type)

    schema_name = f"document_annotation_{doc_type}"

    # Try SDK helper with Pydantic model (preferred - handles schema extraction automatically)
    pydantic_model = schemas.get_document_pydantic_model(doc_type)
    response_format_from_pydantic_model = attr("response_format_from_pydantic_model")
    if pydantic_model is not None and response_format_from_pydantic_model is not None:
        try:
            fmt = response_format_from_pydantic_model(pydantic_model)
            logger.debug(
                "Using SDK response_format_from_pydantic_model for document annotation (type: %s)",
                doc_type,
            )
            return dict(fmt)  # type: ignore[arg-type]
        except Exception as e:
            logger.debug("SDK helper failed for document annotation: %s, falling back...", e)

    # Fallback: manual JSON schema extraction from Pydantic model
    if pydantic_model is not None:
        try:
            json_schema = _extract_model_json_schema(pydantic_model)
            if json_schema:
                logger.debug(
                    "Using Pydantic-derived JSON schema for document annotation (type: %s)",
                    doc_type,
                )
                return _wrap_response_format(json_schema, schema_name)
        except Exception as e:
            logger.debug(
                "Could not get JSON schema from Pydantic model: %s, falling back to predefined schema",
                e,
            )

    # Fallback to predefined JSON schema from schemas.py
    document_schema = schemas.get_document_schema(doc_type)
    raw = document_schema.get("schema")
    if raw:
        return _wrap_response_format(raw, schema_name)
    return None


def _filename_has_keyword(name: str, keywords: List[str]) -> bool:
    """Return True when a keyword appears as a filename token."""
    return any(re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", name) for keyword in keywords)


def classify_document_type(file_path: Path) -> str:  # noqa: C901
    """Classify document type into generic, invoice, financial_statement, contract, or form.

    Uses a hybrid approach:
    1. Regex checks on filename/path.
    2. Parsing first-page text (if PDF or text file) for key identifiers.
    3. Cheap LLM classification fallback (using ministral-8b-latest or mistral-small-latest).
    """
    name = file_path.name.lower()

    # 1. Filename heuristic
    if _filename_has_keyword(name, ["invoice", "receipt", "bill"]):
        logger.debug("Classified %s as 'invoice' via filename", file_path.name)
        return "invoice"
    if _filename_has_keyword(name, ["contract", "agreement", "nda", "lease"]):
        logger.debug("Classified %s as 'contract' via filename", file_path.name)
        return "contract"
    if _filename_has_keyword(name, ["statement", "financial", "balance_sheet", "income"]):
        logger.debug("Classified %s as 'financial_statement' via filename", file_path.name)
        return "financial_statement"
    if _filename_has_keyword(name, ["form", "w9", "w2", "tax"]):
        logger.debug("Classified %s as 'form' via filename", file_path.name)
        return "form"

    # 2. First-page text content check (for text PDFs/txt files)
    ext = file_path.suffix.lower().lstrip(".")
    first_text = ""

    if ext == "pdf":
        try:
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                if pdf.pages:
                    first_text = (pdf.pages[0].extract_text() or "")[:1000]
        except Exception as e:
            logger.debug("Could not extract PDF text for classification: %s", e)
    elif ext == "txt":
        try:
            first_text = file_path.read_text(errors="ignore")[:1000]
        except Exception as e:
            logger.debug("Could not read text file for classification: %s", e)

    if first_text:
        first_text_lower = first_text.lower()
        if any(w in first_text_lower for w in ["invoice", "receipt", "purchase order", "amount due"]):
            logger.debug("Classified %s as 'invoice' via page 1 text", file_path.name)
            return "invoice"
        if any(w in first_text_lower for w in ["agreement", "contract", "parties", "hereby", "undersigned"]):
            logger.debug("Classified %s as 'contract' via page 1 text", file_path.name)
            return "contract"
        financial_signals = (
            "balance sheet",
            "cash flow",
            "income statement",
            "statement of",
        )
        if sum(1 for phrase in financial_signals if phrase in first_text_lower) >= 2:
            logger.debug("Classified %s as 'financial_statement' via page 1 text", file_path.name)
            return "financial_statement"
        if any(w in first_text_lower for w in ["form ", "w-9", "w-2", "tax return", "filer"]):
            logger.debug("Classified %s as 'form' via page 1 text", file_path.name)
            return "form"

    # 3. LLM fallback check (opt-in; requires API key)
    if config.MISTRAL_ENABLE_LLM_DOC_CLASSIFICATION and config.MISTRAL_API_KEY:
        try:
            client = attr("get_mistral_client")()
            if client:
                prompt = (
                    f"Classify the document '{file_path.name}' into one of these types: "
                    "invoice, financial_statement, contract, form, generic.\n"
                    "Reply with only the lowercase name of the type.\n"
                )
                if first_text:
                    prompt += f"First page excerpt:\n{first_text[:500]}\n"

                model = config.MISTRAL_DOCUMENT_QNA_MODEL or "ministral-8b-latest"
                response = client.chat.complete(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                )
                if response and response.choices:
                    result = response.choices[0].message.content.strip().lower()
                    for t in ["invoice", "financial_statement", "contract", "form"]:
                        if t in result:
                            logger.debug("Classified %s as '%s' via LLM", file_path.name, t)
                            return t
        except Exception as e:
            logger.debug("LLM classification check failed: %s", e)

    logger.debug("Classified %s as 'generic'", file_path.name)
    return "generic"


def _ocr_shared_optional_params(file_path: Optional[Path] = None, doc_type: str = "auto") -> Dict[str, Any]:
    """OCR fields shared by sync ``ocr.process`` and batch JSONL ``body``."""
    fields: Dict[str, Any] = {}
    bbox_format = attr("get_bbox_annotation_format")()

    doc_format = None
    if config.MISTRAL_ENABLE_STRUCTURED_OUTPUT and config.MISTRAL_ENABLE_DOCUMENT_ANNOTATION:
        # Resolve dynamic classification only when document annotation can use it.
        resolved_doc_type = doc_type
        if resolved_doc_type == "auto":
            nested = config.MISTRAL_DOCUMENT_SCHEMA_TYPE
            if nested == "auto":
                if file_path is not None:
                    resolved_doc_type = attr("classify_document_type")(file_path)
                else:
                    resolved_doc_type = "generic"
            else:
                resolved_doc_type = nested

        doc_format = attr("get_document_annotation_format")(doc_type=resolved_doc_type)
    if bbox_format is not None:
        fields["bbox_annotation_format"] = bbox_format
    if doc_format is not None:
        fields["document_annotation_format"] = doc_format
    if config.MISTRAL_DOCUMENT_ANNOTATION_PROMPT:
        fields["document_annotation_prompt"] = config.MISTRAL_DOCUMENT_ANNOTATION_PROMPT
    if config.MISTRAL_TABLE_FORMAT:
        fields["table_format"] = config.MISTRAL_TABLE_FORMAT
    fields["extract_header"] = config.MISTRAL_EXTRACT_HEADER
    fields["extract_footer"] = config.MISTRAL_EXTRACT_FOOTER
    if config.MISTRAL_IMAGE_LIMIT > 0:
        fields["image_limit"] = config.MISTRAL_IMAGE_LIMIT
    if config.MISTRAL_IMAGE_MIN_SIZE > 0:
        fields["image_min_size"] = config.MISTRAL_IMAGE_MIN_SIZE
    return fields


def build_ocr_process_kwargs(
    *,
    document: Any,
    model: str,
    include_retries: bool,
    pages: Optional[List[int]] = None,
    request_id: Optional[str] = None,
    file_path: Optional[Path] = None,
    doc_type: str = "auto",
) -> Dict[str, Any]:
    """Build kwargs for ``client.ocr.process`` or a batch JSONL line ``body``."""
    ocr_params: Dict[str, Any] = {
        "model": model,
        "document": document,
        "include_image_base64": config.MISTRAL_INCLUDE_IMAGES,
    }
    if include_retries:
        ocr_params["retries"] = attr("get_retry_config")()
    if request_id is not None:
        ocr_params["id"] = request_id
    if pages is not None:
        ocr_params["pages"] = pages
    ocr_params.update(_ocr_shared_optional_params(file_path=file_path, doc_type=doc_type))
    return ocr_params


# ============================================================================
# OCR Processing
# ============================================================================


def _get_mistralai_package_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("mistralai")
    except Exception:
        return None


def _document_annotation_prompt_sha256() -> str:
    prompt = (config.MISTRAL_DOCUMENT_ANNOTATION_PROMPT or "").strip()
    if not prompt:
        return ""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_mistral_ocr_cache_contract_metadata(
    improve_weak: bool = True,
) -> Dict[str, Any]:
    """Stored with ``mistral_ocr`` cache entries; must match on read for a hit."""
    resolved_schema = _resolve_document_schema_type("auto")
    return {
        "contract_type": "mistral_ocr",
        "contract_version": config.MISTRAL_OCR_CACHE_CONTRACT_VERSION,
        "sdk_version": _get_mistralai_package_version(),
        "ocr_model": config.get_ocr_model(),
        "include_images": bool(config.MISTRAL_INCLUDE_IMAGES),
        "table_format": config.MISTRAL_TABLE_FORMAT or "",
        "extract_header": bool(config.MISTRAL_EXTRACT_HEADER),
        "extract_footer": bool(config.MISTRAL_EXTRACT_FOOTER),
        "image_limit": config.MISTRAL_IMAGE_LIMIT,
        "image_min_size": config.MISTRAL_IMAGE_MIN_SIZE,
        "structured_output_enabled": bool(config.MISTRAL_ENABLE_STRUCTURED_OUTPUT),
        "bbox_annotation_enabled": bool(
            config.MISTRAL_ENABLE_STRUCTURED_OUTPUT and config.MISTRAL_ENABLE_BBOX_ANNOTATION
        ),
        "document_annotation_enabled": bool(
            config.MISTRAL_ENABLE_STRUCTURED_OUTPUT and config.MISTRAL_ENABLE_DOCUMENT_ANNOTATION
        ),
        "document_schema_type": resolved_schema,
        "document_annotation_prompt_hash": _document_annotation_prompt_sha256(),
        "quality_assessment_enabled": bool(config.ENABLE_OCR_QUALITY_ASSESSMENT),
        "weak_page_improvement_enabled": bool(config.ENABLE_OCR_WEAK_PAGE_IMPROVEMENT),
        "improve_weak": bool(improve_weak),
        "image_preprocessing_enabled": bool(config.MISTRAL_ENABLE_IMAGE_PREPROCESSING),
        "image_optimization_enabled": bool(config.MISTRAL_ENABLE_IMAGE_OPTIMIZATION),
    }


def mistral_ocr_cache_contract_matches(stored: Any, current: Dict[str, Any]) -> bool:
    if not isinstance(stored, dict):
        return False
    if stored.get("contract_type") != "mistral_ocr":
        return False
    for key, val in current.items():
        if stored.get(key) != val:
            return False
    return True
