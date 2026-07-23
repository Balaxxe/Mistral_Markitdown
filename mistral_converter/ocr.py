"""Mistral OCR processing, parsing, quality assessment, and conversion pipeline."""

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

from .facade import attr
from .sdk_shims import Mistral
from .session import (
    _commit_session_pages,
    _estimate_session_pages_for_ocr,
    _ocr_session_page_delta,
    _release_session_pages_reservation,
    _reserve_session_pages,
)
logger = utils.logger


def _validate_file_for_ocr(
    file_path: Path, file_size_mb: float
) -> Optional[Tuple[bool, Optional[Dict[str, Any]], Optional[str]]]:
    """Return an early-exit error tuple if the file cannot be processed, else None.

    The size cap and its message live in :func:`utils.mistral_ocr_size_error`
    so up-front CLI validation and this runtime check stay in sync.
    """
    msg = utils.mistral_ocr_size_error(file_size_mb)
    if msg is not None:
        return False, None, msg
    return None


def _prepare_ocr_document(
    client: Mistral,
    file_path: Path,
    signed_url: Optional[str],
    file_size_mb: float,
    progress_callback: Optional[Callable[[str, float], None]],
) -> Tuple[Any, str]:
    """Upload (if needed) and build the SDK document object for OCR."""
    ext = file_path.suffix.lower().lstrip(".")
    is_image = ext in config.IMAGE_EXTENSIONS

    if signed_url:
        logger.debug("Using provided signed URL for OCR")
    else:
        if progress_callback:
            progress_callback(f"Uploading file ({file_size_mb:.1f} MB)...", 0.3)
        signed_url = attr("upload_file_for_ocr")(client, file_path)
        if not signed_url:
            raise RuntimeError("Failed to upload file")
        if progress_callback:
            progress_callback("Upload complete", 0.4)

    ImageURLChunk = attr("ImageURLChunk")
    DocumentURLChunk = attr("DocumentURLChunk")
    if is_image:
        if ImageURLChunk is not None:
            document = ImageURLChunk(image_url=signed_url)
            logger.debug("Using ImageURLChunk for %s file", ext)
        else:
            document = {"type": "image_url", "image_url": signed_url}
            logger.debug("Using image_url dict for %s file", ext)
    else:
        if DocumentURLChunk is not None:
            document = DocumentURLChunk(document_url=signed_url, document_name=file_path.name)
            logger.debug("Using DocumentURLChunk for %s file", ext)
        else:
            document = {"type": "document_url", "document_url": signed_url}
            logger.debug("Using document_url dict for %s file", ext)

    return document, signed_url


def process_with_ocr(
    client: Mistral,
    file_path: Path,
    model: Optional[str] = None,
    pages: Optional[List[int]] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    signed_url: Optional[str] = None,
    ocr_id: Optional[str] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Process file with Mistral OCR.

    Args:
        client: Mistral client instance
        file_path: Path to file
        model: Optional model override
        pages: Optional specific pages to process (0-indexed)
        progress_callback: Optional callback for progress updates (message, progress_0_to_1)
        signed_url: Optional pre-obtained signed URL (avoids re-uploading for weak page improvement)
        ocr_id: Optional task identifier for tracking/debugging

    Returns:
        Tuple of (success, ocr_result_dict, error_message)
    """

    def _report_progress(message: str, progress: float = 0.0):
        """Report progress if callback is provided."""
        if progress_callback:
            progress_callback(message, progress)

    estimated_pages = _estimate_session_pages_for_ocr(file_path, pages)
    reserved_pages = 0
    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        validation_error = _validate_file_for_ocr(file_path, file_size_mb)
        if validation_error is not None:
            return validation_error

        if config.MAX_PAGES_PER_SESSION > 0:
            if not _reserve_session_pages(estimated_pages):
                return (
                    False,
                    None,
                    (
                        f"Session page limit reached ({config.MAX_PAGES_PER_SESSION}). "
                        "Start a new session or increase MAX_PAGES_PER_SESSION."
                    ),
                )
            reserved_pages = estimated_pages

        _report_progress("Analyzing file...", 0.1)

        if model is None:
            model = config.get_ocr_model()

        logger.info("Processing with Mistral OCR using model: %s", model)
        _report_progress("Preparing document...", 0.2)

        try:
            document, signed_url = _prepare_ocr_document(client, file_path, signed_url, file_size_mb, progress_callback)
        except RuntimeError as upload_err:
            return False, None, str(upload_err)

        _report_progress("Processing with Mistral OCR...", 0.5)

        ocr_params = attr("build_ocr_process_kwargs")(
            document=document,
            model=model,
            include_retries=True,
            pages=pages,
            request_id=ocr_id,
            file_path=file_path,
        )

        response = client.ocr.process(**ocr_params)
        _report_progress("Parsing OCR response...", 0.8)

        if response:
            result = _parse_ocr_response(response, file_path)

            if result.get("parse_error"):
                logger.warning(
                    "OCR response parsing encountered an error for %s: %s",
                    file_path.name,
                    result["parse_error"],
                )

            if not result.get("full_text", "").strip():
                parse_hint = ""
                if result.get("parse_error"):
                    parse_hint = f" Parse error: {result['parse_error']}"
                error_msg = (
                    "Mistral OCR returned empty text. Possible causes: "
                    "API key lacks OCR access, document is empty/corrupted, "
                    "or response parsing failed.%s" % parse_hint
                )
                logger.warning(error_msg)
                return False, None, error_msg

            actual_pages = _ocr_session_page_delta(result)
            if reserved_pages != actual_pages:
                logger.debug(
                    "OCR page delta (%d) differs from reserved estimate (%d) for %s",
                    actual_pages,
                    reserved_pages,
                    file_path.name,
                )
            if config.MAX_PAGES_PER_SESSION > 0:
                if not _commit_session_pages(reserved_pages, actual_pages):
                    logger.warning(
                        "Session page limit (%d) reached during processing of %s. "
                        "Returning result but further OCR requests will be refused.",
                        config.MAX_PAGES_PER_SESSION,
                        file_path.name,
                    )
                reserved_pages = 0
            _report_progress("OCR processing complete", 1.0)
            return True, result, None
        else:
            return False, None, "Empty response from Mistral OCR"

    except Exception as e:
        status_code = getattr(e, "status_code", None) or getattr(e, "code", None)
        err_str = str(e)
        if status_code == 401 or (status_code is None and ("401" in err_str or "Unauthorized" in err_str)):
            error_msg = (
                "Mistral API authentication failed (401 Unauthorized). "
                "Please verify your API key has OCR access at https://console.mistral.ai/"
            )
            logger.error(error_msg)
        elif status_code == 403 or (status_code is None and ("403" in err_str or "Forbidden" in err_str)):
            error_msg = "Access denied to Mistral OCR (403 Forbidden). This feature may require a paid plan."
            logger.error(error_msg)
        else:
            error_msg = f"Error processing with Mistral OCR: {e}"
            # Unexpected failure path: preserve traceback for debugging.
            logger.exception("Unexpected error processing with Mistral OCR")
        return False, None, error_msg
    finally:
        if reserved_pages:
            _release_session_pages_reservation(reserved_pages)


def _extract_page_text(page: Any) -> str:
    """Extract text content from a single OCR page object."""
    if hasattr(page, "markdown") and page.markdown:
        return page.markdown
    if hasattr(page, "text") and page.text:
        return page.text
    if hasattr(page, "content") and page.content:
        return page.content
    if isinstance(page, dict):
        return page.get("markdown", page.get("text", page.get("content", "")))
    if isinstance(page, str):
        return page
    return ""


def _parse_page_object(page: Any, idx: int) -> Dict[str, Any]:
    """Parse a single OCR page object into a standardised dict."""
    raw_text = _extract_page_text(page)
    page_text = utils.clean_consecutive_duplicates(raw_text)
    # Decode HTML entities (e.g. &amp; -> &) that the OCR API may return
    page_text = html.unescape(page_text)

    raw_idx = getattr(page, "index", None)
    if raw_idx is None and isinstance(page, dict):
        raw_idx = page.get("index")
    if raw_idx is not None:
        try:
            api_page_index = int(raw_idx)
        except (TypeError, ValueError):
            api_page_index = idx
    else:
        api_page_index = idx
    # 1-based page label for markdown output (SDK ``pages`` selector is 0-based)
    page_number = api_page_index + 1

    page_data: Dict[str, Any] = {
        "page_number": page_number,
        "api_page_index": api_page_index,
        "text": page_text,
        "images": [],
        "dimensions": None,
        "tables": [],
        "hyperlinks": [],
        "header": None,
        "footer": None,
    }

    # Images
    if hasattr(page, "images") and page.images:
        for img in page.images:
            page_data["images"].append(
                {
                    "id": getattr(img, "id", None),
                    "top_left_x": getattr(img, "top_left_x", None),
                    "top_left_y": getattr(img, "top_left_y", None),
                    "bottom_right_x": getattr(img, "bottom_right_x", None),
                    "bottom_right_y": getattr(img, "bottom_right_y", None),
                    "bbox": getattr(img, "bbox", None),
                    "base64": (
                        (getattr(img, "image_base64", None) or getattr(img, "base64", None))
                        if config.MISTRAL_INCLUDE_IMAGES
                        else None
                    ),
                }
            )

    # Dimensions
    if hasattr(page, "dimensions") and page.dimensions:
        dims = page.dimensions
        page_data["dimensions"] = {
            "dpi": getattr(dims, "dpi", None),
            "height": getattr(dims, "height", None),
            "width": getattr(dims, "width", None),
        }

    # Tables
    if hasattr(page, "tables") and page.tables:
        page_data["tables"] = [t.model_dump() if hasattr(t, "model_dump") else t for t in page.tables]
        # Expand table placeholder links in page text with actual table content.
        # The API returns placeholders like [tbl-0.md](tbl-0.md) and stores the
        # real table data in the tables array.
        for tbl in page_data["tables"]:
            tbl_id = tbl.get("id", "") if isinstance(tbl, dict) else ""
            tbl_content = tbl.get("content", "") if isinstance(tbl, dict) else ""
            if tbl_id and tbl_content:
                page_data["text"] = page_data["text"].replace(f"[{tbl_id}]({tbl_id})", tbl_content)

    # Hyperlinks
    if hasattr(page, "hyperlinks") and page.hyperlinks:
        page_data["hyperlinks"] = [h.model_dump() if hasattr(h, "model_dump") else h for h in page.hyperlinks]

    # Header / footer
    if hasattr(page, "header") and page.header:
        page_data["header"] = page.header
    if hasattr(page, "footer") and page.footer:
        page_data["footer"] = page.footer

    return page_data


def _parse_pages_response(response: Any, result: Dict[str, Any]) -> None:
    """Parse a multi-page OCR response (``response.pages``) into *result*."""
    for idx, page in enumerate(response.pages):
        page_data = _parse_page_object(page, idx)
        result["pages"].append(page_data)
        if page_data["text"]:
            result["full_text"] += page_data["text"] + "\n\n"


def _parse_single_text_response(text: str, result: Dict[str, Any]) -> None:
    """Handle responses that carry a single text field (markdown / text / content)."""
    cleaned = html.unescape(utils.clean_consecutive_duplicates(text))
    result["full_text"] = cleaned
    result["pages"].append({"page_number": 1, "api_page_index": 0, "text": cleaned, "images": []})


def _parse_dict_response(response: dict, result: Dict[str, Any]) -> None:
    """Handle responses that arrive as plain Python dicts."""
    if "pages" in response:
        for idx, page in enumerate(response["pages"]):
            page_text = page.get("markdown", page.get("text", page.get("content", "")))
            page_text = html.unescape(utils.clean_consecutive_duplicates(page_text))
            # Expand table placeholder links with actual content
            tables = page.get("tables", [])
            for tbl in tables:
                tbl_id = tbl.get("id", "") if isinstance(tbl, dict) else ""
                tbl_content = tbl.get("content", "") if isinstance(tbl, dict) else ""
                if tbl_id and tbl_content:
                    page_text = page_text.replace(f"[{tbl_id}]({tbl_id})", tbl_content)
            raw_index = page.get("index")
            if raw_index is not None:
                try:
                    api_page_index = int(raw_index)
                except (TypeError, ValueError):
                    api_page_index = idx
            else:
                api_page_index = idx
            page_num = api_page_index + 1
            result["pages"].append(
                {
                    "page_number": page_num,
                    "api_page_index": api_page_index,
                    "text": page_text,
                    "images": page.get("images", []),
                    "tables": tables,
                }
            )
            if page_text:
                result["full_text"] += page_text + "\n\n"
    else:
        text = response.get("markdown", response.get("text", ""))
        if text:
            _parse_single_text_response(text, result)


def _extract_structured_outputs(response: Any, result: Dict[str, Any]) -> None:
    """Extract bbox_annotations and document_annotation from the response."""
    if hasattr(response, "bbox_annotations") and response.bbox_annotations:
        result["bbox_annotations"] = [
            bbox.model_dump() if hasattr(bbox, "model_dump") else bbox for bbox in response.bbox_annotations
        ]

    if hasattr(response, "document_annotation") and response.document_annotation:
        annotation = response.document_annotation
        if isinstance(annotation, str):
            try:
                result["document_annotation"] = json.loads(annotation)
            except (json.JSONDecodeError, TypeError):
                result["document_annotation"] = annotation
        elif hasattr(annotation, "model_dump"):
            result["document_annotation"] = annotation.model_dump()
        else:
            result["document_annotation"] = annotation


def _extract_response_metadata(response: Any, result: Dict[str, Any]) -> None:
    """Extract metadata, usage_info, and model from the response."""
    if hasattr(response, "metadata"):
        result["metadata"] = response.metadata
    elif isinstance(response, dict) and "metadata" in response:
        result["metadata"] = response["metadata"]

    if hasattr(response, "usage_info") and response.usage_info:
        usage = response.usage_info
        result["usage_info"] = {
            "pages_processed": getattr(usage, "pages_processed", None),
            "doc_size_bytes": getattr(usage, "doc_size_bytes", None),
        }
    elif isinstance(response, dict) and "usage_info" in response:
        result["usage_info"] = response["usage_info"]

    if hasattr(response, "model") and response.model:
        result["model"] = response.model
    elif isinstance(response, dict) and "model" in response:
        result["model"] = response["model"]


def _parse_ocr_response(response: Any, file_path: Path) -> Dict[str, Any]:
    """
    Parse OCR response into structured dictionary.

    Delegates to focused helpers for pages, single-text, dict, annotations
    and metadata extraction.

    Args:
        response: Mistral OCR response
        file_path: Original file path

    Returns:
        Parsed OCR result
    """
    result: Dict[str, Any] = {
        "file_name": file_path.name,
        "pages": [],
        "full_text": "",
        "images": [],
        "metadata": {},
        "bbox_annotations": [],
        "document_annotation": None,
        "usage_info": {},
        "model": None,
        "parse_error": None,
    }

    try:
        _extract_structured_outputs(response, result)

        if hasattr(response, "pages") and response.pages:
            _parse_pages_response(response, result)
        elif hasattr(response, "markdown") and response.markdown:
            _parse_single_text_response(response.markdown, result)
        elif hasattr(response, "text") and response.text:
            _parse_single_text_response(response.text, result)
        elif hasattr(response, "content") and response.content:
            _parse_single_text_response(response.content, result)
        elif isinstance(response, dict):
            _parse_dict_response(response, result)

        _extract_response_metadata(response, result)

        logger.debug(
            "Extracted %d pages, %d chars",
            len(result["pages"]),
            len(result["full_text"]),
        )

    except Exception as e:
        # Preserve full traceback so unexpected parser failures can be diagnosed from logs.
        logger.exception("Error parsing OCR response: %s", e)
        result["parse_error"] = str(e)

    return result


# ============================================================================
# Per-Page OCR Improvements
# ============================================================================

# Cap concurrency for weak-page re-OCR to avoid nested thread-pool explosion.
# When improve_weak_pages runs inside _process_files_concurrently (which uses
# MAX_CONCURRENT_FILES threads), an uncapped inner pool could spawn M*M threads.
# Configurable via config.OCR_MAX_WEAK_PAGE_WORKERS.


def _is_weak_page(text: str) -> bool:
    """
    Detect if OCR page text is weak or low-quality.

    Checks for:
    - Very short text
    - High repetition rate (same words/phrases repeated)
    - Repeated page-header patterns
    - Low average line length
    - Low unique token ratio

    (Digit-density checks were removed; ``assess_ocr_quality`` still reports ``digit_count``.)

    All thresholds are configurable via config.py (from .env).

    Args:
        text: Page text to analyze

    Returns:
        True if page appears to have weak OCR results
    """
    if not text or len(text.strip()) < 10:
        return True

    # Check 1: Very short text (configurable via OCR_MIN_TEXT_LENGTH)
    if len(text.strip()) < config.OCR_MIN_TEXT_LENGTH:
        return True

    # Check 2: Token uniqueness ratio (detect heavy repetition)
    # Configurable via OCR_MIN_UNIQUENESS_RATIO
    tokens = text.split()
    if not tokens:  # pragma: no cover – unreachable after len(text.strip()) >= 10
        return True

    unique_tokens = set(tokens)
    uniqueness_ratio = len(unique_tokens) / len(tokens)

    if uniqueness_ratio < config.OCR_MIN_UNIQUENESS_RATIO:
        logger.debug("Low uniqueness ratio: %.2f", uniqueness_ratio)
        return True

    # Check 3: Detect repeated header patterns
    # Use regex to catch all "Page N" patterns, not just a hardcoded few
    # Configurable via OCR_MAX_PHRASE_REPETITIONS
    page_refs = re.findall(r"Page\s+\d+", text)
    if len(page_refs) > config.OCR_MAX_PHRASE_REPETITIONS:
        logger.debug("Repeated page references found %s times", len(page_refs))
        return True

    # Check 4: Average line length (very short lines suggest parsing issues)
    # Configurable via OCR_MIN_AVG_LINE_LENGTH
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        if avg_line_length < config.OCR_MIN_AVG_LINE_LENGTH:
            logger.debug("Short average line length: %.1f", avg_line_length)
            return True

    return False


def assess_ocr_quality(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess the quality of OCR results to determine if they should be used.

    Args:
        ocr_result: OCR result dictionary

    Returns:
        Dictionary with quality assessment:
        {
            "is_usable": bool,           # Overall quality verdict
            "quality_score": float,       # 0-100 score
            "issues": List[str],          # List of quality issues found
            "weak_page_count": int,       # Number of weak pages
            "total_page_count": int,      # Total pages analyzed
            "digit_count": int,           # Total digits extracted
            "uniqueness_ratio": float,    # Token uniqueness across all pages
        }
    """
    full_text = ocr_result.get("full_text", "")
    pages = ocr_result.get("pages", [])

    assessment = {
        "is_usable": True,
        "quality_score": 100.0,
        "issues": [],
        "weak_page_count": 0,
        "total_page_count": len(pages),
        "digit_count": 0,
        "uniqueness_ratio": 0.0,
    }

    if not full_text or len(full_text.strip()) < 50:
        assessment["is_usable"] = False
        assessment["quality_score"] = 0.0
        assessment["issues"].append("Minimal text extracted")
        return assessment

    # Count weak pages
    for page in pages:
        page_text = page.get("text", "")
        if _is_weak_page(page_text):
            assessment["weak_page_count"] += 1

    # Calculate metrics
    assessment["digit_count"] = sum(1 for char in full_text if char.isdigit())

    tokens = full_text.split()
    if tokens:
        unique_tokens = set(tokens)
        assessment["uniqueness_ratio"] = len(unique_tokens) / len(tokens)

    # Deduct points for issues
    if assessment["weak_page_count"] > 0:
        weak_ratio = assessment["weak_page_count"] / max(1, assessment["total_page_count"])
        points_lost = weak_ratio * config.OCR_QUALITY_PENALTY_WEAK_PAGES_MAX
        assessment["quality_score"] -= points_lost
        assessment["issues"].append(
            f"{assessment['weak_page_count']}/{assessment['total_page_count']} pages are weak quality"
        )

    if assessment["uniqueness_ratio"] < config.OCR_MIN_UNIQUENESS_RATIO:
        assessment["quality_score"] -= config.OCR_QUALITY_PENALTY_HIGH_REPETITION
        assessment["issues"].append(f"High repetition (uniqueness: {assessment['uniqueness_ratio']:.1%})")

    # Clamp score to [0, 100]
    assessment["quality_score"] = max(0.0, min(100.0, assessment["quality_score"]))

    # Final verdict (configurable via OCR_QUALITY_THRESHOLD_ACCEPTABLE)
    if assessment["quality_score"] < config.OCR_QUALITY_THRESHOLD_ACCEPTABLE:
        assessment["is_usable"] = False
        assessment["issues"].append("Overall quality too low for inclusion")

    logger.info(
        "OCR Quality Assessment: Score=%.1f/100, Usable=%s, Issues=%d",
        assessment["quality_score"],
        assessment["is_usable"],
        len(assessment["issues"]),
    )

    return assessment


def _detect_weak_pages(ocr_result: Dict[str, Any]) -> List[int]:
    """Return indices of pages whose OCR text is considered weak."""
    weak_pages = []
    for i, page in enumerate(ocr_result["pages"]):
        text = page.get("text", "")
        if _is_weak_page(text):
            weak_pages.append(i)
            logger.debug("Page %s has weak OCR result (%s chars)", i + 1, len(text))
    return weak_pages


def _run_weak_page_improvements(
    weak_pages: List[int],
    improve_fn: Callable[[int], Tuple[int, Optional[Dict[str, Any]]]],
    ocr_result: Dict[str, Any],
) -> None:
    """Execute *improve_fn* across *weak_pages* in a thread pool, mutating *ocr_result*."""
    max_workers = min(len(weak_pages), config.OCR_MAX_WEAK_PAGE_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(improve_fn, idx): idx for idx in weak_pages}
        for future in as_completed(futures):
            try:
                page_idx, improved_page = future.result()
            except Exception as e:
                logger.warning("Unexpected error retrieving page improvement result: %s", e)
                continue
            if improved_page is None:
                continue
            original_page = ocr_result["pages"][page_idx]
            original_len = len(original_page.get("text", ""))
            improved_len = len(improved_page.get("text", ""))
            if improved_len > original_len:
                logger.info("Improved page %d", page_idx + 1)
                merged_page = dict(improved_page)
                if "page_number" not in merged_page and "page_number" in original_page:
                    merged_page["page_number"] = original_page["page_number"]
                if "api_page_index" not in merged_page and "api_page_index" in original_page:
                    merged_page["api_page_index"] = original_page["api_page_index"]
                # Re-OCR may omit images; keep originals so inline assets are not dropped.
                if not merged_page.get("images") and original_page.get("images"):
                    merged_page["images"] = list(original_page["images"])
                ocr_result["pages"][page_idx] = merged_page


def improve_weak_pages(client: Mistral, file_path: Path, ocr_result: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Re-OCR weak pages with low confidence or short text.

    Uses enhanced detection heuristics to identify:
    - Short text
    - Repetitive content
    - Low information density
    - Missing numerical data

    .. note::
        This function **mutates** *ocr_result* in place (replacing pages and
        rebuilding ``full_text``).  The same dict is returned for convenience.

    Args:
        client: Mistral client instance
        file_path: Path to original file
        ocr_result: Initial OCR result (modified in place)
        model: Model to use

    Returns:
        The same *ocr_result* dict, with weak pages replaced where improvement was found.
    """
    if not ocr_result.get("pages"):
        return ocr_result

    logger.info("Analyzing pages for weak OCR results...")

    weak_pages = _detect_weak_pages(ocr_result)

    if not weak_pages:
        logger.info("No weak pages detected")
        return ocr_result

    logger.info("Re-processing %s weak pages...", len(weak_pages))

    # Upload file ONCE and reuse the signed URL for all weak pages.
    # Re-upload if the URL is nearing expiry (within 10% of the TTL).
    signed_url = None
    upload_time = 0.0
    url_ttl_seconds = config.MISTRAL_SIGNED_URL_EXPIRY * 3600
    try:
        logger.debug("Uploading file once for weak page improvements...")
        signed_url = attr("upload_file_for_ocr")(client, file_path)
        upload_time = time.time()
    except Exception as e:
        logger.warning("Failed to pre-upload for weak pages: %s", e)

    _url_lock = threading.Lock()

    def _refresh_url_if_needed() -> Optional[str]:
        nonlocal signed_url, upload_time
        with _url_lock:
            if (
                signed_url
                and (time.time() - upload_time) > url_ttl_seconds * config.MISTRAL_SIGNED_URL_REFRESH_THRESHOLD
            ):
                try:
                    logger.debug("Signed URL nearing expiry, re-uploading...")
                    signed_url = attr("upload_file_for_ocr")(client, file_path)
                    upload_time = time.time()
                except Exception as e:
                    logger.warning("Re-upload failed: %s", e)
            return signed_url

    def _improve_page(page_idx: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Re-OCR a single page; returns (page_idx, improved_page_data) or (page_idx, None)."""
        try:
            url = _refresh_url_if_needed()
            page_entry = ocr_result["pages"][page_idx]
            ocr_page_spec = page_entry.get("api_page_index", page_idx)
            ok, improved_result, _ = attr("process_with_ocr")(
                client,
                file_path,
                model=model,
                pages=[ocr_page_spec],
                signed_url=url,
            )
            if ok and improved_result and improved_result.get("pages"):
                return page_idx, improved_result["pages"][0]
        except Exception as e:
            logger.warning("Error improving page %d: %s", page_idx + 1, e)
        return page_idx, None

    _run_weak_page_improvements(weak_pages, _improve_page, ocr_result)

    # Rebuild full text
    ocr_result["full_text"] = "\n\n".join(page.get("text", "") for page in ocr_result["pages"])

    return ocr_result


# ============================================================================
# Image Extraction and Saving
# ============================================================================


def _process_ocr_result_pipeline(
    client: Mistral,
    file_path: Path,
    ocr_result: Dict[str, Any],
    use_cache: bool = True,
    improve_weak: bool = True,
    from_cache: bool = False,
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Common pipeline for processing OCR results (quality check, improvement, saving).

    Args:
        client: Mistral client instance
        file_path: Path to file
        ocr_result: OCR result dictionary
        use_cache: Whether to cache the result
        improve_weak: Whether to improve weak pages
        from_cache: Whether this result came from cache (skips re-improvement and image saving)

    Returns:
        Tuple of (success, output_md_path, error_message)
    """
    quality_assessment: Optional[Dict[str, Any]] = None

    # If from cache, reuse stored quality assessment if available
    if from_cache and "quality_assessment" in ocr_result:
        logger.info("Using cached OCR result with stored quality assessment")
        quality_assessment = ocr_result["quality_assessment"]
    elif config.ENABLE_OCR_QUALITY_ASSESSMENT:
        # Assess OCR quality (only for fresh results)
        logger.info("Assessing OCR quality...")
        quality_assessment = assess_ocr_quality(ocr_result)
        ocr_result["quality_assessment"] = quality_assessment
    else:
        logger.info("OCR quality assessment disabled by configuration")

    # Re-process weak pages if requested and quality is low
    # IMPORTANT: Skip re-improvement for cached results to avoid redundant API calls
    # Cached results have already been improved (if improvement was enabled when they were created)
    if (
        not from_cache  # Only improve fresh results, not cached ones
        and config.ENABLE_OCR_QUALITY_ASSESSMENT
        and config.ENABLE_OCR_WEAK_PAGE_IMPROVEMENT
        and improve_weak
        and ocr_result.get("pages")
        and quality_assessment
        and quality_assessment.get("weak_page_count", 0) > 0
    ):
        logger.info(
            "Attempting to improve %d weak pages...",
            quality_assessment["weak_page_count"],
        )
        model = config.get_ocr_model()
        # Note: improve_weak_pages is synchronous
        ocr_result = improve_weak_pages(client, file_path, ocr_result, model)

        # Re-assess quality after improvement
        quality_assessment = assess_ocr_quality(ocr_result)
        ocr_result["quality_assessment"] = quality_assessment
        logger.info("Quality after improvement: %.1f/100", quality_assessment["quality_score"])

    # Cache result (only for fresh results)
    if use_cache and not from_cache:
        utils.cache.set(
            file_path,
            ocr_result,
            cache_type="mistral_ocr",
            metadata=attr("build_mistral_ocr_cache_contract_metadata")(
                improve_weak=improve_weak,
            ),
        )

    # Save extracted images (skip for cached results to avoid redundant IO)
    if not from_cache:
        attr("save_extracted_images")(ocr_result, file_path)
    else:
        logger.debug("Skipping image extraction for cached result")

    # Generate markdown output
    output_path = _create_markdown_output(file_path, ocr_result)

    # Save JSON metadata if requested (non-fatal -- OCR already succeeded)
    if config.SAVE_MISTRAL_JSON:
        try:
            json_path = config.OUTPUT_MD_DIR / f"{utils.safe_output_stem(file_path)}_ocr_metadata.json"
            utils.atomic_write_text(json_path, json.dumps(ocr_result, indent=2, ensure_ascii=False))
            logger.info("Saved OCR metadata: %s", json_path.name)
        except Exception as e:
            logger.warning("Failed to save OCR metadata JSON: %s", e)

    # Save structured outputs if they exist (non-fatal -- OCR already succeeded)
    try:
        _save_structured_outputs(file_path, ocr_result)
    except Exception as e:
        logger.warning("Failed to save structured outputs: %s", e)

    return True, output_path, None


# ============================================================================
# Main Conversion Function
# ============================================================================


def convert_with_mistral_ocr(
    file_path: Path, use_cache: bool = True, improve_weak: bool = True
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Convert file using Mistral OCR with full pipeline.

    Args:
        file_path: Path to file
        use_cache: Use cached results if available
        improve_weak: Re-process weak pages

    Returns:
        Tuple of (success, output_md_path, error_message)
    """
    client = attr("get_mistral_client")()
    if client is None:
        error_msg = "Mistral client not available (see errors above for details)"
        logger.warning(error_msg)
        return False, None, error_msg

    # Check cache (payload must match current OCR request contract metadata)
    from_cache = False
    if use_cache:
        cache_entry = utils.cache.get_entry(file_path, cache_type="mistral_ocr")
        contract = attr("build_mistral_ocr_cache_contract_metadata")(
            improve_weak=improve_weak,
        )
        if cache_entry and attr("mistral_ocr_cache_contract_matches")(cache_entry.get("metadata"), contract):
            cached_result = cache_entry.get("data")
            if isinstance(cached_result, dict):
                logger.info("Using cached Mistral OCR result for %s", file_path.name)
                ocr_result = cached_result
                success = True
                error = None
                from_cache = True
            else:
                logger.warning(
                    "Cached OCR result for %s is not a dict (type=%s), invalidating and re-processing",
                    file_path.name,
                    type(cached_result).__name__,
                )
                success, ocr_result, error = attr("process_with_ocr")(client, file_path)
        else:
            if cache_entry and not attr("mistral_ocr_cache_contract_matches")(cache_entry.get("metadata"), contract):
                logger.debug(
                    "Mistral OCR cache ignored for %s (contract metadata mismatch)",
                    file_path.name,
                )
            success, ocr_result, error = attr("process_with_ocr")(client, file_path)
    else:
        success, ocr_result, error = attr("process_with_ocr")(client, file_path)

    if not success or not ocr_result:
        return False, None, error

    # Process result using common pipeline
    # Pass from_cache flag to skip redundant API calls and IO for cached results
    return _process_ocr_result_pipeline(client, file_path, ocr_result, use_cache, improve_weak, from_cache)


def _save_structured_outputs(file_path: Path, ocr_result: Dict[str, Any]) -> None:
    """
    Save structured outputs from bbox and document annotations.

    Args:
        file_path: Original file path
        ocr_result: OCR result dictionary containing structured outputs
    """
    # Save bounding box annotations if present
    if "bbox_annotations" in ocr_result and ocr_result["bbox_annotations"]:
        bbox_path = config.OUTPUT_MD_DIR / f"{utils.safe_output_stem(file_path)}_bbox_annotations.json"
        utils.atomic_write_text(
            bbox_path,
            json.dumps(ocr_result["bbox_annotations"], indent=2, ensure_ascii=False),
        )
        logger.info("Saved bbox annotations: %s", bbox_path.name)

    # Save document annotations if present
    if "document_annotation" in ocr_result and ocr_result["document_annotation"]:
        doc_path = config.OUTPUT_MD_DIR / f"{utils.safe_output_stem(file_path)}_document_annotation.json"
        utils.atomic_write_text(
            doc_path,
            json.dumps(ocr_result["document_annotation"], indent=2, ensure_ascii=False),
        )
        logger.info("Saved document annotation: %s", doc_path.name)


def _create_markdown_output(file_path: Path, ocr_result: Dict[str, Any]) -> Path:
    """
    Create markdown output from OCR result.

    Args:
        file_path: Original file path
        ocr_result: OCR result dictionary

    Returns:
        Path to created markdown file
    """
    # Calculate total image count from all pages (images are stored per-page)
    total_image_count = sum(len(page.get("images", [])) for page in ocr_result.get("pages", []))

    # Generate frontmatter
    frontmatter = utils.generate_yaml_frontmatter(
        title=f"OCR: {file_path.stem}",
        file_name=file_path.name,
        conversion_method="Mistral OCR",
        additional_fields={
            "page_count": len(ocr_result.get("pages", [])),
            "image_count": total_image_count,
        },
    )

    # Build markdown content
    md_content = frontmatter + f"\n# OCR Result: {file_path.name}\n\n"

    # Add page-by-page breakdown (no "Full Text" section to avoid duplication)
    if ocr_result.get("pages"):
        total_pages = len(ocr_result["pages"])
        md_content += f"## OCR Content ({total_pages} page{'s' if total_pages != 1 else ''})\n\n"

        # page_number is now preserved as the API's 1-based index
        for page in ocr_result["pages"]:
            text = page.get("text", "")
            display_page_num = page.get("page_number", 1)

            md_content += f"### Page {display_page_num}\n\n"

            # Header (if extracted separately from page content)
            header = page.get("header")
            if header:
                header_text = header if isinstance(header, str) else getattr(header, "text", str(header))
                if header_text.strip():
                    md_content += f"> **Header:** {header_text.strip()}\n\n"

            md_content += text

            # Footer (if extracted separately from page content)
            footer = page.get("footer")
            if footer:
                footer_text = footer if isinstance(footer, str) else getattr(footer, "text", str(footer))
                if footer_text.strip():
                    md_content += f"\n\n> **Footer:** {footer_text.strip()}"

            md_content += "\n\n---\n\n"
    else:
        # Fallback if pages aren't available (shouldn't happen, but be defensive)
        md_content += "## OCR Content\n\n"
        md_content += ocr_result.get("full_text", "")
        md_content += "\n\n---\n\n"

    # Save markdown
    output_path = config.OUTPUT_MD_DIR / f"{utils.safe_output_stem(file_path)}_mistral_ocr.md"
    utils.atomic_write_text(output_path, md_content)

    # Save text version
    utils.save_text_output(output_path, md_content)

    logger.info("Saved Mistral OCR output: %s", output_path.name)

    return output_path


# ============================================================================
# Document QnA (NEW - from updated Mistral docs)
# Query documents using chat.complete with document_url content type
# ============================================================================

