"""Session page budget globals and helpers."""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import utils

logger = utils.logger
_session_pages_processed = 0
_session_pages_inflight = 0  # reserved estimate while OCR HTTP call is in flight
_session_pages_warned = False
_session_pages_lock = threading.Lock()


def _session_page_limit() -> int:
    """Return the configured page cap, or 0 when a direct override is unsafe."""
    try:
        return int(config.MAX_PAGES_PER_SESSION)
    except (TypeError, ValueError):
        return 0


def _estimate_session_pages_for_ocr(file_path: Path, pages: Optional[List[int]]) -> int:
    """Upper-bound page estimate for session budgeting (concurrent-safe reserve).

    Uses an explicit ``pages`` slice when provided, else local PDF page count when
    available, else a conservative default so parallel workers cannot all slip
    past ``MAX_PAGES_PER_SESSION`` before committing real counts.
    """
    if pages is not None:
        return max(1, len(pages))

    ext = file_path.suffix.lower().lstrip(".")
    if ext in config.IMAGE_EXTENSIONS:
        return 1

    if ext == "pdf":
        try:
            import local_converter as _lc

            analysis = _lc.analyze_file_content(file_path)
            pc = int(analysis.get("page_count") or 0)
            if pc > 0:
                return pc
        except (OSError, ImportError, ValueError):
            pass
        # Unknown PDFs reserve the entire remaining budget so only one can be
        # in flight at a time without blocking later sequential work after a
        # smaller-than-reserved response commits its actual page count.
        page_limit = _session_page_limit()
        if page_limit > 0:
            with _session_pages_lock:
                remaining = page_limit - _session_pages_processed - _session_pages_inflight
            return max(1, remaining)
        return 1

    # Office / other non-image non-PDF: conservative estimate from config.
    cap = _session_page_limit()
    if cap <= 0:
        return 1
    return max(1, min(cap, config.OCR_OFFICE_PAGE_ESTIMATE_DEFAULT))


def _reserve_session_pages(estimated: int) -> bool:
    """Reserve *estimated* pages against the session cap before starting OCR."""
    global _session_pages_inflight
    est = max(1, estimated)
    with _session_pages_lock:
        page_limit = _session_page_limit()
        if page_limit <= 0:
            return False
        if _session_pages_processed + _session_pages_inflight + est > page_limit:
            return False
        _session_pages_inflight += est
        return True


def _commit_session_pages(reserved: int, actual: int) -> bool:
    """Release *reserved* inflight credit and commit *actual* processed pages."""
    global _session_pages_processed, _session_pages_inflight, _session_pages_warned
    with _session_pages_lock:
        _session_pages_inflight = max(0, _session_pages_inflight - max(0, reserved))
        page_limit = _session_page_limit()
        if page_limit <= 0:
            return False
        new_total = _session_pages_processed + actual
        _session_pages_processed = new_total
        if new_total >= page_limit:
            if not _session_pages_warned:
                _session_pages_warned = True
                if new_total > page_limit:
                    logger.warning(
                        "Session page budget exceeded (%d > %d pages). This can happen when "
                        "parallel OCR jobs underestimated page counts or the API returned "
                        "more pages than reserved. Further OCR requests will be refused.",
                        new_total,
                        page_limit,
                    )
                else:
                    logger.warning(
                        "Session page limit reached (%d/%d). Further OCR requests will be refused.",
                        new_total,
                        page_limit,
                    )
            return False
        return True


def _release_session_pages_reservation(reserved: int) -> None:
    """Return reserved inflight credit when OCR fails before a result exists."""
    global _session_pages_inflight
    if reserved <= 0:
        return
    with _session_pages_lock:
        _session_pages_inflight = max(0, _session_pages_inflight - reserved)


def _is_page_limit_reached() -> bool:
    """Return ``True`` if the session page limit has already been reached."""
    with _session_pages_lock:
        page_limit = _session_page_limit()
        return page_limit <= 0 or _session_pages_processed >= page_limit


def _ocr_session_page_delta(result: Dict[str, Any]) -> int:
    """Pages to count against ``MAX_PAGES_PER_SESSION`` for one OCR response.

    Prefer ``len(pages)``; if the parser left ``pages`` empty but text exists,
    use ``usage_info.pages_processed`` when present, otherwise count as one page.
    """
    pages = result.get("pages") or []
    if pages:
        return len(pages)
    usage = result.get("usage_info") or {}
    if isinstance(usage, dict):
        pp = usage.get("pages_processed")
        if isinstance(pp, int) and pp > 0:
            return pp
    if (result.get("full_text") or "").strip():
        return 1
    return 0


def reset_session_page_counter() -> None:
    """Reset the session page counter so a new logical session can start fresh.

    Useful when embedding the converter in a long-lived process (e.g. a web
    service) where each request should have its own page budget. A reset is
    skipped while OCR work is active so neither committed nor reserved usage
    can be detached from requests that are still completing.
    """
    global _session_pages_processed, _session_pages_warned
    with _session_pages_lock:
        if _session_pages_inflight > 0:
            logger.warning(
                "Session page counter reset skipped while %d page(s) are reserved",
                _session_pages_inflight,
            )
            return
        _session_pages_processed = 0
        _session_pages_warned = False
