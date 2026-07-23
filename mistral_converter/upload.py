"""Mistral Files API upload helpers and local upload registry."""

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
import utils

from .sdk_shims import Mistral
from .facade import attr

logger = utils.logger
_UPLOAD_REGISTRY_LOCK = threading.Lock()
_UPLOAD_REGISTRY_FILENAME = "mistral_upload_registry.json"


def _upload_registry_path() -> Path:
    return config.CACHE_DIR / _UPLOAD_REGISTRY_FILENAME


def _load_upload_registry() -> List[Dict[str, Any]]:
    """Load the local upload registry; return [] if missing or corrupt."""
    path = _upload_registry_path()
    try:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return [e for e in data if isinstance(e, dict) and e.get("id")]
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.debug("Could not load upload registry: %s", e)
        return []


def _save_upload_registry(entries: List[Dict[str, Any]]) -> bool:
    """Persist the upload registry atomically. Returns False on I/O failure."""
    path = _upload_registry_path()
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        utils.atomic_write_text(path, json.dumps(entries, indent=2))
        return True
    except OSError as e:
        logger.warning("Could not save upload registry: %s", e)
        return False


def _register_uploaded_file(file_id: str, purpose: str) -> bool:
    """Record an uploaded file id in the local registry.

    Returns False when the id is empty or the registry cannot be persisted, so
    callers can delete the remote upload instead of orphaning it under
    ``CLEANUP_UPLOAD_SCOPE=registry``.
    """
    if not file_id:
        return False
    now = datetime.now(timezone.utc).isoformat()
    with _UPLOAD_REGISTRY_LOCK:
        entries = _load_upload_registry()
        for entry in entries:
            if entry.get("id") == file_id:
                entry["purpose"] = purpose
                if not entry.get("created_at"):
                    entry["created_at"] = now
                return _save_upload_registry(entries)
        entries.append({"id": file_id, "purpose": purpose, "created_at": now})
        return _save_upload_registry(entries)


def _unregister_uploaded_file(file_id: str) -> None:
    """Remove a file id from the local upload registry."""
    if not file_id:
        return
    with _UPLOAD_REGISTRY_LOCK:
        entries = _load_upload_registry()
        new_entries = [e for e in entries if e.get("id") != file_id]
        if len(new_entries) != len(entries):
            _save_upload_registry(new_entries)


def _parse_registry_created_at(value: Any) -> Optional[datetime]:
    """Parse a registry created_at value into an aware UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            return None
    return None


def cleanup_uploaded_files(client: Mistral, days_old: Optional[int] = None) -> int:
    """
    Clean up old files uploaded to Mistral Files API.

    When ``CLEANUP_UPLOAD_SCOPE`` is ``"registry"`` (default), only file IDs
    present in the local upload registry and older than *days_old* are deleted.
    When ``"all"``, performs account-wide age-based cleanup for ``ocr`` and
    ``batch`` purposes and prunes matching registry entries.

    Args:
        client: Mistral client instance
        days_old: Delete files older than N days (default: from config)

    Returns:
        Number of files deleted
    """
    if days_old is None:
        days_old = config.UPLOAD_RETENTION_DAYS

    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
        scope = getattr(config, "CLEANUP_UPLOAD_SCOPE", "registry")

        if scope == "registry":
            deleted = _cleanup_registry_scoped(client, cutoff_date)
        else:
            deleted_ids: List[str] = []
            deleted = _cleanup_files_by_purpose(client, "ocr", cutoff_date, deleted_ids)
            deleted += _cleanup_files_by_purpose(client, "batch", cutoff_date, deleted_ids)
            for fid in deleted_ids:
                _unregister_uploaded_file(fid)

        if deleted > 0:
            logger.info(
                "Cleaned up %s old uploaded files (older than %s days, scope=%s)",
                deleted,
                days_old,
                scope,
            )

        return deleted

    except Exception as e:
        logger.warning("Error cleaning up uploaded files: %s", e)
        return 0


def _cleanup_registry_scoped(client: Mistral, cutoff_date: datetime) -> int:
    """Delete only registry-tracked uploads older than *cutoff_date*."""
    deleted = 0
    with _UPLOAD_REGISTRY_LOCK:
        entries = _load_upload_registry()
        remaining: List[Dict[str, Any]] = []
        for entry in entries:
            file_id = entry.get("id")
            if not file_id:
                continue
            file_created = _parse_registry_created_at(entry.get("created_at"))
            # Missing/unparsable timestamps must not permanently shield uploads.
            if file_created is not None and file_created >= cutoff_date:
                remaining.append(entry)
                continue
            try:
                client.files.delete(file_id=file_id)
                deleted += 1
                logger.debug(
                    "Deleted registry-tracked %s file: %s (created %s)",
                    entry.get("purpose", "unknown"),
                    file_id,
                    file_created,
                )
            except Exception as e:
                logger.debug("Error deleting registry file %s: %s", file_id, e)
                remaining.append(entry)
        _save_upload_registry(remaining)
    return deleted


def _cleanup_files_by_purpose(
    client: Mistral,
    purpose: str,
    cutoff_date: datetime,
    deleted_ids: List[str],
) -> int:
    """Delete files older than cutoff_date for a given purpose (account-wide)."""
    deleted = 0
    page = 0
    page_size = 100

    while True:
        try:
            files_response = client.files.list(purpose=purpose, page=page, page_size=page_size)
            files_list = files_response.data if hasattr(files_response, "data") else files_response
        except Exception as e:
            logger.debug(
                "Error listing %s files for cleanup (page %s): %s",
                purpose,
                page,
                e,
            )
            break

        if not files_list:
            break

        for file in files_list:
            try:
                if not hasattr(file, "created_at"):
                    continue

                if isinstance(file.created_at, str):
                    file_created = datetime.fromisoformat(file.created_at.replace("Z", "+00:00"))
                elif hasattr(file.created_at, "replace"):
                    file_created = file.created_at
                    if file_created.tzinfo is None:
                        file_created = file_created.replace(tzinfo=timezone.utc)
                else:
                    logger.debug(
                        "Unexpected created_at type for file %s: %s",
                        file.id,
                        type(file.created_at),
                    )
                    continue

                if file_created < cutoff_date:
                    client.files.delete(file_id=file.id)
                    deleted += 1
                    deleted_ids.append(file.id)
                    logger.debug(
                        "Deleted old %s file: %s (created %s)",
                        purpose,
                        file.id,
                        file_created,
                    )
            except Exception as e:
                logger.debug("Error processing %s file %s: %s", purpose, file.id, e)
                continue

        total = getattr(files_response, "total", None)
        if isinstance(total, int) and total >= 0 and (page + 1) * page_size >= total:
            break
        if len(files_list) < page_size:
            break
        page += 1

    return deleted


def _delete_ocr_file_ids(client: Mistral, file_ids: List[str]) -> None:
    """Best-effort delete for orphaned OCR uploads (e.g. failed batch assembly)."""
    for fid in file_ids:
        try:
            client.files.delete(file_id=fid)
            _unregister_uploaded_file(fid)
        except Exception as e:
            logger.warning("Failed to delete uploaded file %s: %s", fid, e)


def _upload_file_for_ocr_pair(
    client: Mistral,
    file_path: Path,
    expiry_hours: Optional[int] = None,
) -> Optional[Tuple[str, str]]:
    """
    Upload for OCR; return (signed_url, file_id) or None on failure.

    See ``upload_file_for_ocr`` for behavior notes on preprocessing.
    """
    temp_files_to_cleanup: List[Path] = []

    try:
        if expiry_hours is None:
            expiry_hours = config.MISTRAL_SIGNED_URL_EXPIRY
        processed_file_path = file_path
        if file_path.suffix.lower().lstrip(".") in config.IMAGE_EXTENSIONS:
            logger.debug("Image file detected: %s", file_path.suffix)

            if config.MISTRAL_ENABLE_IMAGE_PREPROCESSING:
                preprocessed_path = attr("preprocess_image")(file_path)
                if preprocessed_path and preprocessed_path != file_path:
                    processed_file_path = preprocessed_path
                    temp_files_to_cleanup.append(preprocessed_path)
                    logger.info("Image preprocessed: %s", processed_file_path.name)

            if config.MISTRAL_ENABLE_IMAGE_OPTIMIZATION:
                optimized_path = attr("optimize_image")(processed_file_path)
                if optimized_path and processed_file_path != optimized_path:
                    processed_file_path = optimized_path
                    temp_files_to_cleanup.append(optimized_path)
                    logger.info("Image optimized: %s", processed_file_path.name)
        else:
            logger.debug("PDF/document file - preprocessing skipped (not applicable)")

        logger.info("Uploading file to Mistral: %s", processed_file_path.name)

        with open(processed_file_path, "rb") as f:
            response = client.files.upload(
                file={
                    "file_name": file_path.name,
                    "content": f,
                },
                purpose="ocr",
            )

        if not hasattr(response, "id"):
            logger.error("Upload response missing file ID")
            return None

        file_id = response.id
        logger.info("File uploaded successfully: %s", file_id)

        try:
            signed_url_response = client.files.get_signed_url(
                file_id=file_id,
                expiry=expiry_hours,
            )
        except Exception as e:
            logger.error("Error getting signed URL for uploaded file %s: %s", file_id, e)
            attr("_delete_ocr_file_ids")(client, [file_id])
            return None

        url = getattr(signed_url_response, "url", None)
        if url:
            logger.debug("Got signed URL for file %s", file_id)
            if not attr("_register_uploaded_file")(file_id, "ocr"):
                logger.error("Failed to persist upload registry for file %s; deleting remote upload", file_id)
                attr("_delete_ocr_file_ids")(client, [file_id])
                return None
            return url, file_id

        logger.error("Failed to get signed URL for uploaded file")
        attr("_delete_ocr_file_ids")(client, [file_id])
        return None

    except Exception as e:
        logger.error("Error uploading file: %s", e)
        return None
    finally:
        attr("_cleanup_temp_files")(temp_files_to_cleanup)


def upload_file_for_ocr(
    client: Mistral,
    file_path: Path,
    expiry_hours: Optional[int] = None,
) -> Optional[str]:
    """
    Upload file to Mistral using Files API with purpose="ocr" and get signed URL.

    For PDFs, this uploads directly. For images, preprocessing is applied first if enabled.
    Temporary files created during preprocessing are always cleaned up.

    Note: Image preprocessing (optimization/enhancement) only works on individual image files,
    NOT on PDFs. PDFs are processed as-is by Mistral OCR which handles them natively.

    Args:
        client: Mistral client instance
        file_path: Path to file to upload
        expiry_hours: Signed URL expiry in hours (default: from config)

    Returns:
        Signed URL if successful, None otherwise
    """
    pair = _upload_file_for_ocr_pair(client, file_path, expiry_hours=expiry_hours)
    return pair[0] if pair else None


def _cleanup_temp_files(temp_files: List[Path]) -> None:
    """
    Clean up temporary files created during image preprocessing.

    Args:
        temp_files: List of temporary file paths to delete
    """
    if not temp_files:
        return

    for temp_file in temp_files:
        try:
            if temp_file and temp_file.exists():
                temp_file.unlink()
                logger.debug("Deleted temporary file: %s", temp_file.name)
        except Exception as e:
            logger.warning("Could not delete temporary file %s: %s", temp_file.name, e)


