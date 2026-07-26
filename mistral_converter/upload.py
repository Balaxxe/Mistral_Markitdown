"""Mistral Files API upload helpers and local upload registry."""

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Set, Tuple

import config
import utils

from .facade import attr
from .sdk_shims import Mistral

try:
    import fcntl as _fcntl_import
except ImportError:  # pragma: no cover - Windows
    _fcntl_import = None

try:
    import msvcrt as _msvcrt_import
except ImportError:  # pragma: no cover - POSIX
    _msvcrt_import = None

# Typed as Any: typeshed defines each module's attributes only on its own
# platform, and exactly one of the two is None at runtime.
fcntl: Any = _fcntl_import
msvcrt: Any = _msvcrt_import

logger = utils.logger
_UPLOAD_REGISTRY_LOCK = threading.Lock()
_UPLOAD_REGISTRY_FILENAME = "mistral_upload_registry.json"
_UPLOAD_REGISTRY_LOCK_FILENAME = "mistral_upload_registry.lock"
_UPLOAD_REGISTRY_LOCK_TIMEOUT = 30.0
_UPLOAD_REGISTRY_LOCK_RETRY_INTERVAL = 0.1


def _upload_registry_path() -> Path:
    return config.CACHE_DIR / _UPLOAD_REGISTRY_FILENAME


def _upload_registry_lock_path() -> Path:
    return config.CACHE_DIR / _UPLOAD_REGISTRY_LOCK_FILENAME


def _registry_file_locking_available() -> bool:
    """Return whether this platform can take a cross-process registry lock."""
    return fcntl is not None or msvcrt is not None


def _acquire_registry_file_lock(fd: int) -> bool:
    """Take an exclusive lock on *fd*; return False when it could not be taken."""
    deadline = time.monotonic() + _UPLOAD_REGISTRY_LOCK_TIMEOUT
    if fcntl is not None:
        # Non-blocking retries against the deadline: a blocking flock would pin
        # the process-wide thread lock behind a stalled peer for as long as it
        # holds the file lock.
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError as e:
                if time.monotonic() >= deadline:
                    logger.debug("Could not lock the upload registry: %s", e)
                    return False
                time.sleep(_UPLOAD_REGISTRY_LOCK_RETRY_INTERVAL)
    while True:
        try:
            # LK_LOCK retries internally for ~10s before raising.
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            return True
        except OSError as e:
            if time.monotonic() >= deadline:
                logger.debug("Could not lock the upload registry: %s", e)
                return False


def _release_registry_file_lock(fd: int) -> None:
    """Release the exclusive lock held on *fd*."""
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        else:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError as e:
        logger.debug("Could not release the upload registry lock: %s", e)


@contextmanager
def _registry_lock() -> Iterator[None]:
    """Hold an exclusive cross-process lock over a registry read-modify-write.

    Enter this only while ``_UPLOAD_REGISTRY_LOCK`` is held: the thread lock must
    stay outermost so re-entry cannot deadlock against mandatory Windows locks.
    Degrades to a no-op when the platform offers no locking module.
    """
    if not _registry_file_locking_available():
        logger.debug("No file locking available; upload registry updates are not cross-process safe")
        yield
        return

    try:
        config.ensure_private_dir(config.CACHE_DIR)
        fd = os.open(str(_upload_registry_lock_path()), os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as e:
        logger.debug("Could not open the upload registry lock file: %s", e)
        yield
        return

    try:
        locked = _acquire_registry_file_lock(fd)
        try:
            yield
        finally:
            if locked:
                _release_registry_file_lock(fd)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


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
        config.ensure_private_dir(config.CACHE_DIR)
        utils.atomic_write_text(path, json.dumps(entries, indent=2))
        return True
    except OSError as e:
        logger.warning("Could not save upload registry: %s", e)
        return False


def _upload_registry_needs_rewrite(entries: List[Dict[str, Any]]) -> bool:
    """Return whether persisting *entries* would change the registry on disk."""
    path = _upload_registry_path()
    try:
        existing = path.read_text(encoding="utf-8")
    except OSError:
        # Missing or unreadable: a rewrite is needed unless there is nothing to write.
        return bool(entries)
    return existing != json.dumps(entries, indent=2)


def _register_uploaded_file(file_id: str, purpose: str) -> bool:
    """Record an uploaded file id in the local registry.

    Returns False when the id is empty or the registry cannot be persisted, so
    callers can delete the remote upload instead of orphaning it under
    ``CLEANUP_UPLOAD_SCOPE=registry``.
    """
    if not file_id:
        return False
    now = datetime.now(timezone.utc).isoformat()
    with _UPLOAD_REGISTRY_LOCK, _registry_lock():
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
    with _UPLOAD_REGISTRY_LOCK, _registry_lock():
        entries = _load_upload_registry()
        new_entries = [e for e in entries if e.get("id") != file_id]
        if len(new_entries) != len(entries):
            _save_upload_registry(new_entries)


def _parse_registry_created_at(value: Any) -> Optional[datetime]:
    """Parse a registry created_at value into an aware UTC datetime."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
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


def cleanup_uploaded_files(
    client: Mistral,
    days_old: Optional[int] = None,
    *,
    raise_on_error: bool = False,
) -> int:
    """
    Clean up old files uploaded to Mistral Files API.

    When ``CLEANUP_UPLOAD_SCOPE`` is ``"registry"`` (default), only file IDs
    present in the local upload registry and older than *days_old* are deleted.
    When ``"all"``, performs account-wide age-based cleanup for ``ocr`` and
    ``batch`` purposes and prunes matching registry entries.

    Args:
        client: Mistral client instance
        days_old: Delete files older than N days (default: from config)
        raise_on_error: Raise after any listing/deletion/registry failure instead
            of preserving the historical best-effort behavior.

    Returns:
        Number of files deleted
    """
    retention_days = config.UPLOAD_RETENTION_DAYS if days_old is None else days_old

    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        scope = getattr(config, "CLEANUP_UPLOAD_SCOPE", "registry")

        if scope == "registry":
            deleted = _cleanup_registry_scoped(client, cutoff_date, raise_on_error=raise_on_error)
        elif scope == "all":
            deleted_ids: List[str] = []
            try:
                deleted = _cleanup_files_by_purpose(
                    client,
                    "ocr",
                    cutoff_date,
                    deleted_ids,
                    raise_on_error=raise_on_error,
                )
                deleted += _cleanup_files_by_purpose(
                    client,
                    "batch",
                    cutoff_date,
                    deleted_ids,
                    raise_on_error=raise_on_error,
                )
            finally:
                # Do not leave successfully-deleted IDs registered just
                # because a later item or purpose failed.
                for fid in deleted_ids:
                    _unregister_uploaded_file(fid)
        else:
            message = f"Invalid CLEANUP_UPLOAD_SCOPE={scope!r}; expected 'registry' or 'all'"
            logger.warning("Refusing upload cleanup: %s", message)
            if raise_on_error:
                raise ValueError(message)
            return 0

        if deleted > 0:
            logger.info(
                "Cleaned up %s old uploaded files (older than %s days, scope=%s)",
                deleted,
                retention_days,
                scope,
            )

        return deleted

    except Exception as e:
        logger.warning("Error cleaning up uploaded files: %s", e)
        if raise_on_error:
            raise
        return 0


def _cleanup_registry_scoped(
    client: Mistral,
    cutoff_date: datetime,
    *,
    raise_on_error: bool = False,
) -> int:
    """Delete only registry-tracked uploads older than *cutoff_date*."""
    deleted = 0
    failures: List[Exception] = []
    # Read a snapshot under the lock, then release it: remote deletes must not
    # block other processes, and entries registered meanwhile must survive.
    with _UPLOAD_REGISTRY_LOCK, _registry_lock():
        entries = _load_upload_registry()

    deleted_ids: Set[str] = set()
    for entry in entries:
        file_id = entry.get("id")
        if not file_id:
            continue
        file_created = _parse_registry_created_at(entry.get("created_at"))
        # Missing/unparsable timestamps must not permanently shield uploads.
        if file_created is not None and file_created >= cutoff_date:
            continue
        try:
            client.files.delete(file_id=file_id)
            deleted += 1
            deleted_ids.add(file_id)
            logger.debug(
                "Deleted registry-tracked %s file: %s (created %s)",
                entry.get("purpose", "unknown"),
                file_id,
                file_created,
            )
        except Exception as e:
            logger.debug("Error deleting registry file %s: %s", file_id, e)
            failures.append(e)

    registry_saved = True
    with _UPLOAD_REGISTRY_LOCK, _registry_lock():
        # ``_load_upload_registry`` already drops malformed (id-less) rows, so
        # rewriting whenever the result differs from the file on disk prunes
        # both deleted ids and corrupt rows, and writes nothing when the
        # registry is already exactly what it should be.
        current = _load_upload_registry()
        remaining: List[Dict[str, Any]] = [e for e in current if e.get("id") and e.get("id") not in deleted_ids]
        if _upload_registry_needs_rewrite(remaining):
            registry_saved = _save_upload_registry(remaining)
    if raise_on_error and not registry_saved:
        raise OSError("Could not persist the upload registry after cleanup")
    if raise_on_error and failures:
        raise RuntimeError(f"Failed to delete {len(failures)} registry-tracked upload(s)") from failures[0]
    return deleted


def _cleanup_files_by_purpose(
    client: Mistral,
    purpose: Literal["ocr", "batch"],
    cutoff_date: datetime,
    deleted_ids: List[str],
    *,
    raise_on_error: bool = False,
) -> int:
    """Delete files older than cutoff_date for a given purpose (account-wide)."""
    deleted = 0
    page = 0
    page_size = 100

    while True:
        try:
            files_response = client.files.list(purpose=purpose, page=page, page_size=page_size)
            raw_files: Any = files_response.data if hasattr(files_response, "data") else files_response
            files_list: List[Any] = list(raw_files) if raw_files is not None else []
        except Exception as e:
            logger.debug(
                "Error listing %s files for cleanup (page %s): %s",
                purpose,
                page,
                e,
            )
            if raise_on_error:
                raise RuntimeError(f"Could not list {purpose} uploads for cleanup") from e
            break

        if not files_list:
            break

        for file in files_list:
            try:
                if not hasattr(file, "created_at"):
                    continue

                file_created = _parse_registry_created_at(file.created_at)
                if file_created is None:
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
                if raise_on_error:
                    raise RuntimeError(f"Could not delete {purpose} upload {getattr(file, 'id', '')}") from e
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
    mode: str = "mistral_ocr",
) -> Optional[Tuple[str, str]]:
    """
    Upload for OCR; return (signed_url, file_id) or None on failure.

    The path is re-validated with :func:`utils.validate_file` before it is
    opened, so callers that bypass the entry-point checks cannot stream an
    arbitrary path to the API. Pass *mode* to validate under the caller's own
    limits (for example ``"qna"``), so a file the caller accepts is not refused
    with an OCR-limit message.

    See ``upload_file_for_ocr`` for behavior notes on preprocessing.
    """
    is_valid, validation_message = utils.validate_file(file_path, mode=mode)
    if not is_valid:
        logger.error("Refusing to upload %s: %s", file_path.name, validation_message)
        return None

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
    mode: str = "mistral_ocr",
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
        mode: Validation mode for :func:`utils.validate_file` (e.g. ``"qna"``)

    Returns:
        Signed URL if successful, None otherwise
    """
    pair = _upload_file_for_ocr_pair(client, file_path, expiry_hours=expiry_hours, mode=mode)
    return pair[0] if pair else None


def _cleanup_temp_files(temp_files: List[Path]) -> None:
    """
    Clean up temporary files created during image preprocessing.

    Args:
        temp_files: List of temporary file paths to delete
    """
    if not temp_files:
        return

    try:
        cache_dir = config.CACHE_DIR.resolve()
    except OSError:
        return

    for temp_file in temp_files:
        try:
            if not temp_file:
                continue
            resolved = temp_file.resolve()
            if cache_dir not in resolved.parents or not resolved.name.startswith("mistral_"):
                logger.warning("Refusing to clean up non-owned temporary file: %s", temp_file)
                continue
            if resolved.exists():
                resolved.unlink()
                logger.debug("Deleted temporary file: %s", resolved.name)
        except Exception as e:
            logger.warning("Could not delete temporary file %s: %s", temp_file.name, e)
