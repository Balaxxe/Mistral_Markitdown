"""
Enhanced Document Converter - Utility Functions

This module provides helper functions for logging, caching, file operations,
and metadata tracking.

Documentation references:
- MarkItDown: https://github.com/microsoft/markitdown
- Mistral OCR: https://docs.mistral.ai/capabilities/document_ai/basic_ocr/
"""

import hashlib
import itertools
import json
import logging
import re
import sys
import tempfile
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import config


class ConversionResult(NamedTuple):
    """Canonical return type for converter and orchestration functions.

    Internal converters may still return ``Tuple[bool, Optional[T], Optional[str]]``
    for backward compatibility; use :func:`to_conversion_result` at module
    boundaries (thread pool, CLI dispatch) to normalise any of those shapes
    into this one.
    """

    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None


def to_conversion_result(result: Any) -> "ConversionResult":
    """Normalise a converter's output into a :class:`ConversionResult`.

    Accepted shapes:

    * ``ConversionResult`` — returned as-is.
    * ``(success, output_path_or_payload, error)`` 3-tuple — the second slot
      is preserved as ``output_path`` when it is a :class:`pathlib.Path`,
      otherwise it is dropped and the error (if present) is preserved.
    * ``(success, error)`` 2-tuple — mapped to ``(success, None, error)``.
    * ``bool`` — mapped to ``(bool, None, None)``.

    Any other input yields ``(False, None, "unknown converter result")``
    so the caller surfaces a clear failure rather than crashing.
    """
    if isinstance(result, ConversionResult):
        return result
    if isinstance(result, tuple):
        if len(result) >= 3:
            success = bool(result[0])
            maybe_path = result[1]
            output_path = maybe_path if isinstance(maybe_path, Path) else None
            err = result[2]
            return ConversionResult(success=success, output_path=output_path, error=err)
        if len(result) == 2:
            return ConversionResult(success=bool(result[0]), output_path=None, error=result[1])
        if len(result) == 1:
            return ConversionResult(success=bool(result[0]), output_path=None, error=None)
    if isinstance(result, bool):
        return ConversionResult(success=result, output_path=None, error=None)
    return ConversionResult(success=False, output_path=None, error="unknown converter result")


__all__ = [
    "setup_logging",
    "IntelligentCache",
    "atomic_write_text",
    "atomic_write_binary",
    "format_table_to_markdown",
    "detect_month_header_row",
    "clean_table_cell",
    "is_page_artifact_row",
    "clean_table",
    "normalize_table_headers",
    "clean_consecutive_duplicates",
    "markdown_to_text",
    "save_text_output",
    "print_progress",
    "validate_file",
    "sanitize_stdin_filename_hint",
    "read_stdin_bytes_limited",
    "mistral_ocr_size_error",
    "pdf_exceeds_heavy_work_limit",
    "safe_output_stem",
    "generate_yaml_frontmatter",
    "strip_yaml_frontmatter",
    "sanitize_for_terminal",
    "ui_print",
    "ConversionResult",
    "to_conversion_result",
]

# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("document_converter")
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler if requested
    if log_file and config.SAVE_PROCESSING_LOGS:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# Default logger
logger = setup_logging()

_FRONTMATTER_RE = re.compile(r"\A---\s*\r?\n.*?\r?\n---\s*(?:\r?\n)?", re.DOTALL)

# ANSI/C0/C1 escape-sequence pattern for terminal sanitization.
_ANSI_ESCAPE_RE = re.compile(
    r"(\x1b"  # ESC
    r"(?:[@-Z\\-_]"  # Fe sequences (C1 controls)
    r"|\[[0-?]*[ -/]*[@-~]"  # CSI sequences
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC sequences
    r"|[PX^_][^\x1b]*\x1b\\)"  # DCS/SOS/PM/APC sequences
    r")|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # remaining C0 controls (keep \t \n \r)
)


def sanitize_for_terminal(text: str) -> str:
    """Strip ANSI escape sequences and non-printable C0/C1 control characters.

    Preserves tab, newline, and carriage return.  Use this before printing
    untrusted text (e.g. OCR output, LLM answers) to prevent terminal
    manipulation attacks.
    """
    return _ANSI_ESCAPE_RE.sub("", text)


def ui_print(*args, **kwargs) -> None:
    """Print user-facing CLI output.

    A thin wrapper around :func:`print` that marks output as intentional CLI
    messaging (as opposed to debug/operational logging via :data:`logger`).
    Future callers can redirect or format this output without grepping for
    bare ``print()`` calls.
    """
    print(*args, **kwargs)


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8", newline: Optional[str] = None) -> None:
    """Write *content* to *path* atomically via a temporary file and rename.

    This prevents partial / corrupt files when the process is interrupted
    mid-write.  The temporary file is created in the same directory as *path*
    so that ``Path.replace`` is guaranteed to be an atomic same-filesystem
    operation.

    Args:
        path: Destination file path.
        content: Text to write.
        encoding: File encoding (default ``"utf-8"``).
        newline: Newline translation mode passed to the underlying
            ``open()`` call.  Pass ``""`` when writing content that
            already contains correct line terminators (e.g. CSV output
            from ``csv.writer``, which emits ``\\r\\n`` per RFC 4180)
            to prevent the OS text-mode layer from double-translating
            on Windows.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline=newline,
            dir=str(path.parent),
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(path)
    except BaseException:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def atomic_write_binary(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically via a temporary file and rename.

    Binary counterpart of :func:`atomic_write_text`.  Prevents partial /
    corrupt files when the process is interrupted mid-write.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_file.write(data)
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(path)
    except BaseException:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


# ============================================================================
# Intelligent Caching System
# ============================================================================


class IntelligentCache:
    """
    Hash-based caching system for OCR results to avoid reprocessing.

    Uses file content hashing to detect changes and cache invalidation.
    Statistics are tracked per instance.
    """

    def __init__(self, cache_dir: Path = config.CACHE_DIR):
        """
        Initialize the cache system.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        # In-memory hash cache keyed by (path, mtime_ns, size) to avoid
        # re-reading file contents on every cache lookup.
        # Keep this bounded to avoid unbounded growth in long-running processes.
        self._hash_memo: "OrderedDict[Tuple[str, int, int], str]" = OrderedDict()
        self._hash_memo_max_entries = 1000

    def _get_file_hash(self, file_path: Path) -> str:
        """
        Generate SHA-256 hash of file contents.

        The cache key is content-based by design: two files with identical
        bytes produce the same hash and share cached results.  This is
        correct because OCR / conversion output is deterministic for a
        given input.  Cache *type* segregation (``_get_cache_path``)
        prevents cross-type collisions.

        Results are memoized by (path, mtime_ns, size) so repeated lookups
        for the same unchanged file avoid re-reading the entire file.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal hash string
        """
        stat = file_path.stat()
        memo_key = (str(file_path), stat.st_mtime_ns, stat.st_size)

        with self._lock:
            cached_hash = self._hash_memo.get(memo_key)
            if cached_hash is not None:
                # LRU refresh
                self._hash_memo.move_to_end(memo_key)
                return cached_hash

        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in larger chunks for better throughput on modern disks (64KB)
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()

        with self._lock:
            self._hash_memo[memo_key] = file_hash
            self._hash_memo.move_to_end(memo_key)
            while len(self._hash_memo) > self._hash_memo_max_entries:
                self._hash_memo.popitem(last=False)
        return file_hash

    def _get_cache_path(self, file_hash: str, cache_type: str = "ocr") -> Path:
        """
        Get cache file path for a given hash and type.

        Args:
            file_hash: SHA-256 hash of file contents
            cache_type: Type of cache (e.g., "ocr", "mistral_ocr", "table")

        Returns:
            Path to cache file, segregated by type to avoid collisions
        """
        # Include cache_type in filename to prevent collisions between different cache types
        return self.cache_dir / f"{file_hash}_{cache_type}.json"

    def get_entry(self, file_path: Path, cache_type: str = "ocr") -> Optional[Dict[str, Any]]:  # noqa: C901
        """
        Retrieve the full validated cache entry (``data``, ``metadata``, etc.).

        Returns:
            Parsed cache JSON dict, or ``None`` if missing/invalid/expired.
        """
        cache_path: Optional[Path] = None
        try:
            if not file_path.exists():
                logger.debug("Cache lookup skipped (file missing): %s", file_path)
                with self._lock:
                    self.misses += 1
                return None

            file_hash = self._get_file_hash(file_path)
            cache_path = self._get_cache_path(file_hash, cache_type)

            with self._lock:
                if not cache_path.exists():
                    self.misses += 1
                    return None

                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                except FileNotFoundError:
                    self.misses += 1
                    return None

                if not isinstance(cache_data, dict):
                    raise ValueError("cache entry is not a dict")
                for required_key in ("timestamp", "type", "data"):
                    if required_key not in cache_data:
                        raise ValueError(f"cache entry missing required key: {required_key}")

                cached_time = datetime.fromisoformat(cache_data.get("timestamp", ""))
                if cached_time.tzinfo is None:
                    cached_time = cached_time.replace(tzinfo=timezone.utc)
                max_age = timedelta(hours=config.CACHE_DURATION_HOURS)

                if datetime.now(timezone.utc) - cached_time > max_age:
                    logger.debug("Cache expired for %s", file_path.name)
                    cache_path.unlink(missing_ok=True)
                    self.misses += 1
                    return None

                if cache_data.get("type") != cache_type:
                    logger.debug(
                        "Cache type mismatch (expected %s, got %s)",
                        cache_type,
                        cache_data.get("type"),
                    )
                    self.misses += 1
                    return None

                self.hits += 1
            logger.info("Cache hit for %s", file_path.name)
            return cache_data

        except FileNotFoundError:
            logger.debug("Cache lookup failed (file not found): %s", file_path)
            with self._lock:
                self.misses += 1
            return None
        except (json.JSONDecodeError, ValueError):
            if cache_path is not None:
                logger.warning("Corrupt or tampered cache file, removing: %s", cache_path)
                try:
                    cache_path.unlink(missing_ok=True)
                except OSError:
                    pass
            with self._lock:
                self.misses += 1
            return None
        except Exception as e:
            logger.warning("Error reading cache for %s: %s", file_path.name, e)
            with self._lock:
                self.misses += 1
            return None

    def get(self, file_path: Path, cache_type: str = "ocr") -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result for a file (payload only).

        Args:
            file_path: Path to the file
            cache_type: Type of cache (ocr, table, etc.)

        Returns:
            Cached data if valid, None otherwise
        """
        entry = self.get_entry(file_path, cache_type)
        if entry is None:
            return None
        return entry.get("data")  # type: ignore[no-any-return]

    def set(
        self,
        file_path: Path,
        data: Dict[str, Any],
        cache_type: str = "ocr",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store data in cache.

        Args:
            file_path: Path to the file
            data: Data to cache
            cache_type: Type of cache
            metadata: Optional metadata to store
        """
        try:
            if not file_path.exists():
                logger.warning("Cannot cache missing file: %s", file_path)
                return

            file_hash = self._get_file_hash(file_path)
            # Use type-segregated cache path to avoid collisions
            cache_path = self._get_cache_path(file_hash, cache_type)

            cache_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "type": cache_type,
                "data": data,
                "metadata": metadata or {},
            }

            # Atomic write to avoid partial/corrupt cache files under concurrency.
            # os.replace/Path.replace is atomic on the same filesystem.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.cache_dir),
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                json.dump(cache_entry, tmp_file, indent=2, ensure_ascii=False)
                temp_path = Path(tmp_file.name)

            temp_path.replace(cache_path)

            # Restrict permissions on cache files (may contain sensitive OCR text)
            if sys.platform != "win32":  # pragma: no cover
                import os

                try:
                    os.chmod(cache_path, 0o600)
                except OSError:
                    pass

            logger.debug("Cached result for %s", file_path.name)

        except Exception as e:
            logger.warning("Error writing cache for %s: %s", file_path, e)

    def clear_old_entries(self) -> int:
        """
        Remove cache entries older than CACHE_DURATION_HOURS.

        Returns:
            Number of entries removed
        """
        removed = 0
        max_age = timedelta(hours=config.CACHE_DURATION_HOURS)

        with self._lock:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)

                    cached_time = datetime.fromisoformat(cache_data.get("timestamp", ""))
                    if cached_time.tzinfo is None:
                        cached_time = cached_time.replace(tzinfo=timezone.utc)

                    if datetime.now(timezone.utc) - cached_time > max_age:
                        cache_file.unlink()
                        removed += 1

                except Exception as e:
                    logger.debug("Error processing cache file %s: %s", cache_file.name, e)

        return removed

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        with self._lock:
            hits = self.hits
            misses = self.misses

        total_requests = hits + misses

        return {
            "total_entries": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_hits": hits,
            "cache_misses": misses,
            "hit_rate": (hits / total_requests * 100) if total_requests > 0 else 0,
        }


# Global cache instance
cache = IntelligentCache()

# ============================================================================
# Markdown Table Formatting
# ============================================================================


def format_table_to_markdown(data: List[List[str]], headers: Optional[List[str]] = None) -> str:
    """
    Convert table data to Markdown format.

    Args:
        data: List of rows (each row is a list of cells)
        headers: Optional list of column headers

    Returns:
        Markdown-formatted table string
    """
    if not data:
        return ""

    # Use first row as headers if not provided
    if headers is None and data:
        headers = data[0]
        data = data[1:]

    if not headers:
        return ""

    # Build markdown table
    lines = []

    # Header row
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")

    # Separator row
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    # Data rows
    for row in data:
        # Pad row to match header length
        padded_row = list(row) + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(str(cell) for cell in padded_row[: len(headers)]) + " |")

    return "\n".join(lines)


# ============================================================================
# Table Header Normalization & Cleanup
# ============================================================================

# Common month headers found in financial documents
MONTH_HEADERS = [
    "Beginning",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
    "Current",
]


def detect_month_header_row(table: List[List[str]]) -> Optional[int]:
    """
    Detect which row contains month headers (for financial documents).

    Args:
        table: List of table rows

    Returns:
        Row index containing month headers, or None if not found
    """
    if not table:
        return None

    for row_idx, row in enumerate(table):
        # Join all cells in the row
        row_text = " ".join(str(cell or "") for cell in row).strip()

        # Check if this row contains multiple month names
        month_count = sum(1 for month in MONTH_HEADERS if month in row_text)

        # If we find at least 3 month names (or Beginning/Current), likely a header
        if month_count >= 3:
            return row_idx

    return None


def clean_table_cell(cell: str) -> str:
    """
    Clean individual table cell.

    - Removes extra whitespace and newlines within cells
    - Normalizes "Acct\nAccount Title" to "Acct Account Title"
    - Strips leading/trailing whitespace

    Args:
        cell: Cell content string

    Returns:
        Cleaned cell content
    """
    if not cell:
        return ""

    # Replace newlines with spaces
    cell = cell.replace("\n", " ").replace("\r", " ")

    # Collapse multiple spaces
    cell = " ".join(cell.split())

    return cell.strip()


def is_page_artifact_row(row: List[str]) -> bool:
    """
    Detect if a row is a page artifact (page numbers, repeated headers, etc.).

    Common artifacts:
    - "Page 1", "Page 2", etc.
    - Date stamps like "December 31, 2010"
    - Repeated document headers

    Args:
        row: Table row

    Returns:
        True if row appears to be a page artifact
    """
    if not row:
        return False

    # Join all cells
    row_text = " ".join(str(cell or "") for cell in row).strip()

    # Check for page number artifacts (e.g., "Page 1", "Page 42", etc.)
    if re.match(r"^Page\s+\d+$", row_text):
        return True

    # Check if the row is just a date (e.g., "December 31, 2010")
    # Pattern: single cell or cells that form a date
    if len(row_text) < 30 and any(month in row_text for month in MONTH_HEADERS):
        # Check if it looks like "Month DD, YYYY"
        date_pattern = r"^[A-Za-z]+\s+\d{1,2},?\s+\d{4}$"
        if re.match(date_pattern, row_text.replace(",", "")):
            return True

    # Empty or near-empty rows
    if len(row_text.strip()) < 3:
        return True

    return False


def clean_table(table: List[List[str]]) -> List[List[str]]:
    """
    Clean a table by:
    - Removing newlines in cells
    - Removing page artifact rows
    - Normalizing cell content

    Args:
        table: List of table rows

    Returns:
        Cleaned table
    """
    if not table:
        return []

    cleaned = []

    for row in table:
        # Skip page artifact rows
        if is_page_artifact_row(row):
            continue

        # Clean each cell
        cleaned_row = [clean_table_cell(cell) for cell in row]

        # Only add non-empty rows
        if any(cell.strip() for cell in cleaned_row):
            cleaned.append(cleaned_row)

    return cleaned


def normalize_table_headers(
    table: List[List[str]],
) -> Tuple[List[str], List[List[str]]]:
    """
    Normalize table headers by detecting month headers and cleaning cells.

    Args:
        table: List of table rows

    Returns:
        Tuple of (headers, data_rows)
    """
    if not table or len(table) < 1:
        return [], []

    # First, clean the table
    table = clean_table(table)

    if not table:
        return [], []

    # Detect month header row
    header_idx = detect_month_header_row(table)

    if header_idx is not None:
        # Use detected month header
        headers = table[header_idx]
        data_rows = table[:header_idx] + table[header_idx + 1 :]
    else:
        # Fall back to first row as header
        headers = table[0]
        data_rows = table[1:]

    return headers, data_rows


# ============================================================================
# Text Export Functions
# ============================================================================


def clean_consecutive_duplicates(text: str) -> str:
    """
    Clean text by collapsing consecutive identical lines into a single occurrence.

    This helps remove artifacts from OCR output where headers, footers, or
    other elements are mistakenly recognized multiple times in sequence.
    It compares lines exactly as they appear (including whitespace and Markdown).

    Args:
        text: The input text (e.g., OCR output for a page).
    Returns:
        Cleaned text with consecutive duplicates collapsed.
    """
    if not text:
        return ""

    # Use splitlines() to handle different line endings correctly
    lines = text.splitlines()

    # Use itertools.groupby to group consecutive identical lines.
    # The 'key' represents the unique line content for each group.
    # By taking only the key, we effectively collapse the group into one line.
    cleaned_lines = [key for key, group in itertools.groupby(lines)]

    # Rejoin the lines
    return "\n".join(cleaned_lines)


def markdown_to_text(markdown_content: str) -> str:
    """
    Convert Markdown to plain text by removing formatting.

    Args:
        markdown_content: Markdown string

    Returns:
        Plain text string
    """
    text = strip_yaml_frontmatter(markdown_content)

    # Remove images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Remove links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Remove headers #
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)

    # Remove bold/italic
    text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^\*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)

    # Remove code blocks
    text = re.sub(r"```[^\n]*\n.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Clean up multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def save_text_output(markdown_path: Path, markdown_content: str) -> Optional[Path]:
    """
    Save plain text version of markdown content.

    Args:
        markdown_path: Path to markdown file
        markdown_content: Markdown content

    Returns:
        Path to text file if successful, None otherwise
    """
    if not config.GENERATE_TXT_OUTPUT:
        return None

    try:
        text_path = config.OUTPUT_TXT_DIR / f"{markdown_path.stem}.txt"
        text_content = markdown_to_text(markdown_content)
        atomic_write_text(text_path, text_content)
        logger.debug("Saved text output: %s", text_path.name)
        return text_path

    except Exception as e:
        logger.error("Error saving text output: %s", e)
        return None


# ============================================================================
# Progress Indicators
# ============================================================================


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    """
    Print progress bar to console.

    Args:
        current: Current item number
        total: Total number of items
        prefix: Prefix text
    """
    if not config.VERBOSE_PROGRESS:
        return

    if total <= 0:
        print(
            f"\r{prefix}: [----------------------------------------] 0.0% (0/0)",
            end="",
            flush=True,
        )
        return

    current = max(0, min(current, total))
    percent = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = "=" * filled + "-" * (bar_length - filled)

    print(f"\r{prefix}: [{bar}] {percent:.1f}% ({current}/{total})", end="", flush=True)

    if current == total:
        print()  # New line when complete


# ============================================================================
# File Validation
# ============================================================================


def sanitize_stdin_filename_hint(hint: str) -> Tuple[bool, str, Optional[str]]:
    """
    Normalize ``--stdin-filename`` to a single path segment (basename only).

    Rejects empty hints and embedded path separators so output paths cannot
    escape ``OUTPUT_MD_DIR``.
    """
    stripped = (hint or "").strip()
    if not stripped:
        return False, "", "Empty --stdin-filename"
    normalized = stripped.replace("\\", "/")
    base = normalized.rsplit("/", 1)[-1] if "/" in normalized else normalized
    if not base or base in (".", ".."):
        return False, "", "Invalid --stdin-filename"
    if "/" in base or "\\" in base:
        return False, "", "Invalid --stdin-filename"
    return True, base, None


def read_stdin_bytes_limited(max_bytes: int) -> Tuple[bool, bytes, Optional[str]]:
    """
    Read stdin until EOF, failing if more than *max_bytes* are received.

    Reads in chunks so oversized input is not fully buffered before rejection.
    """
    if max_bytes < 0:
        return False, b"", "Invalid stdin size limit"
    chunks: List[bytes] = []
    total = 0
    while True:
        to_read = 1024 * 1024
        if max_bytes > 0:
            to_read = min(to_read, max_bytes - total + 1)
        chunk = sys.stdin.buffer.read(to_read)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            mb = max_bytes / (1024 * 1024)
            return (
                False,
                b"",
                f"Stdin exceeds MarkItDown size limit ({mb:.1f} MB).",
            )
        chunks.append(chunk)
    return True, b"".join(chunks), None


def _resolved_path_under_input_dir(file_path: Path) -> Tuple[bool, Optional[str]]:
    """If strict resolution is enabled, ensure *file_path* resolves under ``INPUT_DIR``."""
    if not config.STRICT_INPUT_PATH_RESOLUTION:
        return True, None
    try:
        resolved = file_path.resolve()
        base = config.INPUT_DIR.resolve()
        if not resolved.is_relative_to(base):
            return (
                False,
                f"Resolved path must be under input directory: {file_path.name}",
            )
    except (OSError, ValueError) as e:
        return False, f"Cannot resolve file path: {e}"
    return True, None


def validate_file(file_path: Path, mode: Optional[str] = None) -> Tuple[bool, Optional[str]]:  # noqa: C901
    """
    Validate if a file can be processed.

    Args:
        file_path: Path to file
        mode: Conversion mode to validate against. When ``"markitdown"``,
              only MarkItDown-supported extensions are accepted. When
              ``"mistral_ocr"``, ``"qna"``, or ``"batch_ocr"``, only
              Mistral-OCR-supported extensions are accepted. ``"pdf_to_images"``
              accepts PDF only. ``None`` or ``"smart"`` accepts the union of both
              extension sets; the **size cap** for ``smart`` uses the MarkItDown
              limit alone for types that always route to MarkItDown, and the
              larger of MarkItDown and Mistral OCR limits when the extension may
              use cloud OCR (e.g. PDF, images, docx, pptx).

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    if not file_path.is_file():
        return False, f"Not a file: {file_path}"

    ok_path, path_err = _resolved_path_under_input_dir(file_path)
    if not ok_path:
        return False, path_err

    if file_path.stat().st_size == 0:
        return False, f"File is empty: {file_path.name}"

    # Check file extension against the correct set for the requested mode
    ext = file_path.suffix.lower().lstrip(".")

    if mode == "markitdown":
        supported = config.MARKITDOWN_SUPPORTED
    elif mode in ("mistral_ocr", "qna", "batch_ocr"):
        supported = config.MISTRAL_OCR_SUPPORTED
    elif mode == "pdf_to_images":
        supported = config.PDF_EXTENSIONS
    else:
        # smart / None — union (routing picks engine per file)
        supported = config.MARKITDOWN_SUPPORTED | config.MISTRAL_OCR_SUPPORTED

    if ext not in supported:
        return False, f"Unsupported file type for {mode or 'this'} mode: .{ext}"

    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
    except OSError as e:
        return False, f"Cannot read file: {e}"

    max_mb: Optional[float] = None
    if mode == "markitdown":
        max_mb = float(config.MARKITDOWN_MAX_FILE_SIZE_MB)
    elif mode in ("mistral_ocr", "batch_ocr"):
        max_mb = float(config.MISTRAL_OCR_MAX_FILE_SIZE_MB)
    elif mode == "qna":
        max_mb = float(config.MISTRAL_QNA_MAX_FILE_SIZE_MB)
    elif mode == "pdf_to_images":
        if ext == "pdf":
            max_mb = float(config.pdf_heavy_work_max_file_size_mb())
    else:
        # smart / None — types that may route to Mistral OCR use the larger cap
        if ext in config.MISTRAL_OCR_SUPPORTED:
            max_mb = float(
                max(
                    config.MARKITDOWN_MAX_FILE_SIZE_MB,
                    config.MISTRAL_OCR_MAX_FILE_SIZE_MB,
                )
            )
        else:
            max_mb = float(config.MARKITDOWN_MAX_FILE_SIZE_MB)

    if max_mb is not None and size_mb > max_mb:
        return False, (f"File too large ({size_mb:.1f} MB) for {mode or 'this'} mode (limit {int(max_mb)} MB).")

    return True, None


def mistral_ocr_size_error(file_size_mb: float) -> Optional[str]:
    """Return a user-facing message if *file_size_mb* exceeds the Mistral OCR cap.

    Centralizes the size check used both by up-front validation (see
    :func:`validate_file` with ``mode="mistral_ocr"``) and by the runtime
    OCR path (see ``mistral_converter._validate_file_for_ocr``) so the rule
    and its wording stay in one place.

    Returns ``None`` when the file is within limits.
    """
    if file_size_mb > config.MISTRAL_OCR_MAX_FILE_SIZE_MB:
        return (
            f"File too large for Mistral OCR ({file_size_mb:.1f} MB). "
            f"Maximum allowed: {config.MISTRAL_OCR_MAX_FILE_SIZE_MB} MB"
        )
    return None


def pdf_exceeds_heavy_work_limit(file_path: Path) -> Tuple[bool, Optional[str]]:
    """Return (True, error_message) if *file_path* is a PDF over the heavy-work size cap."""
    if file_path.suffix.lower() != ".pdf":
        return False, None
    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
    except OSError as e:
        return True, f"Cannot read file: {e}"
    cap = config.pdf_heavy_work_max_file_size_mb()
    if size_mb > cap:
        return True, (
            f"PDF too large ({size_mb:.1f} MB) for table extraction / PDF-to-images "
            f"(limit {cap} MB; max of MARKITDOWN_MAX_FILE_SIZE_MB and MISTRAL_OCR_MAX_FILE_SIZE_MB)."
        )
    return False, None


def safe_output_stem(file_path: Path) -> str:
    """Return a unique output stem for a file, adding a short hash if needed to
    avoid collisions when multiple files share the same name (e.g. from different
    directories passed programmatically).

    For files in the standard input directory, this checks for same-stem collisions
    (e.g. report.pdf and report.docx) and appends ``_<ext>`` when needed.
    For files elsewhere, it appends ``_<6-char hash>`` derived from the full path.
    """
    stem = file_path.stem
    ext = file_path.suffix.lower().lstrip(".")
    try:
        resolved = file_path.resolve()
        input_dir = config.INPUT_DIR.resolve()
        if resolved.parent == input_dir:
            collisions = [
                p for p in input_dir.glob(f"{stem}.*") if p.is_file() and p.suffix.lower() != file_path.suffix.lower()
            ]
            if collisions:
                return f"{stem}_{ext}"
        else:
            path_hash = hashlib.sha256(str(resolved).encode()).hexdigest()[:6]
            return f"{stem}_{path_hash}"
    except (OSError, ValueError):
        pass
    return stem


# ============================================================================
# YAML Frontmatter Generation
# ============================================================================


def generate_yaml_frontmatter(
    title: str,
    file_name: str,
    conversion_method: str,
    additional_fields: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate YAML frontmatter for markdown files.

    Args:
        title: Document title
        file_name: Original file name
        conversion_method: Method used for conversion
        additional_fields: Optional additional metadata fields

    Returns:
        YAML frontmatter string
    """
    if not config.INCLUDE_METADATA:
        return ""

    metadata: Dict[str, Any] = {
        "title": title,
        "source_file": file_name,
        "conversion_method": conversion_method,
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "converter_version": config.VERSION,
    }

    if additional_fields:
        metadata.update(additional_fields)

    # Build YAML
    # Use json.dumps for strings so quotes/newlines are escaped safely.
    lines = ["---"]
    for key, value in metadata.items():
        if isinstance(value, str):
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---\n")

    return "\n".join(lines)


def strip_yaml_frontmatter(content: str) -> str:
    """
    Remove YAML frontmatter from markdown content.

    Frontmatter is detected as content between two '---' markers at the start.

    Args:
        content: Markdown content that may contain frontmatter

    Returns:
        Content without frontmatter
    """
    return _FRONTMATTER_RE.sub("", content, count=1).strip()
