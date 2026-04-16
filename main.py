"""
Enhanced Document Converter - Main Application

Interactive CLI for document conversion with 8 modes:
1. Convert (Smart)       - Auto-picks best engine per file type
2. Convert (MarkItDown)  - Force local conversion (no API)
3. Convert (Mistral OCR) - Force cloud OCR
4. PDF to Images         - Page rendering
5. Document QnA          - Query documents in natural language
6. Batch OCR             - Reduced-cost batch jobs
7. System Status         - Cache and performance metrics
8. Maintenance           - Clear cache, clean up old uploads

Usage:
    python3 main.py                      # Interactive menu
    python3 main.py --mode smart         # Smart auto-routing
    python3 main.py --mode markitdown    # Force MarkItDown
    python3 main.py --test               # Test mode

Documentation references:
- MarkItDown: https://github.com/microsoft/markitdown
- Mistral OCR: https://docs.mistral.ai/capabilities/document_ai/basic_ocr/
"""

import argparse
import io
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress harmless dependency-version warning from requests (often
# RequestsDependencyWarning from ``requests``, mentioning urllib3/chardet/
# charset_normalizer) when transitives are newer than requests pins expect.
warnings.filterwarnings(
    "ignore",
    message=r".*urllib3.*chardet.*charset_normalizer.*",
)

import config
import local_converter
import mistral_converter
import utils
from modes.batch import mode_batch_ocr
from modes.qna import mode_document_qna
from modes.system import mode_maintenance, mode_system_status

logger = utils.logger


# ============================================================================
# Helper: file listing and validation
# ============================================================================


def _list_input_files() -> List[Path]:
    """Return sorted list of files in the input directory (includes extensionless files).

    Dotfiles (e.g. ``.gitkeep``, ``.DS_Store``) are silently excluded to avoid
    spurious "File is empty" warnings during non-interactive runs.
    """
    return sorted(
        (f for f in config.INPUT_DIR.iterdir() if f.is_file() and not f.name.startswith(".")),
        key=lambda p: p.name.lower(),
    )


def _filter_valid_files(files: List[Path], mode: Optional[str] = None) -> List[Path]:
    """Return only valid files, logging warnings for invalid ones.

    Args:
        files: List of file paths to validate.
        mode: Conversion mode (``"markitdown"``, ``"mistral_ocr"``, etc.).
              Passed through to ``utils.validate_file`` for per-mode extension checks.
    """
    valid_files = []
    for file_path in files:
        is_valid, error = utils.validate_file(file_path, mode=mode)
        if is_valid:
            valid_files.append(file_path)
        else:
            logger.warning(error)
    return valid_files


# ============================================================================
# Helper: concurrent file processing
# ============================================================================


def _unpack_result(result) -> Tuple[bool, Optional[str]]:
    """Extract (success, error) from a converter result.

    Delegates to :func:`utils.to_conversion_result` so every shape accepted
    at the thread-pool boundary (``ConversionResult``, 3-tuple, 2-tuple,
    bool) is normalised in one place.
    """
    normalised = utils.to_conversion_result(result)
    return normalised.success, normalised.error


def _process_files_concurrently(
    file_paths: List[Path],
    process_fn,
    label: str = "Processing files",
) -> Tuple[int, int]:
    """Run *process_fn* on each file, using threads when there are multiple files.

    Args:
        file_paths: Files to process.
        process_fn: Callable(Path) -> ConversionResult | Tuple[bool, ...].
        label: Progress bar label.

    Returns:
        (successful_count, failed_count)
    """
    successful = 0
    failed = 0

    total = len(file_paths)

    if total == 1:
        try:
            result = process_fn(file_paths[0])
            ok, err = _unpack_result(result)
            if ok:
                successful += 1
            else:
                failed += 1
                logger.error("Failed: %s - %s", file_paths[0].name, err)
        except Exception as e:
            failed += 1
            logger.error("Error processing %s: %s", file_paths[0].name, e)
    else:
        with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_FILES) as executor:
            futures = {executor.submit(process_fn, fp): fp for fp in file_paths}

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    ok, err = _unpack_result(result)
                    if ok:
                        successful += 1
                    else:
                        failed += 1
                        logger.error("Failed: %s - %s", file_path.name, err)
                except Exception as e:
                    failed += 1
                    logger.error("Error processing %s: %s", file_path.name, e)

    utils.ui_print(f"\n{label}: {successful + failed}/{total} complete")

    return successful, failed


# ============================================================================
# Mode 1: Convert (Smart) -- auto-routes by file type
# ============================================================================


def _content_prefers_mistral_ocr(file_path: Path) -> bool:
    """Content-based routing only (ignores per-engine size limits).

    - Images -> OCR
    - Scanned / image-only PDFs -> OCR
    - Text-based PDFs -> MarkItDown
    - Office docs, etc. -> MarkItDown
    """
    ext = file_path.suffix.lower().lstrip(".")

    if ext in config.IMAGE_EXTENSIONS:
        return True

    if ext == "pdf":
        analysis = local_converter.analyze_file_content(file_path)
        if analysis.get("is_text_based"):
            return False
        return True

    return False


def _should_use_ocr(file_path: Path) -> bool:
    """Decide whether a file should be routed to Mistral OCR or MarkItDown.

    Uses content heuristics, then adjusts for per-engine size caps so smart mode
    does not send an oversized file to the wrong engine (e.g. text PDF that
    exceeds MarkItDown's limit but fits Mistral OCR).
    """
    if not config.MISTRAL_API_KEY:
        return False

    content_wants_ocr = _content_prefers_mistral_ocr(file_path)

    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
    except OSError:
        return content_wants_ocr

    md_ok = size_mb <= float(config.MARKITDOWN_MAX_FILE_SIZE_MB)
    ocr_ok = size_mb <= float(config.MISTRAL_OCR_MAX_FILE_SIZE_MB)

    if content_wants_ocr:
        if ocr_ok:
            return True
        if md_ok:
            return False
        return True

    if md_ok:
        return False
    if ocr_ok:
        return True
    return False


def _route_label_cached(file_path: Path, use_ocr: bool) -> str:
    """Return a human-readable label for the engine that will handle this file."""
    ext = file_path.suffix.lower().lstrip(".")
    if use_ocr:
        label = "Mistral OCR"
        if ext == "pdf":
            label += " (scanned + table extraction)"
        return label
    label = "MarkItDown (local)"
    if ext == "pdf":
        label += " (text-based + table extraction)"
    return label


def _extract_pdf_tables(file_path: Path) -> None:
    """Run pdfplumber table extraction for a PDF and save sidecar files.

    Skips silently when the PDF exceeds the heavy-work size cap.
    Logs but does not raise on extraction failures.
    """
    too_large, size_err = utils.pdf_exceeds_heavy_work_limit(file_path)
    if too_large:
        logger.warning("Skipping table extraction for %s: %s", file_path.name, size_err)
        return
    try:
        table_result = local_converter.extract_all_tables(file_path)
        if table_result["table_count"] > 0:
            local_converter.save_tables_to_files(file_path, table_result["tables"])
            logger.info(
                "Extracted %d tables from %s",
                table_result["table_count"],
                file_path.name,
            )
    except (OSError, ValueError) as e:
        logger.warning("Table extraction failed for %s: %s", file_path.name, e)


def _process_single_smart(
    file_path: Path, *, use_ocr: Optional[bool] = None
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """Process one file with smart routing based on file content analysis.

    Args:
        file_path: File to process.
        use_ocr: Pre-computed routing decision.  When *None* (legacy callers),
                 routing is decided on the fly (same rules as ``_should_use_ocr``).
    """
    ext = file_path.suffix.lower().lstrip(".")

    # PDF table extraction runs regardless of engine choice
    if ext == "pdf":
        _extract_pdf_tables(file_path)

    if use_ocr is None:
        use_ocr = _should_use_ocr(file_path)

    if use_ocr:
        return mistral_converter.convert_with_mistral_ocr(file_path)
    else:
        success, content, error = local_converter.convert_with_markitdown(file_path)
        output_path = config.OUTPUT_MD_DIR / f"{utils.safe_output_stem(file_path)}.md" if success else None
        return success, output_path, error


def mode_convert_smart(file_paths: List[Path]) -> Tuple[bool, str]:
    """
    Smart conversion mode: auto-routes each file to the best engine.

    Routing logic:
    - Images (no text layer) -> Mistral OCR
    - Scanned PDFs (no extractable text) -> Mistral OCR + table extraction
    - Text-based PDFs -> MarkItDown + table extraction
    - Office docs, data files, etc. -> MarkItDown

    Multiple files are processed concurrently.
    """
    logger.info("SMART CONVERT MODE: Processing %d file(s)", len(file_paths))

    if config.MAX_BATCH_FILES > 0 and len(file_paths) > config.MAX_BATCH_FILES:
        return False, (
            f"Batch size ({len(file_paths)}) exceeds MAX_BATCH_FILES ({config.MAX_BATCH_FILES}). "
            "Increase the limit or split into smaller batches."
        )

    # Pre-compute routing decisions once so we don't analyse each file twice
    # (once for the routing plan display, once for actual processing).
    routing_cache: Dict[Path, bool] = {fp: _should_use_ocr(fp) for fp in file_paths}

    # Show routing plan
    utils.ui_print("\nRouting plan:")
    if not config.MISTRAL_API_KEY:
        utils.ui_print("  NOTE: No MISTRAL_API_KEY set. All files will use MarkItDown (local).\n")

    for fp in file_paths:
        label = _route_label_cached(fp, routing_cache[fp])
        utils.ui_print(f"  {utils.sanitize_for_terminal(fp.name):<40} -> {label}")
    utils.ui_print()

    def _process_fn(file_path: Path) -> Tuple[bool, Optional[Path], Optional[str]]:
        return _process_single_smart(file_path, use_ocr=routing_cache[file_path])

    successful, failed = _process_files_concurrently(file_paths, _process_fn, "Converting files")

    total = len(file_paths)
    return failed == 0, f"Processed {successful}/{total} files successfully"


# ============================================================================
# Mode 2: Convert (MarkItDown) -- force local, no API
# ============================================================================


def _process_single_markitdown_with_pdf_tables(
    file_path: Path,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """MarkItDown conversion with pdfplumber table sidecars for PDFs (smart-mode parity)."""
    if file_path.suffix.lower().lstrip(".") == "pdf":
        _extract_pdf_tables(file_path)
    return local_converter.convert_with_markitdown(file_path)


def mode_markitdown_only(file_paths: List[Path]) -> Tuple[bool, str]:
    """Force all files through MarkItDown (local conversion, no API calls)."""
    logger.info("MARKITDOWN MODE: Processing %d file(s)", len(file_paths))

    successful, failed = _process_files_concurrently(
        file_paths,
        _process_single_markitdown_with_pdf_tables,
        "Converting files",
    )

    return failed == 0, f"Processed {successful}/{len(file_paths)} files successfully"


def mode_markitdown_stdin(stdin_bytes: bytes, filename_hint: str) -> Tuple[bool, str]:
    """Convert stdin bytes with MarkItDown using *filename_hint* for format detection."""
    ok, base, sanit_err = utils.sanitize_stdin_filename_hint(filename_hint)
    if not ok:
        return False, sanit_err or "Invalid --stdin-filename"

    logger.info("MARKITDOWN STDIN: %s (%d bytes)", base, len(stdin_bytes))

    stream = io.BytesIO(stdin_bytes)
    success, markdown, error = local_converter.convert_stream_with_markitdown(stream, filename=base)
    if not success or not markdown:
        return False, error or "MarkItDown stream conversion failed"

    hint_path = Path(base)
    if hint_path.suffix:
        stem_raw = hint_path.stem if hint_path.stem else "document"
    else:
        stem_raw = hint_path.name if hint_path.name else "stdin"
    stem = re.sub(r"[^\w\-. ]+", "_", stem_raw).strip("._ ") or "stdin"
    doc_metadata = {
        "file_size_bytes": len(stdin_bytes),
        "file_extension": hint_path.suffix.lower() or "",
        "source": "stdin",
    }
    doc_title = stem_raw if stem_raw else stem
    frontmatter = utils.generate_yaml_frontmatter(
        title=doc_title,
        file_name=base,
        conversion_method="MarkItDown (stream)",
        additional_fields=doc_metadata,
    )
    full_content = frontmatter + markdown
    output_path = config.OUTPUT_MD_DIR / f"{stem}.md"
    utils.atomic_write_text(output_path, full_content)
    utils.save_text_output(output_path, full_content)
    logger.info("Saved: %s", output_path.name)
    return True, f"Saved {output_path}"


# ============================================================================
# Mode 3: Convert (Mistral OCR) -- force cloud OCR
# ============================================================================


def mode_mistral_ocr_only(file_paths: List[Path]) -> Tuple[bool, str]:
    """Force all files through Mistral OCR (cloud processing)."""
    if not config.MISTRAL_API_KEY:
        return False, "Mistral OCR requires MISTRAL_API_KEY to be set"

    if config.MAX_BATCH_FILES > 0 and len(file_paths) > config.MAX_BATCH_FILES:
        return False, (
            f"Batch size ({len(file_paths)}) exceeds MAX_BATCH_FILES ({config.MAX_BATCH_FILES}). "
            "Increase the limit or split into smaller batches."
        )

    logger.info("MISTRAL OCR MODE: Processing %d file(s)", len(file_paths))

    successful, failed = _process_files_concurrently(
        file_paths, mistral_converter.convert_with_mistral_ocr, "OCR processing"
    )

    return failed == 0, f"Processed {successful}/{len(file_paths)} files successfully"


# ============================================================================
# Mode 4: PDF to Images
# ============================================================================


def mode_pdf_to_images(file_paths: List[Path]) -> Tuple[bool, str]:
    """Render each PDF page to PNG images."""
    logger.info("PDF TO IMAGES MODE: Converting %d PDF(s)", len(file_paths))

    pdf_files: List[Path] = []
    for fp in file_paths:
        if fp.suffix.lower() != ".pdf":
            continue
        too_large, size_err = utils.pdf_exceeds_heavy_work_limit(fp)
        if too_large:
            logger.warning("Skipping %s: %s", fp.name, size_err)
            continue
        pdf_files.append(fp)

    non_pdf = sum(1 for fp in file_paths if fp.suffix.lower() != ".pdf")
    if non_pdf:
        logger.warning("Skipping %d non-PDF file(s)", non_pdf)

    if not pdf_files:
        return False, "No PDF files to convert (or all exceeded size limit)"

    # Cap Poppler threads per PDF when the outer pool also runs in parallel.
    if len(pdf_files) > 1:
        inner_threads = max(1, config.PDF_IMAGE_THREAD_COUNT // max(1, config.MAX_CONCURRENT_FILES))
    else:
        inner_threads = config.PDF_IMAGE_THREAD_COUNT

    def _convert_one_pdf(pdf_path: Path) -> Tuple[bool, List[Path], Optional[str]]:
        return local_converter.convert_pdf_to_images(pdf_path, thread_count=inner_threads)

    successful, failed = _process_files_concurrently(pdf_files, _convert_one_pdf, "Converting PDFs")

    return failed == 0, f"Converted {successful} PDFs"


# ============================================================================
# Modes 5-6 (QnA, Batch OCR) implemented in modes/qna.py and modes/batch.py
# ============================================================================

# Re-export private helpers so legacy callers (e.g. tests referencing
# ``main._qna_print_complete``) keep working without importing the modes
# package directly. F401 suppression is intentional.
from modes.batch import _JOB_ID_RE  # noqa: E402,F401
from modes.batch import _batch_download  # noqa: E402,F401
from modes.batch import _batch_list  # noqa: E402,F401
from modes.batch import _batch_status  # noqa: E402,F401
from modes.batch import _batch_submit  # noqa: E402,F401
from modes.batch import _validate_job_id  # noqa: E402,F401
from modes.qna import _qna_print_complete  # noqa: E402,F401
from modes.qna import _qna_print_stream  # noqa: E402,F401

# ============================================================================
# Modes 7-8 (System Status, Maintenance) implemented in modes/system.py
# ============================================================================


# ============================================================================
# File Selection
# ============================================================================


def select_files() -> List[Path]:  # pragma: no cover
    """Prompt user to select files from input directory."""
    input_files = _list_input_files()

    out = utils.ui_print

    if not input_files:
        logger.warning("No files found in %s", config.INPUT_DIR)
        out(f"\nNo files found in '{config.INPUT_DIR}'")
        out("Please add files to the input directory and try again.\n")
        return []

    out(f"\nFound {len(input_files)} file(s) in input directory:\n")

    for i, file_path in enumerate(input_files, 1):
        try:
            file_size = file_path.stat().st_size / 1024
            size_str = f"({file_size:.1f} KB)"
        except OSError:
            size_str = "(size unavailable)"
        out(f"  {i}. {utils.sanitize_for_terminal(file_path.name)} {size_str}")

    out(f"\n  {len(input_files) + 1}. Process ALL files")
    out("  0. Cancel\n")

    while True:
        try:
            choice = input("Select file(s) to process (comma-separated or single number): ").strip()

            if choice == "0":
                return []

            if choice == str(len(input_files) + 1):
                return input_files

            indices = [int(c.strip()) for c in choice.split(",")]

            selected = []
            seen_idx = set()
            for idx in indices:
                if 1 <= idx <= len(input_files):
                    if idx not in seen_idx:
                        seen_idx.add(idx)
                        selected.append(input_files[idx - 1])
                else:
                    utils.ui_print(f"Invalid selection: {idx}")
                    selected = []
                    break

            if selected:
                return selected

        except (ValueError, IndexError):
            utils.ui_print("Invalid input. Please enter numbers separated by commas.\n")
        except (KeyboardInterrupt, EOFError):
            return []


# ============================================================================
# Interactive Menu
# ============================================================================


def show_menu():
    """Display the interactive menu."""
    out = utils.ui_print
    out("\n" + "=" * 60)
    out(f"  ENHANCED DOCUMENT CONVERTER v{config.VERSION}")
    out("=" * 60)
    out("\nSelect conversion mode:\n")
    out("  1. Convert (Smart)")
    out("     Auto-picks best engine per file type")
    out()
    out("  2. Convert (MarkItDown)")
    out("     Force local conversion (no API calls)")
    out()
    out("  3. Convert (Mistral OCR)")
    out("     Force cloud OCR for highest accuracy")
    out()
    out("  4. PDF to Images")
    out("     Render each PDF page to PNG images")
    out()
    out("  5. Document QnA")
    out("     Query documents in natural language")
    out()
    out("  6. Batch OCR (reduced cost)")
    out("     Submit batch jobs to Mistral Batch API")
    out()
    out("  7. System Status")
    out("     Cache stats, config info, and diagnostics")
    out()
    out("  8. Maintenance")
    out("     Clear cache, clean up old uploads")
    out()
    out("  0. Exit")
    out("\n" + "=" * 60 + "\n")


# Dispatch table: menu_choice -> (cli_mode_name, handler)
MODE_DISPATCH: Dict[str, Tuple[str, Any]] = {
    "1": ("smart", mode_convert_smart),
    "2": ("markitdown", mode_markitdown_only),
    "3": ("mistral_ocr", mode_mistral_ocr_only),
    "4": ("pdf_to_images", mode_pdf_to_images),
    "5": ("qna", mode_document_qna),
    "6": ("batch_ocr", mode_batch_ocr),
}

# Reverse lookup: cli mode name -> handler
_CLI_MODE_DISPATCH = {cli_name: handler for _, (cli_name, handler) in MODE_DISPATCH.items()}


def interactive_menu():  # pragma: no cover
    """Run the interactive menu loop."""
    while True:
        show_menu()

        try:
            choice = input("Enter your choice (0-8): ").strip()

            if choice == "0":
                utils.ui_print("\nExiting. Goodbye!\n")
                return

            if choice == "7":
                mode_system_status()
                input("\nPress Enter to continue...")
                continue

            if choice == "8":
                mode_maintenance()
                input("\nPress Enter to continue...")
                continue

            if choice == "6":
                start_time = time.time()
                success, message = mode_batch_ocr([], non_interactive=False)
                utils.ui_print(f"\n{message}")
                elapsed = time.time() - start_time
                utils.ui_print(f"Total processing time: {elapsed:.2f} seconds")
                input("\nPress Enter to continue...")
                continue

            if choice not in MODE_DISPATCH:
                utils.ui_print("\nInvalid choice. Please enter a number between 0 and 8.\n")
                continue

            files = select_files()
            if not files:
                continue

            cli_mode, handler = MODE_DISPATCH[choice]
            valid_files = _filter_valid_files(files, mode=cli_mode)

            if not valid_files:
                utils.ui_print("\nNo valid files to process.\n")
                input("Press Enter to continue...")
                continue

            start_time = time.time()
            success, message = handler(valid_files)
            utils.ui_print(f"\n{message}")

            elapsed = time.time() - start_time
            utils.ui_print(f"Total processing time: {elapsed:.2f} seconds")

            input("\nPress Enter to continue...")

        except KeyboardInterrupt:
            utils.ui_print("\n\nInterrupted. Exiting...\n")
            return

        except Exception as e:
            logger.error("Unexpected error: %s", e)
            utils.ui_print(f"\nError: {e}\n")
            input("Press Enter to continue...")


# ============================================================================
# Command-Line Interface
# ============================================================================


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject invalid CLI argument combinations."""
    if args.stdin and args.mode != "markitdown":
        parser.error("--stdin can only be used with --mode markitdown")
    if args.stdin and not args.no_interactive:
        parser.error("--stdin requires --no-interactive")
    if args.stdin_filename and not args.stdin:
        parser.error("--stdin-filename requires --stdin")
    if args.qna_question and args.mode != "qna":
        parser.error("--qna-question can only be used with --mode qna")
    if args.qna_document_url and args.mode != "qna":
        parser.error("--qna-document-url can only be used with --mode qna")
    if args.qna_document_url and not args.no_interactive:
        parser.error("--qna-document-url requires --no-interactive")
    if args.qna_no_stream and args.mode != "qna":
        parser.error("--qna-no-stream can only be used with --mode qna")
    if args.batch_action and args.mode != "batch_ocr":
        parser.error("--batch-action can only be used with --mode batch_ocr")
    if args.batch_job_id and args.mode != "batch_ocr":
        parser.error("--batch-job-id can only be used with --mode batch_ocr")


def _run_stdin_mode(args: argparse.Namespace) -> None:
    """Handle --mode markitdown --stdin pipeline."""
    if not args.stdin_filename:
        utils.ui_print("--stdin requires --stdin-filename (e.g. report.pdf)")
        sys.exit(1)
    max_stdin_bytes = int(config.MARKITDOWN_MAX_FILE_SIZE_MB * 1024 * 1024)
    ok_stdin, stdin_data, stdin_err = utils.read_stdin_bytes_limited(max_stdin_bytes)
    if not ok_stdin:
        utils.ui_print(stdin_err or "Stdin read failed")
        sys.exit(1)
    start_time = time.time()
    success, message = mode_markitdown_stdin(stdin_data, args.stdin_filename)
    utils.ui_print(f"\n{message}")
    elapsed = time.time() - start_time
    utils.ui_print(f"\nTotal processing time: {elapsed:.2f} seconds")
    sys.exit(0 if success else 1)


def _collect_files_non_interactive(args: argparse.Namespace) -> List[Path]:
    """Gather input files for non-interactive direct-mode execution."""
    qna_url = (args.qna_document_url or "").strip()

    if args.mode == "qna" and qna_url:
        utils.ui_print("Non-interactive QnA using --qna-document-url (no input/ files).")
        return []

    if args.mode == "batch_ocr":
        ba = (args.batch_action or "").lower().strip()
        if ba in ("status", "list", "download"):
            utils.ui_print(f"Non-interactive batch: --batch-action {ba} (no input/ files required).")
            return []

    files = _list_input_files()
    if not files:
        utils.ui_print(f"No files found in {config.INPUT_DIR}")
        sys.exit(1)
    utils.ui_print(f"Non-interactive mode: Processing {len(files)} files from input directory")
    return files


def _run_direct_mode(args: argparse.Namespace) -> None:
    """Execute a specific mode non-interactively or with file picker."""
    if args.mode == "markitdown" and args.stdin:
        _run_stdin_mode(args)
        return

    if args.no_interactive:
        files = _collect_files_non_interactive(args)
    else:
        files = select_files()
        if not files:
            return

    # Validate and filter files (skip for modes that don't need them)
    qna_url = (args.qna_document_url or "").strip()
    needs_files = not (args.mode == "qna" and qna_url)
    batch_no_files = (
        args.mode == "batch_ocr"
        and args.no_interactive
        and (args.batch_action or "").lower().strip() in ("status", "list", "download")
    )
    if needs_files and not batch_no_files:
        files = _filter_valid_files(files, mode=args.mode)
        if not files:
            utils.ui_print("No valid files to process.")
            sys.exit(1)

    start_time = time.time()
    handler = _CLI_MODE_DISPATCH.get(args.mode)
    if handler is None:
        utils.ui_print(f"Unknown mode: {args.mode}")
        sys.exit(1)
    if args.mode == "qna":
        success, message = mode_document_qna(
            files,
            initial_question=args.qna_question,
            non_interactive=args.no_interactive,
            qna_document_url=args.qna_document_url,
            qna_use_stream=not args.qna_no_stream,
        )
    elif args.mode == "batch_ocr":
        success, message = mode_batch_ocr(
            files,
            batch_action=args.batch_action,
            batch_job_id=args.batch_job_id,
            non_interactive=args.no_interactive,
        )
    else:
        success, message = handler(files)
    utils.ui_print(f"\n{message}")

    elapsed = time.time() - start_time
    utils.ui_print(f"\nTotal processing time: {elapsed:.2f} seconds")

    sys.exit(0 if success else 1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=f"Enhanced Document Converter v{config.VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py                     # Interactive menu
  python3 main.py --mode smart        # Smart auto-routing
  python3 main.py --mode markitdown   # Force MarkItDown
  python3 main.py --mode mistral_ocr  # Force Mistral OCR
  python3 main.py --mode qna --no-interactive --qna-question "Summary?"
  python3 main.py --mode qna --no-interactive --qna-document-url "https://example.com/doc.pdf" --qna-question "Summary?"
  python3 main.py --mode qna --no-interactive --qna-no-stream --qna-question "Summary?"
  python3 main.py --mode markitdown --no-interactive --stdin --stdin-filename report.pdf < report.pdf
  python3 main.py --mode batch_ocr --no-interactive --batch-action submit
  python3 main.py --test              # Test mode
        """,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "smart",
            "markitdown",
            "mistral_ocr",
            "pdf_to_images",
            "maintenance",
            "status",
            "qna",
            "batch_ocr",
        ],
        help="Run specific mode directly",
    )

    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Use all files in input/ without the selection menu; for qna/batch_ocr, supply --qna-question / --batch-action",
    )

    parser.add_argument("--test", action="store_true", help="Run in test mode")

    parser.add_argument(
        "--batch-action",
        choices=["submit", "status", "list", "download"],
        default=None,
        help="With --mode batch_ocr and --no-interactive: run this action without prompts",
    )
    parser.add_argument(
        "--batch-job-id",
        default=None,
        help="Job ID for batch status or download in non-interactive mode",
    )
    parser.add_argument(
        "--qna-question",
        default=None,
        help="Single question for --mode qna when using --no-interactive",
    )
    parser.add_argument(
        "--qna-document-url",
        default=None,
        help="HTTPS document URL for QnA (requires --no-interactive; no input/ file)",
    )
    parser.add_argument(
        "--qna-no-stream",
        action="store_true",
        help="Use chat.complete (non-streaming) for QnA instead of streaming",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="With --mode markitdown and --no-interactive: read document bytes from stdin",
    )
    parser.add_argument(
        "--stdin-filename",
        default=None,
        help="Filename hint for --stdin (extension selects the MarkItDown converter)",
    )

    args = parser.parse_args()
    _validate_args(parser, args)

    # Print header
    utils.ui_print("\n" + "=" * 60)
    utils.ui_print(f"  Enhanced Document Converter v{config.VERSION}")
    utils.ui_print("  https://github.com/microsoft/markitdown")
    utils.ui_print("  https://docs.mistral.ai/capabilities/document_ai/basic_ocr/")
    utils.ui_print("=" * 60 + "\n")

    # Initialize config (creates directories, validates settings)
    issues = config.initialize()
    if issues:
        for issue in issues:
            utils.ui_print(issue)

    # Wire up file-based processing log if enabled
    if config.SAVE_PROCESSING_LOGS:
        utils.setup_logging(log_file=str(config.LOGS_DIR / "processing.log"))
        utils.ui_print()

    # Test mode
    if args.test:
        logger.info("Running in test mode...")
        mode_system_status()
        sys.exit(0)

    # Status/maintenance modes don't need file selection
    if args.mode == "status":
        mode_system_status()
        sys.exit(0)

    if args.mode == "maintenance":
        success, message = mode_maintenance()
        utils.ui_print(f"\n{message}")
        sys.exit(0 if success else 1)

    # Direct mode execution
    if args.mode:
        _run_direct_mode(args)
        return

    # Interactive menu
    interactive_menu()


if __name__ == "__main__":
    main()
