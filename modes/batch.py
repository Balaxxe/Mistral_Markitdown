"""Batch OCR CLI mode.

Implements ``mode_batch_ocr`` (menu option 6, ``--mode batch_ocr``) plus its
four sub-actions: submit, status, list, download. Interactive file selection
delegates back to :mod:`main` via lazy imports so this module has no static
dependency on the CLI entry point.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import config
import mistral_converter
import utils

logger = utils.logger


# Batch job ID validation
_JOB_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


def _validate_job_id(job_id: str) -> bool:
    """Return True if *job_id* looks like a valid Mistral batch job identifier."""
    return bool(_JOB_ID_RE.match(job_id))


def _batch_submit(file_paths: List[Path], *, non_interactive: bool) -> Tuple[bool, str]:
    """Batch sub-action: submit a new batch OCR job."""
    submit_paths = list(file_paths)
    if not submit_paths:
        if non_interactive:
            return (
                False,
                "Batch submit requires files in input/ (use interactive mode to pick files).",
            )
        # Lazy import to avoid a hard dependency between the modes package and
        # the CLI entry point. main imports this module at startup.
        import main as _main

        picked = _main.select_files()
        if not picked:
            return False, "No files selected"
        submit_paths = _main._filter_valid_files(picked, mode="batch_ocr")
        if not submit_paths:
            return False, "No valid files to process for batch OCR."

    if config.MAX_BATCH_FILES > 0 and len(submit_paths) > config.MAX_BATCH_FILES:
        return False, (
            f"Batch size ({len(submit_paths)}) exceeds MAX_BATCH_FILES ({config.MAX_BATCH_FILES}). "
            "Increase the limit or split into smaller batches."
        )

    if len(submit_paths) < config.MISTRAL_BATCH_MIN_FILES:
        utils.ui_print(f"\nNote: Batch processing is most cost-effective with {config.MISTRAL_BATCH_MIN_FILES}+ files.")
        utils.ui_print(f"You selected {len(submit_paths)} file(s). Proceeding anyway.\n")

    batch_file: Optional[Path] = None
    try:
        fd, batch_path_str = tempfile.mkstemp(
            suffix=".jsonl",
            prefix="batch_ocr_",
            dir=str(config.CACHE_DIR),
        )
        os.close(fd)
        batch_file = Path(batch_path_str)
        utils.ui_print(f"\nCreating batch file for {len(submit_paths)} document(s)...")

        success, batch_path, error = mistral_converter.create_batch_ocr_file(submit_paths, batch_file)
        if not success or batch_path is None:
            return False, f"Failed to create batch file: {error}"

        utils.ui_print("Submitting batch job...")
        success, job_id, error = mistral_converter.submit_batch_ocr_job(batch_path)
        if success:
            utils.ui_print(f"\nBatch job submitted: {job_id}")
            # Machine-readable line for automation / scripting
            utils.ui_print(f"BATCH_JOB_ID={job_id}")
            utils.ui_print("Use option 2 to check status, option 4 to download results when complete.")
            return True, f"Batch job submitted: {job_id}"
        return False, f"Failed to submit batch job: {error}"
    finally:
        if batch_file is not None:
            batch_file.unlink(missing_ok=True)


def _batch_status(*, batch_job_id: Optional[str], non_interactive: bool) -> Tuple[bool, str]:
    """Batch sub-action: check job status."""
    if non_interactive:
        job_id = (batch_job_id or "").strip()
        if not job_id:
            return False, "Non-interactive batch status requires --batch-job-id"
    else:
        job_id = input("Enter job ID: ").strip()
        if not job_id:
            return False, "No job ID provided"
    if not _validate_job_id(job_id):
        return False, "Invalid job ID format"
    success, status, error = mistral_converter.get_batch_job_status(job_id)
    if success and status is not None:
        utils.ui_print(f"\nJob: {job_id}")
        utils.ui_print(f"  Status: {status['status']}")
        utils.ui_print(f"  Progress: {status['progress_percent']}%")
        utils.ui_print(f"  Succeeded: {status['succeeded_requests']}")
        utils.ui_print(f"  Failed: {status['failed_requests']}")
        return True, f"Job {job_id}: {status['status']}"
    return False, f"Error: {error}"


def _batch_list() -> Tuple[bool, str]:
    """Batch sub-action: list all batch jobs."""
    success, jobs, error = mistral_converter.list_batch_jobs()
    if success and jobs:
        utils.ui_print(f"\n{len(jobs)} batch job(s):\n")
        for job in jobs:
            utils.ui_print(f"  {job['id']} | {job['status']} | {job['total_requests']} requests | {job['created_at']}")
        return True, f"Listed {len(jobs)} batch jobs"
    elif success:
        utils.ui_print("\nNo batch jobs found.")
        return True, "No batch jobs"
    return False, f"Error: {error}"


def _batch_download(*, batch_job_id: Optional[str], non_interactive: bool) -> Tuple[bool, str]:
    """Batch sub-action: download batch results."""
    if non_interactive:
        job_id = (batch_job_id or "").strip()
        if not job_id:
            return False, "Non-interactive batch download requires --batch-job-id"
    else:
        job_id = input("Enter job ID: ").strip()
        if not job_id:
            return False, "No job ID provided"
    if not _validate_job_id(job_id):
        return False, "Invalid job ID format"
    success, path, error = mistral_converter.download_batch_results(job_id)
    if success:
        utils.ui_print(f"\nResults saved to: {path}")
        return True, f"Results downloaded: {path}"
    return False, f"Error: {error}"


def mode_batch_ocr(
    file_paths: List[Path],
    *,
    batch_action: Optional[str] = None,
    batch_job_id: Optional[str] = None,
    non_interactive: bool = False,
) -> Tuple[bool, str]:
    """Submit files for batch OCR processing at reduced cost."""
    logger.info("BATCH OCR MODE: %d file(s) in initial selection", len(file_paths))

    if not config.MISTRAL_API_KEY:
        return False, "Batch OCR requires MISTRAL_API_KEY to be set"

    if not config.MISTRAL_BATCH_ENABLED:
        return False, "Batch processing is disabled (set MISTRAL_BATCH_ENABLED=true)"

    choice: Optional[str] = None
    if non_interactive:
        if not batch_action:
            return (
                False,
                "Non-interactive batch mode requires --batch-action (submit|status|list|download)",
            )
        _batch_map = {"submit": "1", "status": "2", "list": "3", "download": "4"}
        choice = _batch_map.get(batch_action.lower().strip())
        if choice is None:
            return False, f"Unknown --batch-action: {batch_action!r}"
    else:
        utils.ui_print("\nBatch OCR Options:")
        utils.ui_print("  1. Submit new batch job")
        utils.ui_print("  2. Check job status")
        utils.ui_print("  3. List all batch jobs")
        utils.ui_print("  4. Download batch results")
        utils.ui_print("  0. Cancel\n")

        try:
            choice = input("Select option: ").strip()
        except (KeyboardInterrupt, EOFError):
            return False, "Cancelled"

    if choice == "0":
        return False, "Cancelled"

    _dispatch = {
        "1": lambda: _batch_submit(file_paths, non_interactive=non_interactive),
        "2": lambda: _batch_status(batch_job_id=batch_job_id, non_interactive=non_interactive),
        "3": _batch_list,
        "4": lambda: _batch_download(batch_job_id=batch_job_id, non_interactive=non_interactive),
    }
    action = _dispatch.get(choice)
    if action:
        return action()
    return False, "Cancelled"
