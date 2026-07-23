"""Mistral Batch OCR job helpers."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
import utils

from .client import _http_client_exceptions
from .facade import attr
from .sdk_shims import httpx

logger = utils.logger


def _prepare_batch_entries(
    client: Any,
    file_paths: List[Path],
    model: str,
    include_image_base64: bool,
    batch_signed_url_expiry: int,
) -> Tuple[List[Dict[str, Any]], List[str], Optional[str]]:
    """Upload files and build JSONL entries; returns (entries, uploaded_ids, error_or_None)."""
    entries: List[Dict[str, Any]] = []
    uploaded_file_ids: List[str] = []

    for idx, file_path in enumerate(file_paths):
        pair = attr("_upload_file_for_ocr_pair")(client, file_path, expiry_hours=batch_signed_url_expiry)
        if not pair:
            logger.warning("Failed to upload %s, skipping...", file_path.name)
            if config.MISTRAL_BATCH_STRICT:
                attr("_delete_ocr_file_ids")(client, uploaded_file_ids)
                return (
                    [],
                    [],
                    f"Batch strict mode: upload failed for {file_path.name}",
                )
            continue

        signed_url, file_id = pair
        uploaded_file_ids.append(file_id)

        ext = file_path.suffix.lower().lstrip(".")
        is_image = ext in config.IMAGE_EXTENSIONS

        custom_id = f"{idx}_{utils.safe_output_stem(file_path)}"
        if is_image:
            document = {"type": "image_url", "image_url": signed_url}
        else:
            document = {
                "type": "document_url",
                "document_url": signed_url,
                "document_name": file_path.name,
            }

        body = attr("build_ocr_process_kwargs")(
            document=document,
            model=model,
            include_retries=False,
            pages=None,
            request_id=custom_id,
            file_path=file_path,
        )
        body["include_image_base64"] = include_image_base64

        entry = {"custom_id": custom_id, "body": body}
        entries.append(entry)
        logger.debug("Added %s to batch (id: %s)", file_path.name, custom_id)

    return entries, uploaded_file_ids, None


def create_batch_ocr_file(
    file_paths: List[Path],
    output_file: Path,
    model: Optional[str] = None,
    include_image_base64: Optional[bool] = None,
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Create a JSONL batch file for OCR processing.

    This creates a file in the format required by Mistral's Batch API
    for processing multiple documents at reduced cost.

    Args:
        file_paths: List of file paths to process
        output_file: Path where to save the JSONL batch file
        model: OCR model to use (default: mistral-ocr-latest)
        include_image_base64: Whether to include image base64 in results

    Returns:
        Tuple of (success, batch_file_path, error_message)

    Example:
        >>> success, batch_file, error = create_batch_ocr_file(
        ...     [Path("doc1.pdf"), Path("doc2.pdf")],
        ...     Path("batch_input.jsonl")
        ... )

    Documentation:
        https://docs.mistral.ai/capabilities/batch/
    """
    client = attr("get_mistral_client")()
    if client is None:
        return False, None, "Mistral client not available"

    if model is None:
        model = config.get_ocr_model()

    resolved_include_images = config.MISTRAL_INCLUDE_IMAGES if include_image_base64 is None else include_image_base64

    batch_signed_url_expiry = max(
        config.MISTRAL_SIGNED_URL_EXPIRY,
        config.MISTRAL_BATCH_TIMEOUT_HOURS + 1,
    )

    try:
        logger.info("Creating batch OCR file for %s documents...", len(file_paths))

        entries, uploaded_file_ids, prep_error = _prepare_batch_entries(
            client, file_paths, model, resolved_include_images, batch_signed_url_expiry
        )
        if prep_error:
            return False, None, prep_error

        if not entries:
            attr("_delete_ocr_file_ids")(client, uploaded_file_ids)
            return False, None, "No files could be prepared for batch processing"

        # Write JSONL (signed URLs). Prefer writing under ``config.CACHE_DIR`` (POSIX
        # 0o700 from ``ensure_directories``). On Windows, tighten ACLs on ``cache/``
        # or the output path if these URLs must stay secret on disk.
        content = "".join(json.dumps(entry) + "\n" for entry in entries)
        utils.atomic_write_text(output_file, content)

        if sys.platform != "win32":
            os.chmod(output_file, 0o600)

        logger.info("Created batch file with %s entries: %s", len(entries), output_file)
        return True, output_file, None

    except Exception as e:
        error_msg = f"Error creating batch OCR file: {e}"
        logger.exception("Unexpected error creating batch OCR file")
        return False, None, error_msg


def submit_batch_ocr_job(  # noqa: C901
    batch_file_path: Path,
    model: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Submit a batch OCR job to Mistral's Batch API.

    This submits the batch file for processing and returns a job ID
    that can be used to monitor progress and retrieve results.

    Args:
        batch_file_path: Path to the JSONL batch file
        model: OCR model to use (default: mistral-ocr-latest)
        metadata: Optional metadata dictionary for the job

    Returns:
        Tuple of (success, job_id, error_message)

    Example:
        >>> success, job_id, error = submit_batch_ocr_job(
        ...     Path("batch_input.jsonl"),
        ...     metadata={"job_type": "document_processing"}
        ... )
        >>> if success:
        ...     print(f"Job submitted: {job_id}")
    """
    client = attr("get_mistral_client")()
    if client is None:
        return False, None, "Mistral client not available"

    if model is None:
        model = config.get_ocr_model()

    batch_file_id: Optional[str] = None
    try:
        logger.info("Uploading batch file: %s", batch_file_path.name)

        # Stream file handle directly to avoid loading the full JSONL into memory.
        with open(batch_file_path, "rb") as f:
            batch_data = client.files.upload(
                file={
                    "file_name": batch_file_path.name,
                    "content": f,
                },
                purpose="batch",
            )

        uploaded_id = getattr(batch_data, "id", None)
        if not isinstance(uploaded_id, str) or not uploaded_id:
            # Early return bypasses the except cleanup; still scrub local JSONL
            # (signed URLs) and best-effort delete a non-string remote id.
            try:
                batch_file_path.unlink(missing_ok=True)
            except OSError:
                pass
            if uploaded_id is not None:
                try:
                    client.files.delete(file_id=str(uploaded_id))
                except Exception as del_err:
                    logger.debug(
                        "Could not delete batch upload with invalid id %r: %s",
                        uploaded_id,
                        del_err,
                    )
            return False, None, "Batch upload response missing file ID"
        batch_file_id = uploaded_id
        logger.info("Batch file uploaded: %s", batch_file_id)
        if not attr("_register_uploaded_file")(batch_file_id, "batch"):
            raise RuntimeError(f"Failed to persist upload registry for batch file {batch_file_id}")

        # Create the batch job
        job_params = {
            "input_files": [batch_file_id],
            "model": model,
            "endpoint": "/v1/ocr",
        }

        if metadata:
            job_params["metadata"] = metadata

        if config.MISTRAL_BATCH_TIMEOUT_HOURS != config.MISTRAL_BATCH_DEFAULT_TIMEOUT_HOURS:
            job_params["timeout_hours"] = config.MISTRAL_BATCH_TIMEOUT_HOURS

        created_job = client.batch.jobs.create(**job_params)

        logger.info("Batch job created: %s", created_job.id)

        if config.ENABLE_BATCH_METADATA:
            try:
                meta_path = config.METADATA_DIR / f"batch_job_{created_job.id}.json"
                meta_payload = {
                    "job_id": created_job.id,
                    "model": model,
                    "batch_file": batch_file_path.name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                utils.atomic_write_text(meta_path, json.dumps(meta_payload, indent=2))
            except OSError as meta_err:
                logger.warning("Could not write batch job metadata: %s", meta_err)

        # Clean up the local JSONL file (contains signed URLs)
        try:
            batch_file_path.unlink(missing_ok=True)
            logger.debug("Cleaned up batch JSONL file: %s", batch_file_path.name)
        except OSError as cleanup_err:  # pragma: no cover
            logger.warning(
                "Could not remove batch JSONL file %s: %s",
                batch_file_path.name,
                cleanup_err,
            )

        return True, created_job.id, None

    except Exception as e:  # pragma: no cover
        if batch_file_id:
            try:
                client.files.delete(file_id=batch_file_id)
                attr("_unregister_uploaded_file")(batch_file_id)
            except Exception as del_err:
                logger.debug(
                    "Could not delete orphaned batch upload %s: %s",
                    batch_file_id,
                    del_err,
                )
        try:
            batch_file_path.unlink(missing_ok=True)
        except OSError:
            pass
        error_msg = f"Error submitting batch OCR job: {e}"
        logger.error(error_msg)
        return False, None, error_msg


def get_batch_job_status(
    job_id: str,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Get the status of a batch OCR job.

    Args:
        job_id: The batch job ID

    Returns:
        Tuple of (success, status_dict, error_message)

        status_dict contains:
        - status: Job status (QUEUED, RUNNING, SUCCESS, FAILED, etc.)
        - total_requests: Total number of requests in batch
        - succeeded_requests: Number of successful requests
        - failed_requests: Number of failed requests
        - output_file: Output file ID (when complete)
    """
    client = attr("get_mistral_client")()
    if client is None:
        return False, None, "Mistral client not available"

    try:
        job = client.batch.jobs.get(job_id=job_id)

        total_req = getattr(job, "total_requests", None) or 0
        succeeded_req = getattr(job, "succeeded_requests", None) or 0
        failed_req = getattr(job, "failed_requests", None) or 0

        status = {
            "status": job.status,
            "total_requests": total_req,
            "succeeded_requests": succeeded_req,
            "failed_requests": failed_req,
            "output_file": getattr(job, "output_file", None),
            "error_file": getattr(job, "error_file", None),
        }

        if total_req > 0:
            status["progress_percent"] = round(((succeeded_req + failed_req) / total_req) * 100, 2)
        else:
            status["progress_percent"] = 0

        logger.info(
            "Batch job %s: %s - %s%% complete",
            job_id,
            status["status"],
            status["progress_percent"],
        )

        return True, status, None

    except Exception as e:
        error_msg = f"Error getting batch job status: {e}"
        http_types = _http_client_exceptions()
        if http_types and isinstance(e, http_types):
            logger.error(error_msg)
        else:
            logger.exception("Unexpected error getting batch job status")
        return False, None, error_msg


def download_batch_results(
    job_id: str,
    output_dir: Optional[Path] = None,
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Download results from a completed batch OCR job.

    Args:
        job_id: The batch job ID
        output_dir: Directory to save results (default: output_md/)

    Returns:
        Tuple of (success, results_file_path, error_message)
    """
    client = attr("get_mistral_client")()
    if client is None:
        return False, None, "Mistral client not available"

    if output_dir is None:
        output_dir = config.OUTPUT_MD_DIR

    try:
        # Get job status to get output file ID
        job = client.batch.jobs.get(job_id=job_id)

        if job.status not in ["SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"]:
            return False, None, f"Job not complete. Status: {job.status}"

        if not job.output_file:
            return False, None, "No output file available"

        # Download the output file
        logger.info("Downloading batch results for job %s...", job_id)

        output_path = output_dir / f"batch_ocr_results_{job_id}.jsonl"

        # Download file content. SDK versions differ here:
        # - older/test paths may return raw bytes
        # - current SDK returns an httpx.Response stream wrapper
        file_content = client.files.download(file_id=job.output_file)

        if httpx is not None and isinstance(file_content, httpx.Response):
            payload = file_content.read()
            file_content.close()
        elif isinstance(file_content, (bytes, bytearray, memoryview)):
            payload = bytes(file_content)
        elif hasattr(file_content, "content"):
            payload = file_content.content
        elif hasattr(file_content, "read"):
            payload = file_content.read()
        else:
            raise TypeError(f"Unsupported batch download payload type: {type(file_content)!r}")

        utils.atomic_write_binary(output_path, payload)

        logger.info("Batch results saved to: %s", output_path)
        return True, output_path, None

    except Exception as e:
        error_msg = f"Error downloading batch results: {e}"
        http_types = _http_client_exceptions()
        if http_types and isinstance(e, http_types):
            logger.error(error_msg)
        else:
            logger.exception("Unexpected error downloading batch results")
        return False, None, error_msg


def list_batch_jobs(
    status: Optional[str] = None,
    page: int = 0,
    page_size: int = 100,
) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    List batch OCR jobs with optional status filtering and pagination.

    Args:
        status: Optional filter by status (QUEUED, RUNNING, SUCCESS, FAILED, etc.)
        page: Page number (0-indexed) for pagination
        page_size: Number of results per page (default: 100)

    Returns:
        Tuple of (success, jobs_list, error_message)
    """
    client = attr("get_mistral_client")()
    if client is None:
        return False, None, "Mistral client not available"

    try:
        list_kwargs: Dict[str, Any] = {}
        if status is not None:
            list_kwargs["status"] = status
        if page > 0:
            list_kwargs["page"] = page
        if page_size != 100:
            list_kwargs["page_size"] = page_size

        try:
            jobs_response = client.batch.jobs.list(**list_kwargs)
        except TypeError:
            # SDK version doesn't accept status=; fall back to client-side filtering.
            list_kwargs.pop("status", None)
            jobs_response = client.batch.jobs.list(**list_kwargs)
        jobs_data = (jobs_response.data or []) if hasattr(jobs_response, "data") else jobs_response

        jobs_list = []
        for job in jobs_data:
            job_info = {
                "id": job.id,
                "status": job.status,
                "model": getattr(job, "model", None),
                "total_requests": getattr(job, "total_requests", 0),
                "succeeded_requests": getattr(job, "succeeded_requests", 0),
                "failed_requests": getattr(job, "failed_requests", 0),
                "created_at": str(getattr(job, "created_at", "")),
            }

            # Client-side filter acts as safety net when server-side filtering
            # is unsupported or as a no-op when the server already filtered.
            if status is None or job_info["status"] == status:
                jobs_list.append(job_info)

        logger.info("Found %s batch jobs", len(jobs_list))
        return True, jobs_list, None

    except Exception as e:
        error_msg = f"Error listing batch jobs: {e}"
        http_types = _http_client_exceptions()
        if http_types and isinstance(e, http_types):
            logger.error(error_msg)
        else:
            logger.exception("Unexpected error listing batch jobs")
        return False, None, error_msg
