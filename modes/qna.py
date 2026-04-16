"""Document QnA CLI mode.

Implements ``mode_document_qna`` (menu option 5, ``--mode qna``). Uses
Mistral's chat completions against either an uploaded file's signed URL or
a user-supplied HTTPS document URL. Includes a single-retry path for signed
URL expiry detected by :func:`mistral_converter.is_signed_url_expiry_error`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import config
import mistral_converter
import utils

logger = utils.logger


def _qna_print_stream(document_url: str, question: str) -> Tuple[bool, str]:
    """Run streaming QnA and print to stdout; returns (ok, message)."""
    success, stream, error = mistral_converter.query_document_stream(document_url, question)
    if success and stream is not None:
        utils.ui_print("\nAnswer: ", end="", flush=True)
        emitted_any = False
        try:
            for chunk in stream:
                if chunk.data.choices and chunk.data.choices[0].delta.content:
                    emitted_any = True
                    safe_text = utils.sanitize_for_terminal(chunk.data.choices[0].delta.content)
                    utils.ui_print(safe_text, end="", flush=True)
        except Exception as e:
            utils.ui_print(f"\n\nStream error: {e}")
            return False, f"QnA stream failed: {e}"
        utils.ui_print("\n")
        if not emitted_any:
            return False, "QnA stream returned no answer content"
        return True, "ok"
    return False, error or "QnA stream failed"


def _qna_print_complete(document_url: str, question: str) -> Tuple[bool, str]:
    """Run non-streaming QnA and print the full answer."""
    success, answer, error = mistral_converter.query_document(document_url, question)
    if success and answer:
        utils.ui_print("\nAnswer:\n")
        utils.ui_print(utils.sanitize_for_terminal(answer))
        utils.ui_print("\n")
        return True, "ok"
    return False, error or "QnA failed"


def mode_document_qna(
    file_paths: List[Path],
    *,
    initial_question: Optional[str] = None,
    non_interactive: bool = False,
    qna_document_url: Optional[str] = None,
    qna_use_stream: bool = True,
) -> Tuple[bool, str]:
    """Query a document in natural language using Mistral chat + OCR."""
    logger.info("DOCUMENT QnA MODE: %d file(s) selected", len(file_paths))

    if not config.MISTRAL_API_KEY:
        return False, "Document QnA requires MISTRAL_API_KEY to be set"

    url_mode = bool((qna_document_url or "").strip())
    if url_mode:
        doc_url = (qna_document_url or "").strip()
        ok_url, url_err = mistral_converter.validate_https_document_url(doc_url)
        if not ok_url:
            return False, f"Invalid document URL: {url_err}"
        display_name = doc_url[:80] + ("…" if len(doc_url) > 80 else "")
    else:
        if len(file_paths) != 1:
            utils.ui_print("\nPlease select exactly 1 file to query.\n")
            return False, "Document QnA works on one file at a time"
        doc_url = ""
        display_name = file_paths[0].name

    file_path = file_paths[0] if not url_mode else None

    client = None
    if not url_mode:
        client = mistral_converter.get_mistral_client()
        if client is None:
            return False, "Mistral client not available"
        if file_path is None:
            return False, "Internal error: file_path is None in non-URL QnA mode"
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            cap = config.MISTRAL_QNA_MAX_FILE_SIZE_MB
            if file_size_mb > cap:
                return False, (
                    f"File too large for Document QnA ({file_size_mb:.1f} MB). "
                    f"Maximum allowed is {cap} MB (MISTRAL_QNA_MAX_FILE_SIZE_MB). Consider splitting the document."
                )
        except OSError as e:
            return False, f"Cannot read file: {e}"

    if non_interactive and not (initial_question or "").strip():
        return False, "Non-interactive QnA requires --qna-question"

    ttl_seconds = config.MISTRAL_SIGNED_URL_EXPIRY * 3600
    upload_started_at = 0.0
    signed_url: Optional[str] = None

    def _get_document_url() -> Optional[str]:
        nonlocal signed_url, upload_started_at
        if url_mode:
            return doc_url
        if client is None or file_path is None:
            return None
        if signed_url and (time.time() - upload_started_at) < ttl_seconds * config.MISTRAL_SIGNED_URL_REFRESH_THRESHOLD:
            return signed_url
        signed_url = mistral_converter.upload_file_for_ocr(client, file_path)
        upload_started_at = time.time()
        return signed_url

    if not _get_document_url():
        return False, (f"Failed to upload {file_path.name} for QnA" if file_path else "No document URL available")

    utils.ui_print(f"\nQuerying: {display_name}")
    utils.ui_print(f"Model: {config.MISTRAL_DOCUMENT_QNA_MODEL}")

    def _ask_with_retry(question: str, use_stream: bool) -> Tuple[bool, str]:
        """Ask a QnA question, retrying once with a fresh URL on expiry-related errors."""
        document_url = _get_document_url()
        if not document_url:
            return False, "Failed to resolve document URL for QnA"
        qna_fn = _qna_print_stream if use_stream else _qna_print_complete
        ok, msg = qna_fn(document_url, question)
        if ok:
            return True, msg
        # Only retry once when the error looks like a signed-URL expiry (403
        # from the object store, expired signature, etc.). Permanent auth or
        # quota failures are not retried. See mistral_converter for the
        # classifier.
        if mistral_converter.is_signed_url_expiry_error(msg):
            nonlocal signed_url, upload_started_at
            signed_url = None
            upload_started_at = 0.0
            document_url = _get_document_url()
            if document_url:
                ok, msg = qna_fn(document_url, question)
        return ok, msg

    if non_interactive:
        question = (initial_question or "").strip()
        ok, msg = _ask_with_retry(question, qna_use_stream)
        if not ok:
            return False, msg
        label = "URL" if url_mode else (file_path.name if file_path else "document")
        return True, f"Asked 1 question ({label})"

    utils.ui_print("Type 'exit' or 'quit' to return to menu.\n")

    questions_asked = 0
    while True:  # pragma: no cover
        try:
            question = input("Question: ").strip()
            if not question or question.lower() in ("exit", "quit"):
                break

            ok, qmsg = _ask_with_retry(question, qna_use_stream)
            if ok:
                questions_asked += 1
            else:
                utils.ui_print(f"\nError: {qmsg}\n")

        except KeyboardInterrupt:
            break

    label = display_name if url_mode else (file_path.name if file_path else "document")
    return True, f"Asked {questions_asked} question(s) about {label}"
