"""Tests for Mistral Document QnA helpers (query_document, stream,
query_document_file). Split out of test_mistral_converter.py for navigability."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

import config

# Initialize config dirs so imports work
config.ensure_directories()

import mistral_converter

# ============================================================================
# _validate_document_url Tests
# ============================================================================


class TestQueryDocument:
    """Test document querying with mocks."""

    def test_rejects_invalid_url(self):
        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            ok, answer, err = mistral_converter.query_document("http://insecure.com/doc.pdf", "what?")
        assert ok is False
        assert "HTTPS" in err or "https" in err.lower()

    def test_rejects_private_url(self):
        ok, answer, err = mistral_converter.query_document("https://192.168.1.1/doc.pdf", "what?")
        assert ok is False

    def test_no_client_available(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "")
        mistral_converter.reset_mistral_client()
        with patch(
            "socket.getaddrinfo",
            return_value=[(None, None, None, None, ("93.184.216.34", 0))],
        ):
            ok, answer, err = mistral_converter.query_document("https://example.com/doc.pdf", "what?")
        assert ok is False
        mistral_converter.reset_mistral_client()

    def test_custom_server_rejects_arbitrary_document_url(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_SERVER_URL", "https://private-api.example")
        monkeypatch.setattr(config, "MISTRAL_QNA_ALLOW_URL_WITH_CUSTOM_SERVER", False, raising=False)
        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()) as get_client:
            ok, answer, err = mistral_converter.query_document("https://example.com/doc.pdf", "what?")
        assert ok is False
        assert answer is None
        assert "MISTRAL_SERVER_URL" in (err or "")
        get_client.assert_not_called()

    def test_custom_server_rejects_strict_dns_override_bypass(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_SERVER_URL", "https://private-api.example")
        monkeypatch.setattr(config, "MISTRAL_QNA_ALLOW_URL_WITH_CUSTOM_SERVER", False, raising=False)
        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()) as get_client:
            ok, answer, err = mistral_converter.query_document(
                "https://example.com/doc.pdf",
                "what?",
                strict_dns=False,
            )
        assert ok is False
        assert answer is None
        assert "MISTRAL_SERVER_URL" in (err or "")
        get_client.assert_not_called()

    def test_custom_server_allows_opted_in_document_url(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_SERVER_URL", "https://private-api.example")
        monkeypatch.setattr(config, "MISTRAL_QNA_ALLOW_URL_WITH_CUSTOM_SERVER", True, raising=False)
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_response = MagicMock(choices=[mock_choice])
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = mock_response
        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch("mistral_converter.url_validation._resolve_dns_in_subprocess", return_value=["8.8.8.8"]):
                ok, answer, err = mistral_converter.query_document("https://example.com/doc.pdf", "what?")
        assert (ok, answer, err) == (True, "ok", None)


# ============================================================================
# query_document_stream Tests
# ============================================================================


class TestQueryDocumentStream:
    """Test streaming document querying."""

    def test_rejects_invalid_url(self):
        ok, stream, err = mistral_converter.query_document_stream("http://bad.com/doc.pdf", "what?")
        assert ok is False

    def test_no_client_available(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "")
        mistral_converter.reset_mistral_client()
        with patch(
            "socket.getaddrinfo",
            return_value=[(None, None, None, None, ("93.184.216.34", 0))],
        ):
            ok, stream, err = mistral_converter.query_document_stream("https://example.com/doc.pdf", "what?")
        assert ok is False
        mistral_converter.reset_mistral_client()


# ============================================================================
# save_extracted_images Additional Tests
# ============================================================================


class TestQueryDocumentFile:
    """Test file-based document QnA."""

    def test_no_client(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        with patch.object(mistral_converter, "get_mistral_client", return_value=None):
            ok, answer, err = mistral_converter.query_document_file(pdf, "what?")
        assert ok is False

    def test_file_too_large(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        cap = config.MISTRAL_QNA_MAX_FILE_SIZE_MB
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"x" * ((cap + 1) * 1024 * 1024))

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            ok, answer, err = mistral_converter.query_document_file(pdf, "what?")
        assert ok is False
        assert "too large" in err.lower()

    def test_upload_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF small file")

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch.object(mistral_converter, "upload_file_for_ocr", return_value=None):
                ok, answer, err = mistral_converter.query_document_file(pdf, "what?")
        assert ok is False
        assert "upload" in err.lower()

    def test_upload_runs_under_qna_validation_mode(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF small file")

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch.object(mistral_converter, "upload_file_for_ocr", return_value=None) as mock_upload:
                mistral_converter.query_document_file(pdf, "what?")

        assert mock_upload.call_args.kwargs["mode"] == "qna"

    def test_successful_query(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF small content")
        mock_choice = MagicMock()
        mock_choice.message.content = "The answer is 42"
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = MagicMock(choices=[mock_choice])

        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch.object(
                mistral_converter,
                "upload_file_for_ocr",
                return_value="https://signed.url/doc",
            ):
                ok, answer, err = mistral_converter.query_document_file(pdf, "what?")
        assert ok is True
        assert answer == "The answer is 42"

    def test_custom_server_keeps_uploaded_file_qna_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF small content")
        monkeypatch.setattr(config, "MISTRAL_SERVER_URL", "https://private-api.example")
        monkeypatch.setattr(config, "MISTRAL_QNA_ALLOW_URL_WITH_CUSTOM_SERVER", False, raising=False)
        mock_choice = MagicMock()
        mock_choice.message.content = "answer"
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = MagicMock(choices=[mock_choice])

        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch.object(mistral_converter, "upload_file_for_ocr", return_value="https://signed.url/doc"):
                ok, answer, err = mistral_converter.query_document_file(pdf, "what?")

        assert (ok, answer, err) == (True, "answer", None)
        document_part = mock_client.chat.complete.call_args.kwargs["messages"][1]["content"][1]
        assert document_part["document_url"] == "https://signed.url/doc"

    def test_custom_server_stream_rejects_strict_dns_override_bypass(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_SERVER_URL", "https://private-api.example")
        monkeypatch.setattr(config, "MISTRAL_QNA_ALLOW_URL_WITH_CUSTOM_SERVER", False, raising=False)
        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()) as get_client:
            ok, stream, err = mistral_converter.query_document_stream(
                "https://example.com/doc.pdf",
                "what?",
                strict_dns=False,
            )
        assert ok is False
        assert stream is None
        assert "MISTRAL_SERVER_URL" in (err or "")
        get_client.assert_not_called()

    def test_exception_error_redacts_signed_url(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.chat.complete.side_effect = Exception(
            "download https://bucket.example/file?X-Amz-Signature=secret&token=also-secret failed"
        )
        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch("mistral_converter.url_validation._resolve_dns_in_subprocess", return_value=["8.8.8.8"]):
                ok, answer, err = mistral_converter.query_document("https://example.com/doc.pdf", "what?")
        assert ok is False
        assert answer is None
        assert "secret" not in (err or "")
        assert "X-Amz-Signature=<redacted>" in (err or "")


# ============================================================================
# submit_batch_ocr_job Tests
# ============================================================================


class TestQueryDocumentFull:
    """Test query_document with all parameter paths."""

    def test_successful_query_with_limits(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_QNA_MODEL", "mistral-small-latest")
        monkeypatch.setattr(config, "MISTRAL_QNA_SYSTEM_PROMPT", "You are helpful.")
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_IMAGE_LIMIT", 5)
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_PAGE_LIMIT", 10)

        mock_choice = MagicMock()
        mock_choice.message.content = "The answer is 42"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = mock_response

        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch.object(
                mistral_converter,
                "get_retry_config",
                return_value=MagicMock(),
            ):
                with patch(
                    "socket.getaddrinfo",
                    return_value=[(None, None, None, None, ("93.184.216.34", 0))],
                ):
                    ok, answer, err = mistral_converter.query_document("https://example.com/doc.pdf", "What is 6*7?")

        assert ok is True
        assert answer == "The answer is 42"
        call_kwargs = mock_client.chat.complete.call_args[1]
        assert call_kwargs["document_image_limit"] == 5
        assert call_kwargs["document_page_limit"] == 10

    def test_empty_response(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_QNA_MODEL", "mistral-small-latest")
        monkeypatch.setattr(config, "MISTRAL_QNA_SYSTEM_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_PAGE_LIMIT", 0)

        mock_response = MagicMock()
        mock_response.choices = []

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = mock_response

        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch(
                    "socket.getaddrinfo",
                    return_value=[(None, None, None, None, ("93.184.216.34", 0))],
                ):
                    ok, answer, err = mistral_converter.query_document("https://example.com/doc.pdf", "What?")

        assert ok is False
        assert "empty" in err.lower()

    def test_api_exception(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_QNA_MODEL", "mistral-small-latest")
        monkeypatch.setattr(config, "MISTRAL_QNA_SYSTEM_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_PAGE_LIMIT", 0)

        mock_client = MagicMock()
        mock_client.chat.complete.side_effect = Exception("timeout")

        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch(
                    "socket.getaddrinfo",
                    return_value=[(None, None, None, None, ("93.184.216.34", 0))],
                ):
                    ok, answer, err = mistral_converter.query_document("https://example.com/doc.pdf", "What?")

        assert ok is False
        assert "timeout" in err.lower()


# ============================================================================
# query_document_stream - full coverage
# ============================================================================


class TestQueryDocumentStreamFull:
    """Test streaming document QnA."""

    def test_successful_stream(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_QNA_MODEL", "mistral-small-latest")
        monkeypatch.setattr(config, "MISTRAL_QNA_SYSTEM_PROMPT", "Be helpful.")
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_IMAGE_LIMIT", 3)
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_PAGE_LIMIT", 5)

        mock_stream = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.stream.return_value = mock_stream

        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch.object(
                mistral_converter,
                "get_retry_config",
                return_value=MagicMock(),
            ):
                with patch(
                    "socket.getaddrinfo",
                    return_value=[(None, None, None, None, ("93.184.216.34", 0))],
                ):
                    ok, stream, err = mistral_converter.query_document_stream(
                        "https://example.com/doc.pdf", "Summarize this"
                    )

        assert ok is True
        assert stream is mock_stream
        call_kwargs = mock_client.chat.stream.call_args[1]
        assert call_kwargs["document_image_limit"] == 3
        assert call_kwargs["document_page_limit"] == 5

    def test_stream_no_system_prompt(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_QNA_MODEL", "mistral-small-latest")
        monkeypatch.setattr(config, "MISTRAL_QNA_SYSTEM_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_PAGE_LIMIT", 0)

        mock_stream = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.stream.return_value = mock_stream

        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch(
                    "socket.getaddrinfo",
                    return_value=[(None, None, None, None, ("93.184.216.34", 0))],
                ):
                    ok, stream, err = mistral_converter.query_document_stream("https://example.com/doc.pdf", "What?")

        assert ok is True

    def test_stream_invalid_url(self):
        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            ok, stream, err = mistral_converter.query_document_stream("http://example.com/doc.pdf", "What?")
        assert ok is False
        assert "HTTPS" in err

    def test_stream_no_client(self):
        with patch.object(mistral_converter, "get_mistral_client", return_value=None):
            ok, stream, err = mistral_converter.query_document_stream("https://example.com/doc.pdf", "What?")
        assert ok is False

    def test_stream_api_exception(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_QNA_MODEL", "mistral-small-latest")
        monkeypatch.setattr(config, "MISTRAL_QNA_SYSTEM_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_PAGE_LIMIT", 0)

        mock_client = MagicMock()
        mock_client.chat.stream.side_effect = Exception("stream error")

        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch(
                    "socket.getaddrinfo",
                    return_value=[(None, None, None, None, ("93.184.216.34", 0))],
                ):
                    ok, stream, err = mistral_converter.query_document_stream("https://example.com/doc.pdf", "What?")

        assert ok is False
        assert "stream error" in err.lower()

    def test_stream_strict_dns_false_allows_unresolvable_signed_url(self, monkeypatch):
        """Signed-URL callers can bypass fail-closed DNS like query_document_file."""
        import socket

        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_QNA_MODEL", "mistral-small-latest")
        monkeypatch.setattr(config, "MISTRAL_QNA_SYSTEM_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_PAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_URL_STRICT_DNS", True)

        mock_stream = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.stream.return_value = mock_stream

        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch(
                    "socket.getaddrinfo",
                    side_effect=socket.gaierror(8, "nodename nor servname"),
                ):
                    ok, stream, err = mistral_converter.query_document_stream(
                        "https://signed.example/doc.pdf",
                        "What?",
                        strict_dns=False,
                    )

        assert ok is True
        assert stream is mock_stream
        assert err is None


# ============================================================================
# query_document_file - additional paths
# ============================================================================


class TestQueryDocumentFileFull:
    """Test file-based QnA additional paths."""

    def test_file_not_readable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        pdf = tmp_path / "nonexistent.pdf"
        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            ok, answer, err = mistral_converter.query_document_file(pdf, "what?")
        assert ok is False

    def test_exception_during_upload(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF small")

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch.object(
                mistral_converter,
                "upload_file_for_ocr",
                side_effect=Exception("upload boom"),
            ):
                ok, answer, err = mistral_converter.query_document_file(pdf, "what?")
        assert ok is False


# ============================================================================
# create_batch_ocr_file - full coverage
# ============================================================================


# ============================================================================
# query_document_file path confinement
# ============================================================================


class TestQueryDocumentFilePathConfinement:
    """query_document_file applies utils.validate_file before uploading."""

    @staticmethod
    def _input_dir(tmp_path, monkeypatch, strict=True):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setattr(config, "STRICT_INPUT_PATH_RESOLUTION", strict)
        monkeypatch.setattr(config, "INPUT_DIR", input_dir)
        return input_dir

    def test_rejects_file_outside_input_dir(self, tmp_path, monkeypatch):
        self._input_dir(tmp_path, monkeypatch)
        outside = tmp_path / "outside.pdf"
        outside.write_bytes(b"%PDF small")

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch.object(mistral_converter, "upload_file_for_ocr") as mock_upload:
                ok, answer, err = mistral_converter.query_document_file(outside, "what?")

        assert (ok, answer) == (False, None)
        assert "input directory" in (err or "")
        mock_upload.assert_not_called()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
    def test_rejects_symlink_escaping_input_dir(self, tmp_path, monkeypatch):
        input_dir = self._input_dir(tmp_path, monkeypatch)
        outside = tmp_path / "outside.pdf"
        outside.write_bytes(b"%PDF small")
        link = input_dir / "link.pdf"
        link.symlink_to(outside)

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch.object(mistral_converter, "upload_file_for_ocr") as mock_upload:
                ok, answer, err = mistral_converter.query_document_file(link, "what?")

        assert (ok, answer) == (False, None)
        assert "input directory" in (err or "")
        mock_upload.assert_not_called()

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
    def test_rejects_fifo_even_without_strict_resolution(self, tmp_path, monkeypatch):
        input_dir = self._input_dir(tmp_path, monkeypatch, strict=False)
        fifo = input_dir / "pipe.pdf"
        os.mkfifo(fifo)

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch.object(mistral_converter, "upload_file_for_ocr") as mock_upload:
                ok, answer, err = mistral_converter.query_document_file(fifo, "what?")

        assert (ok, answer) == (False, None)
        assert "Not a file" in (err or "")
        mock_upload.assert_not_called()

    def test_accepts_file_inside_input_dir(self, tmp_path, monkeypatch):
        input_dir = self._input_dir(tmp_path, monkeypatch)
        pdf = input_dir / "test.pdf"
        pdf.write_bytes(b"%PDF small content")
        mock_choice = MagicMock()
        mock_choice.message.content = "answer"
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = MagicMock(choices=[mock_choice])

        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch.object(mistral_converter, "upload_file_for_ocr", return_value="https://signed.url/doc"):
                ok, answer, err = mistral_converter.query_document_file(pdf, "what?")

        assert (ok, answer, err) == (True, "answer", None)


class TestUploadValidationMode:
    """The upload layer must validate under the caller's own mode."""

    def test_qna_mode_accepts_a_file_over_the_ocr_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        monkeypatch.setattr(config, "MISTRAL_OCR_MAX_FILE_SIZE_MB", 1)
        monkeypatch.setattr(config, "MISTRAL_QNA_MAX_FILE_SIZE_MB", 8)
        pdf = tmp_path / "big.pdf"
        pdf.write_bytes(b"%PDF" + b"x" * (2 * 1024 * 1024))

        client = MagicMock()
        client.files.upload.return_value = MagicMock(id="file_qna")
        client.files.get_signed_url.return_value = MagicMock(url="https://signed.example/doc")

        with patch.object(mistral_converter, "_register_uploaded_file", return_value=True):
            refused = mistral_converter._upload_file_for_ocr_pair(client, pdf)
            accepted = mistral_converter._upload_file_for_ocr_pair(client, pdf, mode="qna")

        assert refused is None
        assert accepted == ("https://signed.example/doc", "file_qna")
