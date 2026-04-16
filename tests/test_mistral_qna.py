"""Tests for Mistral Document QnA helpers (query_document, stream,
query_document_file). Split out of test_mistral_converter.py for navigability."""

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

    def test_no_client(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        with patch.object(mistral_converter, "get_mistral_client", return_value=None):
            ok, answer, err = mistral_converter.query_document_file(pdf, "what?")
        assert ok is False

    def test_file_too_large(self, tmp_path):
        cap = config.MISTRAL_QNA_MAX_FILE_SIZE_MB
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"x" * ((cap + 1) * 1024 * 1024))

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            ok, answer, err = mistral_converter.query_document_file(pdf, "what?")
        assert ok is False
        assert "too large" in err.lower()

    def test_upload_failure(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF small file")

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch.object(mistral_converter, "upload_file_for_ocr", return_value=None):
                ok, answer, err = mistral_converter.query_document_file(pdf, "what?")
        assert ok is False
        assert "upload" in err.lower()

    def test_successful_query(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF small content")

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch.object(
                mistral_converter,
                "upload_file_for_ocr",
                return_value="https://signed.url/doc",
            ):
                with patch.object(
                    mistral_converter,
                    "query_document",
                    return_value=(True, "The answer is 42", None),
                ):
                    ok, answer, err = mistral_converter.query_document_file(pdf, "what?")
        assert ok is True
        assert answer == "The answer is 42"


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


# ============================================================================
# query_document_file - additional paths
# ============================================================================


class TestQueryDocumentFileFull:
    """Test file-based QnA additional paths."""

    def test_file_not_readable(self, tmp_path):
        pdf = tmp_path / "nonexistent.pdf"
        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            ok, answer, err = mistral_converter.query_document_file(pdf, "what?")
        assert ok is False

    def test_exception_during_upload(self, tmp_path):
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
