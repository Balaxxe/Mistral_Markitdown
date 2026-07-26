"""End-to-end admission tests for STRICT_INPUT_PATH_RESOLUTION.

Every public entry point that accepts a caller-supplied path must confine it to
``config.INPUT_DIR`` when the shipped default is in force. These tests run against
the real validator with no relaxation fixture.
"""

from unittest.mock import MagicMock, patch

import pytest

import cli_files
import config
import mistral_converter

config.ensure_directories()


@pytest.fixture
def strict_input_dir(tmp_path, monkeypatch):
    """Confined input directory plus an outside file that must never be admitted."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(config, "STRICT_INPUT_PATH_RESOLUTION", True)
    monkeypatch.setattr(config, "INPUT_DIR", input_dir)
    return input_dir


def _write_pdf(path):
    path.write_bytes(b"%PDF-1.4\n%EOF")
    return path


class TestFilterValidFilesConfinement:
    def test_accepts_inside_and_rejects_outside(self, tmp_path, strict_input_dir):
        inside = _write_pdf(strict_input_dir / "inside.pdf")
        outside = _write_pdf(tmp_path / "outside.pdf")

        assert cli_files.filter_valid_files([inside, outside]) == [inside]

    def test_rejects_symlink_escaping_input_dir(self, tmp_path, strict_input_dir):
        outside = _write_pdf(tmp_path / "secret.pdf")
        link = strict_input_dir / "trap.pdf"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")

        assert cli_files.filter_valid_files([link]) == []

    def test_rejects_parent_traversal(self, tmp_path, strict_input_dir):
        outside = _write_pdf(tmp_path / "outside.pdf")
        traversal = strict_input_dir / ".." / "outside.pdf"

        assert cli_files.filter_valid_files([traversal]) == []
        assert outside.exists()

    def test_documented_opt_out_admits_outside_path(self, tmp_path, strict_input_dir, relax_strict_input_paths):
        outside = _write_pdf(tmp_path / "outside.pdf")

        assert cli_files.filter_valid_files([outside]) == [outside]


class TestBatchCreationConfinement:
    def test_create_batch_ocr_file_refuses_outside_path(self, tmp_path, strict_input_dir):
        outside = _write_pdf(tmp_path / "outside.pdf")

        with patch.object(mistral_converter, "get_mistral_client") as client:
            ok, path, error = mistral_converter.create_batch_ocr_file([outside], strict_input_dir / "batch.jsonl")

        assert (ok, path) == (False, None)
        assert "input directory" in (error or "").lower()
        client.assert_not_called()


class TestOcrConfinement:
    def test_process_with_ocr_refuses_outside_path(self, tmp_path, strict_input_dir):
        outside = _write_pdf(tmp_path / "outside.pdf")
        client = MagicMock()

        ok, result, error = mistral_converter.process_with_ocr(client, outside)

        assert (ok, result) == (False, None)
        assert "input directory" in (error or "").lower()
        client.files.upload.assert_not_called()
        client.ocr.process.assert_not_called()

    def test_process_with_ocr_admits_inside_path(self, strict_input_dir):
        inside = _write_pdf(strict_input_dir / "inside.pdf")
        client = MagicMock()

        with patch.object(mistral_converter, "upload_file_for_ocr", return_value=None):
            ok, result, error = mistral_converter.process_with_ocr(client, inside)

        assert (ok, result) == (False, None)
        assert "input directory" not in (error or "").lower()


class TestQnaConfinement:
    def test_query_document_file_refuses_outside_path(self, tmp_path, strict_input_dir):
        outside = _write_pdf(tmp_path / "outside.pdf")
        client = MagicMock()

        with patch.object(mistral_converter, "get_mistral_client", return_value=client):
            ok, answer, error = mistral_converter.query_document_file(outside, "What is this?")

        assert (ok, answer) == (False, None)
        assert "input directory" in (error or "").lower()
        client.files.upload.assert_not_called()
        client.chat.complete.assert_not_called()
