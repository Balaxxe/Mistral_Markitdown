"""Thin on-disk integration tests for MarkItDown / smart routing write paths.

These complement heavily mocked pipeline tests by asserting real output files.
"""

from unittest.mock import MagicMock, patch

import pytest

import config
import local_converter
import main
import utils


@pytest.mark.integration
class TestRealMarkItDownParse:
    """Unmocked MarkItDown conversions: assert substrings only a real parse produces.

    Exact output is not pinned because the installed MarkItDown may differ from the pin.
    """

    @pytest.fixture(autouse=True)
    def _real_markitdown(self, tmp_path, monkeypatch):
        pytest.importorskip("markitdown")
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 100)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path / "md")
        monkeypatch.setattr(config, "OUTPUT_TXT_DIR", tmp_path / "txt")
        monkeypatch.setattr(config, "INCLUDE_METADATA", False)
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", False)
        config.OUTPUT_MD_DIR.mkdir(parents=True, exist_ok=True)

    def test_real_text_file_round_trips_to_markdown(self, tmp_path):
        src = tmp_path / "memo.txt"
        src.write_text("plain text sentinel line\nsecond line\n", encoding="utf-8")

        success, content, error = local_converter.convert_with_markitdown(src)

        assert (success, error) == (True, None)
        assert "plain text sentinel line" in content
        output_path = config.OUTPUT_MD_DIR / "memo.md"
        assert output_path.exists()
        assert "plain text sentinel line" in output_path.read_text(encoding="utf-8")

    def test_real_html_file_becomes_markdown_structure(self, tmp_path):
        src = tmp_path / "report.html"
        src.write_text(
            "<html><body><h1>Quarterly Report</h1>"
            "<p>Revenue rose to <strong>42 units</strong>.</p>"
            "<table><tr><th>Item</th><th>Qty</th></tr>"
            "<tr><td>Widget</td><td>7</td></tr></table>"
            "</body></html>",
            encoding="utf-8",
        )

        success, content, error = local_converter.convert_with_markitdown(src)

        assert (success, error) == (True, None)
        output_path = config.OUTPUT_MD_DIR / "report.md"
        assert output_path.exists()
        written = output_path.read_text(encoding="utf-8")
        assert written == content
        assert "# Quarterly Report" in written
        assert "**42 units**" in written
        assert "| Item | Qty |" in written
        assert "<table>" not in written


class TestMarkItDownWritesOutput:
    def test_convert_writes_markdown_under_output_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 100)
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path / "md")
        monkeypatch.setattr(config, "OUTPUT_TXT_DIR", tmp_path / "txt")
        monkeypatch.setattr(config, "INCLUDE_METADATA", False)
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", False)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        config.OUTPUT_MD_DIR.mkdir(parents=True, exist_ok=True)

        src = tmp_path / "note.txt"
        src.write_text("integration hello world", encoding="utf-8")

        mock_result = MagicMock()
        mock_result.markdown = "integration hello world"
        mock_result.title = "note"
        mock_md = MagicMock()
        mock_md.convert.return_value = mock_result

        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            success, content, error = local_converter.convert_with_markitdown(src)

        assert success is True
        assert error is None
        assert isinstance(content, str)
        assert "integration hello world" in content
        output_path = config.OUTPUT_MD_DIR / "note.md"
        assert output_path.exists()
        assert output_path.parent == config.OUTPUT_MD_DIR
        assert output_path.read_text(encoding="utf-8") == content


class TestSmartRoutingSidecarIsolation:
    def test_table_extraction_exception_does_not_fail_conversion(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "")
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path / "md")
        monkeypatch.setattr(config, "OUTPUT_TXT_DIR", tmp_path / "txt")
        monkeypatch.setattr(config, "INCLUDE_METADATA", False)
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", False)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        config.OUTPUT_MD_DIR.mkdir(parents=True, exist_ok=True)

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%EOF")

        mock_result = MagicMock()
        mock_result.markdown = "body text for smart path"
        mock_result.title = "doc"
        mock_md = MagicMock()
        mock_md.convert.return_value = mock_result

        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            with patch.object(local_converter, "extract_all_tables", side_effect=RuntimeError("boom")):
                success, output_path, error = main._process_single_smart(pdf, use_ocr=False)

        assert success is True
        assert error is None
        assert output_path is not None and output_path.exists()


class TestToConversionResultPathContract:
    def test_markitdown_shaped_path_preserved(self, tmp_path):
        out = tmp_path / "x.md"
        out.write_text("ok", encoding="utf-8")
        result = utils.to_conversion_result((True, out, None))
        assert result.success is True
        assert result.output_path == out
        assert result.error is None
