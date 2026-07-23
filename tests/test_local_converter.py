"""
Tests for local_converter.py module.

Tests cover:
- _fix_merged_currency_cells (dollar pairs, bare numbers, text preservation)
- _fix_split_headers (fragmented header rejoining)
- coalesce_tables (merging tables with identical headers)
- _deduplicate_tables (content-based deduplication)
- analyze_file_content (file type analysis)
- get_markitdown_instance / reset (singleton lifecycle)
- convert_with_markitdown (MarkItDown integration, mocked)
- convert_stream_with_markitdown (stream-based conversion, mocked)
- extract_tables_pdfplumber / extract_tables_pdfplumber_text (mocked)
- extract_all_tables (integration, mocked)
- save_tables_to_files (file output, mocked)
- convert_pdf_to_images (pdf2image integration, mocked)
- Import fallbacks
- Exception handlers
"""

import io
import queue
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

import config

config.ensure_directories()

import local_converter

# ============================================================================
# _fix_merged_currency_cells Tests
# ============================================================================


class TestFixMergedCurrencyCells:
    """Test splitting of merged currency/number cells."""

    def test_splits_dollar_sign_pairs(self):
        table = [["Item", "$ 1,234.56 $ 5,678.90"]]
        result = local_converter._fix_merged_currency_cells(table)
        assert len(result[0]) == 3
        assert "$" in result[0][1]
        assert "$" in result[0][2]

    def test_splits_bare_number_pairs(self):
        table = [["153,990.37 (235,497.83)"]]
        result = local_converter._fix_merged_currency_cells(table)
        assert len(result[0]) == 2
        assert "153,990.37" in result[0][0]
        assert "(235,497.83)" in result[0][1]

    def test_preserves_text_cells(self):
        """Cells containing letters should never be split."""
        table = [["10201 Cash - Operating 1"]]
        result = local_converter._fix_merged_currency_cells(table)
        assert len(result[0]) == 1
        assert result[0][0] == "10201 Cash - Operating 1"

    def test_preserves_single_number_cell(self):
        table = [["1,234.56"]]
        result = local_converter._fix_merged_currency_cells(table)
        assert len(result[0]) == 1
        assert result[0][0] == "1,234.56"

    def test_handles_empty_cells(self):
        table = [["", None, "value"]]
        result = local_converter._fix_merged_currency_cells(table)
        assert result[0][0] == ""
        assert result[0][1] is None
        assert result[0][2] == "value"

    def test_handles_zero_shorthands(self):
        table = [[".00 .00"]]
        result = local_converter._fix_merged_currency_cells(table)
        assert len(result[0]) == 2

    def test_splits_parenthetical_negatives(self):
        table = [["(18,954.54) (31,090.86)"]]
        result = local_converter._fix_merged_currency_cells(table)
        assert len(result[0]) == 2


# ============================================================================
# _fix_split_headers Tests
# ============================================================================


class TestFixSplitHeaders:
    """Test rejoining of split header text."""

    def test_rejoins_fragmented_word(self):
        """'Be' + 'ginning' -> '' + 'Beginning'"""
        table = [["Be", "ginning", "January"]]
        result = local_converter._fix_split_headers(table)
        assert "Beginning" in result[0][1] or "Beginning" in " ".join(result[0])

    def test_rejoins_trailing_fragment(self):
        """'Acct Account Title B' + 'alance' -> 'Acct Account Title' + 'Balance'"""
        table = [["Acct Account Title B", "alance"]]
        result = local_converter._fix_split_headers(table)
        assert "Balance" in result[0][1] or "Balance" in " ".join(result[0])

    def test_leaves_proper_headers_alone(self):
        table = [["Name", "Age", "City"]]
        result = local_converter._fix_split_headers(table)
        assert result[0] == ["Name", "Age", "City"]

    def test_does_not_merge_legitimate_lowercase_columns(self):
        """Legitimate lowercase column names (>= 3 chars) should not be merged."""
        table = [["Account ", "units", "Total"]]
        result = local_converter._fix_split_headers(table)
        # "units" should remain its own column since current cell ends with space
        assert "units" in result[0]

    def test_does_not_merge_long_trailing_fragment(self):
        """Trailing fragment > 2 chars in current cell should not trigger merge."""
        table = [["Some Header", "value", "Other"]]
        result = local_converter._fix_split_headers(table)
        assert result[0][1] == "value"

    def test_only_touches_first_rows(self):
        """Data rows below max_header_rows should not be modified."""
        table = [
            ["Header1", "Header2"],
            ["data1", "data2"],
            ["data3", "data4"],
            ["Da", "ta5"],  # row 3, beyond default max_header_rows=3
        ]
        result = local_converter._fix_split_headers(table, max_header_rows=2)
        assert result[3] == ["Da", "ta5"]

    def test_skips_uppercase_next_cell(self):
        """Next cell starting with uppercase is a real column, not a fragment."""
        table = [["Account", "Balance"]]
        result = local_converter._fix_split_headers(table)
        assert result[0] == ["Account", "Balance"]

    def test_skip_merge_long_lowercase_with_space(self):
        """Lines 437-438: next cell >= 3 chars, current ends with space -> skip."""
        table = [["Revenue ", "total", "Other"]]
        result = local_converter._fix_split_headers(table)
        assert result[0][0] == "Revenue "
        assert result[0][1] == "total"


# ============================================================================
# coalesce_tables Tests
# ============================================================================


class TestCoalesceTables:
    """Test merging tables with identical headers across pages."""

    def test_merges_matching_headers(self):
        t1 = [["Name", "Value"], ["A", "1"], ["B", "2"]]
        t2 = [["Name", "Value"], ["C", "3"], ["D", "4"]]
        result = local_converter.coalesce_tables([t1, t2])
        assert len(result) == 1
        assert len(result[0]) == 5  # header + 4 data rows

    def test_keeps_different_headers_separate(self):
        t1 = [["Name", "Value"], ["A", "1"]]
        t2 = [["ID", "Count"], ["X", "9"]]
        result = local_converter.coalesce_tables([t1, t2])
        assert len(result) == 2

    def test_handles_empty_list(self):
        result = local_converter.coalesce_tables([])
        assert result == []

    def test_handles_single_table(self):
        t1 = [["H1", "H2"], ["a", "b"]]
        result = local_converter.coalesce_tables([t1])
        assert len(result) == 1
        assert result[0] == t1

    def test_three_way_merge(self):
        header = ["Col1", "Col2"]
        t1 = [header, ["a", "1"]]
        t2 = [header, ["b", "2"]]
        t3 = [header, ["c", "3"]]
        result = local_converter.coalesce_tables([t1, t2, t3])
        assert len(result) == 1
        assert len(result[0]) == 4  # header + 3 data rows


# ============================================================================
# _deduplicate_tables Tests
# ============================================================================


class TestDeduplicateTables:
    """Test content-based table deduplication."""

    def test_removes_identical_tables(self):
        t1 = [["A", "B"], ["1", "2"]]
        t2 = [["A", "B"], ["1", "2"]]
        result = local_converter._deduplicate_tables([t1, t2])
        assert len(result) == 1

    def test_keeps_different_tables(self):
        t1 = [["A", "B"], ["1", "2"]]
        t2 = [["A", "B"], ["3", "4"]]
        result = local_converter._deduplicate_tables([t1, t2])
        assert len(result) == 2

    def test_skips_empty_tables(self):
        result = local_converter._deduplicate_tables([[], [["A"]], []])
        assert len(result) == 1

    def test_handles_empty_input(self):
        result = local_converter._deduplicate_tables([])
        assert result == []


# ============================================================================
# analyze_file_content Tests
# ============================================================================


class TestAnalyzeFileContent:
    """Test file content analysis."""

    def test_image_file_analysis(self, tmp_path):
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        result = local_converter.analyze_file_content(img_file)
        assert result["file_type"] == "png"
        assert result["has_images"] is True

    def test_non_pdf_no_pages(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello world")
        result = local_converter.analyze_file_content(txt_file)
        assert result["page_count"] == 0
        assert result["file_type"] == "txt"

    def test_large_file_is_complex(self, tmp_path):
        large_file = tmp_path / "big.docx"
        large_file.write_bytes(b"\x00" * (11 * 1024 * 1024))
        result = local_converter.analyze_file_content(large_file)
        assert result["is_complex"] is True

    def test_file_size_calculated(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("a,b,c\n1,2,3\n")
        result = local_converter.analyze_file_content(f)
        assert result["file_size_mb"] > 0


# ============================================================================
# MarkItDown Singleton Lifecycle Tests
# ============================================================================


class TestMarkItDownSingleton:
    """Test get/reset of MarkItDown instance."""

    def test_reset_allows_reinit(self):
        local_converter.reset_markitdown_instance()
        # After reset, the sentinel is restored
        assert local_converter._markitdown_instance is local_converter._MARKITDOWN_UNSET

    def test_get_instance_returns_value(self):
        local_converter.reset_markitdown_instance()
        instance = local_converter.get_markitdown_instance()
        # It should return either a MarkItDown instance or None (if not installed)
        # but not the sentinel
        assert instance is not local_converter._MARKITDOWN_UNSET

    def test_cached_on_second_call(self):
        local_converter.reset_markitdown_instance()
        first = local_converter.get_markitdown_instance()
        second = local_converter.get_markitdown_instance()
        assert first is second

    def test_thread_local_isolation(self):
        local_converter.reset_markitdown_instance()

        instances = []

        def worker():
            inst = local_converter.get_markitdown_instance()
            instances.append(inst)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        if instances[0] is not None and instances[1] is not None:
            assert instances[0] is not instances[1]

    def test_reset_invalidates_other_thread_cache(self):
        local_converter.reset_markitdown_instance()

        class FakeMarkItDown:
            created = 0

            def __init__(self, **kwargs):
                type(self).created += 1
                self.created = type(self).created

        commands = queue.Queue()
        responses = queue.Queue()

        def worker():
            while True:
                command = commands.get(timeout=5)
                if command == "stop":
                    return
                responses.put(local_converter.get_markitdown_instance())

        with patch.object(local_converter, "MarkItDown", FakeMarkItDown):
            worker_thread = threading.Thread(target=worker)
            worker_thread.start()
            try:
                commands.put("get")
                first = responses.get(timeout=5)

                local_converter.reset_markitdown_instance()

                commands.put("get")
                second = responses.get(timeout=5)
            finally:
                commands.put("stop")
                worker_thread.join(timeout=5)
                local_converter.reset_markitdown_instance()

        assert first is not second
        assert first.created == 1
        assert second.created == 2


# ============================================================================
# convert_with_markitdown Tests (mocked)
# ============================================================================


class TestConvertWithMarkItDown:
    """Test convert_with_markitdown with mocked MarkItDown."""

    def test_file_too_large(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 1)
        big_file = tmp_path / "big.pdf"
        big_file.write_bytes(b"\x00" * (2 * 1024 * 1024))
        success, content, error = local_converter.convert_with_markitdown(big_file)
        assert success is False
        assert "too large" in error

    def test_markitdown_not_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 100)
        small_file = tmp_path / "test.txt"
        small_file.write_text("content")
        # Force get_markitdown_instance to return None
        with patch.object(local_converter, "get_markitdown_instance", return_value=None):
            success, content, error = local_converter.convert_with_markitdown(small_file)
        assert success is False
        assert "not available" in error

    def test_successful_conversion(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 100)
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "INCLUDE_METADATA", False)
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", False)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)

        test_file = tmp_path / "doc.txt"
        test_file.write_text("hello world")

        mock_result = MagicMock()
        mock_result.markdown = "# Converted\n\nhello world"
        mock_result.title = "doc"

        mock_md = MagicMock()
        mock_md.convert.return_value = mock_result

        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            success, content, error = local_converter.convert_with_markitdown(test_file)

        assert success is True
        assert content is not None
        assert "hello world" in content
        assert (tmp_path / "doc.md").read_text(encoding="utf-8") == content
        assert error is None

    def test_conversion_exception(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 100)
        test_file = tmp_path / "doc.txt"
        test_file.write_text("hello")

        mock_md = MagicMock()
        mock_md.convert.side_effect = Exception("conversion error")

        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            success, content, error = local_converter.convert_with_markitdown(test_file)

        assert success is False
        assert "conversion error" in error

    def test_low_quality_warning_for_image(self, tmp_path, monkeypatch, caplog):
        """Image files with minimal text output trigger a warning."""
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 100)
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "INCLUDE_METADATA", False)
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", False)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)

        test_file = tmp_path / "scan.png"
        test_file.write_bytes(b"\x89PNG")

        mock_result = MagicMock()
        mock_result.markdown = ""  # No text extracted
        mock_result.title = "scan"

        mock_md = MagicMock()
        mock_md.convert.return_value = mock_result

        import logging

        with caplog.at_level(logging.WARNING):
            with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
                success, content, error = local_converter.convert_with_markitdown(test_file)

        assert success is True
        assert any("no meaningful text" in r.message.lower() for r in caplog.records)

    def test_no_low_quality_warning_for_text_file(self, tmp_path, monkeypatch, caplog):
        """Short text files do NOT trigger the low-quality warning."""
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 100)
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "INCLUDE_METADATA", False)
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", False)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)

        test_file = tmp_path / "note.txt"
        test_file.write_text("hi")

        mock_result = MagicMock()
        mock_result.markdown = "hi"
        mock_result.title = "note"

        mock_md = MagicMock()
        mock_md.convert.return_value = mock_result

        import logging

        with caplog.at_level(logging.WARNING):
            with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
                local_converter.convert_with_markitdown(test_file)

        assert not any("no meaningful text" in r.message.lower() for r in caplog.records)


# ============================================================================
# convert_stream_with_markitdown Tests
# ============================================================================


class TestConvertStreamWithMarkItDown:
    """Test stream-based conversion."""

    def test_markitdown_not_available(self):
        import io

        stream = io.BytesIO(b"data")
        with patch.object(local_converter, "get_markitdown_instance", return_value=None):
            success, content, error = local_converter.convert_stream_with_markitdown(stream)
        assert success is False
        assert "not available" in error

    def test_convert_stream_not_supported(self):
        import io

        stream = io.BytesIO(b"data")
        mock_md = MagicMock(spec=[])  # no convert_stream attribute
        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            success, content, error = local_converter.convert_stream_with_markitdown(stream)
        assert success is False
        assert "not available" in error

    def test_successful_stream_conversion(self):
        import io

        stream = io.BytesIO(b"data")
        mock_result = MagicMock()
        mock_result.markdown = "# Stream Content"

        mock_md = MagicMock()
        mock_md.convert_stream.return_value = mock_result

        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            success, content, error = local_converter.convert_stream_with_markitdown(stream, "test.txt")

        assert success is True
        assert "Stream Content" in content


# ============================================================================
# extract_tables_pdfplumber Tests (mocked)
# ============================================================================


class TestExtractTablesPdfplumber:
    """Test pdfplumber table extraction with mocks."""

    def test_returns_empty_when_not_installed(self, tmp_path):
        with patch.object(local_converter, "pdfplumber", None):
            result = local_converter.extract_tables_pdfplumber(tmp_path / "test.pdf")
        assert result == []

    def test_extracts_tables(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_page = MagicMock()
        mock_page.extract_tables.return_value = [[["H1", "H2"], ["A", "B"]]]

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch.object(local_converter, "pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = mock_pdf
            result = local_converter.extract_tables_pdfplumber(pdf_file)

        assert len(result) == 1
        assert result[0] == [["H1", "H2"], ["A", "B"]]

    def test_handles_exception(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        with patch.object(local_converter, "pdfplumber") as mock_plumber:
            mock_plumber.open.side_effect = Exception("corrupt PDF")
            result = local_converter.extract_tables_pdfplumber(pdf_file)

        assert result == []


# ============================================================================
# extract_tables_pdfplumber_text Tests (mocked)
# ============================================================================


class TestExtractTablesPdfplumberText:
    """Test pdfplumber text-strategy table extraction with mocks."""

    def test_returns_empty_when_not_installed(self, tmp_path):
        with patch.object(local_converter, "pdfplumber", None):
            result = local_converter.extract_tables_pdfplumber_text(tmp_path / "test.pdf")
        assert result == []

    def test_handles_exception(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [MagicMock()]
        mock_pdf.pages[0].extract_tables.side_effect = Exception("parse error")

        with patch.object(local_converter.pdfplumber, "open", return_value=mock_pdf):
            result = local_converter.extract_tables_pdfplumber_text(pdf_file)

        assert result == []

    def test_extracts_tables_with_text_strategy(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_page = MagicMock()
        mock_page.extract_tables.return_value = [[["A", "B"], ["1", "2"]]]

        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]

        with patch.object(local_converter.pdfplumber, "open", return_value=mock_pdf):
            result = local_converter.extract_tables_pdfplumber_text(pdf_file)

        assert len(result) == 1
        assert result[0] == [["A", "B"], ["1", "2"]]
        mock_page.extract_tables.assert_called_once_with({"vertical_strategy": "text", "horizontal_strategy": "text"})


# ============================================================================
# save_tables_to_files Tests
# ============================================================================


class TestSaveTablesToFiles:
    """Test table file saving."""

    def test_saves_markdown_and_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "INCLUDE_METADATA", False)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        monkeypatch.setattr(config, "TABLE_OUTPUT_FORMATS", ["markdown"])

        pdf_file = tmp_path / "report.pdf"
        pdf_file.touch()

        tables = [[["Name", "Value"], ["A", "1"], ["B", "2"]]]
        result = local_converter.save_tables_to_files(pdf_file, tables)

        assert len(result) > 0
        # Check markdown file exists
        md_files = list(tmp_path.glob("*.md"))
        assert len(md_files) > 0

    def test_empty_tables_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)

        pdf_file = tmp_path / "report.pdf"
        pdf_file.touch()

        result = local_converter.save_tables_to_files(pdf_file, [])
        assert result == []


# ============================================================================
# convert_pdf_to_images Tests (mocked)
# ============================================================================


class TestConvertPdfToImages:
    """Test PDF to images conversion with mocks."""

    @pytest.fixture(autouse=True)
    def _disable_page_limit_for_legacy_mock_pdfs(self, monkeypatch):
        monkeypatch.setattr(config, "PDF_IMAGE_MAX_PAGES", 0)

    def test_not_installed(self, tmp_path):
        with patch.object(local_converter, "convert_from_path", None):
            success, paths, error = local_converter.convert_pdf_to_images(tmp_path / "test.pdf")
        assert success is False
        assert "not installed" in error.lower() or "not available" in error.lower()

    def test_successful_conversion(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "test_pages"
        monkeypatch.setattr(config, "OUTPUT_IMAGES_DIR", tmp_path)
        monkeypatch.setattr(config, "PDF_IMAGE_DPI", 200)
        monkeypatch.setattr(config, "PDF_IMAGE_FORMAT", "png")
        monkeypatch.setattr(config, "POPPLER_PATH", "")
        monkeypatch.setattr(config, "PDF_IMAGE_THREAD_COUNT", 1)
        monkeypatch.setattr(config, "PDF_IMAGE_USE_PDFTOCAIRO", False)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        from pathlib import Path

        output_dir.mkdir(parents=True, exist_ok=True)
        temp_file1 = str(output_dir / "page-1.png")
        temp_file2 = str(output_dir / "page-2.png")
        Path(temp_file1).touch()
        Path(temp_file2).touch()

        with patch.object(local_converter, "convert_from_path", return_value=[temp_file1, temp_file2]) as mock_convert:
            success, paths, error = local_converter.convert_pdf_to_images(pdf_file, output_dir=output_dir)

        assert success is True
        assert len(paths) == 2
        assert paths[0] == output_dir / "page_001.png"
        assert paths[1] == output_dir / "page_002.png"
        assert (output_dir / "page_001.png").exists()
        assert (output_dir / "page_002.png").exists()

        mock_convert.assert_called_once()
        kwargs = mock_convert.call_args[1]
        assert kwargs["paths_only"] is True

    def test_unknown_page_count_fails_closed_when_limit_enabled(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "unknown_pages"
        pdf_file = tmp_path / "unknown.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(config, "PDF_IMAGE_MAX_PAGES", 5)

        with patch.object(local_converter, "analyze_file_content", return_value={"page_count": 0}):
            with patch.object(local_converter, "convert_from_path") as mock_convert:
                success, paths, error = local_converter.convert_pdf_to_images(pdf_file, output_dir=output_dir)

        assert success is False
        assert paths == []
        assert error is not None and "Unable to determine PDF page count" in error
        mock_convert.assert_not_called()

    def test_known_page_count_renders_without_last_page_truncation(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "known_pages"
        output_dir.mkdir()
        pdf_file = tmp_path / "known.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        rendered = output_dir / "page-1.png"
        rendered.touch()
        monkeypatch.setattr(config, "PDF_IMAGE_MAX_PAGES", 5)
        monkeypatch.setattr(config, "PDF_IMAGE_FORMAT", "png")
        monkeypatch.setattr(config, "PDF_IMAGE_DPI", 200)
        monkeypatch.setattr(config, "PDF_IMAGE_THREAD_COUNT", 1)
        monkeypatch.setattr(config, "PDF_IMAGE_USE_PDFTOCAIRO", False)
        monkeypatch.setattr(config, "POPPLER_PATH", "")

        with patch.object(local_converter, "analyze_file_content", return_value={"page_count": 1}):
            with patch.object(local_converter, "convert_from_path", return_value=[str(rendered)]) as mock_convert:
                success, paths, error = local_converter.convert_pdf_to_images(pdf_file, output_dir=output_dir)

        assert success is True
        assert error is None
        assert paths == [output_dir / "page_001.png"]
        assert "last_page" not in mock_convert.call_args.kwargs

    def test_handles_conversion_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_IMAGES_DIR", tmp_path)
        monkeypatch.setattr(config, "PDF_IMAGE_DPI", 200)
        monkeypatch.setattr(config, "PDF_IMAGE_FORMAT", "png")
        monkeypatch.setattr(config, "POPPLER_PATH", "")
        monkeypatch.setattr(config, "PDF_IMAGE_THREAD_COUNT", 1)
        monkeypatch.setattr(config, "PDF_IMAGE_USE_PDFTOCAIRO", False)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        with patch.object(
            local_converter,
            "convert_from_path",
            side_effect=Exception("poppler not found"),
        ):
            success, paths, error = local_converter.convert_pdf_to_images(pdf_file)

        assert success is False
        assert "poppler" in error.lower()


# ============================================================================
# extract_all_tables Tests (mocked)
# ============================================================================


class TestExtractAllTables:
    """Test combined table extraction."""

    def test_combines_pdfplumber_and_text_strategy(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        table1 = [["A", "B"], ["1", "2"]]
        table2 = [["C", "D"], ["3", "4"]]

        with patch.object(local_converter, "extract_tables_pdfplumber", return_value=[table1]):
            with patch.object(local_converter, "extract_tables_pdfplumber_text", return_value=[table2]):
                result = local_converter.extract_all_tables(pdf_file)

        assert result["table_count"] >= 1
        assert "pdfplumber" in result["methods_used"]

    def test_all_methods_fail_returns_empty(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        with patch.object(local_converter, "extract_tables_pdfplumber", return_value=[]):
            with patch.object(local_converter, "extract_tables_pdfplumber_text", return_value=[]):
                result = local_converter.extract_all_tables(pdf_file)

        assert result["table_count"] == 0
        assert result["tables"] == []


# ============================================================================
# Import Fallback Tests (Lines 42-62)
# ============================================================================


class TestImportFallbacks:
    """Lines 42-62: Import fallback branches when optional deps are missing."""

    def test_markitdown_import_failure(self):
        """Lines 42-47: MarkItDown import failure sets all to None."""
        import importlib

        original_markitdown = sys.modules.get("markitdown")
        try:
            sys.modules["markitdown"] = None
            importlib.reload(local_converter)
            # After reload with failed import, MarkItDown should be None
            assert local_converter.MarkItDown is None
        finally:
            if original_markitdown is not None:
                sys.modules["markitdown"] = original_markitdown
            else:
                sys.modules.pop("markitdown", None)
            importlib.reload(local_converter)

    def test_pdfplumber_import_failure(self):
        """Lines 51-52: pdfplumber import failure."""
        import importlib

        original = sys.modules.get("pdfplumber")
        try:
            sys.modules["pdfplumber"] = None
            importlib.reload(local_converter)
            assert local_converter.pdfplumber is None
        finally:
            if original is not None:
                sys.modules["pdfplumber"] = original
            else:
                sys.modules.pop("pdfplumber", None)
            importlib.reload(local_converter)

    def test_pdf2image_import_failure(self):
        """Lines 61-62: pdf2image import failure."""
        import importlib

        original = sys.modules.get("pdf2image")
        try:
            sys.modules["pdf2image"] = None
            importlib.reload(local_converter)
            assert local_converter.convert_from_path is None
        finally:
            if original is not None:
                sys.modules["pdf2image"] = original
            else:
                sys.modules.pop("pdf2image", None)
            importlib.reload(local_converter)


# ============================================================================
# get_markitdown_instance Branch Tests (Lines 95-136)
# ============================================================================


class TestGetMarkItDownInstanceBranches:
    """Test all branches of get_markitdown_instance()."""

    def test_markitdown_none_returns_none(self):
        """Line 95: MarkItDown is None."""
        local_converter.reset_markitdown_instance()
        with patch.object(local_converter, "MarkItDown", None):
            result = local_converter.get_markitdown_instance()
        assert result is None
        local_converter.reset_markitdown_instance()

    def test_llm_descriptions_enabled(self, monkeypatch):
        """Lines 109, 112-119: LLM client initialization."""
        local_converter.reset_markitdown_instance()
        monkeypatch.setattr(config, "MARKITDOWN_ENABLE_LLM_DESCRIPTIONS", True)
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "test-key")
        monkeypatch.setattr(config, "MARKITDOWN_LLM_MODEL", "test-model")

        mock_md_class = MagicMock()
        mock_openai = MagicMock()

        with patch.object(local_converter, "MarkItDown", mock_md_class):
            with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai)}):
                local_converter.get_markitdown_instance()

        assert mock_md_class.called
        local_converter.reset_markitdown_instance()

    def test_llm_openai_not_installed(self, monkeypatch):
        """Lines 118-119: OpenAI import failure."""
        local_converter.reset_markitdown_instance()
        monkeypatch.setattr(config, "MARKITDOWN_ENABLE_LLM_DESCRIPTIONS", True)
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "test-key")

        mock_md_class = MagicMock()

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("no openai")
            return original_import(name, *args, **kwargs)

        import builtins

        original_import = builtins.__import__

        with patch.object(local_converter, "MarkItDown", mock_md_class):
            with patch("builtins.__import__", side_effect=fake_import):
                local_converter.get_markitdown_instance()

        local_converter.reset_markitdown_instance()

    def test_style_map_configured(self, monkeypatch):
        """Line 122: style_map configuration."""
        local_converter.reset_markitdown_instance()
        monkeypatch.setattr(config, "MARKITDOWN_STYLE_MAP", "p.custom => custom")
        monkeypatch.setattr(config, "MARKITDOWN_ENABLE_LLM_DESCRIPTIONS", False)

        mock_md_class = MagicMock()

        with patch.object(local_converter, "MarkItDown", mock_md_class):
            local_converter.get_markitdown_instance()

        call_kwargs = mock_md_class.call_args[1]
        assert call_kwargs.get("style_map") == "p.custom => custom"
        local_converter.reset_markitdown_instance()

    def test_exiftool_path_configured(self, monkeypatch):
        """Line 125: exiftool_path configuration."""
        local_converter.reset_markitdown_instance()
        monkeypatch.setattr(config, "MARKITDOWN_EXIFTOOL_PATH", "/usr/bin/exiftool")
        monkeypatch.setattr(config, "MARKITDOWN_ENABLE_LLM_DESCRIPTIONS", False)
        monkeypatch.setattr(config, "MARKITDOWN_STYLE_MAP", "")

        mock_md_class = MagicMock()

        with patch.object(local_converter, "MarkItDown", mock_md_class):
            local_converter.get_markitdown_instance()

        call_kwargs = mock_md_class.call_args[1]
        assert call_kwargs.get("exiftool_path") == "/usr/bin/exiftool"
        local_converter.reset_markitdown_instance()

    def test_llm_prompt_configured(self, monkeypatch):
        """Line 128 (already likely covered path) with explicit LLM prompt."""
        local_converter.reset_markitdown_instance()
        monkeypatch.setattr(config, "MARKITDOWN_LLM_PROMPT", "Describe this image")
        monkeypatch.setattr(config, "MARKITDOWN_ENABLE_LLM_DESCRIPTIONS", False)
        monkeypatch.setattr(config, "MARKITDOWN_STYLE_MAP", "")
        monkeypatch.setattr(config, "MARKITDOWN_EXIFTOOL_PATH", "")

        mock_md_class = MagicMock()

        with patch.object(local_converter, "MarkItDown", mock_md_class):
            local_converter.get_markitdown_instance()

        call_kwargs = mock_md_class.call_args[1]
        assert call_kwargs.get("llm_prompt") == "Describe this image"
        local_converter.reset_markitdown_instance()

    def test_keep_data_uris(self, monkeypatch):
        """Line 109: MARKITDOWN_KEEP_DATA_URIS branch."""
        local_converter.reset_markitdown_instance()
        monkeypatch.setattr(config, "MARKITDOWN_KEEP_DATA_URIS", True)
        monkeypatch.setattr(config, "MARKITDOWN_ENABLE_LLM_DESCRIPTIONS", False)
        monkeypatch.setattr(config, "MARKITDOWN_STYLE_MAP", "")
        monkeypatch.setattr(config, "MARKITDOWN_EXIFTOOL_PATH", "")
        monkeypatch.setattr(config, "MARKITDOWN_LLM_PROMPT", "")

        mock_md_class = MagicMock()

        with patch.object(local_converter, "MarkItDown", mock_md_class):
            local_converter.get_markitdown_instance()

        call_kwargs = mock_md_class.call_args[1]
        assert call_kwargs.get("keep_data_uris") is True
        local_converter.reset_markitdown_instance()

    def test_init_exception(self, monkeypatch):
        """Lines 133-136: MarkItDown() constructor raises."""
        local_converter.reset_markitdown_instance()
        monkeypatch.setattr(config, "MARKITDOWN_ENABLE_LLM_DESCRIPTIONS", False)
        monkeypatch.setattr(config, "MARKITDOWN_STYLE_MAP", "")
        monkeypatch.setattr(config, "MARKITDOWN_EXIFTOOL_PATH", "")
        monkeypatch.setattr(config, "MARKITDOWN_LLM_PROMPT", "")

        mock_md_class = MagicMock(side_effect=RuntimeError("init fail"))

        with patch.object(local_converter, "MarkItDown", mock_md_class):
            result = local_converter.get_markitdown_instance()

        assert result is None
        local_converter.reset_markitdown_instance()

    def test_concurrent_lock_returns_cached(self):
        """Line 95: second call inside lock finds instance already set by first call."""
        local_converter.reset_markitdown_instance()

        results = [None, None]

        mock_md = MagicMock()
        MagicMock(return_value=mock_md)

        # Both threads will see _MARKITDOWN_UNSET on fast-path, both enter lock.
        # Thread 1 gets lock first, does init. Thread 2 gets lock second, hits line 95.
        barrier = threading.Barrier(2, timeout=5)

        def slow_markitdown(**kwargs):
            """Simulate slow init to ensure both threads queue up."""
            barrier.wait()  # Wait for both threads to start
            import time

            time.sleep(0.05)
            return mock_md

        with patch.object(local_converter, "MarkItDown", side_effect=slow_markitdown):

            def worker(idx):
                results[idx] = local_converter.get_markitdown_instance()

            t1 = threading.Thread(target=worker, args=(0,))
            t2 = threading.Thread(target=worker, args=(1,))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

        # Both should return the same instance
        assert results[0] is results[1]
        local_converter.reset_markitdown_instance()


# ============================================================================
# convert_with_markitdown Exception Handler Tests (Lines 210-220)
# ============================================================================


class TestConvertWithMarkItDownExceptions:
    """Lines 210-220: exception handlers for specific exception types."""

    def _setup_convert(self, tmp_path, monkeypatch, exception):
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 100)
        test_file = tmp_path / "doc.txt"
        test_file.write_text("hello")
        mock_md = MagicMock()
        mock_md.convert.side_effect = exception
        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            return local_converter.convert_with_markitdown(test_file)

    def test_unsupported_format_exception(self, tmp_path, monkeypatch):
        """Line 210: UnsupportedFormatException handler."""
        if local_converter.UnsupportedFormatException is None:
            pytest.skip("UnsupportedFormatException not available")
        exc = local_converter.UnsupportedFormatException("bad format")
        success, content, error = self._setup_convert(tmp_path, monkeypatch, exc)
        assert success is False
        assert "Unsupported format" in error

    def test_missing_dependency_exception(self, tmp_path, monkeypatch):
        """Lines 213-214: MissingDependencyException handler."""
        if local_converter.MissingDependencyException is None:
            pytest.skip("MissingDependencyException not available")
        exc = local_converter.MissingDependencyException("need dep")
        success, content, error = self._setup_convert(tmp_path, monkeypatch, exc)
        assert success is False
        assert "Missing dependency" in error

    def test_file_conversion_exception(self, tmp_path, monkeypatch):
        """Lines 216-217: FileConversionException handler."""
        if local_converter.FileConversionException is None:
            pytest.skip("FileConversionException not available")
        exc = local_converter.FileConversionException("convert fail")
        success, content, error = self._setup_convert(tmp_path, monkeypatch, exc)
        assert success is False
        assert "Conversion failed" in error

    def test_no_content_returned(self, tmp_path, monkeypatch):
        """Lines 219-220: no content returned from MarkItDown."""
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 100)
        test_file = tmp_path / "doc.txt"
        test_file.write_text("hello")
        mock_md = MagicMock()
        mock_md.convert.return_value = None  # No result
        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            success, content, error = local_converter.convert_with_markitdown(test_file)
        assert success is False
        assert "No content" in error


# ============================================================================
# convert_stream_with_markitdown Exception Tests (Lines 261-276)
# ============================================================================


class TestConvertStreamExceptions:
    """Lines 261-276: stream conversion exception handlers."""

    def test_unsupported_format_stream(self):
        """Line 261: UnsupportedFormatException in stream."""
        if local_converter.UnsupportedFormatException is None:
            pytest.skip("UnsupportedFormatException not available")
        mock_md = MagicMock()
        mock_md.convert_stream.side_effect = local_converter.UnsupportedFormatException("bad")
        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            success, content, error = local_converter.convert_stream_with_markitdown(io.BytesIO(b"data"))
        assert success is False
        assert "Unsupported format" in error

    def test_missing_dependency_stream(self):
        """Lines 266-267: MissingDependencyException in stream."""
        if local_converter.MissingDependencyException is None:
            pytest.skip("MissingDependencyException not available")
        mock_md = MagicMock()
        mock_md.convert_stream.side_effect = local_converter.MissingDependencyException("need dep")
        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            success, content, error = local_converter.convert_stream_with_markitdown(io.BytesIO(b"data"))
        assert success is False
        assert "Missing dependency" in error

    def test_file_conversion_exception_stream(self):
        """FileConversionException in stream matches path-based conversion."""
        if local_converter.FileConversionException is None:
            pytest.skip("FileConversionException not available")
        mock_md = MagicMock()
        mock_md.convert_stream.side_effect = local_converter.FileConversionException("conv fail")
        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            success, content, error = local_converter.convert_stream_with_markitdown(io.BytesIO(b"data"))
        assert success is False
        assert "Conversion failed" in error

    def test_generic_exception_stream(self):
        """Lines 268-276: generic exception in stream."""
        mock_md = MagicMock()
        mock_md.convert_stream.side_effect = RuntimeError("stream fail")
        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            success, content, error = local_converter.convert_stream_with_markitdown(io.BytesIO(b"data"))
        assert success is False
        assert "stream fail" in error

    def test_no_content_from_stream(self):
        """Stream returns no result object."""
        mock_md = MagicMock()
        mock_md.convert_stream.return_value = None
        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            success, content, error = local_converter.convert_stream_with_markitdown(io.BytesIO(b"data"))
        assert success is False

    def test_no_convert_stream_method(self):
        """Stream conversion when convert_stream not available."""
        mock_md = MagicMock(spec=[])  # no convert_stream attribute
        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            success, content, error = local_converter.convert_stream_with_markitdown(io.BytesIO(b"data"))
        assert success is False
        assert "not available" in error

    def test_stream_without_streaminfo(self):
        """Line 261: fallback when StreamInfo is None."""
        mock_md = MagicMock()
        mock_result = MagicMock()
        mock_result.markdown = "# Stream Content"
        mock_md.convert_stream.return_value = mock_result

        with patch.object(local_converter, "get_markitdown_instance", return_value=mock_md):
            with patch.object(local_converter, "StreamInfo", None):
                success, content, error = local_converter.convert_stream_with_markitdown(
                    io.BytesIO(b"data"), filename="test.pdf"
                )
        assert success is True
        assert "Stream Content" in content


# ============================================================================
# extract_all_tables Text Strategy Fallback
# ============================================================================


class TestExtractAllTablesTextFallback:
    """Text strategy runs when <2 tables found by line-based extraction."""

    def test_text_strategy_runs_when_few_tables(self, tmp_path):
        """When pdfplumber finds < 2 tables, text strategy is tried."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        table1 = [["A", "B"], ["1", "2"]]
        text_table = [["X", "Y"], ["3", "4"]]

        with patch.object(local_converter, "extract_tables_pdfplumber", return_value=[table1]):
            with patch.object(
                local_converter,
                "extract_tables_pdfplumber_text",
                return_value=[text_table],
            ):
                result = local_converter.extract_all_tables(pdf_file)

        assert "pdfplumber-text" in result["methods_used"]


# ============================================================================
# save_tables_to_files CSV Output Tests (Lines 583-601)
# ============================================================================


class TestSaveTablesToCsv:
    """Lines 583-584, 601: CSV output in save_tables_to_files."""

    def test_csv_output_created(self, tmp_path, monkeypatch):
        """CSV files are created when 'csv' in TABLE_OUTPUT_FORMATS."""
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "INCLUDE_METADATA", False)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", False)
        monkeypatch.setattr(config, "TABLE_OUTPUT_FORMATS", ["markdown", "csv"])

        pdf_file = tmp_path / "report.pdf"
        pdf_file.touch()

        tables = [[["Name", "Value"], ["A", "1"]]]
        local_converter.save_tables_to_files(pdf_file, tables)

        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 1

    def test_csv_write_exception(self, tmp_path, monkeypatch):
        """Line 601: exception during CSV write."""
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "INCLUDE_METADATA", False)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", False)
        monkeypatch.setattr(config, "TABLE_OUTPUT_FORMATS", ["csv"])

        pdf_file = tmp_path / "report.pdf"
        pdf_file.touch()

        tables = [[["Name", "Value"], ["A", "1"]]]

        with patch("builtins.open", side_effect=[MagicMock(), PermissionError("denied")]):
            # First open for markdown, second for CSV raises
            local_converter.save_tables_to_files(pdf_file, tables)
        # Should not crash


# ============================================================================
# convert_pdf_to_images PNG Format Tests (Line 688)
# ============================================================================


class TestConvertPdfToImagesPng:
    """Line 688: PNG format save branch."""

    @pytest.fixture(autouse=True)
    def _disable_page_limit_for_legacy_mock_pdfs(self, monkeypatch):
        monkeypatch.setattr(config, "PDF_IMAGE_MAX_PAGES", 0)

    def test_png_format(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "png_pages"
        monkeypatch.setattr(config, "OUTPUT_IMAGES_DIR", tmp_path)
        monkeypatch.setattr(config, "PDF_IMAGE_DPI", 200)
        monkeypatch.setattr(config, "PDF_IMAGE_FORMAT", "png")
        monkeypatch.setattr(config, "POPPLER_PATH", "")
        monkeypatch.setattr(config, "PDF_IMAGE_THREAD_COUNT", 1)
        monkeypatch.setattr(config, "PDF_IMAGE_USE_PDFTOCAIRO", False)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        from pathlib import Path

        output_dir.mkdir(parents=True, exist_ok=True)
        temp_file = str(output_dir / "page-1.png")
        Path(temp_file).touch()

        with patch.object(local_converter, "convert_from_path", return_value=[temp_file]) as mock_convert:
            success, paths, error = local_converter.convert_pdf_to_images(pdf_file, output_dir=output_dir)

        assert success is True
        assert paths[0] == output_dir / "page_001.png"
        mock_convert.assert_called_once()
        kwargs = mock_convert.call_args[1]
        assert kwargs["paths_only"] is True
        assert kwargs["fmt"] == "png"

    def test_jpeg_format(self, tmp_path, monkeypatch):
        """Verify JPEG format branch for completeness."""
        output_dir = tmp_path / "jpeg_pages"
        monkeypatch.setattr(config, "OUTPUT_IMAGES_DIR", tmp_path)
        monkeypatch.setattr(config, "PDF_IMAGE_DPI", 200)
        monkeypatch.setattr(config, "PDF_IMAGE_FORMAT", "jpeg")
        monkeypatch.setattr(config, "POPPLER_PATH", "")
        monkeypatch.setattr(config, "PDF_IMAGE_THREAD_COUNT", 1)
        monkeypatch.setattr(config, "PDF_IMAGE_USE_PDFTOCAIRO", False)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        from pathlib import Path

        output_dir.mkdir(parents=True, exist_ok=True)
        temp_file = str(output_dir / "page-1.jpg")
        Path(temp_file).touch()

        with patch.object(local_converter, "convert_from_path", return_value=[temp_file]) as mock_convert:
            success, paths, error = local_converter.convert_pdf_to_images(pdf_file, output_dir=output_dir)

        assert success is True
        assert paths[0] == output_dir / "page_001.jpg"
        mock_convert.assert_called_once()
        kwargs = mock_convert.call_args[1]
        assert kwargs["paths_only"] is True
        assert kwargs["fmt"] == "jpeg"


# ============================================================================
# coalesce_tables Initialization Tests (Lines 724-725)
# ============================================================================


class TestCoalesceTablesInit:
    """Lines 724-725: coalesce_tables with actual data to cover init variables."""

    def test_single_table_coalesced(self):
        """Covers current_table/current_header init + final append."""
        tables = [[["H1", "H2"], ["A", "B"]]]
        result = local_converter.coalesce_tables(tables)
        assert len(result) == 1
        assert result[0] == [["H1", "H2"], ["A", "B"]]

    def test_empty_table_skipped(self):
        """Line 833: empty table is skipped via continue."""
        tables = [
            [["H1", "H2"], ["A", "B"]],
            [],  # empty table
            [["H1", "H2"], ["C", "D"]],  # same header, should coalesce
        ]
        result = local_converter.coalesce_tables(tables)
        assert len(result) == 1
        # Should have merged rows from both non-empty tables
        assert len(result[0]) == 3  # header + 2 data rows


# ============================================================================
# analyze_file_content PDF Analysis Tests (Lines 791, 795, 887-918)
# ============================================================================


class TestAnalyzeFileContentPdf:
    """Lines 791, 795, 887-918: pdfplumber analysis in analyze_file_content."""

    def test_pdf_with_pdfplumber(self, tmp_path):
        """Full PDF analysis path with mocked pdfplumber."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("x" * 100)  # small file

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is a long text content " * 10
        mock_page.extract_tables.return_value = [["table"]]
        mock_page.images = [{"x0": 0}]

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page, mock_page, mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch.object(local_converter, "pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = mock_pdf
            result = local_converter.analyze_file_content(pdf_file)

        assert result["page_count"] == 3
        assert result["has_tables"] is True
        assert result["has_images"] is True
        assert result["is_text_based"] is True

    def test_pdf_text_less(self, tmp_path):
        """PDF with no extractable text."""
        pdf_file = tmp_path / "scan.pdf"
        pdf_file.write_text("x" * 100)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_page.extract_tables.return_value = []
        mock_page.images = [{"x0": 0}]

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page] * 6  # > 5 pages
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch.object(local_converter, "pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = mock_pdf
            result = local_converter.analyze_file_content(pdf_file)

        assert result["is_text_based"] is False
        assert result["has_images"] is True
        assert result["is_complex"] is True

    def test_pdf_analysis_exception(self, tmp_path):
        """Exception during PDF analysis."""
        pdf_file = tmp_path / "bad.pdf"
        pdf_file.write_text("x")

        with patch.object(local_converter, "pdfplumber") as mock_plumber:
            mock_plumber.open.side_effect = Exception("corrupt pdf")
            result = local_converter.analyze_file_content(pdf_file)

        assert result["page_count"] == 0

    def test_pdf_pypdf_fallback_when_pdfplumber_unavailable(self, tmp_path):
        """Smart routing still gets page_count via pypdf when pdfplumber is None."""
        pytest.importorskip("pypdf", reason="minimal PDF fixtures require pypdf")
        from pypdf import PdfWriter

        pdf_file = tmp_path / "minimal.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with open(pdf_file, "wb") as f:
            writer.write(f)

        with patch.object(local_converter, "pdfplumber", None):
            result = local_converter.analyze_file_content(pdf_file)

        assert result["page_count"] == 1

    def test_pdf_pypdf_fallback_after_pdfplumber_error(self, tmp_path):
        """If pdfplumber.open fails, pypdf can still classify a valid PDF."""
        pytest.importorskip("pypdf", reason="minimal PDF fixtures require pypdf")
        from pypdf import PdfWriter

        pdf_file = tmp_path / "recover.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with open(pdf_file, "wb") as f:
            writer.write(f)

        with patch.object(local_converter, "pdfplumber") as mock_plumber:
            mock_plumber.open.side_effect = RuntimeError("pdfplumber failed")
            result = local_converter.analyze_file_content(pdf_file)

        assert result["page_count"] == 1

    def test_image_file_detection(self, tmp_path, monkeypatch):
        """Image file type detection branch."""
        img_file = tmp_path / "photo.jpg"
        img_file.write_text("fake image data")

        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"jpg", "png", "jpeg"})

        result = local_converter.analyze_file_content(img_file)
        assert result["has_images"] is True


# ============================================================================
# extract_all_tables Coalescing Tests (Line 601)
# ============================================================================


class TestExtractAllTablesCoalescing:
    """Line 601: coalesced_count > 0 log message."""

    def test_coalesced_count_positive(self, tmp_path):
        """Tables with identical headers should be coalesced, hitting line 601."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        # Two tables with identical headers should be coalesced into one
        table1 = [["Name", "Value"], ["A", "1"]]
        table2 = [["Name", "Value"], ["B", "2"]]

        with patch.object(local_converter, "extract_tables_pdfplumber", return_value=[table1, table2]):
            with patch.object(local_converter, "extract_tables_pdfplumber_text", return_value=[]):
                result = local_converter.extract_all_tables(pdf_file)

        # They should be coalesced into one table
        assert result["table_count"] == 1


# ============================================================================
# save_tables_to_files Normalization Fallback (Line 688)
# ============================================================================


class TestSaveTablesNormFallback:
    """Line 688: fallback when normalize_table_headers returns empty."""

    def test_normalization_fails_uses_fallback(self, tmp_path, monkeypatch):
        """normalize_table_headers returns empty → uses format_table_to_markdown(table)."""
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "INCLUDE_METADATA", False)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", False)
        monkeypatch.setattr(config, "TABLE_OUTPUT_FORMATS", ["markdown"])

        pdf_file = tmp_path / "report.pdf"
        pdf_file.touch()

        tables = [[["Header1", "Header2"], ["A", "1"]]]

        with patch("utils.normalize_table_headers", return_value=([], [])):
            result = local_converter.save_tables_to_files(pdf_file, tables)

        # Should still produce output (via fallback path)
        assert len(result) > 0


# ============================================================================
# convert_pdf_to_images Other Format Tests (Line 795)
# ============================================================================


class TestConvertPdfToImagesOtherFormat:
    """Line 795: generic format save (not jpeg, not png)."""

    @pytest.fixture(autouse=True)
    def _disable_page_limit_for_legacy_mock_pdfs(self, monkeypatch):
        monkeypatch.setattr(config, "PDF_IMAGE_MAX_PAGES", 0)

    def test_tiff_format(self, tmp_path, monkeypatch):
        """Other format branch uses PDF_IMAGE_FORMAT.upper()."""
        output_dir = tmp_path / "tiff_pages"
        monkeypatch.setattr(config, "OUTPUT_IMAGES_DIR", tmp_path)
        monkeypatch.setattr(config, "PDF_IMAGE_DPI", 200)
        monkeypatch.setattr(config, "PDF_IMAGE_FORMAT", "tiff")
        monkeypatch.setattr(config, "POPPLER_PATH", "")
        monkeypatch.setattr(config, "PDF_IMAGE_THREAD_COUNT", 1)
        monkeypatch.setattr(config, "PDF_IMAGE_USE_PDFTOCAIRO", False)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        from pathlib import Path

        output_dir.mkdir(parents=True, exist_ok=True)
        temp_file = str(output_dir / "page-1.tiff")
        Path(temp_file).touch()

        with patch.object(local_converter, "convert_from_path", return_value=[temp_file]) as mock_convert:
            success, paths, error = local_converter.convert_pdf_to_images(pdf_file, output_dir=output_dir)

        assert success is True
        assert paths[0] == output_dir / "page_001.tiff"
        mock_convert.assert_called_once()
        kwargs = mock_convert.call_args[1]
        assert kwargs["paths_only"] is True
        assert kwargs["fmt"] == "tiff"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
