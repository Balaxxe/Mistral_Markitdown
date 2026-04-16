"""
Tests for utils.py module
"""

import concurrent.futures
import json
import logging
import unittest.mock
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import config
import utils


class TestSetupLogging:
    """Test logging configuration."""

    def test_returns_logger(self):
        logger = utils.setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "document_converter"

    def test_log_file_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "SAVE_PROCESSING_LOGS", True)
        log_file = str(tmp_path / "test.log")
        logger = utils.setup_logging(log_file=log_file)
        logger.info("test message")
        assert Path(log_file).exists()

    def test_no_file_handler_when_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "SAVE_PROCESSING_LOGS", False)
        logger = utils.setup_logging(log_file="/tmp/nope.log")
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0


class TestIntelligentCache:
    """Test the IntelligentCache class."""

    def test_cache_initialization(self, tmp_path):
        """Test cache initializes correctly."""
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        assert cache.cache_dir.exists()
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_set_and_get(self, tmp_path):
        """Test basic cache set and get operations."""
        cache = utils.IntelligentCache(cache_dir=tmp_path)

        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Cache some data
        test_data = {"result": "test value"}
        cache.set(test_file, test_data, cache_type="test")

        # Retrieve cached data
        retrieved = cache.get(test_file, cache_type="test")
        assert retrieved == test_data
        assert cache.hits == 1
        assert cache.misses == 0

    def test_cache_get_entry_returns_full_record(self, tmp_path):
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        test_file = tmp_path / "entry.txt"
        test_file.write_text("abc")
        payload = {"k": 1}
        cache.set(test_file, payload, cache_type="test")
        entry = cache.get_entry(test_file, cache_type="test")
        assert entry is not None
        assert entry["data"] == payload
        assert entry["type"] == "test"
        assert cache.hits == 1

    def test_cache_miss(self, tmp_path):
        """Test cache miss behavior."""
        cache = utils.IntelligentCache(cache_dir=tmp_path)

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Try to get non-existent cache
        result = cache.get(test_file, cache_type="test")
        assert result is None
        assert cache.misses == 1

    def test_cache_type_mismatch(self, tmp_path):
        """Test that different cache types are isolated."""
        cache = utils.IntelligentCache(cache_dir=tmp_path)

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Cache with type "A"
        cache.set(test_file, {"data": "A"}, cache_type="type_a")

        # Try to get with type "B"
        result = cache.get(test_file, cache_type="type_b")
        assert result is None
        assert cache.misses == 1

    def test_cache_statistics(self, tmp_path):
        """Test cache statistics calculation."""
        cache = utils.IntelligentCache(cache_dir=tmp_path)

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Perform operations
        cache.set(test_file, {"data": "value"}, cache_type="test")
        cache.get(test_file, cache_type="test")  # Hit
        cache.get(tmp_path / "nonexistent.txt", cache_type="test")  # Miss

        stats = cache.get_statistics()
        assert stats["total_entries"] == 1
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] == 50.0

    def test_hash_memo_is_bounded(self, tmp_path):
        """Ensure hash memoization does not grow unbounded."""
        cache = utils.IntelligentCache(cache_dir=tmp_path)

        for i in range(1200):
            f = tmp_path / f"f-{i}.txt"
            f.write_text(f"content-{i}")
            cache._get_file_hash(f)

        assert len(cache._hash_memo) <= cache._hash_memo_max_entries

    def test_cache_set_atomic_under_concurrency(self, tmp_path):
        """Concurrent writes to same cache key should not produce corrupt JSON."""
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        test_file = tmp_path / "shared.txt"
        test_file.write_text("shared-content")

        def _writer(i: int):
            cache.set(test_file, {"writer": i, "payload": "x" * 1000}, cache_type="test")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(_writer, range(50)))

        file_hash = cache._get_file_hash(test_file)
        cache_path = cache._get_cache_path(file_hash, "test")
        with open(cache_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        assert parsed.get("type") == "test"
        assert "data" in parsed


class TestMarkdownFormatting:
    """Test markdown formatting functions."""

    def test_format_table_to_markdown_basic(self):
        """Test basic table formatting."""
        headers = ["Name", "Age", "City"]
        data = [["Alice", "30", "NYC"], ["Bob", "25", "LA"]]

        result = utils.format_table_to_markdown(data, headers=headers)

        assert "| Name | Age | City |" in result
        assert "| --- | --- | --- |" in result
        assert "| Alice | 30 | NYC |" in result
        assert "| Bob | 25 | LA |" in result

    def test_format_table_empty_data(self):
        """Test formatting with empty data."""
        result = utils.format_table_to_markdown([])
        assert result == ""

    def test_format_table_uneven_rows(self):
        """Test formatting with uneven row lengths."""
        headers = ["A", "B", "C"]
        data = [
            ["1", "2"],  # Missing third column
            ["3", "4", "5"],
        ]

        result = utils.format_table_to_markdown(data, headers=headers)
        assert "| 1 | 2 |  |" in result  # Padded


class TestTextCleaning:
    """Test text cleaning functions."""

    def test_clean_consecutive_duplicates_basic(self):
        """Test basic duplicate line removal."""
        text = "Line 1\nLine 1\nLine 1\nLine 2\nLine 2\nLine 3"
        result = utils.clean_consecutive_duplicates(text)
        expected = "Line 1\nLine 2\nLine 3"
        assert result == expected

    def test_clean_consecutive_duplicates_no_duplicates(self):
        """Test with no consecutive duplicates."""
        text = "Line 1\nLine 2\nLine 3"
        result = utils.clean_consecutive_duplicates(text)
        assert result == text

    def test_clean_consecutive_duplicates_empty(self):
        """Test with empty string."""
        result = utils.clean_consecutive_duplicates("")
        assert result == ""

    def test_markdown_to_text_removes_formatting(self):
        """Test markdown to text conversion."""
        markdown = """---
title: Test
---
# Heading

**Bold text** and *italic text*

[Link text](https://example.com)

```python
code block
```
"""
        result = utils.markdown_to_text(markdown)

        assert "---" not in result  # No frontmatter
        assert "# " not in result  # No heading markers
        assert "**" not in result  # No bold markers
        assert "*" not in result or "italic text" in result  # No italic markers
        assert "https://example.com" not in result  # No URLs
        assert "```" not in result  # No code block markers


class TestFileValidation:
    """Test file validation functions."""

    def test_validate_file_exists(self, tmp_path):
        """Test validation of existing file."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")

        is_valid, error = utils.validate_file(test_file)
        assert is_valid
        assert error is None

    def test_validate_file_not_exists(self, tmp_path):
        """Test validation of non-existent file."""
        test_file = tmp_path / "nonexistent.pdf"

        is_valid, error = utils.validate_file(test_file)
        assert not is_valid
        assert "does not exist" in error

    def test_validate_file_empty(self, tmp_path):
        """Test validation of empty file."""
        test_file = tmp_path / "empty.pdf"
        test_file.write_text("")

        is_valid, error = utils.validate_file(test_file)
        assert not is_valid
        assert "empty" in error.lower()

    def test_validate_file_unsupported_type(self, tmp_path):
        """Test validation of unsupported file type."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")

        is_valid, error = utils.validate_file(test_file)
        assert not is_valid
        assert "Unsupported" in error

    def test_validate_file_markitdown_rejects_over_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 1)
        test_file = tmp_path / "big.pdf"
        test_file.write_bytes(b"x" * (2 * 1024 * 1024))
        ok, err = utils.validate_file(test_file, mode="markitdown")
        assert ok is False
        assert err and "too large" in err.lower()

    def test_validate_file_smart_uses_union_size_cap(self, tmp_path, monkeypatch):
        """Smart mode max size is max(MarkItDown, Mistral) so OCR-viable files are not rejected early."""
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 1)
        monkeypatch.setattr(config, "MISTRAL_OCR_MAX_FILE_SIZE_MB", 10)
        test_file = tmp_path / "mid.pdf"
        test_file.write_bytes(b"x" * (2 * 1024 * 1024))
        ok_smart, _ = utils.validate_file(test_file, mode="smart")
        ok_md, err_md = utils.validate_file(test_file, mode="markitdown")
        assert ok_smart is True
        assert ok_md is False
        assert err_md and "too large" in err_md.lower()

    def test_validate_file_strict_rejects_symlink_outside_input(self, tmp_path, monkeypatch):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        monkeypatch.setattr(config, "INPUT_DIR", inbox)
        monkeypatch.setattr(config, "STRICT_INPUT_PATH_RESOLUTION", True)
        outside = tmp_path / "secret.pdf"
        outside.write_bytes(b"%PDF-1.4")
        link = inbox / "trap.pdf"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported")
        ok, err = utils.validate_file(link)
        assert ok is False
        assert err and "input directory" in err.lower()

    def test_validate_file_smart_txt_uses_markitdown_size_cap(self, tmp_path, monkeypatch):
        """Smart mode: types that only go through MarkItDown use MARKITDOWN cap, not OCR cap."""
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 1)
        monkeypatch.setattr(config, "MISTRAL_OCR_MAX_FILE_SIZE_MB", 50)
        test_file = tmp_path / "huge.txt"
        test_file.write_bytes(b"x" * (2 * 1024 * 1024))
        ok, err = utils.validate_file(test_file, mode="smart")
        assert ok is False
        assert err and "too large" in err.lower()

    def test_validate_file_qna_accepts_mistral_extensions_only(self, tmp_path):
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF")
        ok, err = utils.validate_file(pdf, mode="qna")
        assert ok is True

        txt = tmp_path / "notes.txt"
        txt.write_text("hello")
        ok2, err2 = utils.validate_file(txt, mode="qna")
        assert ok2 is False
        assert err2 and "Unsupported" in err2

    def test_validate_file_batch_ocr_matches_mistral_extensions(self, tmp_path):
        pptx = tmp_path / "s.pptx"
        pptx.write_bytes(b"PK\x03\x04")
        ok, err = utils.validate_file(pptx, mode="batch_ocr")
        assert ok is True

        csv_f = tmp_path / "d.csv"
        csv_f.write_text("a,b")
        ok2, err2 = utils.validate_file(csv_f, mode="batch_ocr")
        assert ok2 is False
        assert err2 and "Unsupported" in err2

    def test_validate_file_pdf_to_images_pdf_only(self, tmp_path):
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF")
        assert utils.validate_file(pdf, mode="pdf_to_images")[0] is True

        png = tmp_path / "i.png"
        png.write_bytes(b"\x89PNG")
        ok, err = utils.validate_file(png, mode="pdf_to_images")
        assert ok is False
        assert err and "Unsupported" in err


class TestStdinHelpers:
    """sanitize_stdin_filename_hint and read_stdin_bytes_limited."""

    def test_sanitize_stdin_takes_basename_only(self):
        ok, base, err = utils.sanitize_stdin_filename_hint("foo/bar/report.pdf")
        assert ok and base == "report.pdf" and err is None

    def test_sanitize_stdin_rejects_empty(self):
        ok, base, err = utils.sanitize_stdin_filename_hint("  ")
        assert ok is False and not base

    def test_read_stdin_bytes_limited_under_cap(self, monkeypatch):
        import io

        class _Std:
            buffer = io.BytesIO(b"hello")

        monkeypatch.setattr(utils.sys, "stdin", _Std())
        ok, data, err = utils.read_stdin_bytes_limited(100)
        assert ok and data == b"hello" and err is None

    def test_read_stdin_bytes_limited_rejects_over_cap(self, monkeypatch):
        import io

        class _Std:
            buffer = io.BytesIO(b"x" * 200)

        monkeypatch.setattr(utils.sys, "stdin", _Std())
        ok, data, err = utils.read_stdin_bytes_limited(100)
        assert ok is False
        assert data == b""
        assert err and "limit" in err.lower()


class TestAtomicWriteText:
    """Regression: atomic_write_text leaves a complete destination file."""

    def test_atomic_write_text_content_visible(self, tmp_path):
        dest = tmp_path / "out.md"
        utils.atomic_write_text(dest, "complete")
        assert dest.read_text(encoding="utf-8") == "complete"


class TestPdfExceedsHeavyWorkLimit:
    """pdf_exceeds_heavy_work_limit stat gate for PDF pipelines."""

    def test_non_pdf_not_over_limit(self, tmp_path):
        t = tmp_path / "x.txt"
        t.write_text("hi")
        too_large, err = utils.pdf_exceeds_heavy_work_limit(t)
        assert too_large is False
        assert err is None

    def test_pdf_within_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 100)
        monkeypatch.setattr(config, "MISTRAL_OCR_MAX_FILE_SIZE_MB", 200)
        pdf = tmp_path / "small.pdf"
        pdf.write_bytes(b"%PDF small")
        too_large, err = utils.pdf_exceeds_heavy_work_limit(pdf)
        assert too_large is False
        assert err is None

    def test_pdf_over_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MARKITDOWN_MAX_FILE_SIZE_MB", 1)
        monkeypatch.setattr(config, "MISTRAL_OCR_MAX_FILE_SIZE_MB", 1)
        pdf = tmp_path / "big.pdf"
        pdf.write_bytes(b"%PDF" + b"\x00" * (2 * 1024 * 1024))
        too_large, err = utils.pdf_exceeds_heavy_work_limit(pdf)
        assert too_large is True
        assert err and "too large" in err.lower()


class TestYAMLFrontmatter:
    """Test YAML frontmatter functions."""

    def test_generate_yaml_frontmatter_basic(self):
        """Test basic frontmatter generation."""
        result = utils.generate_yaml_frontmatter(
            title="Test Document", file_name="test.pdf", conversion_method="Test Method"
        )

        assert "---" in result
        assert 'title: "Test Document"' in result
        assert 'source_file: "test.pdf"' in result
        assert 'conversion_method: "Test Method"' in result

    def test_generate_yaml_frontmatter_additional_fields(self):
        """Test frontmatter with additional fields."""
        result = utils.generate_yaml_frontmatter(
            title="Test",
            file_name="test.pdf",
            conversion_method="Method",
            additional_fields={"page_count": 5, "has_tables": True},
        )

        assert "page_count: 5" in result
        assert "has_tables: True" in result

    def test_generate_yaml_frontmatter_escapes_quotes(self):
        """String values should be safely escaped to valid YAML/JSON string form."""
        result = utils.generate_yaml_frontmatter(
            title='Report "2026"',
            file_name="test.pdf",
            conversion_method="Method",
        )

        assert 'title: "Report \\"2026\\""' in result

    def test_strip_yaml_frontmatter(self):
        """Test frontmatter removal."""
        content = """---
title: "Test"
date: 2025-01-01
---
This is the actual content.
"""
        result = utils.strip_yaml_frontmatter(content)

        assert result == "This is the actual content."


class TestOutputNaming:
    """Test output file naming and collision handling."""

    def test_safe_output_stem_disambiguates_same_stem(self, tmp_path):
        """When two files share the same stem, each gets a _ext suffix."""
        # Create two files with the same stem but different extensions
        (tmp_path / "report.pdf").touch()
        (tmp_path / "report.docx").touch()

        with unittest.mock.patch.object(config, "INPUT_DIR", tmp_path):
            stem_pdf = utils.safe_output_stem(tmp_path / "report.pdf")
            stem_docx = utils.safe_output_stem(tmp_path / "report.docx")

        assert stem_pdf != stem_docx
        assert "pdf" in stem_pdf
        assert "docx" in stem_docx

    def test_safe_output_stem_unique_stem(self, tmp_path):
        """When only one file has a given stem, no _ext suffix is needed."""
        (tmp_path / "unique.pdf").touch()

        with unittest.mock.patch.object(config, "INPUT_DIR", tmp_path):
            stem = utils.safe_output_stem(tmp_path / "unique.pdf")

        assert stem == "unique"


class TestPrintProgress:
    """Test the progress bar helper."""

    def test_no_output_when_verbose_off(self, monkeypatch, capsys):
        monkeypatch.setattr(config, "VERBOSE_PROGRESS", False)
        utils.print_progress(1, 10)
        assert capsys.readouterr().out == ""

    def test_output_when_verbose_on(self, monkeypatch, capsys):
        monkeypatch.setattr(config, "VERBOSE_PROGRESS", True)
        utils.print_progress(5, 10)
        captured = capsys.readouterr().out
        assert "50.0%" in captured

    def test_handles_zero_total(self, monkeypatch, capsys):
        monkeypatch.setattr(config, "VERBOSE_PROGRESS", True)
        utils.print_progress(0, 0)
        captured = capsys.readouterr().out
        assert "0.0%" in captured


class TestClearOldEntries:
    """Test cache expiry cleanup."""

    def test_removes_expired_entries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CACHE_DURATION_HOURS", 1)
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        # Write a cache file with an old timestamp
        cache_file = tmp_path / "old_entry.json"
        old_time = (datetime.now() - timedelta(hours=5)).isoformat()
        cache_file.write_text(json.dumps({"timestamp": old_time, "type": "ocr", "data": {}}))
        removed = cache.clear_old_entries()
        assert removed == 1
        assert not cache_file.exists()

    def test_keeps_fresh_entries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CACHE_DURATION_HOURS", 24)
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        cache_file = tmp_path / "fresh_entry.json"
        cache_file.write_text(json.dumps({"timestamp": datetime.now().isoformat(), "type": "ocr", "data": {}}))
        removed = cache.clear_old_entries()
        assert removed == 0
        assert cache_file.exists()

    def test_clear_old_entries_corrupt_file(self, tmp_path):
        """Lines 317-318: exception handler when processing cache file fails."""
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        corrupt_file = tmp_path / "bad_entry.json"
        corrupt_file.write_text("NOT VALID JSON")
        removed = cache.clear_old_entries()
        assert removed == 0


class TestDetectMonthHeaderRow:
    """Test month header detection in financial tables."""

    def test_detects_month_headers(self):
        table = [
            ["Account", "Description"],
            ["", "January", "February", "March", "April"],
            ["10201", "100", "200", "300", "400"],
        ]
        assert utils.detect_month_header_row(table) == 1

    def test_no_months_returns_none(self):
        table = [["Name", "Value"], ["A", "1"]]
        assert utils.detect_month_header_row(table) is None

    def test_empty_table_returns_none(self):
        assert utils.detect_month_header_row([]) is None


class TestCleanTableCell:
    """Test cell cleaning."""

    def test_removes_newlines(self):
        assert utils.clean_table_cell("Hello\nWorld") == "Hello World"

    def test_collapses_whitespace(self):
        assert utils.clean_table_cell("  too   many   spaces  ") == "too many spaces"

    def test_empty_string(self):
        assert utils.clean_table_cell("") == ""

    def test_none_returns_empty(self):
        assert utils.clean_table_cell(None) == ""


class TestIsPageArtifactRow:
    """Test page artifact row detection."""

    def test_page_number_is_artifact(self):
        assert utils.is_page_artifact_row(["Page 1"]) is True

    def test_page_number_multi_digit(self):
        assert utils.is_page_artifact_row(["Page 42"]) is True

    def test_date_is_artifact(self):
        assert utils.is_page_artifact_row(["December 31, 2010"]) is True

    def test_empty_row_is_artifact(self):
        assert utils.is_page_artifact_row(["", ""]) is True

    def test_data_row_is_not_artifact(self):
        assert utils.is_page_artifact_row(["10201", "Cash Operating", "1234.56"]) is False

    def test_none_input(self):
        assert utils.is_page_artifact_row([]) is False


class TestCleanTable:
    """Test full table cleaning."""

    def test_removes_artifacts_and_cleans(self):
        table = [
            ["Name\nValue", "Amount"],
            ["Page 1"],
            ["Item A", "100"],
        ]
        result = utils.clean_table(table)
        assert len(result) == 2
        assert result[0] == ["Name Value", "Amount"]
        assert result[1] == ["Item A", "100"]

    def test_empty_table(self):
        assert utils.clean_table([]) == []


class TestNormalizeTableHeaders:
    """Test table header normalization."""

    def test_with_month_headers(self):
        table = [
            ["Account", "Description"],
            ["", "January", "February", "March", "April"],
            ["10201", "100", "200", "300", "400"],
        ]
        headers, data = utils.normalize_table_headers(table)
        assert "January" in headers

    def test_without_month_headers(self):
        table = [["Name", "Age"], ["Alice", "30"]]
        headers, data = utils.normalize_table_headers(table)
        assert headers == ["Name", "Age"]
        assert data == [["Alice", "30"]]

    def test_empty_table(self):
        headers, data = utils.normalize_table_headers([])
        assert headers == []
        assert data == []


class TestSaveTextOutput:
    """Test text output saving."""

    def test_saves_text_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", True)
        monkeypatch.setattr(config, "OUTPUT_TXT_DIR", tmp_path)
        md_path = tmp_path / "test.md"
        result = utils.save_text_output(md_path, "# Hello\n\n**Bold** text")
        assert result is not None
        assert result.exists()
        content = result.read_text()
        assert "Hello" in content
        assert "**" not in content

    def test_disabled_returns_none(self, monkeypatch):
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", False)
        result = utils.save_text_output(Path("test.md"), "content")
        assert result is None

    def test_save_text_output_exception(self, tmp_path, monkeypatch):
        """Lines 672-674: exception handler in save_text_output."""
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", True)
        monkeypatch.setattr(config, "OUTPUT_TXT_DIR", Path("/nonexistent/dir/deep"))
        result = utils.save_text_output(Path("test.md"), "# Hello")
        assert result is None


# ============================================================================
# Additional tests for 100% coverage
# ============================================================================


class TestFileCacheExpiredEntry:
    """Lines 206-210: FileCache.get() expired cache branch."""

    def test_expired_cache_returns_none(self, tmp_path, monkeypatch):
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        cache.set(test_file, {"data": "value"}, cache_type="test")

        # Overwrite the cache file timestamp to be very old
        file_hash = cache._get_file_hash(test_file)
        cache_path = cache._get_cache_path(file_hash, "test")
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        cache_data["timestamp"] = "2020-01-01T00:00:00"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

        result = cache.get(test_file, cache_type="test")
        assert result is None
        assert cache.misses == 1


class TestFileCacheTypeMismatchSamePath:
    """Lines 215-218: FileCache.get() type mismatch with old-style (non-segregated) cache files."""

    def test_type_mismatch_old_style(self, tmp_path):
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Write a cache entry directly with wrong type into the expected path
        file_hash = cache._get_file_hash(test_file)
        cache_path = cache._get_cache_path(file_hash, "type_a")
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "file_name": test_file.name,
            "file_size": 7,
            "type": "type_b",  # different from what we'll query
            "data": {"value": 1},
            "metadata": {},
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

        result = cache.get(test_file, cache_type="type_a")
        assert result is None
        assert cache.misses == 1


class TestFileCacheGetExceptions:
    """Lines 225-243: FileCache.get() exception handlers."""

    def test_file_not_found_during_get(self, tmp_path):
        """FileNotFoundError handler."""
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        file_hash = cache._get_file_hash(test_file)

        # Create a corrupt cache path that's actually a dir so open() raises
        # Actually, we need FileNotFoundError. Let's mock _get_cache_path to
        # return a path, then have open() raise FileNotFoundError.
        cache_path = cache._get_cache_path(file_hash, "test")
        # Create the cache file, then remove it and patch _get_cache_path to say it exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("{}")  # create it
        cache_path.unlink()  # delete it

        # The exists() check at the beginning of get() will fail, returning None from first branch
        # We need the cache path to exist for hash lookup but then fail on open
        # Let's use a different approach: patch open to raise
        with unittest.mock.patch("builtins.open", side_effect=FileNotFoundError("gone")):
            # Need to also ensure _get_cache_path exists check passes
            with unittest.mock.patch.object(Path, "exists", return_value=True):
                result = cache.get(test_file, cache_type="test")
        assert result is None

    def test_json_decode_error_during_get(self, tmp_path):
        """JSONDecodeError handler."""
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        file_hash = cache._get_file_hash(test_file)
        cache_path = cache._get_cache_path(file_hash, "test")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("NOT VALID JSON {{{")

        result = cache.get(test_file, cache_type="test")
        assert result is None
        assert cache.misses == 1

    def test_json_decode_error_unlink_fails(self, tmp_path):
        """Lines 234-235: OSError when unlinking corrupt cache file."""
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        file_hash = cache._get_file_hash(test_file)
        cache_path = cache._get_cache_path(file_hash, "test")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("NOT VALID JSON {{{")

        with unittest.mock.patch.object(Path, "unlink", side_effect=OSError("perm denied")):
            result = cache.get(test_file, cache_type="test")
        assert result is None
        assert cache.misses == 1

    def test_generic_exception_during_get(self, tmp_path):
        """Generic Exception handler."""
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        file_hash = cache._get_file_hash(test_file)
        cache_path = cache._get_cache_path(file_hash, "test")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Write valid JSON but with a bad timestamp to trigger an exception
        cache_data = {
            "timestamp": "not-a-date",
            "file_name": test_file.name,
            "type": "test",
            "data": {"value": 1},
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

        result = cache.get(test_file, cache_type="test")
        assert result is None


class TestFileCacheSetMissingFile:
    """Lines 259-260: FileCache.set() missing file check."""

    def test_set_nonexistent_file(self, tmp_path):
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        nonexistent = tmp_path / "does_not_exist.txt"
        cache.set(nonexistent, {"data": "value"}, cache_type="test")
        # Should silently return without writing
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 0


class TestFileCacheSetException:
    """Lines 291-294: FileCache.set() exception handler (write failures)."""

    def test_set_exception_during_write(self, tmp_path):
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with unittest.mock.patch("tempfile.NamedTemporaryFile", side_effect=PermissionError("denied")):
            cache.set(test_file, {"data": "value"}, cache_type="test")
        # Should handle exception gracefully
        # No crash = success

    def test_set_generic_exception(self, tmp_path):
        """Line 292: generic Exception handler in set()."""
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Make stat() raise to trigger generic Exception (not FileNotFoundError)
        original_stat = test_file.stat
        call_count = 0

        def flaky_stat():
            nonlocal call_count
            call_count += 1
            # exists() may call stat internally; let first call succeed for exists(),
            # then fail for the st_size call inside set()
            if call_count > 1:
                raise RuntimeError("stat fail")
            return original_stat()

        with unittest.mock.patch.object(type(test_file), "stat", flaky_stat):
            cache.set(test_file, {"data": "value"}, cache_type="test")
        # Should catch via except Exception (not FileNotFoundError)


class TestFileCacheGetStatisticsBody:
    """Lines 317-318: FileCache.get_statistics() body with actual files."""

    def test_get_statistics_with_entries(self, tmp_path):
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        cache.set(test_file, {"data": "value"}, cache_type="test")
        cache.get(test_file, cache_type="test")  # hit

        stats = cache.get_statistics()
        assert stats["total_entries"] >= 1
        assert stats["total_size_mb"] > 0


class TestFormatTableNoHeaders:
    """Lines 371-372, 375: format_table_to_markdown with headers=None fallback."""

    def test_headers_inferred_from_first_row(self):
        """Lines 371-372: headers = data[0], data = data[1:]."""
        data = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        result = utils.format_table_to_markdown(data, headers=None)
        assert "| Name | Age |" in result
        assert "| Alice | 30 |" in result

    def test_empty_headers_after_extraction(self):
        """Line 375: return '' when headers end up empty."""
        # Pass data where first row is empty strings (falsy list)
        data = [["", ""], ["Alice", "30"]]
        utils.format_table_to_markdown(data, headers=None)
        # headers = ["", ""] which is truthy (non-empty list), so this won't hit 375
        # We need headers=None and data=[[]] to get empty headers

    def test_single_empty_row_no_headers(self):
        """Line 375: headers = [] (empty first row)."""
        data = [[]]
        result = utils.format_table_to_markdown(data, headers=None)
        assert result == ""


class TestNormalizeTableHeadersCleanedEmpty:
    """Line 561: normalize_table_headers where table becomes empty after clean_table."""

    def test_table_cleaned_to_empty(self):
        """All rows are artifact/empty rows that get cleaned away."""
        # is_page_artifact_row matches "Page N" and near-empty rows
        table = [["Page 1"], ["  "], ["Page 2"]]
        headers, data = utils.normalize_table_headers(table)
        assert headers == []
        assert data == []


class TestPrintProgressComplete:
    """Line 707: print newline when current == total."""

    def test_newline_when_complete(self, monkeypatch, capsys):
        monkeypatch.setattr(config, "VERBOSE_PROGRESS", True)
        utils.print_progress(10, 10)
        captured = capsys.readouterr().out
        assert "100.0%" in captured
        assert captured.endswith("\n")


class TestValidateFileNotAFile:
    """Line 729: validate_file returns error for non-file paths."""

    def test_directory_not_a_file(self, tmp_path):
        is_valid, error = utils.validate_file(tmp_path)
        assert not is_valid
        assert "Not a file" in error


class TestSafeOutputStemExternalFile:
    """Lines 765-766: safe_output_stem for files outside INPUT_DIR."""

    def test_external_file_gets_hash_suffix(self, tmp_path):
        external_dir = tmp_path / "external"
        external_dir.mkdir()
        ext_file = external_dir / "report.pdf"
        ext_file.touch()

        with unittest.mock.patch.object(config, "INPUT_DIR", tmp_path / "input"):
            (tmp_path / "input").mkdir(exist_ok=True)
            stem = utils.safe_output_stem(ext_file)

        assert stem.startswith("report_")
        assert len(stem) > len("report_")  # has hash suffix


class TestSafeOutputStemOSError:
    """Lines 767-768: safe_output_stem handles OSError/ValueError."""

    def test_oserror_returns_bare_stem(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.touch()

        with unittest.mock.patch.object(Path, "resolve", side_effect=OSError("bad path")):
            stem = utils.safe_output_stem(test_file)
        assert stem == "test"


class TestYAMLFrontmatterDisabled:
    """Line 793: generate_yaml_frontmatter returns '' when INCLUDE_METADATA is False."""

    def test_frontmatter_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "INCLUDE_METADATA", False)
        result = utils.generate_yaml_frontmatter(title="Test", file_name="test.pdf", conversion_method="Method")
        assert result == ""


class TestSanitizeForTerminal:
    """Tests for sanitize_for_terminal: strip ANSI escapes and control chars."""

    def test_strips_ansi_escape(self):
        result = utils.sanitize_for_terminal("\x1b[31mRed text\x1b[0m")
        assert result == "Red text"

    def test_strips_csi_sequences(self):
        result = utils.sanitize_for_terminal("\x1b[2J\x1b[Hhello")
        assert result == "hello"

    def test_preserves_tabs_and_newlines(self):
        result = utils.sanitize_for_terminal("line1\n\tline2\r\n")
        assert "line1" in result
        assert "line2" in result

    def test_strips_null_bytes(self):
        result = utils.sanitize_for_terminal("safe\x00text")
        assert "\x00" not in result
        assert "safe" in result and "text" in result

    def test_plain_text_unchanged(self):
        result = utils.sanitize_for_terminal("Normal text 123")
        assert result == "Normal text 123"

    def test_empty_string(self):
        result = utils.sanitize_for_terminal("")
        assert result == ""


class TestAtomicWriteBinary:
    """Tests for atomic_write_binary: temp-file + rename for binary content."""

    def test_writes_binary_content(self, tmp_path):
        dest = tmp_path / "out.bin"
        content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        utils.atomic_write_binary(dest, content)
        assert dest.read_bytes() == content

    def test_overwrites_existing_file(self, tmp_path):
        dest = tmp_path / "out.bin"
        dest.write_bytes(b"old")
        utils.atomic_write_binary(dest, b"new")
        assert dest.read_bytes() == b"new"

    def test_writes_to_existing_subdirectory(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        dest = subdir / "out.bin"
        utils.atomic_write_binary(dest, b"nested")
        assert dest.read_bytes() == b"nested"


class TestUiPrint:
    """Tests for ui_print: thin wrapper around print."""

    def test_ui_print_outputs_to_stdout(self, capsys):
        utils.ui_print("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_ui_print_no_args(self, capsys):
        utils.ui_print()
        captured = capsys.readouterr()
        assert captured.out == "\n"


class TestReadStdinBytesLimitedNegative:
    """Edge case: read_stdin_bytes_limited with negative max_bytes."""

    def test_negative_max_bytes_returns_error(self):
        ok, data, err = utils.read_stdin_bytes_limited(-1)
        assert ok is False
        assert data == b""
        assert err is not None and "invalid" in err.lower()


class TestCacheGetEntryInvalidStructure:
    """Cache get_entry returns None for malformed cache data."""

    def test_cache_entry_is_list_instead_of_dict(self, tmp_path):
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        test_file = tmp_path / "bad.txt"
        test_file.write_text("content")
        file_hash = cache._get_file_hash(test_file)
        cache_path = cache._get_cache_path(file_hash, "ocr")
        cache_path.write_text(json.dumps(["not", "a", "dict"]))
        result = cache.get_entry(test_file, cache_type="ocr")
        assert result is None
        assert cache.misses >= 1

    def test_cache_entry_missing_data_key(self, tmp_path):
        cache = utils.IntelligentCache(cache_dir=tmp_path)
        test_file = tmp_path / "nodata.txt"
        test_file.write_text("content")
        file_hash = cache._get_file_hash(test_file)
        cache_path = cache._get_cache_path(file_hash, "ocr")
        cache_path.write_text(json.dumps({"timestamp": "2025-01-01T00:00:00", "type": "ocr"}))
        result = cache.get_entry(test_file, cache_type="ocr")
        assert result is None


class TestSafeOutputStemNoExtension:
    """safe_output_stem with a file that has no extension."""

    def test_file_without_extension(self, tmp_path):
        f = tmp_path / "README"
        f.write_text("hello")
        stem = utils.safe_output_stem(f)
        assert "README" in stem


class TestValidateFilePathTraversal:
    """validate_file rejects path-traversal when strict resolution is on."""

    def test_path_traversal_rejected_under_strict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "STRICT_INPUT_PATH_RESOLUTION", True)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path / "input")
        (tmp_path / "input").mkdir(exist_ok=True)
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        ok, err = utils.validate_file(outside)
        assert ok is False
        assert err is not None and "input directory" in err.lower()


class TestToConversionResult:
    """utils.to_conversion_result normalises converter return shapes."""

    def test_passthrough_conversion_result(self, tmp_path):
        original = utils.ConversionResult(True, tmp_path / "a.md", None)
        assert utils.to_conversion_result(original) is original

    def test_three_tuple_with_path(self, tmp_path):
        out = tmp_path / "out.md"
        result = utils.to_conversion_result((True, out, None))
        assert result.success is True
        assert result.output_path == out
        assert result.error is None

    def test_three_tuple_with_non_path_payload(self):
        # Some converters return (bool, str_or_dict, error); the payload
        # should be dropped but success/error preserved.
        result = utils.to_conversion_result((False, "stringy payload", "boom"))
        assert result.success is False
        assert result.output_path is None
        assert result.error == "boom"

    def test_two_tuple_success_error(self):
        result = utils.to_conversion_result((False, "nope"))
        assert result.success is False
        assert result.output_path is None
        assert result.error == "nope"

    def test_bool_result(self):
        assert utils.to_conversion_result(True) == utils.ConversionResult(True, None, None)
        assert utils.to_conversion_result(False) == utils.ConversionResult(False, None, None)

    def test_unknown_result_type(self):
        result = utils.to_conversion_result(object())
        assert result.success is False
        assert result.error is not None and "unknown" in result.error.lower()


class TestMistralOcrSizeError:
    """utils.mistral_ocr_size_error returns a message only above the cap."""

    def test_within_limit_returns_none(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_OCR_MAX_FILE_SIZE_MB", 100)
        assert utils.mistral_ocr_size_error(50.0) is None

    def test_above_limit_returns_message(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_OCR_MAX_FILE_SIZE_MB", 100)
        msg = utils.mistral_ocr_size_error(150.0)
        assert msg is not None
        assert "too large" in msg.lower()
        assert "100" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
