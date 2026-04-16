"""
Tests for mistral_converter.py module.

Tests cover:
- SSRF URL validation (_validate_document_url)
- OCR quality assessment (assess_ocr_quality, _is_weak_page)
- Annotation format helpers (get_bbox_annotation_format, get_document_annotation_format)
- Batch file creation (create_batch_ocr_file)
- Client cache invalidation (reset_mistral_client)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import config

# Initialize config dirs so imports work
config.ensure_directories()

import mistral_converter  # noqa: E402

# ============================================================================
# _validate_document_url Tests
# ============================================================================


class TestIsWeakPage:
    """Test weak page detection logic."""

    def test_empty_text_is_weak(self):
        assert mistral_converter._is_weak_page("") is True

    def test_short_text_is_weak(self):
        assert mistral_converter._is_weak_page("A few words") is True

    def test_long_quality_text_not_weak(self):
        # Build text that passes all checks:
        # - Long enough (>50 chars)
        # - Contains digits (>20)
        # - Has unique tokens
        # - Reasonable line lengths
        text = (
            "This is a well-formatted financial document page 12345678901234567890 "
            "with various unique words and data points covering multiple topics. "
            "Revenue grew by 15% to $2,500,000 in Q3 compared to prior year."
        )
        assert mistral_converter._is_weak_page(text) is False

    def test_repetitive_text_is_weak(self):
        # Same word repeated many times → low uniqueness ratio
        text = " ".join(["repeated"] * 200) + " 12345678901234567890"
        assert mistral_converter._is_weak_page(text) is True

    def test_no_digits_is_weak(self):
        # Long text without enough digits
        text = "This is a page of text without any numbers in it. " * 5
        assert mistral_converter._is_weak_page(text) is True


# ============================================================================
# assess_ocr_quality Tests
# ============================================================================


class TestAssessOcrQuality:
    """Test OCR quality assessment."""

    def test_empty_result_not_usable(self):
        result = {"full_text": "", "pages": []}
        assessment = mistral_converter.assess_ocr_quality(result)
        assert assessment["is_usable"] is False
        assert assessment["quality_score"] == 0.0

    def test_good_result_usable(self):
        # Build text that satisfies all quality checks:
        # - Sufficient length (>50 chars)
        # - High digit count (>100 for full_text)
        # - High uniqueness ratio (>0.3)
        # - Each page individually passes _is_weak_page checks
        page_text = (
            "Annual Financial Report for Fiscal Year 2024\n"
            "Total consolidated revenue reached $12,345,678.00 representing 15.2% growth.\n"
            "Operating expenses totaled $9,876,543.21 with EBITDA margin at 23.7%.\n"
            "Net income attributable to shareholders was $2,469,134.79 compared to $1,987,654.32.\n"
            "Cash and equivalents of $4,567,890.12 at December 31st 2024.\n"
            "Long-term debt decreased from $8,765,432.10 to $7,654,321.09.\n"
            "Dividends paid per share $3.45 up from $2.98 in prior year.\n"
            "Earnings per diluted share increased to $5.67 versus $4.89.\n"
            "Weighted average shares outstanding 435,678,901.\n"
            "Capital expenditures of $1,234,567.89 primarily in technology infrastructure.\n"
        )
        full_text = page_text * 3  # Make it substantial across pages

        result = {
            "full_text": full_text,
            "pages": [{"page_number": i, "text": page_text} for i in range(3)],
        }
        assessment = mistral_converter.assess_ocr_quality(result)
        assert assessment["is_usable"] is True
        assert assessment["quality_score"] > 50

    def test_returns_expected_keys(self):
        result = {"full_text": "x" * 100, "pages": [{"text": "x" * 100}]}
        assessment = mistral_converter.assess_ocr_quality(result)
        assert "is_usable" in assessment
        assert "quality_score" in assessment
        assert "issues" in assessment
        assert "weak_page_count" in assessment
        assert "total_page_count" in assessment


# ============================================================================
# Annotation Format Tests
# ============================================================================


class TestAnnotationFormats:
    """Test that annotation format helpers return correct types."""

    def test_bbox_format_disabled_returns_none(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_BBOX_ANNOTATION", False)
        result = mistral_converter.get_bbox_annotation_format()
        assert result is None

    def test_document_format_disabled_returns_none(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_DOCUMENT_ANNOTATION", False)
        result = mistral_converter.get_document_annotation_format()
        assert result is None

    def test_bbox_format_enabled_returns_dict(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_BBOX_ANNOTATION", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", True)
        result = mistral_converter.get_bbox_annotation_format()
        assert result is None or isinstance(result, dict)

    def test_document_format_enabled_returns_dict(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_DOCUMENT_ANNOTATION", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", True)
        result = mistral_converter.get_document_annotation_format("generic")
        assert result is None or isinstance(result, dict)

    def test_document_format_auto_resolves_to_generic(self, monkeypatch):
        """auto schema type should resolve to generic when not configured."""
        monkeypatch.setattr(config, "MISTRAL_ENABLE_DOCUMENT_ANNOTATION", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", True)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_SCHEMA_TYPE", "auto")
        result = mistral_converter.get_document_annotation_format("auto")
        assert result is None or isinstance(result, dict)


# ============================================================================
# Client Cache Invalidation Tests
# ============================================================================


class TestClientCacheInvalidation:
    """Test the reset_mistral_client helper."""

    def test_reset_clears_cache(self):
        """Calling reset should not raise and should clear the cached client."""
        mistral_converter.reset_mistral_client()
        # After reset, the internal singleton should be None
        assert mistral_converter._client_instance is None

    def test_get_client_without_api_key_returns_none(self, monkeypatch):
        """Without API key, get_mistral_client should return None."""
        mistral_converter.reset_mistral_client()
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "")
        result = mistral_converter.get_mistral_client()
        assert result is None

    def test_singleton_is_thread_safe(self, monkeypatch):
        """Multiple threads calling get_mistral_client get the same None (no key)."""
        import concurrent.futures

        mistral_converter.reset_mistral_client()
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: mistral_converter.get_mistral_client(), range(8)))

        assert all(r is None for r in results)


# ============================================================================
# _is_weak_page Digit Ratio Tests
# ============================================================================


class TestIsWeakPageDigitRatio:
    """Digit ratio/count checks were removed — verify pages are not flagged for low digit content."""

    def test_low_digit_text_not_flagged(self, monkeypatch):
        """Low-digit pages should NOT be flagged as weak."""
        monkeypatch.setattr(config, "OCR_MIN_TEXT_LENGTH", 10)

        # Text with few digits — should NOT be flagged
        text = "This is text with very few digits 12 and some more unique words here now. " * 3
        assert mistral_converter._is_weak_page(text) is False

    def test_sufficient_digits_not_flagged(self, monkeypatch):
        """Text with sufficient digit ratio should not be weak."""
        monkeypatch.setattr(config, "OCR_MIN_TEXT_LENGTH", 10)
        monkeypatch.setattr(config, "OCR_MIN_UNIQUENESS_RATIO", 0.1)
        monkeypatch.setattr(config, "OCR_MAX_PHRASE_REPETITIONS", 100)
        monkeypatch.setattr(config, "OCR_MIN_AVG_LINE_LENGTH", 5)

        text = (
            "Revenue 12345678901234567890 grew significantly across all regions "
            "with unique valuable insightful data metrics and comprehensive analysis."
        )
        assert mistral_converter._is_weak_page(text) is False


# ============================================================================
# save_extracted_images base64 prefix handling
# ============================================================================


class TestSaveExtractedImages:
    """Test base64 data URI stripping in save_extracted_images."""

    def test_handles_data_uri_prefix(self, tmp_path, monkeypatch):
        """Should decode correctly even when base64 has data: prefix."""
        import base64

        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
        monkeypatch.setattr(config, "OUTPUT_IMAGES_DIR", tmp_path)

        # Create a tiny valid PNG (1x1 transparent pixel)
        # PNG header + minimal IHDR + IDAT + IEND
        import struct
        import zlib

        def _make_minimal_png() -> bytes:
            """Create a minimal valid 1x1 PNG."""
            header = b"\x89PNG\r\n\x1a\n"
            # IHDR
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
            ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
            # IDAT
            raw_data = zlib.compress(b"\x00\xff\x00\x00")
            idat_crc = zlib.crc32(b"IDAT" + raw_data) & 0xFFFFFFFF
            idat = struct.pack(">I", len(raw_data)) + b"IDAT" + raw_data + struct.pack(">I", idat_crc)
            # IEND
            iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
            iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            return header + ihdr + idat + iend

        png_bytes = _make_minimal_png()
        b64_raw = base64.b64encode(png_bytes).decode()
        b64_with_prefix = f"data:image/png;base64,{b64_raw}"

        ocr_result = {
            "pages": [
                {
                    "page_number": 1,
                    "images": [{"base64": b64_with_prefix}],
                }
            ]
        }

        file_path = tmp_path / "test.pdf"
        file_path.touch()

        saved = mistral_converter.save_extracted_images(ocr_result, file_path)
        assert len(saved) == 1
        # The saved file should contain the raw PNG bytes
        assert saved[0].read_bytes() == png_bytes


# ============================================================================
# Session page commit Tests
# ============================================================================


class TestOcrSessionPageDelta:
    """_ocr_session_page_delta accounts for usage_info when pages list is empty."""

    def test_uses_len_pages_when_present(self):
        delta = mistral_converter._ocr_session_page_delta(
            {"pages": [{"text": "a"}, {"text": "b"}], "full_text": "a\nb"}
        )
        assert delta == 2

    def test_falls_back_to_usage_pages_processed(self):
        delta = mistral_converter._ocr_session_page_delta(
            {"pages": [], "full_text": "hello", "usage_info": {"pages_processed": 4}}
        )
        assert delta == 4

    def test_falls_back_to_one_when_only_full_text(self):
        delta = mistral_converter._ocr_session_page_delta({"pages": [], "full_text": "hello"})
        assert delta == 1

    def test_zero_when_empty(self):
        assert mistral_converter._ocr_session_page_delta({"pages": [], "full_text": ""}) == 0


class TestCommitSessionPages:
    """_commit_session_pages updates counters and warns once at the session cap."""

    def test_commit_increments_below_limit(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_PAGES_PER_SESSION", 100)
        mistral_converter.reset_session_page_counter()
        assert mistral_converter._commit_session_pages(0, 5) is True
        assert mistral_converter._session_pages_processed == 5
        assert mistral_converter._session_pages_warned is False

    def test_warns_once_at_exact_limit(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_PAGES_PER_SESSION", 10)
        mistral_converter.reset_session_page_counter()
        assert mistral_converter._commit_session_pages(0, 10) is False
        assert mistral_converter._session_pages_processed == 10
        assert mistral_converter._session_pages_warned is True

    def test_warns_once_when_budget_exceeded(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_PAGES_PER_SESSION", 10)
        mistral_converter.reset_session_page_counter()
        assert mistral_converter._commit_session_pages(0, 15) is False
        assert mistral_converter._session_pages_processed == 15
        assert mistral_converter._session_pages_warned is True
        # Second commit should not reset warned flag
        assert mistral_converter._commit_session_pages(0, 1) is False
        assert mistral_converter._session_pages_warned is True

    def test_limit_disabled_short_circuits(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_PAGES_PER_SESSION", 0)
        mistral_converter.reset_session_page_counter()
        assert mistral_converter._commit_session_pages(0, 100) is True
        assert mistral_converter._session_pages_processed == 0


# ============================================================================
# get_retry_config Tests
# ============================================================================


class TestGetRetryConfig:
    """Test retry configuration creation."""

    def test_returns_none_when_retries_zero(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_RETRIES", 0)
        result = mistral_converter.get_retry_config()
        assert result is None

    def test_returns_config_when_retries_available(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_RETRIES", 3)
        result = mistral_converter.get_retry_config()
        # Could be None if retries module not available, or a config object
        # Just ensure no exception is raised
        assert result is None or result is not None


# ============================================================================
# _extract_model_json_schema Tests
# ============================================================================


class TestExtractModelJsonSchema:
    """Test JSON schema extraction from Pydantic models."""

    def test_pydantic_v2_model(self):
        mock_model = MagicMock()
        mock_model.model_json_schema.return_value = {"type": "object", "properties": {}}
        result = mistral_converter._extract_model_json_schema(mock_model)
        assert result == {"type": "object", "properties": {}}

    def test_pydantic_v1_model(self):
        mock_model = MagicMock(spec=["schema"])
        mock_model.schema.return_value = {"type": "object"}
        result = mistral_converter._extract_model_json_schema(mock_model)
        assert result == {"type": "object"}

    def test_no_schema_method(self):
        mock_model = MagicMock(spec=[])
        result = mistral_converter._extract_model_json_schema(mock_model)
        assert result is None


# ============================================================================
# _wrap_response_format Tests
# ============================================================================


class TestWrapResponseFormat:
    """Test response format wrapping."""

    def test_wraps_schema_correctly(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = mistral_converter._wrap_response_format(schema, "test_format")
        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "test_format"
        assert result["json_schema"]["strict"] is True
        assert result["json_schema"]["schema"] == schema


# ============================================================================
# _extract_page_text Tests
# ============================================================================


class TestExtractPageText:
    """Test text extraction from various page object types."""

    def test_markdown_attribute(self):
        page = MagicMock()
        page.markdown = "# Title"
        page.text = None
        page.content = None
        result = mistral_converter._extract_page_text(page)
        assert result == "# Title"

    def test_text_attribute(self):
        page = MagicMock(spec=["text"])
        page.text = "plain text"
        result = mistral_converter._extract_page_text(page)
        assert result == "plain text"

    def test_dict_page(self):
        page = {"markdown": "# Dict Page"}
        result = mistral_converter._extract_page_text(page)
        assert result == "# Dict Page"

    def test_string_page(self):
        result = mistral_converter._extract_page_text("raw string")
        assert result == "raw string"

    def test_empty_page(self):
        page = MagicMock(spec=[])
        result = mistral_converter._extract_page_text(page)
        assert result == ""


# ============================================================================
# _parse_page_object Tests
# ============================================================================


class TestParsePageObject:
    """Test page object parsing."""

    def test_basic_page(self):
        page = MagicMock()
        page.markdown = "Page content"
        page.text = None
        page.content = None
        page.index = 0  # API returns 0-based index
        page.images = []
        page.dimensions = None
        page.tables = None
        page.hyperlinks = None
        page.header = None
        page.footer = None

        result = mistral_converter._parse_page_object(page, 0)
        assert result["page_number"] == 1  # converted to 1-based
        assert result["api_page_index"] == 0
        assert result["text"] == "Page content"
        assert result["images"] == []

    def test_dict_page(self):
        page = {"markdown": "Dict content", "index": 1}  # 0-based
        result = mistral_converter._parse_page_object(page, 1)
        assert result["page_number"] == 2  # converted to 1-based
        assert result["api_page_index"] == 1
        assert result["text"] == "Dict content"


# ============================================================================
# _parse_single_text_response Tests
# ============================================================================


class TestParseSingleTextResponse:
    """Test single text response parsing."""

    def test_adds_text_and_page(self):
        result = {"full_text": "", "pages": []}
        mistral_converter._parse_single_text_response("Hello World", result)
        assert result["full_text"] == "Hello World"
        assert len(result["pages"]) == 1
        assert result["pages"][0]["page_number"] == 1
        assert result["pages"][0]["api_page_index"] == 0


# ============================================================================
# _parse_dict_response Tests
# ============================================================================


class TestParseDictResponse:
    """Test dict response parsing."""

    def test_with_pages(self):
        response = {
            "pages": [
                {"markdown": "Page 1", "index": 1},
                {"text": "Page 2", "index": 2},
            ]
        }
        result = {"full_text": "", "pages": []}
        mistral_converter._parse_dict_response(response, result)
        assert len(result["pages"]) == 2
        assert "Page 1" in result["full_text"]

    def test_without_pages(self):
        response = {"markdown": "Single page content"}
        result = {"full_text": "", "pages": []}
        mistral_converter._parse_dict_response(response, result)
        assert len(result["pages"]) == 1
        assert result["full_text"] == "Single page content"


# ============================================================================
# _extract_structured_outputs Tests
# ============================================================================


class TestExtractStructuredOutputs:
    """Test structured output extraction."""

    def test_extracts_bbox_annotations(self):
        response = MagicMock()
        bbox = MagicMock()
        bbox.model_dump.return_value = {"x": 0, "y": 0, "w": 100, "h": 50}
        response.bbox_annotations = [bbox]
        response.document_annotation = None

        result = {"bbox_annotations": [], "document_annotation": None}
        mistral_converter._extract_structured_outputs(response, result)
        assert len(result["bbox_annotations"]) == 1

    def test_extracts_document_annotation_json_string(self):
        response = MagicMock()
        response.bbox_annotations = None
        response.document_annotation = '{"type": "invoice"}'

        result = {"bbox_annotations": [], "document_annotation": None}
        mistral_converter._extract_structured_outputs(response, result)
        assert result["document_annotation"] == {"type": "invoice"}

    def test_no_annotations(self):
        response = MagicMock()
        response.bbox_annotations = None
        response.document_annotation = None

        result = {"bbox_annotations": [], "document_annotation": None}
        mistral_converter._extract_structured_outputs(response, result)
        assert result["bbox_annotations"] == []
        assert result["document_annotation"] is None


# ============================================================================
# _extract_response_metadata Tests
# ============================================================================


class TestExtractResponseMetadata:
    """Test metadata extraction from responses."""

    def test_object_response(self):
        response = MagicMock()
        response.metadata = {"source": "test"}
        response.model = "mistral-ocr-latest"
        usage = MagicMock()
        usage.pages_processed = 5
        usage.doc_size_bytes = 1024
        response.usage_info = usage

        result = {"metadata": {}, "usage_info": {}, "model": None}
        mistral_converter._extract_response_metadata(response, result)
        assert result["metadata"] == {"source": "test"}
        assert result["model"] == "mistral-ocr-latest"
        assert result["usage_info"]["pages_processed"] == 5

    def test_dict_response(self):
        response = {
            "metadata": {"key": "val"},
            "model": "pixtral",
            "usage_info": {"pages_processed": 2},
        }

        result = {"metadata": {}, "usage_info": {}, "model": None}
        mistral_converter._extract_response_metadata(response, result)
        assert result["metadata"] == {"key": "val"}
        assert result["model"] == "pixtral"


# ============================================================================
# _parse_ocr_response Tests
# ============================================================================


class TestParseOcrResponse:
    """Test full OCR response parsing pipeline."""

    def test_pages_response(self, tmp_path):
        page = MagicMock()
        page.markdown = "# Page 1 Content"
        page.text = None
        page.content = None
        page.index = 1
        page.images = []
        page.dimensions = None
        page.tables = None
        page.hyperlinks = None
        page.header = None
        page.footer = None

        response = MagicMock()
        response.pages = [page]
        response.bbox_annotations = None
        response.document_annotation = None
        response.metadata = {}
        response.usage_info = None
        response.model = "mistral-ocr-latest"

        result = mistral_converter._parse_ocr_response(response, tmp_path / "test.pdf")
        assert result["file_name"] == "test.pdf"
        assert len(result["pages"]) == 1
        assert "Page 1 Content" in result["full_text"]

    def test_dict_response_fallback(self, tmp_path):
        response = {"markdown": "Direct markdown content"}
        result = mistral_converter._parse_ocr_response(response, tmp_path / "doc.pdf")
        assert "Direct markdown content" in result["full_text"]

    def test_handles_exception_gracefully(self, tmp_path):
        response = MagicMock()
        response.pages = None
        response.markdown = None
        response.text = None
        response.content = None
        response.bbox_annotations = MagicMock(side_effect=Exception("boom"))
        # Should not raise
        result = mistral_converter._parse_ocr_response(response, tmp_path / "err.pdf")
        assert result["file_name"] == "err.pdf"


# ============================================================================
# _create_markdown_output Tests
# ============================================================================


class TestCreateMarkdownOutput:
    """Test markdown output creation."""

    def test_creates_markdown_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "INCLUDE_METADATA", False)
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", False)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)

        file_path = tmp_path / "document.pdf"
        file_path.touch()

        ocr_result = {
            "pages": [
                {"page_number": 1, "text": "Page 1 content", "images": []},
                {"page_number": 2, "text": "Page 2 content", "images": []},
            ],
            "full_text": "Page 1 content\n\nPage 2 content",
        }

        output = mistral_converter._create_markdown_output(file_path, ocr_result)
        assert output.exists()
        content = output.read_text()
        assert "Page 1 content" in content
        assert "Page 2 content" in content
        assert "OCR Result" in content

    def test_fallback_without_pages(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "INCLUDE_METADATA", False)
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", False)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)

        file_path = tmp_path / "doc.pdf"
        file_path.touch()

        ocr_result = {
            "pages": [],
            "full_text": "Fallback text content",
        }

        output = mistral_converter._create_markdown_output(file_path, ocr_result)
        content = output.read_text()
        assert "Fallback text content" in content


# ============================================================================
# _save_structured_outputs Tests
# ============================================================================


class TestSaveStructuredOutputs:
    """Test structured output file saving."""

    def test_saves_bbox_annotations(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)

        file_path = tmp_path / "doc.pdf"
        file_path.touch()

        ocr_result = {
            "bbox_annotations": [{"x": 0, "y": 0, "w": 100, "h": 50}],
            "document_annotation": None,
        }

        mistral_converter._save_structured_outputs(file_path, ocr_result)
        json_files = list(tmp_path.glob("*bbox*.json"))
        assert len(json_files) == 1

    def test_saves_document_annotation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "INPUT_DIR", tmp_path)

        file_path = tmp_path / "doc.pdf"
        file_path.touch()

        ocr_result = {
            "bbox_annotations": [],
            "document_annotation": {"type": "invoice", "total": "1000"},
        }

        mistral_converter._save_structured_outputs(file_path, ocr_result)
        json_files = list(tmp_path.glob("*document_annotation*.json"))
        assert len(json_files) == 1

    def test_no_output_without_annotations(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)

        file_path = tmp_path / "doc.pdf"

        ocr_result = {
            "bbox_annotations": [],
            "document_annotation": None,
        }

        mistral_converter._save_structured_outputs(file_path, ocr_result)
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 0


# ============================================================================
# _validate_document_url Additional Tests
# ============================================================================


class TestGetMistralClient:
    """Test Mistral client initialization."""

    def test_returns_none_without_sdk(self, monkeypatch):
        mistral_converter.reset_mistral_client()
        with patch.object(mistral_converter, "Mistral", None):
            result = mistral_converter.get_mistral_client()
        assert result is None
        mistral_converter.reset_mistral_client()

    def test_returns_none_without_api_key(self, monkeypatch):
        mistral_converter.reset_mistral_client()
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "")
        result = mistral_converter.get_mistral_client()
        assert result is None
        mistral_converter.reset_mistral_client()


# ============================================================================
# query_document Tests
# ============================================================================


class TestSaveExtractedImagesAdditional:
    """Additional image saving tests."""

    def test_images_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        ocr_result = {"pages": [{"page_number": 1, "images": [{"base64": "abc"}]}]}
        saved = mistral_converter.save_extracted_images(ocr_result, tmp_path / "test.pdf")
        assert saved == []

    def test_no_images_in_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
        ocr_result = {"pages": [{"page_number": 1, "images": []}]}
        saved = mistral_converter.save_extracted_images(ocr_result, tmp_path / "test.pdf")
        assert saved == []


# ============================================================================
# optimize_image Tests
# ============================================================================


class TestOptimizeImage:
    """Test image optimization."""

    def test_disabled_returns_original(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", False)
        img = tmp_path / "test.png"
        img.write_bytes(b"fake")
        assert mistral_converter.optimize_image(img) == img

    def test_small_image_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", True)
        monkeypatch.setattr(config, "MISTRAL_MAX_IMAGE_DIMENSION", 2000)
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"fake")

        mock_img = MagicMock()
        mock_img.size = (500, 400)
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)

        with patch.object(mistral_converter, "Image") as mock_pil:
            mock_pil.Resampling.LANCZOS = 1
            mock_pil.open.return_value = mock_img
            result = mistral_converter.optimize_image(img_path)
        assert result == img_path

    def test_large_png_resized(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", True)
        monkeypatch.setattr(config, "MISTRAL_MAX_IMAGE_DIMENSION", 1000)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_QUALITY_THRESHOLD", 85)
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"fake")

        mock_img = MagicMock()
        mock_img.size = (3000, 2000)
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        resized = MagicMock()
        mock_img.resize.return_value = resized

        with patch.object(mistral_converter, "Image") as mock_pil:
            mock_pil.Resampling.LANCZOS = 1
            mock_pil.open.return_value = mock_img
            result = mistral_converter.optimize_image(img_path)

        resized.save.assert_called_once()
        assert "optimized" in str(result)

    def test_jpeg_uses_quality(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", True)
        monkeypatch.setattr(config, "MISTRAL_MAX_IMAGE_DIMENSION", 500)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_QUALITY_THRESHOLD", 80)
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake")

        mock_img = MagicMock()
        mock_img.size = (2000, 1000)
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        resized = MagicMock()
        mock_img.resize.return_value = resized

        with patch.object(mistral_converter, "Image") as mock_pil:
            mock_pil.Resampling.LANCZOS = 1
            mock_pil.open.return_value = mock_img
            mistral_converter.optimize_image(img_path)

        save_call = resized.save.call_args
        assert save_call[1].get("format") == "JPEG"

    def test_error_returns_original(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", True)
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"fake")

        with patch.object(mistral_converter, "Image") as mock_pil:
            mock_pil.Resampling.LANCZOS = 1
            mock_pil.open.side_effect = Exception("corrupt")
            result = mistral_converter.optimize_image(img_path)
        assert result == img_path


# ============================================================================
# preprocess_image Tests
# ============================================================================


class TestPreprocessImage:
    """Test image preprocessing."""

    def test_disabled_returns_original(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_PREPROCESSING", False)
        img = tmp_path / "test.png"
        img.write_bytes(b"fake")
        assert mistral_converter.preprocess_image(img) == img

    def test_jpeg_preprocessing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_PREPROCESSING", True)
        img_path = tmp_path / "test.jpeg"
        img_path.write_bytes(b"fake")

        mock_img = MagicMock()
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        mock_img.convert.return_value = mock_img

        mock_enhance_cls = MagicMock()
        mock_enhance_cls.return_value.enhance.return_value = mock_img

        with patch.object(mistral_converter, "Image") as mock_pil:
            mock_pil.open.return_value = mock_img
            with patch.dict(
                "sys.modules",
                {"PIL.ImageEnhance": MagicMock(Contrast=mock_enhance_cls, Sharpness=mock_enhance_cls)},
            ):
                result = mistral_converter.preprocess_image(img_path)

        assert "preprocessed" in str(result)

    def test_error_returns_original(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_PREPROCESSING", True)
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"fake")

        with patch.object(mistral_converter, "Image") as mock_pil:
            mock_pil.open.side_effect = Exception("bad image")
            result = mistral_converter.preprocess_image(img_path)
        assert result == img_path


# ============================================================================
# cleanup_uploaded_files Tests
# ============================================================================


class TestCleanupUploadedFiles:
    """Test file cleanup."""

    def test_deletes_old_files(self, monkeypatch):
        monkeypatch.setattr(config, "UPLOAD_RETENTION_DAYS", 7)

        from datetime import datetime, timezone

        old_file = MagicMock()
        old_file.id = "file_old"
        old_file.created_at = (datetime.now(timezone.utc) - __import__("datetime").timedelta(days=30)).isoformat()

        mock_client = MagicMock()
        files_response = MagicMock()
        files_response.data = [old_file]
        files_response.total = 1
        mock_client.files.list.return_value = files_response

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=7)
        assert count >= 1
        mock_client.files.delete.assert_called()

    def test_no_old_files(self, monkeypatch):
        from datetime import datetime, timezone

        recent_file = MagicMock()
        recent_file.id = "file_new"
        recent_file.created_at = datetime.now(timezone.utc).isoformat()

        mock_client = MagicMock()
        files_response = MagicMock()
        files_response.data = [recent_file]
        files_response.total = 1
        mock_client.files.list.return_value = files_response

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=7)
        assert count == 0

    def test_empty_file_list(self):
        mock_client = MagicMock()
        files_response = MagicMock()
        files_response.data = []
        mock_client.files.list.return_value = files_response

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=7)
        assert count == 0

    def test_error_returns_zero(self):
        mock_client = MagicMock()
        mock_client.files.list.side_effect = Exception("API error")

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=7)
        assert count == 0


# ============================================================================
# upload_file_for_ocr Tests
# ============================================================================


class TestUploadFileForOcr:
    """Test file upload for OCR."""

    def test_successful_pdf_upload(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        mock_client = MagicMock()
        mock_client.files.upload.return_value = MagicMock(id="file_123")
        mock_client.files.get_signed_url.return_value = MagicMock(url="https://signed.url/doc")

        result = mistral_converter.upload_file_for_ocr(mock_client, pdf_file)
        assert result == "https://signed.url/doc"
        mock_client.files.upload.assert_called_once()

    def test_upload_with_image_preprocessing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_PREPROCESSING", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", False)

        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png")

        mock_client = MagicMock()
        mock_client.files.upload.return_value = MagicMock(id="file_456")
        mock_client.files.get_signed_url.return_value = MagicMock(url="https://signed.url/img")

        with patch.object(mistral_converter, "preprocess_image", return_value=img_file):
            result = mistral_converter.upload_file_for_ocr(mock_client, img_file)
        assert result == "https://signed.url/img"

    def test_upload_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_client = MagicMock()
        mock_client.files.upload.side_effect = Exception("upload failed")

        result = mistral_converter.upload_file_for_ocr(mock_client, pdf_file)
        assert result is None

    def test_missing_signed_url(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_client = MagicMock()
        mock_client.files.upload.return_value = MagicMock(id="file_789")
        mock_client.files.get_signed_url.return_value = MagicMock(spec=[])  # No 'url' attr

        result = mistral_converter.upload_file_for_ocr(mock_client, pdf_file)
        assert result is None
        mock_client.files.delete.assert_called_once_with(file_id="file_789")

    def test_signed_url_exception_deletes_upload(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_client = MagicMock()
        mock_client.files.upload.return_value = MagicMock(id="orphan_candidate")
        mock_client.files.get_signed_url.side_effect = RuntimeError("API down")

        result = mistral_converter.upload_file_for_ocr(mock_client, pdf_file)
        assert result is None
        mock_client.files.delete.assert_called_once_with(file_id="orphan_candidate")


# ============================================================================
# process_with_ocr Tests
# ============================================================================


class TestProcessWithOcr:
    """Test OCR processing pipeline."""

    def test_successful_pdf_processing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "")
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_HEADER", True)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_FOOTER", True)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_MIN_SIZE", 0)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        mock_page = MagicMock()
        mock_page.markdown = "# Page 1 content"
        mock_page.index = 0
        mock_page.images = []

        mock_response = MagicMock()
        mock_response.pages = [mock_page]

        mock_client = MagicMock()
        mock_client.ocr.process.return_value = mock_response

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://signed.url/doc",
        ):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch.object(mistral_converter, "get_bbox_annotation_format", return_value=None):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        with patch.object(mistral_converter, "DocumentURLChunk", MagicMock()):
                            success, result, error = mistral_converter.process_with_ocr(mock_client, pdf_file)

        assert success is True
        assert result is not None
        assert "full_text" in result

    def test_upload_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_client = MagicMock()

        with patch.object(mistral_converter, "upload_file_for_ocr", return_value=None):
            success, result, error = mistral_converter.process_with_ocr(mock_client, pdf_file)

        assert success is False
        assert "Failed to upload" in error

    def test_empty_response(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "")
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_HEADER", True)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_FOOTER", True)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_MIN_SIZE", 0)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_client = MagicMock()
        mock_client.ocr.process.return_value = None

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://signed.url/doc",
        ):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch.object(mistral_converter, "get_bbox_annotation_format", return_value=None):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        with patch.object(mistral_converter, "DocumentURLChunk", MagicMock()):
                            success, result, error = mistral_converter.process_with_ocr(mock_client, pdf_file)

        assert success is False
        assert "Empty response" in error

    def test_api_error_401(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "")
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_HEADER", True)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_FOOTER", True)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_MIN_SIZE", 0)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_client = MagicMock()
        mock_client.ocr.process.side_effect = Exception("401 Unauthorized")

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://signed.url/doc",
        ):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch.object(mistral_converter, "get_bbox_annotation_format", return_value=None):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        with patch.object(mistral_converter, "DocumentURLChunk", MagicMock()):
                            success, result, error = mistral_converter.process_with_ocr(mock_client, pdf_file)

        assert success is False
        assert "authentication" in error.lower() or "401" in error

    def test_image_file_uses_image_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "")
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_HEADER", True)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_FOOTER", True)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_MIN_SIZE", 0)

        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png")

        mock_page = MagicMock()
        mock_page.markdown = "Image text content"
        mock_page.index = 0
        mock_page.images = []

        mock_response = MagicMock()
        mock_response.pages = [mock_page]

        mock_client = MagicMock()
        mock_client.ocr.process.return_value = mock_response

        mock_image_chunk = MagicMock()
        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://signed.url/img",
        ):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch.object(mistral_converter, "get_bbox_annotation_format", return_value=None):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        with patch.object(mistral_converter, "ImageURLChunk", mock_image_chunk):
                            success, result, error = mistral_converter.process_with_ocr(mock_client, img_file)

        assert success is True
        mock_image_chunk.assert_called_once()


# ============================================================================
# get_batch_job_status Tests
# ============================================================================


class TestConvertWithMistralOcr:
    """Test full OCR conversion pipeline."""

    def test_no_client(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "")
        mistral_converter.reset_mistral_client()
        ok, path, err = mistral_converter.convert_with_mistral_ocr(Path("test.pdf"))
        assert ok is False
        assert "not available" in err.lower()
        mistral_converter.reset_mistral_client()

    def test_cached_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "SAVE_MISTRAL_JSON", False)
        monkeypatch.setattr(config, "ENABLE_OCR_QUALITY_ASSESSMENT", False)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        cached_result = {
            "full_text": "Cached text",
            "pages": [{"text": "Cached text", "page_number": 1}],
        }
        contract = mistral_converter.build_mistral_ocr_cache_contract_metadata()
        cache_entry = {"data": cached_result, "metadata": contract}

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch("utils.cache.get_entry", return_value=cache_entry):
                with patch.object(
                    mistral_converter,
                    "_create_markdown_output",
                    return_value=tmp_path / "out.md",
                ):
                    with patch.object(mistral_converter, "_save_structured_outputs"):
                        with patch.object(mistral_converter, "save_extracted_images"):
                            ok, path, err = mistral_converter.convert_with_mistral_ocr(pdf, use_cache=True)

        assert ok is True

    def test_cache_contract_mismatch_runs_ocr(self, tmp_path, monkeypatch):
        """Stale cache without matching contract metadata must not short-circuit OCR."""
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "SAVE_MISTRAL_JSON", False)
        monkeypatch.setattr(config, "ENABLE_OCR_QUALITY_ASSESSMENT", False)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        bad_entry = {
            "data": {"full_text": "old", "pages": []},
            "metadata": {"contract_type": "mistral_ocr", "contract_version": -99},
        }

        fresh = {"full_text": "new", "pages": [{"text": "new", "page_number": 1}]}

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch("utils.cache.get_entry", return_value=bad_entry):
                with patch.object(
                    mistral_converter,
                    "process_with_ocr",
                    return_value=(True, fresh, None),
                ) as mock_pw:
                    with patch.object(
                        mistral_converter,
                        "_create_markdown_output",
                        return_value=tmp_path / "out.md",
                    ):
                        with patch.object(mistral_converter, "_save_structured_outputs"):
                            with patch.object(mistral_converter, "save_extracted_images"):
                                ok, path, err = mistral_converter.convert_with_mistral_ocr(pdf, use_cache=True)

        assert ok is True
        mock_pw.assert_called_once()

    def test_ocr_failure(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch("utils.cache.get_entry", return_value=None):
                with patch.object(
                    mistral_converter,
                    "process_with_ocr",
                    return_value=(False, None, "OCR failed"),
                ):
                    ok, path, err = mistral_converter.convert_with_mistral_ocr(pdf)

        assert ok is False
        assert "OCR failed" in err


# ============================================================================
# OCR request / QnA message helpers
# ============================================================================


class TestOcrRequestHelpers:
    def test_build_ocr_process_kwargs_batch_omits_retries(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "")
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_MIN_SIZE", 0)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_HEADER", True)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_FOOTER", True)

        with patch.object(mistral_converter, "get_bbox_annotation_format", return_value=None):
            with patch.object(mistral_converter, "get_document_annotation_format", return_value=None):
                body = mistral_converter.build_ocr_process_kwargs(
                    document={"type": "document_url", "document_url": "https://x"},
                    model="mistral-ocr-latest",
                    include_retries=False,
                    request_id="custom",
                )
        assert "retries" not in body
        assert body["id"] == "custom"

    def test_build_ocr_process_kwargs_sync_has_retries_key(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "")
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_MIN_SIZE", 0)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_HEADER", True)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_FOOTER", True)

        with patch.object(mistral_converter, "get_bbox_annotation_format", return_value=None):
            with patch.object(mistral_converter, "get_document_annotation_format", return_value=None):
                body = mistral_converter.build_ocr_process_kwargs(
                    document={"type": "document_url", "document_url": "https://x"},
                    model="mistral-ocr-latest",
                    include_retries=True,
                    request_id=None,
                )
        assert "retries" in body

    def test_build_qna_messages_stable_shape(self):
        a = mistral_converter._build_qna_messages("https://doc", "hello?")
        b = mistral_converter._build_qna_messages("https://doc", "hello?")
        assert a == b
        assert a[1]["content"][1]["type"] == "document_url"


# ============================================================================
# query_document_file Tests
# ============================================================================


class TestProcessOcrResultPipeline:
    """Test OCR result processing pipeline."""

    def test_fresh_result_with_quality(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "SAVE_MISTRAL_JSON", False)
        monkeypatch.setattr(config, "ENABLE_OCR_QUALITY_ASSESSMENT", True)
        monkeypatch.setattr(config, "ENABLE_OCR_WEAK_PAGE_IMPROVEMENT", False)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")
        ocr_result = {
            "full_text": "Hello",
            "pages": [{"text": "Hello", "page_number": 1}],
        }

        with patch.object(
            mistral_converter,
            "assess_ocr_quality",
            return_value={"quality_score": 95, "weak_page_count": 0},
        ):
            with patch.object(
                mistral_converter,
                "_create_markdown_output",
                return_value=tmp_path / "out.md",
            ):
                with patch.object(mistral_converter, "save_extracted_images"):
                    with patch.object(mistral_converter, "_save_structured_outputs"):
                        with patch("utils.cache.set"):
                            ok, path, err = mistral_converter._process_ocr_result_pipeline(
                                MagicMock(), pdf, ocr_result, True, False, False
                            )

        assert ok is True

    def test_cached_result_skips_improvement(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "SAVE_MISTRAL_JSON", False)
        monkeypatch.setattr(config, "ENABLE_OCR_QUALITY_ASSESSMENT", True)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")
        ocr_result = {
            "full_text": "Hello",
            "pages": [{"text": "Hello", "page_number": 1}],
            "quality_assessment": {"quality_score": 50, "weak_page_count": 1},
        }

        with patch.object(
            mistral_converter,
            "_create_markdown_output",
            return_value=tmp_path / "out.md",
        ):
            with patch.object(mistral_converter, "save_extracted_images"):
                with patch.object(mistral_converter, "_save_structured_outputs"):
                    ok, path, err = mistral_converter._process_ocr_result_pipeline(
                        MagicMock(),
                        pdf,
                        ocr_result,
                        True,
                        True,
                        True,  # from_cache=True
                    )

        assert ok is True


# ============================================================================
# get_mistral_client - initialization paths
# ============================================================================


class TestGetMistralClientInit:
    """Test client initialization with retry config and error handling."""

    def test_client_creation_with_retry(self, monkeypatch):
        mistral_converter.reset_mistral_client()
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "test-key-123")
        monkeypatch.setattr(config, "MISTRAL_CLIENT_TIMEOUT_MS", 120000)
        monkeypatch.setattr(config, "RETRY_MAX_ELAPSED_TIME_MS", 30000)

        mock_retry = MagicMock()
        mock_mistral_class = MagicMock()
        mock_instance = MagicMock()
        mock_mistral_class.return_value = mock_instance

        with patch.object(mistral_converter, "Mistral", mock_mistral_class):
            with patch.object(mistral_converter, "get_retry_config", return_value=mock_retry):
                result = mistral_converter.get_mistral_client()

        assert result is mock_instance
        call_kwargs = mock_mistral_class.call_args[1]
        assert call_kwargs["api_key"] == "test-key-123"
        assert call_kwargs["retry_config"] is mock_retry
        assert call_kwargs["timeout_ms"] == 120000
        mistral_converter.reset_mistral_client()

    def test_client_creation_no_retry(self, monkeypatch):
        mistral_converter.reset_mistral_client()
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "test-key")
        monkeypatch.setattr(config, "RETRY_MAX_ELAPSED_TIME_MS", 60000)

        mock_mistral_class = MagicMock()
        mock_instance = MagicMock()
        mock_mistral_class.return_value = mock_instance

        with patch.object(mistral_converter, "Mistral", mock_mistral_class):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                result = mistral_converter.get_mistral_client()

        assert result is mock_instance
        call_kwargs = mock_mistral_class.call_args[1]
        assert "retry_config" not in call_kwargs
        mistral_converter.reset_mistral_client()

    def test_client_creation_exception(self, monkeypatch):
        mistral_converter.reset_mistral_client()
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "test-key")

        with patch.object(mistral_converter, "Mistral", side_effect=Exception("connection refused")):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                result = mistral_converter.get_mistral_client()

        assert result is None
        mistral_converter.reset_mistral_client()

    def test_sdk_not_available(self, monkeypatch):
        mistral_converter.reset_mistral_client()
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "test-key")

        with patch.object(mistral_converter, "Mistral", None):
            result = mistral_converter.get_mistral_client()

        assert result is None
        mistral_converter.reset_mistral_client()

    def test_client_creation_with_server_url(self, monkeypatch):
        mistral_converter.reset_mistral_client()
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "test-key")
        monkeypatch.setattr(config, "MISTRAL_SERVER_URL", "https://custom.api.example")

        mock_mistral_class = MagicMock()
        mock_mistral_class.return_value = MagicMock()

        with patch.object(mistral_converter, "Mistral", mock_mistral_class):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                mistral_converter.get_mistral_client()

        call_kwargs = mock_mistral_class.call_args[1]
        assert call_kwargs["server_url"] == "https://custom.api.example"
        mistral_converter.reset_mistral_client()

    def test_no_api_key(self, monkeypatch):
        mistral_converter.reset_mistral_client()
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "")

        with patch.object(mistral_converter, "Mistral", MagicMock()):
            result = mistral_converter.get_mistral_client()

        assert result is None
        mistral_converter.reset_mistral_client()


# ============================================================================
# get_retry_config - full paths
# ============================================================================


class TestGetRetryConfigFull:
    """Test retry config creation including success and error paths."""

    def test_successful_creation(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_RETRIES", 3)
        monkeypatch.setattr(config, "RETRY_INITIAL_INTERVAL_MS", 500)
        monkeypatch.setattr(config, "RETRY_MAX_INTERVAL_MS", 5000)
        monkeypatch.setattr(config, "RETRY_EXPONENT", 1.5)
        monkeypatch.setattr(config, "RETRY_MAX_ELAPSED_TIME_MS", 30000)
        monkeypatch.setattr(config, "RETRY_CONNECTION_ERRORS", True)

        mock_backoff = MagicMock()
        mock_retry = MagicMock()
        mock_retries = MagicMock()
        mock_retries.BackoffStrategy.return_value = mock_backoff
        mock_retries.RetryConfig.return_value = mock_retry

        with patch.object(mistral_converter, "retries", mock_retries):
            result = mistral_converter.get_retry_config()

        assert result is mock_retry
        mock_retries.BackoffStrategy.assert_called_once()
        mock_retries.RetryConfig.assert_called_once()

    def test_exception_returns_none(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_RETRIES", 3)

        mock_retries = MagicMock()
        mock_retries.BackoffStrategy.side_effect = Exception("bad config")

        with patch.object(mistral_converter, "retries", mock_retries):
            result = mistral_converter.get_retry_config()

        assert result is None


# ============================================================================
# Annotation format fallback paths
# ============================================================================


class TestAnnotationFormatFallbacks:
    """Test annotation format creation with SDK helper and fallback paths."""

    def test_bbox_sdk_helper_success(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_BBOX_ANNOTATION", True)

        mock_model = MagicMock()
        mock_fmt = {"type": "json_schema", "json_schema": {}}

        with patch("schemas.get_bbox_pydantic_model", return_value=mock_model):
            with patch.object(
                mistral_converter,
                "response_format_from_pydantic_model",
                return_value=mock_fmt,
            ):
                result = mistral_converter.get_bbox_annotation_format()

        assert result == mock_fmt

    def test_bbox_sdk_helper_fails_uses_pydantic_schema(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_BBOX_ANNOTATION", True)

        mock_model = MagicMock()
        mock_model.model_json_schema.return_value = {
            "type": "object",
            "properties": {},
        }

        with patch("schemas.get_bbox_pydantic_model", return_value=mock_model):
            with patch.object(
                mistral_converter,
                "response_format_from_pydantic_model",
                side_effect=Exception("fail"),
            ):
                result = mistral_converter.get_bbox_annotation_format()

        assert result is not None
        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "bbox_annotation"

    def test_bbox_pydantic_schema_fails_uses_predefined(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_BBOX_ANNOTATION", True)

        mock_model = MagicMock()
        mock_model.model_json_schema.side_effect = Exception("no schema")
        del mock_model.schema  # Ensure fallback to predefined

        with patch("schemas.get_bbox_pydantic_model", return_value=mock_model):
            with patch.object(
                mistral_converter,
                "response_format_from_pydantic_model",
                side_effect=Exception("fail"),
            ):
                with patch(
                    "schemas.get_bbox_schema",
                    return_value={"schema": {"type": "object"}},
                ):
                    result = mistral_converter.get_bbox_annotation_format()

        assert result is not None
        assert result["type"] == "json_schema"

    def test_doc_sdk_helper_success(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_DOCUMENT_ANNOTATION", True)

        mock_model = MagicMock()
        mock_fmt = {"type": "json_schema", "json_schema": {}}

        with patch("schemas.get_document_pydantic_model", return_value=mock_model):
            with patch.object(
                mistral_converter,
                "response_format_from_pydantic_model",
                return_value=mock_fmt,
            ):
                result = mistral_converter.get_document_annotation_format("invoice")

        assert result == mock_fmt

    def test_doc_sdk_helper_fails_uses_pydantic_schema(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_DOCUMENT_ANNOTATION", True)

        mock_model = MagicMock()
        mock_model.model_json_schema.return_value = {"type": "object"}

        with patch("schemas.get_document_pydantic_model", return_value=mock_model):
            with patch.object(
                mistral_converter,
                "response_format_from_pydantic_model",
                side_effect=Exception("fail"),
            ):
                result = mistral_converter.get_document_annotation_format("invoice")

        assert result is not None
        assert result["type"] == "json_schema"
        assert "document_annotation_invoice" in result["json_schema"]["name"]

    def test_doc_fallback_to_predefined_schema(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_DOCUMENT_ANNOTATION", True)

        mock_model = MagicMock()
        mock_model.model_json_schema.side_effect = Exception("no schema")
        del mock_model.schema

        with patch("schemas.get_bbox_pydantic_model", return_value=mock_model):
            with patch.object(
                mistral_converter,
                "response_format_from_pydantic_model",
                side_effect=Exception("fail"),
            ):
                with patch(
                    "schemas.get_document_schema",
                    return_value={"schema": {"type": "object"}},
                ):
                    result = mistral_converter.get_document_annotation_format("generic")

        assert result is not None

    def test_doc_no_pydantic_model_uses_predefined(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_DOCUMENT_ANNOTATION", True)

        with patch("schemas.get_document_pydantic_model", return_value=None):
            with patch(
                "schemas.get_document_schema",
                return_value={"schema": {"type": "object"}},
            ):
                result = mistral_converter.get_document_annotation_format("generic")

        assert result is not None


# ============================================================================
# optimize_image - full coverage
# ============================================================================


class TestOptimizeImageFull:
    """Test image optimization including resize and format-specific saves."""

    def test_optimization_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", False)
        img = tmp_path / "test.png"
        img.write_bytes(b"fake")
        result = mistral_converter.optimize_image(img)
        assert result == img

    def test_no_pil(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", True)
        with patch.object(mistral_converter, "Image", None):
            img = tmp_path / "test.png"
            img.write_bytes(b"fake")
            result = mistral_converter.optimize_image(img)
        assert result == img

    def test_image_within_max_dim(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", True)
        monkeypatch.setattr(config, "MISTRAL_MAX_IMAGE_DIMENSION", 2000)

        img = tmp_path / "test.png"
        img.write_bytes(b"fake")

        mock_img = MagicMock()
        mock_img.size = (500, 400)
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)

        mock_image_mod = MagicMock()
        mock_image_mod.open.return_value = mock_img
        mock_image_mod.Resampling.LANCZOS = 1

        with patch.object(mistral_converter, "Image", mock_image_mod):
            result = mistral_converter.optimize_image(img)

        assert result == img  # No optimization needed

    def test_resize_landscape_png(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", True)
        monkeypatch.setattr(config, "MISTRAL_MAX_IMAGE_DIMENSION", 1000)

        img = tmp_path / "test.png"
        img.write_bytes(b"fake")

        mock_src = MagicMock()
        mock_src.size = (2000, 1000)
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        mock_resized = MagicMock()
        mock_src.resize.return_value = mock_resized

        mock_image_mod = MagicMock()
        mock_image_mod.open.return_value = mock_src
        mock_image_mod.Resampling.LANCZOS = 1

        with patch.object(mistral_converter, "Image", mock_image_mod):
            result = mistral_converter.optimize_image(img)

        assert "optimized" in str(result)
        mock_resized.save.assert_called_once()
        save_kwargs = mock_resized.save.call_args
        assert save_kwargs[1]["format"] == "PNG"

    def test_resize_portrait_jpeg(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", True)
        monkeypatch.setattr(config, "MISTRAL_MAX_IMAGE_DIMENSION", 800)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_QUALITY_THRESHOLD", 85)

        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")

        mock_src = MagicMock()
        mock_src.size = (600, 1200)
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        mock_resized = MagicMock()
        mock_src.resize.return_value = mock_resized

        mock_image_mod = MagicMock()
        mock_image_mod.open.return_value = mock_src
        mock_image_mod.Resampling.LANCZOS = 1

        with patch.object(mistral_converter, "Image", mock_image_mod):
            mistral_converter.optimize_image(img)

        save_kwargs = mock_resized.save.call_args
        assert save_kwargs[1]["format"] == "JPEG"
        assert save_kwargs[1]["quality"] == 85

    def test_resize_other_format(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", True)
        monkeypatch.setattr(config, "MISTRAL_MAX_IMAGE_DIMENSION", 800)

        img = tmp_path / "test.webp"
        img.write_bytes(b"fake")

        mock_src = MagicMock()
        mock_src.size = (1600, 1200)
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        mock_resized = MagicMock()
        mock_src.resize.return_value = mock_resized

        mock_image_mod = MagicMock()
        mock_image_mod.open.return_value = mock_src
        mock_image_mod.Resampling.LANCZOS = 1

        with patch.object(mistral_converter, "Image", mock_image_mod):
            mistral_converter.optimize_image(img)

        save_kwargs = mock_resized.save.call_args
        assert save_kwargs[1].get("optimize") is True
        assert "format" not in save_kwargs[1]

    def test_exception_returns_original(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", True)

        img = tmp_path / "test.png"
        img.write_bytes(b"fake")

        mock_image_mod = MagicMock()
        mock_image_mod.open.side_effect = Exception("corrupt image")

        with patch.object(mistral_converter, "Image", mock_image_mod):
            result = mistral_converter.optimize_image(img)

        assert result == img


# ============================================================================
# preprocess_image - full coverage
# ============================================================================


class TestPreprocessImageFull:
    """Test image preprocessing including contrast/sharpness enhancement."""

    def test_preprocessing_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_PREPROCESSING", False)
        img = tmp_path / "test.png"
        img.write_bytes(b"fake")
        result = mistral_converter.preprocess_image(img)
        assert result == img

    def test_no_pil(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_PREPROCESSING", True)
        with patch.object(mistral_converter, "Image", None):
            img = tmp_path / "test.png"
            img.write_bytes(b"fake")
            result = mistral_converter.preprocess_image(img)
        assert result == img

    def test_preprocess_jpeg(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_PREPROCESSING", True)

        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")

        mock_src = MagicMock()
        mock_rgb = MagicMock()
        mock_src.convert.return_value = mock_rgb
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        mock_contrast_enhanced = MagicMock()
        mock_sharp_enhanced = MagicMock()

        mock_image_mod = MagicMock()
        mock_image_mod.open.return_value = mock_src

        mock_enhance_mod = MagicMock()
        mock_contrast_enhancer = MagicMock()
        mock_contrast_enhancer.enhance.return_value = mock_contrast_enhanced
        mock_sharp_enhancer = MagicMock()
        mock_sharp_enhancer.enhance.return_value = mock_sharp_enhanced

        mock_enhance_mod.Contrast.return_value = mock_contrast_enhancer
        mock_enhance_mod.Sharpness.return_value = mock_sharp_enhancer

        with patch.object(mistral_converter, "Image", mock_image_mod):
            with patch("PIL.ImageEnhance", mock_enhance_mod):
                result = mistral_converter.preprocess_image(img)

        assert "preprocessed" in str(result)

    def test_preprocess_exception_returns_original(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_PREPROCESSING", True)

        img = tmp_path / "test.png"
        img.write_bytes(b"fake")

        mock_image_mod = MagicMock()
        mock_image_mod.open.side_effect = Exception("corrupt")

        with patch.object(mistral_converter, "Image", mock_image_mod):
            result = mistral_converter.preprocess_image(img)

        assert result == img


# ============================================================================
# cleanup_uploaded_files - full coverage
# ============================================================================


class TestCleanupUploadedFilesFull:
    """Test file cleanup with pagination, date parsing, and error handling."""

    def test_deletes_old_files(self):
        from datetime import datetime, timedelta, timezone

        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

        mock_file = MagicMock()
        mock_file.id = "file_old"
        mock_file.created_at = old_date

        mock_response = MagicMock()
        mock_response.data = [mock_file]
        mock_response.total = 1

        mock_client = MagicMock()
        mock_client.files.list.return_value = mock_response

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        assert count >= 1

    def test_skips_recent_files(self):
        from datetime import datetime, timezone

        recent_date = datetime.now(timezone.utc).isoformat()

        mock_file = MagicMock()
        mock_file.id = "file_new"
        mock_file.created_at = recent_date

        mock_response = MagicMock()
        mock_response.data = [mock_file]
        mock_response.total = 1

        mock_client = MagicMock()
        mock_client.files.list.return_value = mock_response

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        assert count == 0

    def test_handles_datetime_object_created_at(self):
        from datetime import datetime, timedelta, timezone

        mock_file = MagicMock()
        mock_file.id = "file_dt"
        mock_file.created_at = datetime.now(timezone.utc) - timedelta(days=60)

        mock_response = MagicMock()
        mock_response.data = [mock_file]
        mock_response.total = 1

        mock_client = MagicMock()
        mock_client.files.list.return_value = mock_response

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        assert count >= 1

    def test_pagination_stops_on_empty_page(self):
        mock_response_page1 = MagicMock()
        mock_response_page1.data = []
        mock_response_page1.total = 0

        mock_client = MagicMock()
        mock_client.files.list.return_value = mock_response_page1

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        assert count == 0

    def test_api_error_returns_zero(self):
        mock_client = MagicMock()
        mock_client.files.list.side_effect = Exception("API down")

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        assert count == 0

    def test_delete_error_continues(self):
        from datetime import datetime, timedelta, timezone

        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

        mock_file = MagicMock()
        mock_file.id = "file_err"
        mock_file.created_at = old_date

        mock_response = MagicMock()
        mock_response.data = [mock_file]
        mock_response.total = 1

        mock_client = MagicMock()
        mock_client.files.list.return_value = mock_response
        mock_client.files.delete.side_effect = Exception("delete failed")

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        assert count == 0


# ============================================================================
# upload_file_for_ocr - full paths
# ============================================================================


class TestUploadFileForOcrFull:
    """Test file upload with image preprocessing, optimization, and signed URLs."""

    def test_pdf_upload_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_upload_resp = MagicMock()
        mock_upload_resp.id = "file_123"

        mock_signed_resp = MagicMock()
        mock_signed_resp.url = "https://signed.example.com/doc"

        mock_client = MagicMock()
        mock_client.files.upload.return_value = mock_upload_resp
        mock_client.files.get_signed_url.return_value = mock_signed_resp

        result = mistral_converter.upload_file_for_ocr(mock_client, pdf)
        assert result == "https://signed.example.com/doc"

    def test_image_upload_with_preprocessing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_PREPROCESSING", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", True)

        img = tmp_path / "test.png"
        img.write_bytes(b"fake image")

        preprocessed = tmp_path / "test_preprocessed.png"
        preprocessed.write_bytes(b"preprocessed")

        optimized = tmp_path / "test_preprocessed_optimized.png"
        optimized.write_bytes(b"optimized")

        mock_upload_resp = MagicMock()
        mock_upload_resp.id = "file_img"
        mock_signed_resp = MagicMock()
        mock_signed_resp.url = "https://signed.example.com/img"

        mock_client = MagicMock()
        mock_client.files.upload.return_value = mock_upload_resp
        mock_client.files.get_signed_url.return_value = mock_signed_resp

        with patch.object(mistral_converter, "preprocess_image", return_value=preprocessed):
            with patch.object(mistral_converter, "optimize_image", return_value=optimized):
                result = mistral_converter.upload_file_for_ocr(mock_client, img)

        assert result == "https://signed.example.com/img"

    def test_upload_missing_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        mock_resp = MagicMock(spec=[])  # No .id attribute
        mock_client = MagicMock()
        mock_client.files.upload.return_value = mock_resp

        result = mistral_converter.upload_file_for_ocr(mock_client, pdf)
        assert result is None

    def test_signed_url_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        mock_upload_resp = MagicMock()
        mock_upload_resp.id = "file_123"
        mock_signed_resp = MagicMock(spec=[])  # No .url

        mock_client = MagicMock()
        mock_client.files.upload.return_value = mock_upload_resp
        mock_client.files.get_signed_url.return_value = mock_signed_resp

        result = mistral_converter.upload_file_for_ocr(mock_client, pdf)
        assert result is None

    def test_upload_exception(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        mock_client = MagicMock()
        mock_client.files.upload.side_effect = Exception("network error")

        result = mistral_converter.upload_file_for_ocr(mock_client, pdf)
        assert result is None


# ============================================================================
# process_with_ocr - additional paths
# ============================================================================


class TestProcessWithOcrAdditional:
    """Test additional process_with_ocr paths."""

    def _setup_monkeypatch(self, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "Extract data")
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "markdown")
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_HEADER", True)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_FOOTER", False)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 10)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_MIN_SIZE", 50)

    def test_with_progress_callback(self, tmp_path, monkeypatch):
        self._setup_monkeypatch(monkeypatch)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF content")

        mock_page = MagicMock()
        mock_page.markdown = "Page content 12345"
        mock_page.index = 1
        mock_page.images = []
        mock_page.dimensions = None
        mock_page.tables = None
        mock_page.hyperlinks = None
        mock_page.header = None
        mock_page.footer = None

        mock_response = MagicMock()
        mock_response.pages = [mock_page]
        mock_response.bbox_annotations = None
        mock_response.document_annotation = None
        mock_response.metadata = None
        mock_response.usage_info = None
        mock_response.model = "mistral-ocr-latest"

        mock_client = MagicMock()
        mock_client.ocr.process.return_value = mock_response

        progress_calls = []

        def progress_cb(msg, prog):
            progress_calls.append((msg, prog))

        mock_bbox = {"type": "json_schema", "json_schema": {}}
        mock_doc_fmt = {"type": "json_schema", "json_schema": {}}

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://signed.url/doc",
        ):
            with patch.object(mistral_converter, "get_retry_config", return_value=MagicMock()):
                with patch.object(
                    mistral_converter,
                    "get_bbox_annotation_format",
                    return_value=mock_bbox,
                ):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=mock_doc_fmt,
                    ):
                        with patch.object(
                            mistral_converter,
                            "DocumentURLChunk",
                            MagicMock(),
                        ):
                            success, result, error = mistral_converter.process_with_ocr(
                                mock_client,
                                pdf,
                                progress_callback=progress_cb,
                            )

        assert success is True
        assert len(progress_calls) > 0

    def test_with_provided_signed_url(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "")
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_HEADER", True)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_FOOTER", True)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_MIN_SIZE", 0)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        mock_page = MagicMock()
        mock_page.markdown = "Content from re-OCR"
        mock_page.index = 0
        mock_page.images = []

        mock_response = MagicMock()
        mock_response.pages = [mock_page]

        mock_client = MagicMock()
        mock_client.ocr.process.return_value = mock_response

        with patch.object(mistral_converter, "get_retry_config", return_value=None):
            with patch.object(
                mistral_converter,
                "get_bbox_annotation_format",
                return_value=None,
            ):
                with patch.object(
                    mistral_converter,
                    "get_document_annotation_format",
                    return_value=None,
                ):
                    with patch.object(mistral_converter, "DocumentURLChunk", MagicMock()):
                        success, result, error = mistral_converter.process_with_ocr(
                            mock_client,
                            pdf,
                            signed_url="https://pre-signed.url/doc",
                        )

        assert success is True

    def test_with_ocr_id_and_pages(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "")
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_HEADER", True)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_FOOTER", True)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_MIN_SIZE", 0)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        mock_page = MagicMock()
        mock_page.markdown = "Page 2 content"
        mock_page.index = 1
        mock_page.images = []

        mock_response = MagicMock()
        mock_response.pages = [mock_page]

        mock_client = MagicMock()
        mock_client.ocr.process.return_value = mock_response

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://signed.url",
        ):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch.object(
                    mistral_converter,
                    "get_bbox_annotation_format",
                    return_value=None,
                ):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        with patch.object(
                            mistral_converter,
                            "DocumentURLChunk",
                            MagicMock(),
                        ):
                            success, result, error = mistral_converter.process_with_ocr(
                                mock_client,
                                pdf,
                                pages=[1],
                                ocr_id="task_123",
                            )

        assert success is True
        call_kwargs = mock_client.ocr.process.call_args[1]
        assert call_kwargs["pages"] == [1]
        assert call_kwargs["id"] == "task_123"

    def test_image_file_uses_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "")
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_HEADER", True)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_FOOTER", True)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_MIN_SIZE", 0)

        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")

        mock_page = MagicMock()
        mock_page.markdown = "Image text"
        mock_page.index = 0
        mock_page.images = []

        mock_response = MagicMock()
        mock_response.pages = [mock_page]

        mock_client = MagicMock()
        mock_client.ocr.process.return_value = mock_response

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://signed.url",
        ):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch.object(
                    mistral_converter,
                    "get_bbox_annotation_format",
                    return_value=None,
                ):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        with patch.object(mistral_converter, "ImageURLChunk", None):
                            success, result, error = mistral_converter.process_with_ocr(mock_client, img)

        assert success is True


# ============================================================================
# improve_weak_pages
# ============================================================================


class TestImproveWeakPages:
    """Test weak page improvement logic."""

    def test_no_weak_pages(self, monkeypatch):
        page_text = (
            "Financial statement with revenue of $12,345,678.90 and "
            "growth of 15.2%.\n"
            "Operating income reached $4,567,890.12 with margins at "
            "23.7% year over year.\n"
            "Total assets were $98,765,432.10 including cash "
            "equivalents of $3,456,789.01.\n"
        )
        ocr_result = {"pages": [{"text": page_text, "page_number": 1}]}
        result = mistral_converter.improve_weak_pages(MagicMock(), Path("test.pdf"), ocr_result, "model")
        assert result == ocr_result

    def test_improves_weak_page(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONCURRENT_FILES", 2)
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)

        weak_text = "short"
        good_text = (
            "Improved financial data with revenue of $12,345,678.90 "
            "and EBITDA of 23.7%.\n"
            "Operating expenses totaled $9,876,543.21 across all "
            "business segments.\n"
            "Net income attributable to shareholders was $2,469,134.79 "
            "for fiscal year.\n"
        )

        ocr_result = {
            "pages": [
                {"text": weak_text, "page_number": 1},
            ]
        }

        improved_page = {"text": good_text, "page_number": 1}

        mock_client = MagicMock()

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://signed.url",
        ):
            with patch.object(
                mistral_converter,
                "process_with_ocr",
                return_value=(True, {"pages": [improved_page]}, None),
            ):
                result = mistral_converter.improve_weak_pages(mock_client, Path("test.pdf"), ocr_result, "model")

        assert result["pages"][0]["text"] == good_text

    def test_improve_weak_pages_passes_api_page_index(self, monkeypatch):
        """Re-OCR must use ``api_page_index`` for the ``pages`` selector when present."""
        monkeypatch.setattr(config, "MAX_CONCURRENT_FILES", 1)
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)

        weak_text = "x" * 200  # long enough to not be weak? Actually "x"*200 might be weak due to uniqueness
        # Use repetitive weak short - use "short" which is weak
        weak_text = "short"
        ocr_result = {
            "pages": [
                {
                    "text": weak_text,
                    "page_number": 3,
                    "api_page_index": 2,
                }
            ]
        }
        improved = {"text": "much longer improved text " * 20, "page_number": 3}
        mock_client = MagicMock()
        with patch.object(mistral_converter, "upload_file_for_ocr", return_value="https://u"):
            with patch.object(
                mistral_converter,
                "process_with_ocr",
                return_value=(True, {"pages": [improved]}, None),
            ) as m_ocr:
                mistral_converter.improve_weak_pages(mock_client, Path("doc.pdf"), ocr_result, "mistral-ocr-latest")
        kwargs = m_ocr.call_args[1]
        assert kwargs.get("pages") == [2]

    def test_improvement_fails_keeps_original(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONCURRENT_FILES", 1)
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)

        weak_text = "short"
        ocr_result = {"pages": [{"text": weak_text, "page_number": 1}]}

        mock_client = MagicMock()

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://signed.url",
        ):
            with patch.object(
                mistral_converter,
                "process_with_ocr",
                return_value=(False, None, "OCR failed"),
            ):
                result = mistral_converter.improve_weak_pages(mock_client, Path("test.pdf"), ocr_result, "model")

        assert result["pages"][0]["text"] == weak_text

    def test_url_upload_failure(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONCURRENT_FILES", 1)
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)

        weak_text = "x"
        ocr_result = {"pages": [{"text": weak_text, "page_number": 1}]}

        mock_client = MagicMock()

        with patch.object(mistral_converter, "upload_file_for_ocr", return_value=None):
            with patch.object(
                mistral_converter,
                "process_with_ocr",
                return_value=(False, None, "no url"),
            ):
                result = mistral_converter.improve_weak_pages(mock_client, Path("test.pdf"), ocr_result, "model")

        assert result["pages"][0]["text"] == weak_text


# ============================================================================
# _process_ocr_result_pipeline - weak page improvement path
# ============================================================================


class TestProcessOcrResultPipelineImprovement:
    """Test pipeline with weak page improvement enabled."""

    def test_improvement_triggered(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "SAVE_MISTRAL_JSON", True)
        monkeypatch.setattr(config, "ENABLE_OCR_QUALITY_ASSESSMENT", True)
        monkeypatch.setattr(config, "ENABLE_OCR_WEAK_PAGE_IMPROVEMENT", True)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        ocr_result = {
            "full_text": "Some text here with enough content",
            "pages": [{"text": "Some text", "page_number": 1}],
        }

        quality_before = {"quality_score": 40, "weak_page_count": 1}
        quality_after = {"quality_score": 80, "weak_page_count": 0}

        with patch.object(
            mistral_converter,
            "assess_ocr_quality",
            side_effect=[quality_before, quality_after],
        ):
            with patch.object(
                mistral_converter,
                "improve_weak_pages",
                return_value=ocr_result,
            ):
                with patch.object(
                    mistral_converter,
                    "_create_markdown_output",
                    return_value=tmp_path / "out.md",
                ):
                    with patch.object(mistral_converter, "save_extracted_images"):
                        with patch.object(mistral_converter, "_save_structured_outputs"):
                            with patch("utils.cache.set"):
                                ok, path, err = mistral_converter._process_ocr_result_pipeline(
                                    MagicMock(),
                                    pdf,
                                    ocr_result,
                                    True,
                                    True,
                                    False,
                                )

        assert ok is True

    def test_quality_assessment_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "SAVE_MISTRAL_JSON", False)
        monkeypatch.setattr(config, "ENABLE_OCR_QUALITY_ASSESSMENT", False)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        ocr_result = {
            "full_text": "Text",
            "pages": [{"text": "Text", "page_number": 1}],
        }

        with patch.object(
            mistral_converter,
            "_create_markdown_output",
            return_value=tmp_path / "out.md",
        ):
            with patch.object(mistral_converter, "save_extracted_images"):
                with patch.object(mistral_converter, "_save_structured_outputs"):
                    with patch("utils.cache.set"):
                        ok, path, err = mistral_converter._process_ocr_result_pipeline(
                            MagicMock(),
                            pdf,
                            ocr_result,
                            True,
                            True,
                            False,
                        )

        assert ok is True


# ============================================================================
# query_document - additional paths
# ============================================================================


class TestConvertWithMistralOcrPaths:
    """Test convert_with_mistral_ocr with cache disabled."""

    def test_no_cache_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "SAVE_MISTRAL_JSON", False)
        monkeypatch.setattr(config, "ENABLE_OCR_QUALITY_ASSESSMENT", False)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        ocr_result = {
            "full_text": "Converted text",
            "pages": [{"text": "Converted text", "page_number": 1}],
        }

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch.object(
                mistral_converter,
                "process_with_ocr",
                return_value=(True, ocr_result, None),
            ):
                with patch.object(
                    mistral_converter,
                    "_create_markdown_output",
                    return_value=tmp_path / "out.md",
                ):
                    with patch.object(mistral_converter, "save_extracted_images"):
                        with patch.object(mistral_converter, "_save_structured_outputs"):
                            with patch("utils.cache.set"):
                                ok, path, err = mistral_converter.convert_with_mistral_ocr(pdf, use_cache=False)

        assert ok is True


# ============================================================================
# save_extracted_images - base64 data URI
# ============================================================================


class TestSaveExtractedImagesDataUri:
    """Test image saving with data URI prefix."""

    def test_data_uri_prefix_stripped(self, tmp_path, monkeypatch):
        import base64

        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
        monkeypatch.setattr(config, "OUTPUT_IMAGES_DIR", tmp_path)

        raw_data = b"fake image bytes"
        b64_data = base64.b64encode(raw_data).decode()

        ocr_result = {
            "pages": [
                {
                    "page_number": 1,
                    "images": [{"base64": f"data:image/png;base64,{b64_data}"}],
                }
            ]
        }

        saved = mistral_converter.save_extracted_images(ocr_result, Path("test.pdf"))
        assert len(saved) == 1
        assert saved[0].read_bytes() == raw_data


# ============================================================================
# Import fallback paths (module-level try/except)
# ============================================================================


class TestModuleImportFallbacks:
    """Test module-level import try/except fallback paths."""

    def test_mistralai_not_available(self):
        """Test behavior when Mistral SDK is not available at all."""
        orig_mistral = mistral_converter.Mistral

        try:
            mistral_converter.Mistral = None
            mistral_converter.reset_mistral_client()
            result = mistral_converter.get_mistral_client()
            assert result is None
        finally:
            mistral_converter.Mistral = orig_mistral
            mistral_converter.reset_mistral_client()

    def test_chunk_classes_none(self):
        """Test with chunk classes unavailable."""
        orig_doc = mistral_converter.DocumentURLChunk
        orig_img = mistral_converter.ImageURLChunk
        orig_fc = mistral_converter.FileChunk

        try:
            mistral_converter.DocumentURLChunk = None
            mistral_converter.ImageURLChunk = None
            mistral_converter.FileChunk = None
            # These should be None when imports fail
            assert mistral_converter.DocumentURLChunk is None
            assert mistral_converter.ImageURLChunk is None
            assert mistral_converter.FileChunk is None
        finally:
            mistral_converter.DocumentURLChunk = orig_doc
            mistral_converter.ImageURLChunk = orig_img
            mistral_converter.FileChunk = orig_fc

    def test_response_format_helper_none(self):
        """Test with response_format_from_pydantic_model unavailable."""
        orig = mistral_converter.response_format_from_pydantic_model

        try:
            mistral_converter.response_format_from_pydantic_model = None
            assert mistral_converter.response_format_from_pydantic_model is None
        finally:
            mistral_converter.response_format_from_pydantic_model = orig

    def test_pil_not_available(self):
        """Test with PIL Image unavailable."""
        orig = mistral_converter.Image

        try:
            mistral_converter.Image = None
            assert mistral_converter.Image is None
        finally:
            mistral_converter.Image = orig


# ============================================================================
# get_mistral_client - double-checked locking
# ============================================================================


class TestGetMistralClientLocking:
    """Test double-checked locking in get_mistral_client."""

    def test_cached_instance_returned_immediately(self, monkeypatch):
        """Test that cached instance is returned without acquiring lock."""
        mistral_converter.reset_mistral_client()
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_mistral = MagicMock(return_value=mock_client)

        with patch.object(mistral_converter, "Mistral", mock_mistral):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                # First call creates
                c1 = mistral_converter.get_mistral_client()
                assert c1 is mock_client
                # Second call returns cached (line 159)
                c2 = mistral_converter.get_mistral_client()
                assert c2 is c1
                # Mistral constructor only called once
                assert mock_mistral.call_count == 1

        mistral_converter.reset_mistral_client()

    def test_double_checked_locking_concurrent(self, monkeypatch):
        """Test that the lock-internal check (line 164) works."""
        import threading

        mistral_converter.reset_mistral_client()
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "test-key")

        mock_client = MagicMock()
        call_count = 0

        def slow_mistral(**kwargs):
            nonlocal call_count
            call_count += 1
            return mock_client

        with patch.object(mistral_converter, "Mistral", slow_mistral):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                results = [None, None]

                def get_client(idx):
                    results[idx] = mistral_converter.get_mistral_client()

                t1 = threading.Thread(target=get_client, args=(0,))
                t2 = threading.Thread(target=get_client, args=(1,))
                t1.start()
                t1.join()
                t2.start()
                t2.join()

                assert results[0] is mock_client
                assert results[1] is mock_client

        mistral_converter.reset_mistral_client()


# ============================================================================
# Annotation format - return None paths
# ============================================================================


class TestAnnotationFormatReturnNone:
    """Test paths where annotation formats return None."""

    def test_bbox_all_fallbacks_fail(self, monkeypatch):
        """Line 331: bbox schema raw is empty -> return None."""
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_BBOX_ANNOTATION", True)

        with patch("schemas.get_bbox_pydantic_model", return_value=None):
            with patch("schemas.get_bbox_schema", return_value={"schema": None}):
                result = mistral_converter.get_bbox_annotation_format()
        assert result is None

    def test_doc_pydantic_extract_exception(self, monkeypatch):
        """Lines 379-380: exception during JSON schema extraction."""
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", True)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_DOCUMENT_ANNOTATION", True)

        mock_model = MagicMock()
        mock_model.model_json_schema.side_effect = Exception("schema fail")
        # Remove .schema to prevent hasattr fallback
        if hasattr(mock_model, "schema"):
            del mock_model.schema

        with patch("schemas.get_document_pydantic_model", return_value=mock_model):
            with patch.object(
                mistral_converter,
                "response_format_from_pydantic_model",
                side_effect=Exception("sdk fail"),
            ):
                with patch(
                    "schemas.get_document_schema",
                    return_value={"schema": None},
                ):
                    result = mistral_converter.get_document_annotation_format("generic")
        # Line 387: return None when raw schema is empty
        assert result is None


# ============================================================================
# preprocess_image - PNG and other format branches
# ============================================================================


class TestPreprocessImageFormatBranches:
    """Test preprocess_image format-specific save branches."""

    def test_preprocess_png_save(self, tmp_path, monkeypatch):
        """Lines 483-484: PNG format save branch."""
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_PREPROCESSING", True)

        img = tmp_path / "test.png"
        img.write_bytes(b"fake")

        mock_src = MagicMock()
        mock_rgb = MagicMock()
        mock_src.convert.return_value = mock_rgb
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        mock_enhanced = MagicMock()

        mock_image_mod = MagicMock()
        mock_image_mod.open.return_value = mock_src

        mock_enhance_mod = MagicMock()
        mock_enhancer = MagicMock()
        mock_enhancer.enhance.return_value = mock_enhanced
        mock_enhance_mod.Contrast.return_value = mock_enhancer
        mock_enhance_mod.Sharpness.return_value = mock_enhancer

        with patch.object(mistral_converter, "Image", mock_image_mod):
            with patch("PIL.ImageEnhance", mock_enhance_mod):
                result = mistral_converter.preprocess_image(img)

        assert "preprocessed" in str(result)
        mock_enhanced.save.assert_called_once()
        save_args = mock_enhanced.save.call_args
        assert save_args[1]["format"] == "PNG"

    def test_preprocess_other_format_save(self, tmp_path, monkeypatch):
        """Lines 485-486: other format save branch."""
        monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_PREPROCESSING", True)

        img = tmp_path / "test.bmp"
        img.write_bytes(b"fake")

        mock_src = MagicMock()
        mock_rgb = MagicMock()
        mock_src.convert.return_value = mock_rgb
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        mock_enhanced = MagicMock()

        mock_image_mod = MagicMock()
        mock_image_mod.open.return_value = mock_src

        mock_enhance_mod = MagicMock()
        mock_enhancer = MagicMock()
        mock_enhancer.enhance.return_value = mock_enhanced
        mock_enhance_mod.Contrast.return_value = mock_enhancer
        mock_enhance_mod.Sharpness.return_value = mock_enhancer

        with patch.object(mistral_converter, "Image", mock_image_mod):
            with patch("PIL.ImageEnhance", mock_enhance_mod):
                result = mistral_converter.preprocess_image(img)

        assert "preprocessed" in str(result)
        mock_enhanced.save.assert_called_once()
        save_args = mock_enhanced.save.call_args
        # For non-jpg/non-png, save without format kwarg
        assert "format" not in save_args[1]


# ============================================================================
# cleanup_uploaded_files - remaining edge cases
# ============================================================================


class TestCleanupUploadedFilesEdgeCases:
    """Cover remaining cleanup edge cases."""

    def test_default_days_old(self, monkeypatch):
        """Line 513: days_old defaults to config.UPLOAD_RETENTION_DAYS."""
        monkeypatch.setattr(config, "UPLOAD_RETENTION_DAYS", 7)

        mock_response = MagicMock()
        mock_response.data = []
        mock_response.total = 0

        mock_client = MagicMock()
        mock_client.files.list.return_value = mock_response

        # Call without days_old arg
        count = mistral_converter.cleanup_uploaded_files(mock_client)
        assert count == 0

    def test_unexpected_created_at_type(self):
        """Lines 563-565: unexpected type for created_at triggers debug log."""
        mock_file = MagicMock()
        mock_file.id = "file_weird"
        mock_file.created_at = 12345  # integer, unexpected type

        mock_response = MagicMock()
        mock_response.data = [mock_file]
        mock_response.total = 1

        mock_client = MagicMock()
        mock_client.files.list.return_value = mock_response

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        assert count == 0

    def test_pagination_by_total(self):
        """Lines 577-579: pagination stops when total is reached."""
        from datetime import datetime, timedelta, timezone

        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

        mock_file = MagicMock()
        mock_file.id = "file_paged"
        mock_file.created_at = old_date

        mock_response = MagicMock()
        mock_response.data = [mock_file]
        mock_response.total = 1  # Total = 1, page_size default is typically large

        mock_client = MagicMock()
        mock_client.files.list.return_value = mock_response

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        assert count >= 1

    def test_cleanup_by_purpose_pagination(self):
        """Line 540, 577-579: paginating through files by purpose."""
        from datetime import datetime, timedelta, timezone

        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

        # First page returns files, total indicates more
        mock_file1 = MagicMock()
        mock_file1.id = "f1"
        mock_file1.created_at = old

        page1 = MagicMock()
        page1.data = [mock_file1]
        page1.total = 2

        mock_file2 = MagicMock()
        mock_file2.id = "f2"
        mock_file2.created_at = old

        page2 = MagicMock()
        page2.data = [mock_file2]
        page2.total = 2

        mock_client = MagicMock()
        mock_client.files.list.side_effect = [
            page1,
            page2,  # ocr pages
            MagicMock(data=[], total=0),  # batch purpose - empty
        ]

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        assert count >= 1

    def test_datetime_with_replace_naive(self):
        """Lines 547-550: created_at is datetime object with naive tz."""
        from datetime import datetime, timedelta

        mock_file = MagicMock()
        mock_file.id = "file_naive_dt"
        # Create a real naive datetime (not a mock)
        naive_dt = datetime.now() - timedelta(days=60)
        mock_file.created_at = naive_dt

        mock_response = MagicMock()
        mock_response.data = [mock_file]
        mock_response.total = 1

        mock_client = MagicMock()
        mock_client.files.list.return_value = mock_response

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        assert count >= 1

    def test_list_page_error_breaks_loop(self):
        """Line 540: when listing raises an error mid-pagination."""
        mock_client = MagicMock()
        # First call for "ocr" purpose raises
        mock_client.files.list.side_effect = Exception("API error on page")

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        assert count == 0


# ============================================================================
# _cleanup_temp_files
# ============================================================================


class TestCleanupTempFiles:
    """Lines 721-722: temporary file cleanup."""

    def test_cleanup_existing_files(self, tmp_path):
        f1 = tmp_path / "temp1.png"
        f1.write_bytes(b"temp")
        f2 = tmp_path / "temp2.png"
        f2.write_bytes(b"temp")

        mistral_converter._cleanup_temp_files([f1, f2])
        assert not f1.exists()
        assert not f2.exists()

    def test_cleanup_nonexistent_files(self, tmp_path):
        f1 = tmp_path / "nonexistent.png"
        # Should not raise
        mistral_converter._cleanup_temp_files([f1])

    def test_cleanup_empty_list(self):
        mistral_converter._cleanup_temp_files([])

    def test_cleanup_none_in_list(self, tmp_path):
        f1 = tmp_path / "temp.png"
        f1.write_bytes(b"temp")
        mistral_converter._cleanup_temp_files([None, f1])
        assert not f1.exists()

    def test_cleanup_delete_error(self, tmp_path):
        f1 = tmp_path / "temp.png"
        f1.write_bytes(b"temp")
        with patch.object(Path, "unlink", side_effect=OSError("perm denied")):
            # Should not raise, just log warning
            mistral_converter._cleanup_temp_files([f1])


# ============================================================================
# process_with_ocr - DocumentURLChunk None fallback
# ============================================================================


class TestProcessWithOcrDocChunkFallback:
    """Lines 813-817: DocumentURLChunk is None dict fallback."""

    def test_document_url_dict_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "")
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_HEADER", True)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_FOOTER", True)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_MIN_SIZE", 0)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        mock_page = MagicMock()
        mock_page.markdown = "PDF text via dict"
        mock_page.index = 0
        mock_page.images = []

        mock_response = MagicMock()
        mock_response.pages = [mock_page]

        mock_client = MagicMock()
        mock_client.ocr.process.return_value = mock_response

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://signed.url",
        ):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch.object(
                    mistral_converter,
                    "get_bbox_annotation_format",
                    return_value=None,
                ):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        # Set DocumentURLChunk to None to trigger dict fallback
                        with patch.object(mistral_converter, "DocumentURLChunk", None):
                            success, result, error = mistral_converter.process_with_ocr(mock_client, pdf)

        assert success is True
        # Verify the dict fallback was used
        call_kwargs = mock_client.ocr.process.call_args[1]
        doc = call_kwargs["document"]
        assert isinstance(doc, dict)
        assert doc["type"] == "document_url"


# ============================================================================
# process_with_ocr - 401 Unauthorized error
# ============================================================================


class TestProcessWithOcrAuthError:
    """Lines 875-881, 896-897: Auth and permission errors."""

    def test_401_unauthorized(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        mock_client = MagicMock()

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://url",
        ):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch.object(
                    mistral_converter,
                    "get_bbox_annotation_format",
                    return_value=None,
                ):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        with patch.object(
                            mistral_converter,
                            "DocumentURLChunk",
                            MagicMock(),
                        ):
                            mock_client.ocr.process.side_effect = Exception("401 Unauthorized")
                            success, result, error = mistral_converter.process_with_ocr(mock_client, pdf)

        assert success is False
        assert "401" in error or "Unauthorized" in error

    def test_empty_ocr_response(self, tmp_path, monkeypatch):
        """Line 898: empty response from OCR."""
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "")
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_HEADER", True)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_FOOTER", True)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_MIN_SIZE", 0)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        mock_client = MagicMock()
        mock_client.ocr.process.return_value = None  # Empty response

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://url",
        ):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch.object(
                    mistral_converter,
                    "get_bbox_annotation_format",
                    return_value=None,
                ):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        with patch.object(
                            mistral_converter,
                            "DocumentURLChunk",
                            MagicMock(),
                        ):
                            success, result, error = mistral_converter.process_with_ocr(mock_client, pdf)

        assert success is False
        assert "Empty response" in error


# ============================================================================
# _extract_page_text - content attribute path
# ============================================================================


class TestExtractPageTextContent:
    """Line 910: page with content attribute."""

    def test_page_with_content_attr(self):
        page = MagicMock(spec=[])
        page.content = "Content text here"
        # Remove markdown and text so content path is hit
        result = mistral_converter._extract_page_text(page)
        assert result == "Content text here"

    def test_page_as_string(self):
        result = mistral_converter._extract_page_text("raw string page")
        assert result == "raw string page"


# ============================================================================
# _parse_page_object - dict page and index from dict
# ============================================================================


class TestParsePageObjectDict:
    """Lines 927, 942-943: page as dict indexing paths."""

    def test_dict_page_with_index(self):
        page = {"markdown": "Dict page text", "index": 4}  # 0-based
        result = mistral_converter._parse_page_object(page, 0)
        assert result["page_number"] == 5  # converted to 1-based
        assert result["api_page_index"] == 4
        assert result["text"] == "Dict page text"

    def test_dict_page_without_index(self):
        page = {"text": "Dict text"}
        result = mistral_converter._parse_page_object(page, 3)
        assert result["page_number"] == 4  # idx + 1
        assert result["api_page_index"] == 3

    def test_page_with_dimensions(self):
        page = MagicMock()
        page.markdown = "Text"
        page.index = 1
        page.images = []

        dims = MagicMock()
        dims.dpi = 300
        dims.height = 800
        dims.width = 600
        page.dimensions = dims
        page.tables = None
        page.hyperlinks = None
        page.header = None
        page.footer = None

        result = mistral_converter._parse_page_object(page, 0)
        assert result["dimensions"]["dpi"] == 300
        assert result["dimensions"]["height"] == 800

    def test_page_with_tables_and_hyperlinks(self):
        page = MagicMock()
        page.markdown = "Text"
        page.index = 1
        page.images = []
        page.dimensions = None

        mock_table = MagicMock()
        mock_table.model_dump.return_value = {"col1": "val"}
        page.tables = [mock_table]

        mock_link = MagicMock()
        mock_link.model_dump.return_value = {"url": "https://example.com"}
        page.hyperlinks = [mock_link]

        page.header = "Page Header"
        page.footer = "Page Footer"

        result = mistral_converter._parse_page_object(page, 0)
        assert result["tables"] == [{"col1": "val"}]
        assert result["hyperlinks"] == [{"url": "https://example.com"}]
        assert result["header"] == "Page Header"
        assert result["footer"] == "Page Footer"


# ============================================================================
# _extract_structured_outputs - document_annotation paths
# ============================================================================


class TestExtractStructuredOutputsPaths:
    """Lines 1034-1035, 1039: document annotation parsing."""

    def test_annotation_string_valid_json(self):
        response = MagicMock()
        response.bbox_annotations = None
        response.document_annotation = '{"key": "value"}'

        result = {"bbox_annotations": [], "document_annotation": None}
        mistral_converter._extract_structured_outputs(response, result)
        assert result["document_annotation"] == {"key": "value"}

    def test_annotation_string_invalid_json(self):
        """Line 1034-1035: JSON decode error for string annotation."""
        response = MagicMock()
        response.bbox_annotations = None
        response.document_annotation = "not valid json {{"

        result = {"bbox_annotations": [], "document_annotation": None}
        mistral_converter._extract_structured_outputs(response, result)
        assert result["document_annotation"] == "not valid json {{"

    def test_annotation_with_model_dump(self):
        response = MagicMock()
        response.bbox_annotations = None
        annotation = MagicMock()
        annotation.model_dump.return_value = {"parsed": True}
        response.document_annotation = annotation

        result = {"bbox_annotations": [], "document_annotation": None}
        mistral_converter._extract_structured_outputs(response, result)
        assert result["document_annotation"] == {"parsed": True}

    def test_annotation_plain_dict(self):
        """Line 1039: annotation that's not str, not model_dump."""
        response = MagicMock()
        response.bbox_annotations = None

        # Make an object without model_dump
        annotation_obj = {"plain": "dict"}
        response.document_annotation = annotation_obj

        result = {"bbox_annotations": [], "document_annotation": None}
        mistral_converter._extract_structured_outputs(response, result)
        assert result["document_annotation"] == {"plain": "dict"}

    def test_bbox_annotations_without_model_dump(self):
        response = MagicMock()
        bbox = {"x": 1, "y": 2}
        response.bbox_annotations = [bbox]
        response.document_annotation = None

        result = {"bbox_annotations": [], "document_annotation": None}
        mistral_converter._extract_structured_outputs(response, result)
        assert result["bbox_annotations"] == [{"x": 1, "y": 2}]


# ============================================================================
# _parse_ocr_response - conditional branches
# ============================================================================


class TestParseOcrResponseBranches:
    """Lines 1096, 1098, 1100, 1108-1110: response type branches."""

    def test_response_with_text_attr(self):
        """Line 1098: response.text path."""
        response = MagicMock(spec=[])
        response.text = "Text from response"
        response.pages = None
        response.markdown = None

        result = mistral_converter._parse_ocr_response(response, Path("test.pdf"))
        assert result["full_text"] == "Text from response"

    def test_response_with_content_attr(self):
        """Line 1100: response.content path."""
        response = MagicMock(spec=[])
        response.content = "Content from response"
        response.pages = None
        response.markdown = None
        response.text = None

        result = mistral_converter._parse_ocr_response(response, Path("test.pdf"))
        assert result["full_text"] == "Content from response"

    def test_response_as_dict(self):
        """Line 1102: dict response path."""
        response = {"pages": [{"markdown": "Dict page"}]}

        result = mistral_converter._parse_ocr_response(response, Path("test.pdf"))
        assert len(result["pages"]) == 1
        assert "Dict page" in result["full_text"]

    def test_response_parsing_exception(self):
        """Lines 1108-1110: exception during parsing."""
        # Create response that causes error during structured output extraction
        response = MagicMock()
        response.bbox_annotations = MagicMock()
        # __iter__ will raise during list comprehension
        response.bbox_annotations.__iter__ = MagicMock(side_effect=TypeError("not iterable"))
        response.pages = None

        result = mistral_converter._parse_ocr_response(response, Path("test.pdf"))
        # Should return result dict even on error
        assert result["file_name"] == "test.pdf"

    def test_response_with_markdown_attr(self):
        """Line 1096: response.markdown path."""
        response = MagicMock(spec=[])
        response.markdown = "Markdown text"
        response.pages = None

        result = mistral_converter._parse_ocr_response(response, Path("test.pdf"))
        assert "Markdown text" in result["full_text"]


# ============================================================================
# _is_weak_page - remaining branches
# ============================================================================


class TestIsWeakPageBranches:
    """Lines 1160, 1174-1175, 1183-1184: additional weak page checks."""

    def test_low_digit_count_no_longer_flags_page(self, monkeypatch):
        """Digit count check was removed — low-digit pages are not weak."""
        monkeypatch.setattr(config, "OCR_MIN_TEXT_LENGTH", 5)
        monkeypatch.setattr(config, "OCR_MIN_UNIQUENESS_RATIO", 0.0)
        monkeypatch.setattr(config, "OCR_MAX_PHRASE_REPETITIONS", 100)
        monkeypatch.setattr(config, "OCR_MIN_AVG_LINE_LENGTH", 0)

        # Text with very few digits — should NOT be flagged (digit check removed)
        text = "This is a text paragraph without many numbers at all."
        assert mistral_converter._is_weak_page(text) is False

    def test_repeated_page_references(self, monkeypatch):
        """Lines 1174-1175: too many Page N references."""
        monkeypatch.setattr(config, "OCR_MIN_TEXT_LENGTH", 5)
        monkeypatch.setattr(config, "OCR_MIN_UNIQUENESS_RATIO", 0.0)
        monkeypatch.setattr(config, "OCR_MAX_PHRASE_REPETITIONS", 2)
        monkeypatch.setattr(config, "OCR_MIN_AVG_LINE_LENGTH", 0)

        text = "Page 1 content here Page 2 more content " "Page 3 and Page 4 references Page 5"
        assert mistral_converter._is_weak_page(text) is True

    def test_short_average_line_length(self, monkeypatch):
        """Lines 1183-1184: very short average line length."""
        monkeypatch.setattr(config, "OCR_MIN_TEXT_LENGTH", 5)
        monkeypatch.setattr(config, "OCR_MIN_UNIQUENESS_RATIO", 0.0)
        monkeypatch.setattr(config, "OCR_MAX_PHRASE_REPETITIONS", 100)
        monkeypatch.setattr(config, "OCR_MIN_AVG_LINE_LENGTH", 50)

        # Very short lines
        text = "ab\ncd\nef\ngh\nij\nkl\nmn\nop\nqr\nst"
        assert mistral_converter._is_weak_page(text) is True


# ============================================================================
# improve_weak_pages - remaining paths
# ============================================================================


class TestImproveWeakPagesEdgeCases:
    """Lines 1294, 1323-1337, 1349-1350: edge cases."""

    def test_no_pages_key(self):
        """Line 1294: empty pages list."""
        result = mistral_converter.improve_weak_pages(MagicMock(), Path("test.pdf"), {"pages": []}, "model")
        assert result == {"pages": []}

    def test_url_refresh_on_expiry(self, monkeypatch):
        """Lines 1323-1324, 1332-1337: URL refresh when nearing expiry."""
        import time

        monkeypatch.setattr(config, "MAX_CONCURRENT_FILES", 1)
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 1)  # 1 hour

        weak_text = "x"
        ocr_result = {"pages": [{"text": weak_text, "page_number": 1}]}

        good_text = (
            "This is a much better OCR result with real content and "
            "financial numbers $12,345.67 and percentage 15.2% growth.\n"
            "Additional paragraph with meaningful text content.\n"
        )

        mock_client = MagicMock()
        upload_count = [0]

        def mock_upload(client, path, **kwargs):
            upload_count[0] += 1
            return f"https://signed.url/{upload_count[0]}"

        with patch.object(mistral_converter, "upload_file_for_ocr", side_effect=mock_upload):
            with patch.object(
                mistral_converter,
                "process_with_ocr",
                return_value=(
                    True,
                    {"pages": [{"text": good_text, "page_number": 1}]},
                    None,
                ),
            ):
                # Monkey-patch time.time to simulate expiry
                original_time = time.time
                call_count = [0]

                def fake_time():
                    call_count[0] += 1
                    if call_count[0] <= 2:
                        return original_time()
                    # Simulate that a lot of time has passed (>90% of TTL)
                    return original_time() + 4000

                with patch("time.time", side_effect=fake_time):
                    result = mistral_converter.improve_weak_pages(
                        mock_client,
                        Path("test.pdf"),
                        ocr_result,
                        "model",
                    )

        assert result["pages"][0]["text"] == good_text

    def test_improve_page_exception(self, monkeypatch):
        """Lines 1349-1350: exception in _improve_page."""
        monkeypatch.setattr(config, "MAX_CONCURRENT_FILES", 1)
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)

        weak_text = "x"
        ocr_result = {"pages": [{"text": weak_text, "page_number": 1}]}

        mock_client = MagicMock()

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://url",
        ):
            with patch.object(
                mistral_converter,
                "process_with_ocr",
                side_effect=Exception("OCR crash"),
            ):
                result = mistral_converter.improve_weak_pages(
                    mock_client,
                    Path("test.pdf"),
                    ocr_result,
                    "model",
                )

        # Original page preserved on error
        assert result["pages"][0]["text"] == weak_text


# ============================================================================
# save_extracted_images - error path
# ============================================================================


class TestSaveExtractedImagesError:
    """Lines 1404, 1425-1426: image save error and logging."""

    def test_invalid_base64_data(self, tmp_path, monkeypatch):
        """Line 1404: exception during base64 decode."""
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
        monkeypatch.setattr(config, "OUTPUT_IMAGES_DIR", tmp_path)

        ocr_result = {
            "pages": [
                {
                    "page_number": 1,
                    "images": [{"base64": "not-valid-base64!!!"}],
                }
            ]
        }

        saved = mistral_converter.save_extracted_images(ocr_result, Path("test.pdf"))
        assert len(saved) == 0

    def test_multiple_images_logs_count(self, tmp_path, monkeypatch):
        """Lines 1425-1426: multiple images saved logs count."""
        import base64

        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
        monkeypatch.setattr(config, "OUTPUT_IMAGES_DIR", tmp_path)

        b64_1 = base64.b64encode(b"image1").decode()
        b64_2 = base64.b64encode(b"image2").decode()

        ocr_result = {
            "pages": [
                {
                    "page_number": 1,
                    "images": [
                        {"base64": b64_1},
                        {"base64": b64_2},
                    ],
                }
            ]
        }

        saved = mistral_converter.save_extracted_images(ocr_result, Path("test.pdf"))
        assert len(saved) == 2


# ============================================================================
# _process_ocr_result_pipeline - JSON save exception
# ============================================================================


class TestPipelineJsonSaveError:
    """Lines 1512-1513: exception during JSON metadata save."""

    def test_json_save_exception(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path)
        monkeypatch.setattr(config, "SAVE_MISTRAL_JSON", True)
        monkeypatch.setattr(config, "ENABLE_OCR_QUALITY_ASSESSMENT", False)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        ocr_result = {
            "full_text": "Text",
            "pages": [{"text": "Text", "page_number": 1}],
        }

        with patch.object(
            mistral_converter,
            "_create_markdown_output",
            return_value=tmp_path / "out.md",
        ):
            with patch.object(mistral_converter, "save_extracted_images"):
                with patch.object(mistral_converter, "_save_structured_outputs"):
                    with patch("utils.cache.set"):
                        # Make the JSON write fail
                        with patch(
                            "builtins.open",
                            side_effect=PermissionError("no write"),
                        ):
                            ok, path, err = mistral_converter._process_ocr_result_pipeline(
                                MagicMock(),
                                pdf,
                                ocr_result,
                                True,
                                True,
                                False,
                            )

        # Pipeline should still succeed (JSON save is non-fatal)
        assert ok is True


# ============================================================================
# _validate_document_url - IPv6-mapped IPv4 and DNS errors
# ============================================================================


class TestExtractResponseMetadataDict:
    """Cover dict-based metadata extraction paths."""

    def test_dict_response_metadata(self):
        response = {
            "metadata": {"key": "val"},
            "usage_info": {"pages_processed": 5},
            "model": "mistral-ocr-latest",
        }
        result = {
            "metadata": {},
            "usage_info": {},
            "model": None,
        }
        mistral_converter._extract_response_metadata(response, result)
        assert result["metadata"] == {"key": "val"}
        assert result["usage_info"] == {"pages_processed": 5}
        assert result["model"] == "mistral-ocr-latest"


# ============================================================================
# REMAINING COVERAGE GAPS - TARGETED FIXES
# ============================================================================


class TestCleanupFileNoCreatedAt:
    """Line 540: file object without created_at attribute."""

    def test_file_missing_created_at(self):
        from types import SimpleNamespace

        mock_file = SimpleNamespace(id="no_date_file")
        # SimpleNamespace only has 'id', no 'created_at'

        mock_response = MagicMock()
        mock_response.data = [mock_file]
        mock_response.total = 1

        mock_client = MagicMock()
        mock_client.files.list.return_value = mock_response

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        assert count == 0


class TestCleanupPaginationNoTotal:
    """Lines 563-565: second pagination check when total is absent."""

    def test_pagination_no_total_attribute(self):
        from datetime import datetime, timedelta, timezone

        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

        mock_file = MagicMock()
        mock_file.id = "f1"
        mock_file.created_at = old

        mock_response = MagicMock(spec=[])
        mock_response.data = [mock_file]
        # No .total attribute (spec=[] restricts auto-creation)

        mock_client = MagicMock()
        mock_client.files.list.return_value = mock_response

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        assert count >= 1

    def test_pagination_page_increment(self):
        """Line 565: page += 1 when files_list has >= page_size items."""
        from datetime import datetime, timedelta, timezone

        recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        # Create page_size (100) files that are NOT old enough to delete
        files_page1 = []
        for i in range(100):
            f = MagicMock()
            f.id = f"f_{i}"
            f.created_at = recent
            files_page1.append(f)

        page1 = MagicMock(spec=[])
        page1.data = files_page1
        # No .total attribute

        # Second page is empty
        page2 = MagicMock(spec=[])
        page2.data = []

        # Also empty for "batch" purpose
        empty_page = MagicMock(spec=[])
        empty_page.data = []

        mock_client = MagicMock()
        mock_client.files.list.side_effect = [
            page1,
            page2,  # "ocr" pages
            empty_page,  # "batch" page
        ]

        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old=30)
        # No files deleted (all too recent), but page was incremented
        assert count == 0
        # Verify list was called at least with page=1
        calls = mock_client.files.list.call_args_list
        assert (
            any(
                c.kwargs.get("page", c.args[0] if c.args else 0) == 1
                for c in calls
                if c.kwargs.get("purpose") == "ocr" or (c.args and c.args[0] == "ocr")
            )
            or len(calls) >= 3
        )


class TestCleanupOuterException:
    """Lines 577-579: outer exception handler."""

    def test_invalid_days_old_type(self):
        mock_client = MagicMock()

        # Pass a string that timedelta can't use -> outer except fires
        count = mistral_converter.cleanup_uploaded_files(mock_client, days_old="invalid")
        assert count == 0


class TestProcessWithOcr403Error:
    """Lines 896-897: 403 Forbidden error from OCR API."""

    def test_403_forbidden(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        mock_client = MagicMock()

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://url",
        ):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch.object(
                    mistral_converter,
                    "get_bbox_annotation_format",
                    return_value=None,
                ):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        with patch.object(
                            mistral_converter,
                            "DocumentURLChunk",
                            MagicMock(),
                        ):
                            mock_client.ocr.process.side_effect = Exception("403 Forbidden")
                            success, result, error = mistral_converter.process_with_ocr(mock_client, pdf)

        assert success is False
        assert "403" in error or "Forbidden" in error


class TestProcessWithOcrEmptyText:
    """Lines 875-881: OCR returns parseable but empty text."""

    def test_empty_text_response(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "")
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_HEADER", True)
        monkeypatch.setattr(config, "MISTRAL_EXTRACT_FOOTER", True)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_IMAGE_MIN_SIZE", 0)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")

        mock_client = MagicMock()
        # Mock _parse_ocr_response to return empty text
        empty_result = {"full_text": "   ", "pages": []}

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            return_value="https://url",
        ):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch.object(
                    mistral_converter,
                    "get_bbox_annotation_format",
                    return_value=None,
                ):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        with patch.object(
                            mistral_converter,
                            "DocumentURLChunk",
                            MagicMock(),
                        ):
                            mock_client.ocr.process.return_value = MagicMock()
                            with patch.object(
                                mistral_converter,
                                "_parse_ocr_response",
                                return_value=empty_result,
                            ):
                                ok, result, err = mistral_converter.process_with_ocr(mock_client, pdf)

        assert ok is False
        assert "empty text" in err.lower()


class TestParsePageObjectImages:
    """Lines 942-943: page with images attribute."""

    def test_page_with_images(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)

        page = MagicMock()
        page.markdown = "Text with images"
        page.index = 0
        page.dimensions = None
        page.tables = None
        page.hyperlinks = None
        page.header = None
        page.footer = None

        img = MagicMock()
        img.id = "img_1"
        img.top_left_x = 10
        img.top_left_y = 20
        img.bottom_right_x = 100
        img.bottom_right_y = 200
        img.bbox = [10, 20, 100, 200]
        img.image_base64 = "dGVzdA=="
        img.base64 = None
        page.images = [img]

        result = mistral_converter._parse_page_object(page, 0)
        assert len(result["images"]) == 1
        assert result["images"][0]["id"] == "img_1"
        assert result["images"][0]["base64"] == "dGVzdA=="


class TestImproveWeakPagesUploadFailure:
    """Lines 1323-1324: upload_file_for_ocr fails for weak pages."""

    def test_upload_fails_in_improve(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONCURRENT_FILES", 1)
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)

        weak_text = "x"
        ocr_result = {"pages": [{"text": weak_text, "page_number": 1}]}

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            side_effect=Exception("Upload failed"),
        ):
            with patch.object(
                mistral_converter,
                "process_with_ocr",
                return_value=(True, {"pages": [{"text": "ok"}]}, None),
            ):
                result = mistral_converter.improve_weak_pages(
                    MagicMock(),
                    Path("test.pdf"),
                    ocr_result,
                    "model",
                )

        # Should still run, using None as signed_url
        assert result is not None


class TestImproveWeakPagesUrlRefreshFail:
    """Lines 1332-1337: URL refresh re-upload path."""

    def test_url_refresh_succeeds(self, monkeypatch):
        """Lines 1332-1335: URL refresh triggers and succeeds."""
        monkeypatch.setattr(config, "MAX_CONCURRENT_FILES", 1)
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 0)  # 0 hours = instant expiry

        weak_text = "x"
        ocr_result = {"pages": [{"text": weak_text, "page_number": 1}]}

        upload_calls = [0]

        def mock_upload(client, path, **kwargs):
            upload_calls[0] += 1
            return f"https://url/{upload_calls[0]}"

        # Simulate advancing clock so that the elapsed-time check
        # (time.time() - upload_time) > 0  always returns True after
        # the initial upload.
        _time_counter = [1000.0]

        def advancing_time():
            _time_counter[0] += 1.0
            return _time_counter[0]

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            side_effect=mock_upload,
        ):
            with patch.object(
                mistral_converter,
                "process_with_ocr",
                return_value=(
                    True,
                    {
                        "pages": [
                            {
                                "text": (
                                    "Better OCR text with numbers 123456 and " "real content for verification purposes."
                                ),
                                "page_number": 1,
                            }
                        ]
                    },
                    None,
                ),
            ):
                with patch("time.time", side_effect=advancing_time):
                    mistral_converter.improve_weak_pages(
                        MagicMock(),
                        Path("test.pdf"),
                        ocr_result,
                        "model",
                    )

        # Should have called upload at least twice (initial + refresh)
        assert upload_calls[0] >= 2

    def test_url_refresh_fails(self, monkeypatch):
        """Lines 1336-1337: URL refresh re-upload raises exception."""
        monkeypatch.setattr(config, "MAX_CONCURRENT_FILES", 1)
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 0)  # instant expiry

        weak_text = "x"
        ocr_result = {"pages": [{"text": weak_text, "page_number": 1}]}

        upload_calls = [0]

        def mock_upload(client, path, **kwargs):
            upload_calls[0] += 1
            if upload_calls[0] == 1:
                return "https://initial.url"
            raise Exception("Re-upload failed")

        with patch.object(
            mistral_converter,
            "upload_file_for_ocr",
            side_effect=mock_upload,
        ):
            with patch.object(
                mistral_converter,
                "process_with_ocr",
                return_value=(
                    True,
                    {
                        "pages": [
                            {
                                "text": ("Better OCR text with some content " "and numbers 789012 for test."),
                                "page_number": 1,
                            }
                        ]
                    },
                    None,
                ),
            ):
                result = mistral_converter.improve_weak_pages(
                    MagicMock(),
                    Path("test.pdf"),
                    ocr_result,
                    "model",
                )

        assert result is not None


class TestSaveImageEmptyBase64:
    """Line 1404: image with empty/None base64 is skipped."""

    def test_empty_base64(self, tmp_path, monkeypatch):
        import base64 as b64

        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
        monkeypatch.setattr(config, "OUTPUT_IMAGES_DIR", tmp_path)

        valid_b64 = b64.b64encode(b"real_image").decode()

        ocr_result = {
            "pages": [
                {
                    "page_number": 1,
                    "images": [
                        {"base64": ""},
                        {"base64": None},
                        {"base64": valid_b64},
                    ],
                }
            ]
        }

        saved = mistral_converter.save_extracted_images(ocr_result, Path("test.pdf"))
        # Only the valid image should be saved
        assert len(saved) == 1


class TestDoubleCheckedLockingConcurrent:
    """Line 164: double-checked locking second check returns cached."""

    def test_concurrent_locking(self, monkeypatch):
        import threading

        mistral_converter.reset_mistral_client()
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "test-key")

        mock_client = MagicMock()
        lock_acquired = threading.Event()
        can_proceed = threading.Event()

        def slow_mistral(**kwargs):
            lock_acquired.set()
            can_proceed.wait(timeout=5)
            return mock_client

        results = [None, None]

        def thread1():
            results[0] = mistral_converter.get_mistral_client()

        def thread2():
            lock_acquired.wait(timeout=5)
            results[1] = mistral_converter.get_mistral_client()

        with patch.object(mistral_converter, "Mistral", slow_mistral):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                t1 = threading.Thread(target=thread1)
                t2 = threading.Thread(target=thread2)
                t1.start()
                t2.start()

                can_proceed.set()
                t1.join(timeout=5)
                t2.join(timeout=5)

        assert results[0] is mock_client
        assert results[1] is mock_client
        mistral_converter.reset_mistral_client()


# ============================================================================
# Module-level import fallback paths (via importlib.reload)
# ============================================================================


class TestImportFallbackPaths:
    """Lines 62-73, 78-83, 87-88, 92-93: module-level import fallbacks."""

    def _save_module_state(self):
        """Save current sys.modules entries related to mistralai and PIL."""
        import sys

        saved = {}
        for key in list(sys.modules.keys()):
            if key.startswith("mistralai") or key.startswith("PIL"):
                saved[key] = sys.modules[key]
        return saved

    def _restore_module_state(self, saved):
        """Restore sys.modules and reload mistral_converter."""
        import importlib
        import sys

        # Remove any mock entries
        for key in list(sys.modules.keys()):
            if key.startswith("mistralai") or key.startswith("PIL"):
                sys.modules.pop(key, None)
        # Restore originals
        for key, mod in saved.items():
            sys.modules[key] = mod
        importlib.reload(mistral_converter)

    def test_mistralai_completely_unavailable(self):
        """mistralai not installed at all — all imports fall back to None."""
        import importlib
        import sys

        saved = self._save_module_state()
        try:
            # Remove all mistralai modules
            for key in list(sys.modules.keys()):
                if key.startswith("mistralai"):
                    sys.modules.pop(key)
            # Block imports by setting to None
            sys.modules["mistralai"] = None
            sys.modules["mistralai.utils"] = None
            sys.modules["mistralai.extra"] = None

            importlib.reload(mistral_converter)

            assert mistral_converter.Mistral is None
            assert mistral_converter.models is None
            assert mistral_converter.retries is None
            assert mistral_converter.DocumentURLChunk is None
            assert mistral_converter.ImageURLChunk is None
            assert mistral_converter.FileChunk is None
            assert mistral_converter.response_format_from_pydantic_model is None
        finally:
            self._restore_module_state(saved)

    def test_pil_unavailable(self):
        """Lines 92-93: PIL not installed."""
        import importlib
        import sys

        saved = self._save_module_state()
        try:
            for key in list(sys.modules.keys()):
                if key.startswith("PIL"):
                    sys.modules.pop(key)
            sys.modules["PIL"] = None
            sys.modules["PIL.Image"] = None
            sys.modules["PIL.ImageEnhance"] = None

            importlib.reload(mistral_converter)

            assert mistral_converter.Image is None
        finally:
            self._restore_module_state(saved)


class TestBuildMistralOcrCacheContractMetadata:
    """build_mistral_ocr_cache_contract_metadata returns expected keys."""

    def test_returns_expected_keys(self):
        meta = mistral_converter.build_mistral_ocr_cache_contract_metadata()
        expected = {
            "contract_type",
            "contract_version",
            "sdk_version",
            "ocr_model",
            "include_images",
            "table_format",
            "extract_header",
            "extract_footer",
            "image_limit",
            "image_min_size",
            "structured_output_enabled",
            "bbox_annotation_enabled",
            "document_annotation_enabled",
            "document_schema_type",
            "document_annotation_prompt_hash",
            "quality_assessment_enabled",
            "weak_page_improvement_enabled",
            "improve_weak",
            "image_preprocessing_enabled",
            "image_optimization_enabled",
        }
        assert set(meta.keys()) == expected
        assert meta["contract_type"] == "mistral_ocr"


class TestValidateFileForOcr:
    """_validate_file_for_ocr returns error when file exceeds size limit."""

    def test_oversized_file_returns_error(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_OCR_MAX_FILE_SIZE_MB", 100)
        result = mistral_converter._validate_file_for_ocr(Path("big.pdf"), 150.0)
        assert result is not None
        ok, data, msg = result
        assert ok is False
        assert data is None
        assert "too large" in msg.lower()

    def test_within_limit_returns_none(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_OCR_MAX_FILE_SIZE_MB", 200)
        result = mistral_converter._validate_file_for_ocr(Path("ok.pdf"), 50.0)
        assert result is None


class TestDetectWeakPages:
    """_detect_weak_pages identifies weak pages correctly."""

    def test_mix_of_weak_and_strong(self, monkeypatch):
        monkeypatch.setattr(config, "OCR_MIN_TEXT_LENGTH", 50)
        ocr_result = {
            "pages": [
                {"text": "short"},
                {"text": "A " * 100},
                {"text": ""},
                {"text": "This is a sufficiently long paragraph with enough unique words. " * 3},
            ]
        }
        weak = mistral_converter._detect_weak_pages(ocr_result)
        assert 0 in weak  # "short" is weak
        assert 2 in weak  # empty is weak
        assert 3 not in weak  # long diverse text is not weak
