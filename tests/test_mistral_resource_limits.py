"""Focused resource-boundary tests for modular Mistral OCR parsing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config
import mistral_converter
from mistral_converter import ocr


def test_table_expansion_is_non_recursive_for_dict_response():
    result = {"pages": [], "full_text": ""}

    ocr._parse_dict_response(
        {
            "pages": [
                {
                    "markdown": "[first.md](first.md)",
                    "tables": [
                        {"id": "first.md", "content": "[second.md](second.md)"},
                        {"id": "second.md", "content": "expanded"},
                    ],
                }
            ]
        },
        result,
    )

    assert result["pages"][0]["text"] == "[second.md](second.md)"


def test_table_expansion_is_non_recursive_for_sdk_response():
    first = MagicMock()
    first.model_dump.return_value = {"id": "first.md", "content": "[second.md](second.md)"}
    second = MagicMock()
    second.model_dump.return_value = {"id": "second.md", "content": "expanded"}
    page = MagicMock(markdown="[first.md](first.md)", index=0, tables=[first, second])
    page.images = []
    page.dimensions = None
    page.hyperlinks = None
    page.header = None
    page.footer = None

    assert ocr._parse_page_object(page, 0)["text"] == "[second.md](second.md)"


def test_table_expansion_rejects_output_over_budget(monkeypatch):
    monkeypatch.setattr(ocr, "_MAX_TABLE_PLACEHOLDER_OUTPUT_BYTES", 24)

    with pytest.raises(ocr._OCRResponseLimitError, match="output byte limit"):
        ocr._expand_table_placeholders("[table.md](table.md)", [{"id": "table.md", "content": "x" * 32}])


def test_table_expansion_rejects_oversized_initial_text(monkeypatch):
    monkeypatch.setattr(ocr, "_MAX_TABLE_PLACEHOLDER_OUTPUT_BYTES", 8)

    with pytest.raises(ocr._OCRResponseLimitError, match="page text"):
        ocr._expand_table_placeholders("x" * 9, [])


def test_table_expansion_rejects_duplicate_ids():
    tables = [{"id": "table.md", "content": "first"}, {"id": "table.md", "content": "second"}]

    with pytest.raises(ocr._OCRResponseLimitError, match="Duplicate"):
        ocr._expand_table_placeholders("[table.md](table.md)", tables)


def test_table_expansion_rejects_excess_replacements(monkeypatch):
    monkeypatch.setattr(ocr, "_MAX_OCR_TABLE_REPLACEMENTS_PER_PAGE", 1)

    with pytest.raises(ocr._OCRResponseLimitError, match="replacements"):
        ocr._expand_table_placeholders(
            "[table.md](table.md) [table.md](table.md)", [{"id": "table.md", "content": "ok"}]
        )


def test_parse_response_clears_partial_data_when_aggregate_text_limit_is_exceeded(monkeypatch):
    monkeypatch.setattr(ocr, "_MAX_TABLE_PLACEHOLDER_OUTPUT_BYTES", 16)
    monkeypatch.setattr(ocr, "_MAX_OCR_TOTAL_TEXT_BYTES", 12)
    response = {"pages": [{"markdown": "first"}, {"markdown": "second"}]}

    result = ocr._parse_ocr_response(response, Path("test.pdf"))

    assert result["pages"] == []
    assert result["full_text"] == ""
    assert "aggregate" in (result["parse_error"] or "")


def test_parse_rejects_image_count_before_retaining_page_payloads(monkeypatch):
    monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
    monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 1)
    response = {
        "pages": [
            {
                "markdown": "text",
                "images": [{"base64": "YQ=="}, {"base64": "Yg=="}],
            }
        ]
    }

    result = ocr._parse_ocr_response(response, Path("test.pdf"))

    assert result["pages"] == []
    assert result["full_text"] == ""
    assert "image count" in (result["parse_error"] or "")


def test_image_count_is_checked_before_payload_property_access(monkeypatch):
    class UnreadablePayload:
        @property
        def image_base64(self):
            raise AssertionError("payload accessed before image-count admission")

    monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
    monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 1)

    with pytest.raises(ocr.OCRResponseLimitError, match="image count"):
        ocr._parse_page_images([UnreadablePayload(), UnreadablePayload()], {})


def test_payloadless_images_count_toward_cross_page_limit(monkeypatch):
    monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
    monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 1)
    response = {
        "pages": [
            {"markdown": "first", "images": [{"id": "one"}]},
            {"markdown": "second", "images": [{"id": "two"}]},
        ]
    }

    result = ocr._parse_ocr_response(response, Path("test.pdf"))

    assert result["pages"] == []
    assert "image count" in (result["parse_error"] or "")


def test_structured_fields_share_one_response_byte_budget(monkeypatch):
    monkeypatch.setattr(ocr, "_MAX_OCR_STRUCTURED_TOTAL_BYTES", 10)
    response = {
        "pages": [
            {"markdown": "first", "header": "123456"},
            {"markdown": "second", "footer": "abcdef"},
        ]
    }

    result = ocr._parse_ocr_response(response, Path("test.pdf"))

    assert result["pages"] == []
    assert "structured fields" in (result["parse_error"] or "")


def test_page_count_is_checked_before_page_property_access(monkeypatch):
    class UnreadablePage:
        @property
        def markdown(self):
            raise AssertionError("page accessed before count admission")

    monkeypatch.setattr(config, "MAX_PAGES_PER_SESSION", 1)
    response = MagicMock()
    response.bbox_annotations = None
    response.document_annotation = None
    response.pages = [UnreadablePage(), UnreadablePage()]

    result = ocr._parse_ocr_response(response, Path("test.pdf"))

    assert result["pages"] == []
    assert "page count" in (result["parse_error"] or "")


def test_parse_checks_data_uri_length_before_copying_payload(monkeypatch):
    monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
    monkeypatch.setattr(ocr, "MAX_EXTRACTED_IMAGE_ENCODED_BYTES", 4)
    response = {"pages": [{"markdown": "text", "images": [{"base64": "data:image/png;base64,AAAAA"}]}]}

    result = ocr._parse_ocr_response(response, Path("test.pdf"))

    assert result["pages"] == []
    assert "encoded-byte" in (result["parse_error"] or "")


def test_parse_checks_payload_length_before_ascii_scan(monkeypatch):
    class ScanDetectingString(str):
        def isascii(self):
            raise AssertionError("ASCII scan ran before encoded-length admission")

    monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
    monkeypatch.setattr(ocr, "MAX_EXTRACTED_IMAGE_ENCODED_BYTES", 4)
    response = {"pages": [{"markdown": "text", "images": [{"base64": ScanDetectingString("AAAAA")}]}]}

    result = ocr._parse_ocr_response(response, Path("test.pdf"))

    assert result["pages"] == []
    assert "encoded-byte" in (result["parse_error"] or "")


@pytest.mark.parametrize(
    "response",
    [
        {"pages": [{"markdown": "ok", "hyperlinks": [{"url": "x" * 9}]}]},
        {"pages": [{"markdown": "ok", "header": "x" * 9}]},
        {"bbox_annotations": [{"label": "x" * 9}], "pages": [{"markdown": "ok"}]},
        {"document_annotation": '{"x":"' + ("x" * 9) + '"}', "pages": [{"markdown": "ok"}]},
    ],
)
def test_parse_rejects_oversized_structured_fields(monkeypatch, response):
    monkeypatch.setattr(ocr, "_MAX_OCR_STRUCTURED_STRING_BYTES", 8)
    monkeypatch.setattr(ocr, "_MAX_OCR_HEADER_FOOTER_BYTES", 8)

    result = ocr._parse_ocr_response(response, Path("test.pdf"))

    assert result["pages"] == []
    assert result["full_text"] == ""
    assert result["parse_error"]


def test_object_response_structured_annotation_is_bounded(monkeypatch):
    monkeypatch.setattr(ocr, "_MAX_OCR_STRUCTURED_DEPTH", 1)
    response = MagicMock()
    response.bbox_annotations = None
    response.document_annotation = {"outer": {"inner": "value"}}
    response.pages = None
    response.markdown = None
    response.text = None
    response.content = None

    result = ocr._parse_ocr_response(response, Path("test.pdf"))

    assert result["pages"] == []
    assert "nesting" in (result["parse_error"] or "")


def test_document_annotation_node_limit_runs_before_json_decode(monkeypatch):
    monkeypatch.setattr(ocr, "_MAX_OCR_STRUCTURED_NODES", 3)
    response = {
        "document_annotation": "[0, 0, 0]",
        "pages": [{"markdown": "ok"}],
    }

    with patch.object(ocr.json, "loads", wraps=ocr.json.loads) as loads:
        result = ocr._parse_ocr_response(response, Path("test.pdf"))

    assert result["pages"] == []
    assert "item limit" in (result["parse_error"] or "")
    loads.assert_not_called()


def _configure_post_improvement_pipeline(monkeypatch, tmp_path, improved_text):
    monkeypatch.setattr(config, "ENABLE_OCR_QUALITY_ASSESSMENT", True)
    monkeypatch.setattr(config, "ENABLE_OCR_WEAK_PAGE_IMPROVEMENT", True)
    monkeypatch.setattr(
        ocr,
        "assess_ocr_quality",
        MagicMock(
            side_effect=[
                {"weak_page_count": 1, "quality_score": 1.0},
                {"weak_page_count": 0, "quality_score": 100.0},
            ]
        ),
    )
    monkeypatch.setattr(
        ocr,
        "improve_weak_pages",
        MagicMock(return_value={"pages": [{"text": improved_text, "images": []}], "full_text": improved_text}),
    )
    cache_set = MagicMock()
    create_markdown = MagicMock(return_value=tmp_path / "result.md")
    save_images = MagicMock()
    monkeypatch.setattr(mistral_converter, "save_extracted_images", save_images)
    monkeypatch.setattr(ocr.utils.cache, "set", cache_set)
    monkeypatch.setattr(ocr, "_create_markdown_output", create_markdown)
    monkeypatch.setattr(ocr, "_save_structured_outputs", MagicMock())
    return save_images, cache_set, create_markdown


def test_post_improvement_over_limit_is_not_cached_or_published(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "_MAX_OCR_TOTAL_TEXT_BYTES", 10)
    save_images, cache_set, create_markdown = _configure_post_improvement_pipeline(monkeypatch, tmp_path, "x" * 11)

    ok, output_path, error = ocr._process_ocr_result_pipeline(
        MagicMock(), tmp_path / "document.pdf", {"pages": [{"text": "weak", "images": []}], "full_text": "weak"}
    )

    assert (ok, output_path) == (False, None)
    assert "aggregate" in (error or "")
    save_images.assert_not_called()
    cache_set.assert_not_called()
    create_markdown.assert_not_called()


def test_post_improvement_under_limit_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "_MAX_OCR_TOTAL_TEXT_BYTES", 11)
    save_images, cache_set, create_markdown = _configure_post_improvement_pipeline(monkeypatch, tmp_path, "x" * 11)

    ok, output_path, error = ocr._process_ocr_result_pipeline(
        MagicMock(), tmp_path / "document.pdf", {"pages": [{"text": "weak", "images": []}], "full_text": "weak"}
    )

    assert (ok, output_path, error) == (True, tmp_path / "result.md", None)
    save_images.assert_called_once()
    cache_set.assert_called_once()
    create_markdown.assert_called_once()
