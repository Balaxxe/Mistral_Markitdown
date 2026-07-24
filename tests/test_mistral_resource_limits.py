"""Focused resource-boundary tests for modular Mistral OCR parsing."""

from pathlib import Path
from unittest.mock import MagicMock

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
