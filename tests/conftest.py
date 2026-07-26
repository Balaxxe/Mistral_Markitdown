"""
Pytest configuration and shared fixtures.

Shared fixtures live here when multiple test modules need the same sample
artifacts. Prefer ``tmp_path`` + local setup for one-off cases.
"""

import pytest

import config


@pytest.fixture
def relax_strict_input_paths(monkeypatch):
    """Opt out of input-directory confinement for tests whose subject is permissive-path behavior.

    Not autouse: the shipped default is confinement, so tests exercise that unless they ask
    for this fixture. Tests that merely need a file outside the input tree should monkeypatch
    ``config.INPUT_DIR`` to ``tmp_path`` instead.
    """
    monkeypatch.setattr(config, "STRICT_INPUT_PATH_RESOLUTION", False)


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Minimal PDF bytes for file-operation tests (not a valid renderable PDF)."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")
    return pdf_path


@pytest.fixture
def sample_text_file(tmp_path):
    """Small text file for path/validation helpers."""
    text_path = tmp_path / "sample.txt"
    text_path.write_text("Sample text content\nLine 2\nLine 3")
    return text_path


@pytest.fixture
def sample_ocr_result():
    """Representative OCR result dict for converter/quality tests."""
    return {
        "file_name": "test.pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "Sample page 1 text with multiple words and numbers 123 456.",
                "images": [],
            },
            {
                "page_number": 2,
                "text": "Sample page 2 text with more content and data 789 012.",
                "images": [],
            },
        ],
        "full_text": (
            "Sample page 1 text with multiple words and numbers 123 456.\n\n"
            "Sample page 2 text with more content and data 789 012."
        ),
        "images": [],
        "metadata": {},
    }
