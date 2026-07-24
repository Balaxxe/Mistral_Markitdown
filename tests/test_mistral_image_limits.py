"""Focused regressions for untrusted OCR images and derived-image temporary files."""

import base64
from unittest.mock import MagicMock, patch

import pytest

import config
import mistral_converter
from mistral_converter import images, ocr, upload
from mistral_converter.resource_limits import OCRResponseLimitError


def _result(*encoded_images: str) -> dict:
    return {"pages": [{"page_number": 1, "images": [{"base64": image} for image in encoded_images]}]}


@pytest.fixture(autouse=True)
def image_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
    monkeypatch.setattr(config, "OUTPUT_IMAGES_DIR", tmp_path / "output_images")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "MISTRAL_IMAGE_LIMIT", 0)


def test_image_count_limit_rejects_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "_MAX_EXTRACTED_IMAGES", 1)
    encoded = base64.b64encode(b"image").decode("ascii")

    with pytest.raises(OCRResponseLimitError, match="count"):
        images.save_extracted_images(_result(encoded, encoded), tmp_path / "document.pdf", fail_on_limit=True)

    assert not (config.OUTPUT_IMAGES_DIR / "document_ocr").exists()


def test_image_aggregate_limit_rejects_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "_MAX_EXTRACTED_IMAGES_TOTAL_DECODED_BYTES", 5)
    encoded = base64.b64encode(b"four").decode("ascii")

    with pytest.raises(OCRResponseLimitError, match="aggregate"):
        images.save_extracted_images(_result(encoded, encoded), tmp_path / "document.pdf", fail_on_limit=True)

    assert not (config.OUTPUT_IMAGES_DIR / "document_ocr").exists()


def test_invalid_base64_is_rejected_before_creating_output(tmp_path):
    with pytest.raises(OCRResponseLimitError, match="invalid base64"):
        images.save_extracted_images(_result("not-base64!"), tmp_path / "document.pdf", fail_on_limit=True)

    assert not (config.OUTPUT_IMAGES_DIR / "document_ocr").exists()


def test_default_image_limit_failure_skips_entire_set(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "_MAX_EXTRACTED_IMAGES", 1)
    encoded = base64.b64encode(b"image").decode("ascii")

    assert images.save_extracted_images(_result(encoded, encoded), tmp_path / "document.pdf") == []
    assert not (config.OUTPUT_IMAGES_DIR / "document_ocr").exists()


def test_new_derived_image_path_is_random_and_cache_owned(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"source")

    first = images._new_image_temp_path(source, "optimized")
    second = images._new_image_temp_path(source, "optimized")

    assert first.parent == config.CACHE_DIR
    assert second.parent == config.CACHE_DIR
    assert first.name.startswith("mistral_optimized_")
    assert first != second
    upload._cleanup_temp_files([first, second])
    assert not first.exists()
    assert not second.exists()


def test_temp_cleanup_refuses_non_owned_paths(tmp_path):
    unowned = tmp_path / "mistral_optimized.png"
    unowned.write_bytes(b"keep")
    cache_unowned_name = config.CACHE_DIR / "not_owned.png"
    cache_unowned_name.parent.mkdir(parents=True)
    cache_unowned_name.write_bytes(b"keep")

    upload._cleanup_temp_files([unowned, cache_unowned_name])

    assert unowned.exists()
    assert cache_unowned_name.exists()


def test_failed_optimization_cleans_its_owned_cache_file(tmp_path, monkeypatch):
    class Resized:
        def save(self, *_args, **_kwargs):
            raise OSError("write failed")

        def close(self):
            pass

    class Source:
        size = (2, 1)

        def resize(self, *_args, **_kwargs):
            return Resized()

    class SourceContext:
        def __enter__(self):
            return Source()

        def __exit__(self, *_args):
            return False

    class FakeImage:
        MAX_IMAGE_PIXELS = 1
        Resampling = type("Resampling", (), {"LANCZOS": object()})

        @staticmethod
        def open(_path):
            return SourceContext()

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    monkeypatch.setattr(config, "MISTRAL_ENABLE_IMAGE_OPTIMIZATION", True)
    monkeypatch.setattr(config, "MISTRAL_MAX_IMAGE_DIMENSION", 1)
    monkeypatch.setattr(mistral_converter, "Image", FakeImage)

    assert images.optimize_image(source) == source
    assert list(config.CACHE_DIR.glob("mistral_optimized_*")) == []


def test_rejected_image_set_is_neither_cached_nor_published(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_OCR_QUALITY_ASSESSMENT", False)
    monkeypatch.setattr(images, "_MAX_EXTRACTED_IMAGES", 1)
    encoded = base64.b64encode(b"image").decode("ascii")
    ocr_result = {"pages": [{"page_number": 1, "images": [{"base64": encoded}, {"base64": encoded}]}]}

    with patch.object(ocr.utils.cache, "set") as cache_set:
        with patch.object(mistral_converter, "_create_markdown_output") as create_markdown:
            ok, output_path, error = mistral_converter._process_ocr_result_pipeline(
                MagicMock(), tmp_path / "document.pdf", ocr_result
            )

    assert (ok, output_path) == (False, None)
    assert "count" in (error or "").lower()
    cache_set.assert_not_called()
    create_markdown.assert_not_called()
