"""Image optimization, preprocessing, and OCR image extraction."""

import base64
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import utils

from .facade import attr
from .resource_limits import OCRResponseLimitError

logger = utils.logger

# OCR responses are untrusted. Keep these local ceilings private so public
# configuration remains compatible while extraction cannot exhaust a worker.
_MAX_EXTRACTED_IMAGES = 100
_MAX_EXTRACTED_IMAGE_ENCODED_BYTES = 10 * 1024 * 1024
_MAX_EXTRACTED_IMAGE_DECODED_BYTES = 7 * 1024 * 1024
_MAX_EXTRACTED_IMAGES_TOTAL_DECODED_BYTES = 50 * 1024 * 1024


def _new_image_temp_path(image_path: Path, operation: str) -> Path:
    """Create a private, invocation-owned derived-image path in the cache."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(
        prefix=f"mistral_{operation}_",
        suffix=image_path.suffix or ".img",
        dir=config.CACHE_DIR,
    )
    os.close(fd)
    return Path(raw_path)


def optimize_image(image_path: Path) -> Optional[Path]:
    """
    Optimize image for better OCR results.

    Args:
        image_path: Path to image file

    Returns:
        Path to optimized image or original if optimization fails
    """
    Image = attr("Image")
    if not config.MISTRAL_ENABLE_IMAGE_OPTIMIZATION or Image is None:
        return image_path

    optimized_path: Optional[Path] = None
    try:
        # Bound decompression work for hostile/huge images (Pillow default may be None
        # depending on version); keep a generous ceiling for legitimate scans.
        if getattr(Image, "MAX_IMAGE_PIXELS", None) in (None, 0):
            Image.MAX_IMAGE_PIXELS = 178_956_970  # ~Pillow default bomb limit
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

        with Image.open(image_path) as src:
            # Check if optimization needed
            width, height = src.size
            max_dim = config.MISTRAL_MAX_IMAGE_DIMENSION

            if width <= max_dim and height <= max_dim:
                return image_path  # No optimization needed

            # Resize while maintaining aspect ratio
            if width > height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))

            img = src.resize((new_width, new_height), resample)

        # Save optimized image with format-appropriate parameters
        optimized_path = _new_image_temp_path(image_path, "optimized")

        try:
            suffix = image_path.suffix.lower()
            if suffix == ".png":
                img.save(optimized_path, format="PNG", optimize=True, compress_level=6)
            elif suffix in {".jpg", ".jpeg"}:
                img.save(
                    optimized_path,
                    format="JPEG",
                    quality=config.MISTRAL_IMAGE_QUALITY_THRESHOLD,
                    optimize=True,
                )
            else:
                img.save(optimized_path, optimize=True)
        finally:
            img.close()

        logger.debug("Optimized image: %s -> %s", image_path.name, optimized_path.name)
        return optimized_path

    except Exception as e:
        if optimized_path is not None:
            attr("_cleanup_temp_files")([optimized_path])
        logger.warning("Error optimizing image %s: %s", image_path.name, e)
        return image_path


def preprocess_image(image_path: Path) -> Optional[Path]:
    """
    Apply preprocessing to image for better OCR (contrast, sharpening, etc.).

    Args:
        image_path: Path to image file

    Returns:
        Path to preprocessed image or original if preprocessing fails
    """
    Image = attr("Image")
    if not config.MISTRAL_ENABLE_IMAGE_PREPROCESSING or Image is None:
        return image_path

    preprocessed_path: Optional[Path] = None
    try:
        from PIL import ImageEnhance

        if getattr(Image, "MAX_IMAGE_PIXELS", None) in (None, 0):
            Image.MAX_IMAGE_PIXELS = 178_956_970

        with Image.open(image_path) as src:
            img = src.convert("RGB")

        try:
            # Enhance contrast
            img = ImageEnhance.Contrast(img).enhance(1.5)

            # Enhance sharpness
            img = ImageEnhance.Sharpness(img).enhance(1.3)

            # Save preprocessed image with format-appropriate parameters
            preprocessed_path = _new_image_temp_path(image_path, "preprocessed")
            if image_path.suffix.lower() in {".jpg", ".jpeg"}:
                img.save(preprocessed_path, format="JPEG", quality=95, optimize=True)
            elif image_path.suffix.lower() == ".png":
                img.save(preprocessed_path, format="PNG", optimize=True)
            else:
                img.save(preprocessed_path)
        finally:
            img.close()

        logger.debug("Preprocessed image: %s", image_path.name)
        return preprocessed_path

    except Exception as e:
        if preprocessed_path is not None:
            attr("_cleanup_temp_files")([preprocessed_path])
        logger.warning("Error preprocessing image %s: %s", image_path.name, e)
        return image_path


# ============================================================================
# Files API Integration
# ============================================================================


def _prepare_extracted_images(ocr_result: Dict[str, Any]) -> List[tuple[Any, bytes]]:  # noqa: C901
    """Validate and decode every response image before creating output files."""
    if not config.MISTRAL_INCLUDE_IMAGES:
        return []

    pending: List[tuple[Any, str]] = []
    estimated_total_bytes = 0
    configured_image_limit = config.MISTRAL_IMAGE_LIMIT
    image_limit = (
        min(configured_image_limit, _MAX_EXTRACTED_IMAGES) if configured_image_limit > 0 else _MAX_EXTRACTED_IMAGES
    )

    for page in ocr_result.get("pages", []):
        page_num = page.get("page_number", 1)
        for image in page.get("images", []):
            image_base64 = image.get("base64")
            if not image_base64:
                continue
            if len(pending) >= image_limit:
                raise OCRResponseLimitError(f"OCR image count exceeds the local limit ({image_limit})")
            if not isinstance(image_base64, str):
                raise OCRResponseLimitError("OCR image base64 data must be text")
            if image_base64.startswith("data:"):
                if "," not in image_base64:
                    raise OCRResponseLimitError("OCR image data URI is malformed")
                image_base64 = image_base64.split(",", 1)[1]
            try:
                encoded_bytes = len(image_base64.encode("ascii"))
            except UnicodeEncodeError as exc:
                raise OCRResponseLimitError("OCR image base64 data contains non-ASCII text") from exc
            if encoded_bytes > _MAX_EXTRACTED_IMAGE_ENCODED_BYTES:
                raise OCRResponseLimitError("OCR image exceeds the local encoded-byte limit")

            # Rounding up is safe before strict base64 decoding and avoids an
            # allocation for an obviously oversized payload.
            estimated_decoded_bytes = ((encoded_bytes + 3) // 4) * 3
            if estimated_decoded_bytes > _MAX_EXTRACTED_IMAGE_DECODED_BYTES:
                raise OCRResponseLimitError("OCR image exceeds the local decoded-byte limit")
            estimated_total_bytes += estimated_decoded_bytes
            if estimated_total_bytes > _MAX_EXTRACTED_IMAGES_TOTAL_DECODED_BYTES:
                raise OCRResponseLimitError("OCR images exceed the local aggregate decoded-byte limit")
            pending.append((page_num, image_base64))

    prepared: List[tuple[Any, bytes]] = []
    actual_total_bytes = 0
    for page_num, image_base64 in pending:
        try:
            image_data = base64.b64decode(image_base64, validate=True)
        except (TypeError, ValueError) as exc:
            raise OCRResponseLimitError("OCR image contains invalid base64 data") from exc
        if len(image_data) > _MAX_EXTRACTED_IMAGE_DECODED_BYTES:
            raise OCRResponseLimitError("OCR image exceeds the local decoded-byte limit")
        actual_total_bytes += len(image_data)
        if actual_total_bytes > _MAX_EXTRACTED_IMAGES_TOTAL_DECODED_BYTES:
            raise OCRResponseLimitError("OCR images exceed the local aggregate decoded-byte limit")
        prepared.append((page_num, image_data))
    return prepared


def save_extracted_images(ocr_result: Dict[str, Any], file_path: Path, *, fail_on_limit: bool = False) -> List[Path]:
    """Save an OCR image set only after the complete set passes local budgets."""
    try:
        prepared = _prepare_extracted_images(ocr_result)
    except OCRResponseLimitError as exc:
        if fail_on_limit:
            raise
        logger.error("Skipping extracted images: %s", exc)
        return []
    if not prepared:
        return []

    image_dir = config.OUTPUT_IMAGES_DIR / f"{utils.safe_output_stem(file_path)}_ocr"
    image_dir.mkdir(parents=True, exist_ok=True)
    saved_images: List[Path] = []
    try:
        for image_count, (page_num, image_data) in enumerate(prepared, 1):
            image_path = image_dir / f"page_{page_num}_image_{image_count}.png"
            utils.atomic_write_binary(image_path, image_data)
            saved_images.append(image_path)
            logger.debug("Saved extracted image: %s", image_path.name)
    except Exception:
        # Preserve all-or-nothing semantics for this extraction invocation.
        for image_path in saved_images:
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove incomplete extracted image: %s", image_path.name)
        raise

    logger.info("Saved %s extracted images to %s", len(saved_images), image_dir)
    return saved_images
