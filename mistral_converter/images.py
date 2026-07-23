"""Image optimization, preprocessing, and OCR image extraction."""

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import utils

from .facade import attr

logger = utils.logger


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
        optimized_path = image_path.parent / f"{image_path.stem}_optimized{image_path.suffix}"

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
            preprocessed_path = image_path.parent / f"{image_path.stem}_preprocessed{image_path.suffix}"
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
        logger.warning("Error preprocessing image %s: %s", image_path.name, e)
        return image_path


# ============================================================================
# Files API Integration
# ============================================================================


def save_extracted_images(ocr_result: Dict[str, Any], file_path: Path) -> List[Path]:
    """
    Save extracted images from OCR result.

    Args:
        ocr_result: OCR result dictionary
        file_path: Original file path

    Returns:
        List of saved image paths
    """
    saved_images = []

    if not config.MISTRAL_INCLUDE_IMAGES:
        return saved_images

    stem = utils.safe_output_stem(file_path)
    image_dir = config.OUTPUT_IMAGES_DIR / f"{stem}_ocr"
    image_count = 0

    for page in ocr_result.get("pages", []):
        page_num = page.get("page_number", 1)

        for img in page.get("images", []):
            image_base64 = img.get("base64")

            if not image_base64:
                continue

            # Create output directory on first actual image (avoids empty folders)
            image_dir.mkdir(parents=True, exist_ok=True)

            try:
                if image_base64.startswith("data:"):
                    image_base64 = image_base64.split(",", 1)[1]

                # Decode base64 image
                image_data = base64.b64decode(image_base64)

                image_count += 1
                image_path = image_dir / f"page_{page_num}_image_{image_count}.png"

                utils.atomic_write_binary(image_path, image_data)

                saved_images.append(image_path)
                logger.debug("Saved extracted image: %s", image_path.name)

            except Exception as e:
                logger.error("Error saving image: %s", e)

    if saved_images:
        logger.info("Saved %s extracted images to %s", len(saved_images), image_dir)

    return saved_images


