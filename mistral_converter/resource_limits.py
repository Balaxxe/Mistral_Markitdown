"""Private resource-limit primitives shared by Mistral conversion modules."""


class OCRResponseLimitError(ValueError):
    """Raised when an untrusted OCR response exceeds a local resource budget."""


# These limits intentionally live alongside the exception rather than in the
# public configuration module: OCR responses are remote, untrusted data and
# these are local safety ceilings, not user-facing tuning knobs.
MAX_EXTRACTED_IMAGES = 100
MAX_EXTRACTED_IMAGE_ENCODED_BYTES = 10 * 1024 * 1024
MAX_EXTRACTED_IMAGE_DECODED_BYTES = 7 * 1024 * 1024
MAX_EXTRACTED_IMAGES_TOTAL_DECODED_BYTES = 50 * 1024 * 1024
MAX_EXTRACTED_IMAGE_DATA_URI_HEADER_CHARS = 1024
