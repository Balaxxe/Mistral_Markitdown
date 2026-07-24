"""Private resource-limit primitives shared by Mistral conversion modules."""


class OCRResponseLimitError(ValueError):
    """Raised when an untrusted OCR response exceeds a local resource budget."""
