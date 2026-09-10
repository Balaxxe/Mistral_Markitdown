# Structure

- `main.py` -- CLI entry point and orchestration
- `cli_files.py` -- input listing, validation, and interactive file selection
- `config.py` -- configuration loading from .env and defaults
- `schemas.py` -- Pydantic data models and validation
- `mistral_converter/` -- Mistral AI OCR/QnA/Batch conversion package (`import mistral_converter`)
- `local_converter.py` -- local MarkItDown-based conversion
- `modes/` -- mode orchestration (`batch.py`, `qna.py`, `system.py`)
- `utils.py` -- shared utilities
- `scripts/` -- helper scripts (test runner, etc.)
- `tests/` -- pytest test suite
- `input/` -- drop files here for conversion (gitignored)
- `output_md/`, `output_txt/`, `output_images/` -- conversion output (gitignored)
- `cache/` -- runtime cache (gitignored)
