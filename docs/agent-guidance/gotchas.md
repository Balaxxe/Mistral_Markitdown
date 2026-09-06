# Gotchas

- `MISTRAL_API_KEY` is optional. Without it, Smart mode falls back to local MarkItDown; PDF-to-images, System Status,
  and Maintenance remain available. Mistral OCR/QnA/Batch features are disabled.
- The `Makefile` and `scripts/test-safe.sh` reference a local `env/` virtualenv. In cloud or CI environments, run tools via `python3 -m <tool>`.
- Pre-existing lint warnings exist in test files (unused imports, unused variables); these are in the upstream code.
- flake8 config is in `.flake8` (120 char line length, black-compatible ignores). pytest config is in `pyproject.toml`.
- Black is configured with `line-length = 120` and isort uses `profile = "black"` — both in `pyproject.toml`.
