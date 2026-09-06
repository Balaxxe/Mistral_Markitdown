# Environment

- Python: 3.10, 3.11, or 3.12
- System deps: `poppler-utils` (needed by pdf2image)
- Setup: `pip install -r requirements.txt && pip install -r requirements-dev.txt`
- Config: if `.env` is absent, copy `.env.example` to it; preserve existing settings. `MISTRAL_API_KEY` is optional for local conversion and non-cloud modes.
