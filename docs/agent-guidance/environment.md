# Environment

- Python: 3.10, 3.11, or 3.12
- System deps: `poppler-utils` (needed by pdf2image)
- Setup: `pip install -r requirements.txt && pip install -r requirements-dev.txt`
- Config: if `.env` is absent, copy `.env.example` to it; preserve existing settings. `MISTRAL_API_KEY` is optional for local conversion and non-cloud modes.

## Dependency and CI runtime pins

Keep the exact runtime dependency pins in `pyproject.toml` and `requirements.txt`
aligned and validate upgrades together. The existing build and development bounds
(`setuptools>=83,<85`, `wheel~=0.48`, `isort>=8,<10`, and `twine>=6.2,<8.0`)
are intentional; a runtime dependency update does not authorize relaxing them.
The previous validated baseline was established in
[PR #50](https://github.com/Balaxxe/Mistral_Markitdown/pull/50).

Lint, security, and release jobs intentionally use Python 3.10, the oldest
supported runtime. The normal test matrix covers Python 3.10, 3.11, and 3.12 on
Linux, macOS, and Windows. Keep these workflow runtimes in place until a
coordinated migration is validated. In particular, changing only the lint,
security, and release jobs to Python 3.14 would leave that version out of the
normal test matrix.

[PR #43](https://github.com/Balaxxe/Mistral_Markitdown/pull/43) was closed because
MarkItDown 0.1.7 required a YouTube transcript dependency that excluded Python
3.14. MarkItDown 0.1.8 removes that specific installation blocker, but does not
by itself establish application support for a new Python runtime. A future
runtime upgrade must cover the normal test matrix, dependency installation,
lint and type checks, security audit, distribution build, and installed CLI
smoke tests. Also follow the upstream SDK request-body checks in
[CONFIGURATION.md](../../CONFIGURATION.md#maintainer-note-upgrading-mistralai-or-markitdown).
