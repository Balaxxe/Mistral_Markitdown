# Mistral Markitdown

Stack: Python 3.10–3.12, MarkItDown, Mistral AI SDK, Pydantic, pdfplumber, pdf2image

## Commands

- Install: `pip install -r requirements.txt`
- Install dev: `pip install -r requirements.txt && pip install -r requirements-dev.txt`
- Test all: `python3 -m pytest tests/` (~720 tests, ~15s) after dev install, or `bash scripts/test-safe.sh`, or `python3 run_tests.py` (bootstraps `./env` + dev deps if pytest is missing)
- Test single file: `python3 -m pytest tests/test_<name>.py -v`
- Lint: `python3 -m flake8 .`
- Format: `python3 -m black . && python3 -m isort .`
- Full check: `make check` (lint + typecheck + test; matches CI)
- Run app: `python3 main.py` (interactive) or `python3 main.py --mode markitdown --no-interactive`
- Self-test: `python3 main.py --test`
- Coverage: `python3 -m pytest tests/ --cov=. --cov-report=html --cov-report=term-missing`
- Security audit: `pip-audit --desc`

## Structure

- `main.py` -- CLI entry point and orchestration
- `config.py` -- configuration loading from .env and defaults
- `schemas.py` -- Pydantic data models and validation
- `mistral_converter.py` -- Mistral AI OCR/QnA/Batch conversion
- `local_converter.py` -- local MarkItDown-based conversion
- `utils.py` -- shared utilities
- `scripts/` -- helper scripts (test runner, etc.)
- `tests/` -- pytest test suite
- `input/` -- drop files here for conversion (gitignored)
- `output_md/`, `output_txt/`, `output_images/` -- conversion output (gitignored)
- `cache/` -- runtime cache (gitignored)
- `docs/` -- documentation

## Rules

- Make the smallest safe change.
- Preserve public APIs unless the task says otherwise.
- Reuse existing patterns/utilities before adding abstractions.
- Add/update tests for behavior changes.
- Ask first before changing schema, auth, CI, infra, or dependencies.
- Never commit secrets or edit generated/vendor files casually.
- Use `python3 -m <tool>` instead of bare commands when not in a virtualenv.

## Environment

- Python: 3.10, 3.11, or 3.12
- System deps: `poppler-utils` (needed by pdf2image)
- Setup: `pip install -r requirements.txt && pip install -r requirements-dev.txt`
- Config: copy `.env.example` to `.env`, set `MISTRAL_API_KEY` (optional — without it, only local MarkItDown works)

## Debugging

- Stale lint: `python3 -m flake8 .` (config is in `.flake8`, 120 char line length, black-compatible ignores)
- Failing tests: `python3 -m pytest tests/ -v --tb=long` — tests mock API calls so they pass without a key
- Type checking: `python3 -m pyright` (CI runs this; `make typecheck` mirrors it). `pyrightconfig.json` uses `typeCheckingMode: basic` with several reports disabled and `tests/` excluded.

## Gotchas

- `MISTRAL_API_KEY` is optional. Without it, only local MarkItDown conversion works; all Mistral OCR/QnA/Batch features are disabled.
- The `Makefile` and `scripts/test-safe.sh` reference a local `env/` virtualenv. In cloud or CI environments, run tools via `python3 -m <tool>`.
- Pre-existing lint warnings exist in test files (unused imports, unused variables); these are in the upstream code.
- flake8 config is in `.flake8` (120 char line length, black-compatible ignores). pytest config is in `pyproject.toml`.
- Black is configured with `line-length = 120` and isort uses `profile = "black"` — both in `pyproject.toml`.

## Subagents

Three custom subagents are defined in `.cursor/agents/`:

- **verifier** (read-only) — reviews changes for correctness, coverage gaps, and rule violations. Use after edits to catch issues before committing.
- **test-runner** — runs scoped pytest for changed files and reports results. Does not fix failures.
- **researcher** (read-only) — explores the codebase for usage sites, patterns, and context before making changes in unfamiliar areas.

**Critical rules subagents must follow** (they do NOT inherit User Rules):

- Use `python3 -m <tool>` instead of bare commands (no virtualenv assumed).
- Never hardcode secrets. `MISTRAL_API_KEY` is loaded via `python-dotenv` and `config.py`.
- Tests mock API calls — they work without a Mistral key.
- Lint: `python3 -m flake8 .` | Format: `python3 -m black . && python3 -m isort .`
- Test: `python3 -m pytest tests/test_<name>.py -v` (scoped) or `python3 -m pytest tests/` (full)

The subagent gate hook (`.cursor/hooks/subagent-gate.sh`) restricts spawning to the three named subagents plus built-in types (explore, bash). Unrecognized agents are denied.

## PRs

- Before PR: `make check` (runs lint + typecheck + tests, mirroring CI)
- Include summary, risk, and validation steps.
