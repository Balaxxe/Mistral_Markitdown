# Debugging

- Stale lint: `python3 -m flake8 .` (config is in `.flake8`, 120 char line length, black-compatible ignores)
- Failing tests: `python3 -m pytest tests/ -v --tb=long` — tests mock API calls so they pass without a key
- Type checking: `python3 -m pyright` (CI runs this; `make typecheck` mirrors it). `pyrightconfig.json` uses `typeCheckingMode: basic` with several reports disabled and `tests/` excluded.
