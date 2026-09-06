# Commands

- Install: `pip install -r requirements.txt`
- Install dev: `pip install -r requirements.txt && pip install -r requirements-dev.txt`
- Test all: `python3 -m pytest tests/` after dev install, or `bash scripts/test-safe.sh`, or `python3 run_tests.py` (bootstraps `./env` + dev deps if pytest is missing)
- Test single file: `python3 -m pytest tests/test_<name>.py -v`
- Lint: `python3 -m flake8 .`
- Format: `python3 -m black . && python3 -m isort .`
- Full check: `make check` (lint + typecheck + test; matches CI)
- Run app: `python3 main.py` (interactive) or `python3 main.py --mode markitdown --no-interactive`
- Self-test: `python3 main.py --test`
- Coverage: `python3 -m pytest tests/ --cov=. --cov-report=html --cov-report=term-missing`
- Security audit: `pip-audit --desc`
