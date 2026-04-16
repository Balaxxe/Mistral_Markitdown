# Enhanced Document Converter - Makefile
# Convenience commands for development and testing

.PHONY: help install install-dev test lint typecheck format clean run check dist publish coverage security-audit dev-setup

help:
	@echo "Enhanced Document Converter - Development Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make test           - Run test suite"
	@echo "  make lint           - Run linters (flake8)"
	@echo "  make typecheck      - Run pyright type checks (matches CI)"
	@echo "  make format         - Format code with black and isort"
	@echo "  make coverage       - Run tests with coverage report"
	@echo "  make security-audit - Run pip-audit dependency scan"
	@echo "  make clean          - Clean build artifacts and cache"
	@echo "  make run            - Run the converter"
	@echo "  make check          - Run all checks (lint + type + test)"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	./scripts/test-safe.sh tests/

lint:
	python3 -m flake8 .

typecheck:
	python3 -m pyright

format:
	python3 -m black .
	python3 -m isort .


clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

run:
	python3 main.py

check: lint typecheck test
	@echo "All checks passed!"

coverage:
	python3 -m pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

security-audit:
	python3 -m pip_audit --desc

dist: clean
	python3 -m build

publish: dist
	python3 -m twine upload dist/*

# Development workflow
dev-setup: install-dev
	pre-commit install
	@echo "Development environment ready!"
	@echo "Run 'make check' to verify everything works"
