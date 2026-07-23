# Contributing to Enhanced Document Converter

Thank you for your interest in contributing to Enhanced Document Converter! This document provides guidelines and setup instructions for developers.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, virtualenv, or conda)

### Quick Start

```bash
# Clone the repository (replace with your fork URL)
git clone https://github.com/your-username/Mistral_Markitdown.git
cd Mistral_Markitdown

# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
make install-dev
# Or manually:
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Optional: Install extended features (if testing audio/YouTube features)
pip install -r requirements-optional.txt

# Verify setup
make check
```

**Note on Optional Dependencies:**

- `requirements-optional.txt` contains MarkItDown plugin features (audio, YouTube, etc.)
- Only install if you're working on features that require these capabilities
- See [README.md](README.md) and [CONFIGURATION.md](CONFIGURATION.md) for details

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
python3 -m pytest tests/test_utils.py -v

# Run tests with coverage
python3 -m pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
```

### Code Quality

```bash
# Run all linters
make lint

# Format code
make format

# Run all checks before committing
make check
```

### Making Changes

1. **Create a branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new features

3. **Test your changes**

   ```bash
   make check
   ```

4. **Commit your changes**

   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

5. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style Guidelines

### Python Style

- **Line Length**: 120 characters maximum
- **Formatting**: Use `black` and `isort`
- **Docstrings**: Google-style docstrings for all public functions
- **Type Hints**: Use type hints for function signatures
- **Imports**: Group imports (stdlib, third-party, local)
- **Logging**: Use lazy `%`-formatting (`logger.info("Processing %s", name)`) instead of f-strings

### Example

```python
from pathlib import Path
from typing import Optional, Tuple

import config
import utils


def process_document(file_path: Path, use_cache: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Process a document file.

    Args:
        file_path: Path to the document file
        use_cache: Whether to use cached results

    Returns:
        Tuple of (success, error_message)
    """
    logger.info("Processing %s", file_path.name)
    # Implementation...
```

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures for common setup
- Mock external API calls

**Testing Optional Features:**

- Audio/YouTube features require `requirements-optional.txt` to be installed
- Use `pytest.importorskip()` to skip tests when optional packages are missing
- Example: `pytest.importorskip("pydub", reason="Audio features not installed")`

### Test Structure

```python
def test_feature_name():
    """Test description."""
    # Arrange
    input_data = create_test_data()

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_output
```

### Fixtures

Common fixtures are available in `tests/conftest.py`:

- `tmp_path`: Temporary directory
- `sample_pdf_path`: Sample PDF file
- `sample_markdown`: Sample markdown content
- `mock_env_vars`: Mocked environment variables

## Project Structure

```
Mistral_Markitdown/
├── main.py                  # Main application entry point: Convert (Smart), Convert (MarkItDown), Convert (Mistral OCR), PDF to Images, Document QnA, Batch OCR, System Status
├── config.py                # Configuration management (90+ options)
├── cli_files.py             # Input discovery, validation, and selection
├── local_converter.py       # MarkItDown integration & table extraction
├── mistral_converter/       # Mistral client, upload, OCR, QnA, batch, URL validation, and image helpers
├── modes/                   # Batch, QnA, maintenance, and system-status orchestration
├── utils.py                 # Utility functions (caching, logging, formatting)
├── schemas.py               # JSON schemas for structured extraction
│
├── requirements.txt         # Core dependencies (required)
├── requirements-optional.txt # Optional features (audio, YouTube, …)
├── requirements-dev.txt     # Development dependencies (testing, linting)
│
├── README.md                # Complete user guide
├── ARCHITECTURE.md          # Architecture and design details
├── CONFIGURATION.md         # Complete configuration reference
├── KNOWN_ISSUES.md          # Known issues and troubleshooting guide
├── CONTRIBUTING.md          # Development guidelines (this file)
├── LICENSE                  # MIT License
│
├── scripts/run_converter.bat  # Windows quick start script
├── scripts/quick_start.sh     # Linux/macOS quick start script
├── Makefile                 # Development commands
├── pyproject.toml           # Tool configuration (black, isort, pytest)
│
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── conftest.py          # Pytest fixtures
│   ├── test_config.py       # Configuration tests
│   ├── test_local_converter.py    # Local converter tests
│   ├── test_mistral_converter.py  # OCR, client, upload, image, and schema-format helper tests
│   ├── test_mistral_batch.py      # Batch API tests
│   ├── test_mistral_qna.py        # Document QnA tests
│   ├── test_mistral_url_validation.py # URL and SSRF validation tests
│   ├── test_integration_markitdown.py # On-disk MarkItDown integration tests
│   ├── test_pipeline.py     # End-to-end pipeline tests
│   ├── test_schemas.py      # Schema validation tests
│   └── test_utils.py        # Utility function tests
│
├── .github/workflows/       # CI/CD automation
│   ├── test.yml             # Multi-platform testing
│   ├── lint.yml             # Code quality checks
│   ├── security.yml         # Security audit (pip-audit, bandit)
│   └── publish.yml          # PyPI publishing on version tags
│
├── input/                   # Place files to convert here
├── output_md/               # Markdown output
├── output_txt/              # Plain text output
├── output_images/           # Extracted images and PDF renders
├── cache/                   # OCR result cache
└── logs/                    # Processing logs
    └── metadata/            # Batch processing metadata
```

## Common Tasks

### Adding a New Feature

1. **Create tests first** (TDD approach)

   ```bash
   # Create test file
   touch tests/test_new_feature.py

   # Write failing tests
   # Then implement feature to make tests pass
   ```

2. **Implement the feature**
   - Add to appropriate module
   - Follow existing patterns
   - Use type hints
   - Add docstrings

3. **Update documentation**
   - Update README.md if user-facing
   - Update docstrings
   - Add comments for complex logic

### Debugging

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python3 main.py

# Use ipdb for interactive debugging
import ipdb; ipdb.set_trace()

# Run specific test with output
python3 -m pytest tests/test_utils.py::TestClassName::test_method_name -v -s
```

### Updating Dependencies

```bash
# Update requirements
pip install --upgrade package-name
pip freeze > requirements.txt

# Test with new versions
make check
```

## Continuous Integration

### GitHub Actions

Four workflows run automatically:

1. **Lint** (`lint.yml`)
   - Runs on every push and PR
   - Checks code formatting (flake8, black, isort)
   - Fast feedback (~2 minutes)

2. **Test** (`test.yml`)
   - Runs on every push and PR
   - Tests on 9 matrix combinations (3 OS x 3 Python versions)
   - Includes coverage upload (~10 minutes)

3. **Security** (`security.yml`)
   - Runs on every push/PR and weekly (Monday 8am UTC)
   - Runs pip-audit for dependency vulnerabilities
   - Runs bandit for Python security linting

4. **Publish** (`publish.yml`)
   - Triggers on `v*` tags
   - Runs full test suite before publishing
   - Publishes to PyPI via trusted publishing

### Pre-commit Hooks

Install pre-commit hooks for automatic checks:

```bash
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## Release Process

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Run full test suite**
   ```bash
   make check
   ```
4. **Create git tag**
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

## Getting Help

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check README.md and docstrings

## Code Standards

- **Docstrings:** Google format (see `mistral_converter/` for examples)
- **Commit messages:** Follow [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`
- **Test coverage:** Maintain 75%+ overall coverage; add tests for all new features
- **PR requirements:** All PRs must pass `make check` (lint + tests) before merge
- **Type hints:** Required for all function signatures
- **Logging:** Use lazy `%`-formatting (`logger.info("Processing %s", name)`) instead of f-strings

## Code Review Guidelines

### For Contributors

- Keep PRs focused and reasonably sized
- Write clear commit messages following Conventional Commits
- Respond to review comments promptly
- Ensure CI passes before requesting review
- Link PRs to related issues when applicable

### For Reviewers

- Be respectful and constructive
- Focus on code quality and maintainability
- Check for test coverage
- Verify documentation is updated

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to Enhanced Document Converter!

**Version:** 3.0.2

**Related Documentation:**

- **[README.md](README.md)** - Complete feature documentation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Architecture and design details
- **[CONFIGURATION.md](CONFIGURATION.md)** - Configuration reference
- **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)** - Known issues
- **[CHANGELOG.md](CHANGELOG.md)** - Release history
- **[SECURITY.md](SECURITY.md)** - Security policy
