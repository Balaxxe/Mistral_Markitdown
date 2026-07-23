"""
Tests for config.py module
"""

import importlib
import warnings

import pytest

import config


class TestDirectoryCreation:
    """Test directory creation functionality."""

    def test_ensure_directories_creates_paths(self):
        """Test that ensure_directories creates required paths."""
        # Directories are created by initialize(), not at import time
        config.initialize()
        assert config.INPUT_DIR.exists()
        assert config.OUTPUT_MD_DIR.exists()
        assert config.OUTPUT_TXT_DIR.exists()
        assert config.OUTPUT_IMAGES_DIR.exists()
        assert config.CACHE_DIR.exists()
        assert config.LOGS_DIR.exists()
        assert config.METADATA_DIR.exists()


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_validate_configuration_returns_list(self):
        """Test that validation returns a list of issues."""
        issues = config.validate_configuration()
        assert isinstance(issues, list)

    def test_validate_configuration_warnings(self, monkeypatch):
        """Test validation warnings for missing API keys."""
        # Temporarily remove API key
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "")

        issues = config.validate_configuration()

        # Should warn about missing Mistral API key
        assert any("MISTRAL_API_KEY" in issue for issue in issues)


class TestModelSelection:
    """Test model selection functionality."""

    def test_get_ocr_model_returns_correct_model(self):
        """Test that get_ocr_model returns the OCR model."""
        # Should always return mistral-ocr-latest for OCR tasks
        model = config.get_ocr_model()
        assert model == config.MISTRAL_OCR_MODEL
        assert model == "mistral-ocr-latest"

    def test_mistral_openai_compatible_base_url_default(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_SERVER_URL", "")
        result = config.mistral_openai_compatible_base_url()
        assert result.startswith("https://api.mistral.ai")
        assert result.endswith("/v1")

    def test_mistral_openai_compatible_base_url_custom(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_SERVER_URL", "https://enterprise.example")
        result = config.mistral_openai_compatible_base_url()
        assert result.startswith("https://enterprise.example")
        assert result.endswith("/v1")

    def test_mistral_openai_compatible_base_url_no_duplicate_v1(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_SERVER_URL", "https://enterprise.example/v1")
        result = config.mistral_openai_compatible_base_url()
        assert result.startswith("https://enterprise.example")
        assert result.endswith("/v1")
        assert "/v1/v1" not in result

    def test_mistral_openai_compatible_base_url_trailing_slash_before_v1(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_SERVER_URL", "https://enterprise.example/v1/")
        result = config.mistral_openai_compatible_base_url()
        assert result.startswith("https://enterprise.example")
        assert result.endswith("/v1")


class TestFileTypeConfiguration:
    """Test file type configuration."""

    def test_markitdown_supported_types(self):
        """Test MarkItDown supported file types."""
        assert "pdf" in config.MARKITDOWN_SUPPORTED
        assert "docx" in config.MARKITDOWN_SUPPORTED
        assert "xlsx" in config.MARKITDOWN_SUPPORTED
        assert "png" in config.MARKITDOWN_SUPPORTED
        assert "webp" in config.MARKITDOWN_SUPPORTED
        assert "avif" in config.MARKITDOWN_SUPPORTED

    def test_mistral_ocr_supported_types(self):
        """Test Mistral OCR supported file types."""
        assert "pdf" in config.MISTRAL_OCR_SUPPORTED
        assert "png" in config.MISTRAL_OCR_SUPPORTED
        assert "jpg" in config.MISTRAL_OCR_SUPPORTED
        assert "docx" in config.MISTRAL_OCR_SUPPORTED

    def test_pdf_extensions(self):
        """Test PDF extensions."""
        assert "pdf" in config.PDF_EXTENSIONS

    def test_image_extensions(self):
        """Test image extensions."""
        assert "png" in config.IMAGE_EXTENSIONS
        assert "jpg" in config.IMAGE_EXTENSIONS
        assert "jpeg" in config.IMAGE_EXTENSIONS

    def test_office_extensions(self):
        """Test office document extensions."""
        assert "docx" in config.OFFICE_EXTENSIONS
        assert "pptx" in config.OFFICE_EXTENSIONS
        assert "xlsx" in config.OFFICE_EXTENSIONS


class TestMistralModels:
    """Test Mistral models configuration."""

    def test_mistral_models_defined(self):
        """Test that Mistral models are properly defined."""
        assert isinstance(config.MISTRAL_MODELS, dict)
        assert len(config.MISTRAL_MODELS) > 0

    def test_ocr_model_exists(self):
        """Test that the OCR model is defined."""
        assert "mistral-ocr-latest" in config.MISTRAL_MODELS

    def test_model_structure(self):
        """Test that models have required fields."""
        for model_id, model_info in config.MISTRAL_MODELS.items():
            assert "name" in model_info
            assert "description" in model_info
            assert "best_for" in model_info
            assert "max_tokens" in model_info


class TestConfigurationDefaults:
    """Test default configuration values."""

    def test_cache_duration_default(self):
        """Test cache duration default value."""
        assert isinstance(config.CACHE_DURATION_HOURS, int)
        assert config.CACHE_DURATION_HOURS > 0

    def test_max_concurrent_files_default(self):
        """Test max concurrent files default."""
        assert isinstance(config.MAX_CONCURRENT_FILES, int)
        assert config.MAX_CONCURRENT_FILES > 0

    def test_log_level_valid(self):
        """Test log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert config.LOG_LEVEL in valid_levels

    def test_reload_invalid_log_level_env_normalizes(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            importlib.reload(config)
        assert config.LOG_LEVEL == "INFO"
        assert any("LOG_LEVEL" in str(r.message) for r in rec)
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        importlib.reload(config)

    def test_ocr_model_correct(self):
        """Test OCR model is set correctly."""
        assert config.MISTRAL_OCR_MODEL == "mistral-ocr-latest"


class TestSafeParsingHelpers:
    """Test edge cases of _safe_int, _safe_float, _safe_bool, _safe_csv."""

    def test_safe_int_below_min(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "-5")
        result = config._safe_int("TEST_INT_VAR", 10, min_val=0)
        assert result == 10

    def test_safe_int_invalid(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "not_a_number")
        result = config._safe_int("TEST_INT_VAR", 42)
        assert result == 42

    def test_safe_float_below_min(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "-1.5")
        result = config._safe_float("TEST_FLOAT_VAR", 0.5, min_val=0.0)
        assert result == 0.5

    def test_safe_float_invalid(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "abc")
        result = config._safe_float("TEST_FLOAT_VAR", 3.14)
        assert result == 3.14

    def test_safe_bool_truthy_values(self, monkeypatch):
        for val in ("1", "true", "yes", "y", "on", "True", "YES"):
            monkeypatch.setenv("TEST_BOOL_VAR", val)
            assert config._safe_bool("TEST_BOOL_VAR", False) is True

    def test_safe_bool_falsy_values(self, monkeypatch):
        for val in ("0", "false", "no", "n", "off"):
            monkeypatch.setenv("TEST_BOOL_VAR", val)
            assert config._safe_bool("TEST_BOOL_VAR", True) is False

    def test_safe_bool_invalid_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL_VAR", "maybe")
        assert config._safe_bool("TEST_BOOL_VAR", True) is True

    def test_safe_csv_empty_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_CSV_VAR", "")
        result = config._safe_csv("TEST_CSV_VAR", "a,b,c")
        assert result == ["a", "b", "c"]

    def test_parse_table_output_formats_unset_defaults_markdown(self):
        assert config._parse_table_output_formats(None) == ["markdown"]

    def test_parse_table_output_formats_blank_disables_sidecars(self):
        assert config._parse_table_output_formats("") == []
        assert config._parse_table_output_formats("   ") == []

    def test_parse_table_output_formats_csv_list(self):
        assert config._parse_table_output_formats("markdown,csv") == ["markdown", "csv"]


class TestSafeIntBelowMinWarning:
    """Line 55: _safe_int value below min_val triggers warning."""

    def test_safe_int_value_below_min_val(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_MIN", "3")
        result = config._safe_int("TEST_INT_MIN", 10, min_val=5)
        assert result == 10

    def test_safe_int_value_meets_min_val(self, monkeypatch):
        """Line 55: return value when value >= min_val."""
        monkeypatch.setenv("TEST_INT_OK", "10")
        result = config._safe_int("TEST_INT_OK", 5, min_val=3)
        assert result == 10


class TestSafeFloatBelowMinWarning:
    """Line 79: _safe_float value below min_val triggers warning."""

    def test_safe_float_value_below_min_val(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_MIN", "0.1")
        result = config._safe_float("TEST_FLOAT_MIN", 1.0, min_val=0.5)
        assert result == 1.0

    def test_safe_float_value_meets_min_val(self, monkeypatch):
        """Line 79: return value when value >= min_val."""
        monkeypatch.setenv("TEST_FLOAT_OK", "2.5")
        result = config._safe_float("TEST_FLOAT_OK", 1.0, min_val=0.5)
        assert result == 2.5


class TestValidateConfigurationBranches:
    """Lines 508-558: all remaining validate_configuration branches."""

    def test_poppler_warning_on_win32(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(config, "POPPLER_PATH", "")
        import sys as sys_mod

        monkeypatch.setattr(sys_mod, "platform", "win32")
        issues = config.validate_configuration()
        assert any("POPPLER_PATH" in i for i in issues)

    def test_structured_output_conflict_bbox(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", False)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_BBOX_ANNOTATION", True)
        issues = config.validate_configuration()
        assert any("BBOX_ANNOTATION" in i and "STRUCTURED_OUTPUT" in i for i in issues)

    def test_structured_output_conflict_document(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(config, "MISTRAL_ENABLE_STRUCTURED_OUTPUT", False)
        monkeypatch.setattr(config, "MISTRAL_ENABLE_DOCUMENT_ANNOTATION", True)
        issues = config.validate_configuration()
        assert any("DOCUMENT_ANNOTATION" in i and "STRUCTURED_OUTPUT" in i for i in issues)

    def test_threshold_ordering_warning(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(config, "OCR_QUALITY_THRESHOLD_EXCELLENT", 50)
        monkeypatch.setattr(config, "OCR_QUALITY_THRESHOLD_GOOD", 90)
        monkeypatch.setattr(config, "OCR_QUALITY_THRESHOLD_ACCEPTABLE", 70)
        issues = config.validate_configuration()
        assert any("descending order" in i for i in issues)

    def test_invalid_log_level(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(config, "LOG_LEVEL", "TRACE")
        issues = config.validate_configuration()
        assert any("LOG_LEVEL" in i and "invalid" in i for i in issues)

    def test_invalid_cleanup_upload_scope_when_monkeypatched(self, monkeypatch):
        """Defensive validate_configuration path for monkeypatched invalid scope."""
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(config, "CLEANUP_UPLOAD_SCOPE", "everywhere")
        issues = config.validate_configuration()
        assert any("CLEANUP_UPLOAD_SCOPE" in i and "invalid" in i for i in issues)

    def test_invalid_schema_type(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_SCHEMA_TYPE", "custom_invalid")
        issues = config.validate_configuration()
        assert any("MISTRAL_DOCUMENT_SCHEMA_TYPE" in i and "invalid" in i for i in issues)

    def test_invalid_table_format(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(config, "MISTRAL_TABLE_FORMAT", "xml")
        issues = config.validate_configuration()
        assert any("MISTRAL_TABLE_FORMAT" in i and "invalid" in i for i in issues)

    def test_invalid_table_output_formats(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(config, "TABLE_OUTPUT_FORMATS", ["markdown", "pdf"])
        issues = config.validate_configuration()
        assert any("TABLE_OUTPUT_FORMATS" in i for i in issues)

    def test_llm_descriptions_without_api_key(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "")
        monkeypatch.setattr(config, "MARKITDOWN_ENABLE_LLM_DESCRIPTIONS", True)
        issues = config.validate_configuration()
        assert any("LLM_DESCRIPTIONS" in i for i in issues)


class TestValidateConfigurationEdgeCases:
    """Validation edge cases for schema type, image format, exiftool path, and ensure_directories."""

    def test_invalid_document_schema_type_warning(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_SCHEMA_TYPE", "nonexistent_type")
        issues = config.validate_configuration()
        assert any("MISTRAL_DOCUMENT_SCHEMA_TYPE" in i and "invalid" in i for i in issues)

    def test_invalid_pdf_image_format_warning(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(config, "PDF_IMAGE_FORMAT", "bmp")
        issues = config.validate_configuration()
        assert any("PDF_IMAGE_FORMAT" in i and "not a recognized format" in i for i in issues)

    def test_relative_exiftool_path_warning(self, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(config, "MARKITDOWN_EXIFTOOL_PATH", "bin/exiftool")
        issues = config.validate_configuration()
        assert any("MARKITDOWN_EXIFTOOL_PATH" in i and "not an absolute path" in i for i in issues)

    def test_ensure_directories_handles_oserror(self, monkeypatch):
        """ensure_directories logs but does not raise on OSError from mkdir."""
        from unittest.mock import patch

        with patch.object(config.Path, "mkdir", side_effect=OSError("permission denied")):
            config.ensure_directories()  # should not raise


class TestInitializeIdempotent:
    """Test that initialize() only runs once."""

    def test_initialize_returns_same_issues_on_second_call(self):
        config._initialized = False
        config._init_issues = []
        result1 = config.initialize()
        result2 = config.initialize()
        assert result2 == result1
        config._initialized = False
        config._init_issues = []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
