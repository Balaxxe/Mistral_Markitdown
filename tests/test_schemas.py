"""
Tests for schemas.py module.

Tests cover:
- Schema retrieval functions (get_document_schema, get_bbox_schema)
- Pydantic model retrieval (get_document_pydantic_model, get_bbox_pydantic_model)
- Schema structure validation (required keys, types)
"""

import pytest

import config
import schemas

# ============================================================================
# get_document_schema Tests
# ============================================================================


class TestGetDocumentSchema:
    """Test document schema retrieval."""

    @pytest.mark.parametrize(
        "schema_type",
        ["invoice", "financial_statement", "contract", "form", "generic"],
    )
    def test_valid_types_return_dict(self, schema_type: str):
        result = schemas.get_document_schema(schema_type)
        assert isinstance(result, dict)

    def test_unknown_type_falls_back(self, monkeypatch):
        """Unknown types should fall back to generic when strict mode is off."""
        monkeypatch.setattr(config, "SCHEMA_STRICT_UNKNOWN_TYPES", False)
        result = schemas.get_document_schema("nonexistent_type")
        assert isinstance(result, dict)

    def test_unknown_type_raises_when_strict(self, monkeypatch):
        monkeypatch.setattr(config, "SCHEMA_STRICT_UNKNOWN_TYPES", True)
        with pytest.raises(ValueError, match="Unknown document schema type"):
            schemas.get_document_schema("nonexistent_type")

    @pytest.mark.parametrize(
        "schema_type",
        ["invoice", "financial_statement", "contract", "form", "generic"],
    )
    def test_schema_has_required_keys(self, schema_type: str):
        result = schemas.get_document_schema(schema_type)
        # Should have at minimum a 'schema' key with the JSON schema
        assert "schema" in result
        assert "name" in result


# ============================================================================
# get_bbox_schema Tests
# ============================================================================


class TestGetBboxSchema:
    """Test bounding box schema retrieval."""

    def test_structured_returns_dict(self):
        result = schemas.get_bbox_schema("structured")
        assert isinstance(result, dict)

    def test_schema_has_required_keys(self):
        result = schemas.get_bbox_schema("structured")
        assert "schema" in result
        assert "name" in result


# ============================================================================
# Pydantic Model Tests
# ============================================================================


class TestPydanticModels:
    """Test Pydantic model retrieval and JSON schema generation."""

    def test_bbox_pydantic_model_returns_class(self):
        model = schemas.get_bbox_pydantic_model()
        # Should return a Pydantic model class or None
        if model is not None:
            # Should be able to generate a JSON schema
            json_schema = model.model_json_schema()
            assert isinstance(json_schema, dict)
            assert "properties" in json_schema or "type" in json_schema

    @pytest.mark.parametrize(
        "doc_type",
        ["invoice", "generic", "financial_statement", "contract", "form"],
    )
    def test_document_pydantic_model_returns_class(self, doc_type: str):
        model = schemas.get_document_pydantic_model(doc_type)
        if model is not None:
            json_schema = model.model_json_schema()
            assert isinstance(json_schema, dict)

    def test_unknown_document_type_returns_none_or_generic(self):
        model = schemas.get_document_pydantic_model("nonexistent")
        # Should return None or fall back to generic
        assert model is None or hasattr(model, "model_json_schema")


# ============================================================================
# Schema Content Validation
# ============================================================================


class TestSchemaContent:
    """Validate that schemas contain sensible content."""

    def test_invoice_schema_has_expected_fields(self):
        result = schemas.get_document_schema("invoice")
        schema_body = result.get("schema", result)
        # The word "invoice" should appear somewhere in the schema
        schema_str = str(schema_body).lower()
        assert "invoice" in schema_str

    def test_generic_schema_is_not_empty(self):
        result = schemas.get_document_schema("generic")
        schema_body = result.get("schema", result)
        assert len(str(schema_body)) > 20  # Non-trivial schema


class TestPydanticModelValidation:
    """Test that Pydantic models enforce field constraints."""

    def test_invoice_line_item_requires_amount(self):
        with pytest.raises(Exception):
            schemas.LineItem(description="test")

    def test_invoice_date_rejects_bad_format(self):
        with pytest.raises(Exception):
            schemas.InvoiceDetails(invoice_number="1", invoice_date="not-a-date")

    def test_invoice_date_accepts_iso_format(self):
        details = schemas.InvoiceDetails(invoice_number="1", invoice_date="2024-01-15")
        assert details.invoice_date == "2024-01-15"

    def test_line_item_uses_decimal(self):
        from decimal import Decimal

        item = schemas.LineItem(description="Widget", amount=Decimal("19.99"))
        assert isinstance(item.amount, Decimal)

    def test_document_schema_returns_all_required_keys(self):
        for schema_type in ["invoice", "financial_statement", "contract", "form", "generic"]:
            result = schemas.get_document_schema(schema_type)
            assert "schema" in result, f"{schema_type} missing 'schema' key"
            assert "name" in result, f"{schema_type} missing 'name' key"
            assert "description" in result, f"{schema_type} missing 'description' key"

    def test_bbox_schema_returns_all_required_keys(self):
        result = schemas.get_bbox_schema("structured")
        assert "schema" in result
        assert "name" in result
        assert "description" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
