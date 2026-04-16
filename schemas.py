"""
Enhanced Document Converter - JSON Schemas for Structured Extraction

This module provides predefined JSON schemas AND Pydantic models for structured
data extraction from various document types using Mistral AI's structured output
capabilities.

Documentation references:
- Mistral Structured Outputs: https://docs.mistral.ai/capabilities/json_mode/
- Mistral OCR Annotations: https://docs.mistral.ai/capabilities/document_ai/annotations
- JSON Schema: https://json-schema.org/
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field

# Use the shared application logger configured in utils.setup_logging so all
# modules emit with the same handlers/formatters. utils does not depend on
# schemas, so importing it here is safe (no cycle).
from utils import logger  # noqa: E402  (import after third-party is intentional)


class _BaseSchema(BaseModel):
    """Shared configuration for all schema models."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


__all__ = [
    "ImageAnnotation",
    "TableAnnotation",
    "ChartAnnotation",
    "BBoxStructuredAnnotation",
    "VendorInfo",
    "CustomerInfo",
    "InvoiceDetails",
    "LineItem",
    "InvoiceTotals",
    "InvoiceDocument",
    "DocumentSection",
    "GenericDocument",
    "CompanyInfo",
    "StatementPeriod",
    "AccountEntry",
    "StatementTotals",
    "FinancialStatementDocument",
    "ContractParty",
    "ContractDates",
    "ContractClause",
    "ContractDocument",
    "FormField",
    "FormSignature",
    "FormDates",
    "FormDocument",
    "get_document_schema",
    "get_bbox_schema",
    "get_bbox_pydantic_model",
    "get_document_pydantic_model",
]

# ============================================================================
# Pydantic Models for BBox Annotations
# These are the recommended way to define schemas for Mistral OCR annotations
# Use with: from mistralai.extra import response_format_from_pydantic_model
# ============================================================================


class ImageAnnotation(_BaseSchema):
    """Pydantic model for image/bbox annotation extraction."""

    image_type: str = Field(..., description="The type of the image (chart, diagram, photo, table, figure, etc.)")
    short_description: str = Field(..., description="A brief description in English describing the image content.")
    summary: str = Field(..., description="A detailed summary of the image including key information and data points.")


class TableAnnotation(_BaseSchema):
    """Pydantic model for table bbox annotation extraction."""

    table_type: str = Field(..., description="The type of table (data, financial, comparison, schedule, etc.)")
    caption: Optional[str] = Field(None, description="Table caption or title if present.")
    rows: int = Field(..., ge=1, description="Number of rows in the table.")
    columns: int = Field(..., ge=1, description="Number of columns in the table.")
    has_header: bool = Field(..., description="Whether the table has a header row.")
    content_summary: str = Field(..., description="Summary of the table's content and purpose.")


class ChartAnnotation(_BaseSchema):
    """Pydantic model for chart/graph bbox annotation extraction."""

    chart_type: str = Field(..., description="The type of chart (bar, line, pie, scatter, histogram, etc.)")
    title: Optional[str] = Field(None, description="Chart title if present.")
    x_axis_label: Optional[str] = Field(None, description="X-axis label if present.")
    y_axis_label: Optional[str] = Field(None, description="Y-axis label if present.")
    data_summary: str = Field(..., description="Summary of the data trends and key insights from the chart.")


class BBoxStructuredAnnotation(_BaseSchema):
    """Comprehensive Pydantic model for bounding box annotation."""

    bbox_type: str = Field(
        ..., description="Type of content in bounding box: text, table, figure, heading, list, chart, other"
    )
    text_content: str = Field(..., description="Extracted text from the bounding box.")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score for extraction (0-1).")
    is_handwritten: Optional[bool] = Field(None, description="Whether content appears handwritten.")
    language: Optional[str] = Field(None, description="Detected language.")


# Document-level Pydantic models
class VendorInfo(_BaseSchema):
    """Vendor information for invoice extraction."""

    name: str = Field(..., description="Vendor name")
    address: Optional[str] = Field(None, description="Vendor address")
    tax_id: Optional[str] = Field(None, description="Tax ID or VAT number")
    contact: Optional[str] = Field(None, description="Contact information")


class CustomerInfo(_BaseSchema):
    """Customer information for invoice extraction."""

    name: Optional[str] = Field(None, description="Customer name")
    address: Optional[str] = Field(None, description="Customer address")
    tax_id: Optional[str] = Field(None, description="Tax ID or VAT number")


class InvoiceDetails(_BaseSchema):
    """Invoice details for extraction."""

    invoice_number: str = Field(..., description="Invoice number")
    invoice_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Invoice date (ISO 8601, YYYY-MM-DD)")
    due_date: Optional[str] = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Payment due date (ISO 8601, YYYY-MM-DD)"
    )
    purchase_order: Optional[str] = Field(None, description="PO number if applicable")


class LineItem(_BaseSchema):
    """Line item for invoice extraction."""

    description: str = Field(..., description="Item description")
    quantity: Optional[Decimal] = Field(None, description="Quantity")
    unit_price: Optional[Decimal] = Field(None, description="Unit price")
    amount: Decimal = Field(..., description="Total amount for this line")
    tax_rate: Optional[Decimal] = Field(None, description="Tax rate if applicable")


class InvoiceTotals(_BaseSchema):
    """Invoice totals for extraction."""

    subtotal: Optional[Decimal] = Field(None, description="Subtotal before tax")
    tax: Optional[Decimal] = Field(None, description="Total tax amount")
    total: Decimal = Field(..., description="Total amount due")
    currency: Optional[str] = Field(None, description="Currency code (USD, EUR, etc.)")


class InvoiceDocument(_BaseSchema):
    """Complete invoice document extraction model."""

    document_type: str = Field(..., description="Type of financial document: invoice, receipt, bill")
    vendor: VendorInfo = Field(..., description="Vendor information")
    customer: Optional[CustomerInfo] = Field(None, description="Customer information")
    invoice_details: InvoiceDetails = Field(..., description="Invoice details")
    line_items: Optional[List[LineItem]] = Field(None, description="List of invoice line items")
    totals: InvoiceTotals = Field(..., description="Invoice totals")
    payment_terms: Optional[str] = Field(None, description="Payment terms and conditions")
    notes: Optional[str] = Field(None, description="Additional notes or comments")


class DocumentSection(_BaseSchema):
    """Document section for generic extraction."""

    heading: str = Field(..., description="Section heading")
    level: Optional[int] = Field(None, ge=1, le=6, description="Heading level (1-6)")
    content_summary: Optional[str] = Field(None, description="Brief summary of section content")


class GenericDocument(_BaseSchema):
    """Generic document extraction model."""

    document_type: str = Field(..., description="Type or category of document")
    title: Optional[str] = Field(None, description="Document title")
    authors: Optional[List[str]] = Field(None, description="Document authors or creators")
    date: Optional[str] = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Document date (ISO 8601, YYYY-MM-DD)"
    )
    sections: Optional[List[DocumentSection]] = Field(None, description="Document sections and headings")
    summary: Optional[str] = Field(None, description="Brief document summary")


# Financial statement models
class CompanyInfo(_BaseSchema):
    """Company information for financial statement extraction."""

    name: str = Field(..., description="Company name")
    registration_number: Optional[str] = Field(None, description="Registration or tax ID number")
    address: Optional[str] = Field(None, description="Company address")


class StatementPeriod(_BaseSchema):
    """Reporting period for financial statement extraction."""

    start_date: Optional[str] = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Period start (ISO 8601, YYYY-MM-DD)"
    )
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Period end (ISO 8601, YYYY-MM-DD)")
    fiscal_year: Optional[int] = Field(None, description="Fiscal year")


class AccountEntry(_BaseSchema):
    """Individual account line in a financial statement."""

    account_number: Optional[str] = Field(None, description="Account number")
    account_name: str = Field(..., description="Account name")
    category: Optional[str] = Field(None, description="Category: Assets, Liabilities, Revenue, etc.")
    debit: Optional[Decimal] = Field(None, description="Debit amount")
    credit: Optional[Decimal] = Field(None, description="Credit amount")
    balance: Optional[Decimal] = Field(None, description="Account balance")


class StatementTotals(_BaseSchema):
    """Aggregate totals for financial statement extraction."""

    total_assets: Optional[Decimal] = Field(None, description="Total assets")
    total_liabilities: Optional[Decimal] = Field(None, description="Total liabilities")
    total_equity: Optional[Decimal] = Field(None, description="Total equity")
    total_revenue: Optional[Decimal] = Field(None, description="Total revenue")
    total_expenses: Optional[Decimal] = Field(None, description="Total expenses")
    net_income: Optional[Decimal] = Field(None, description="Net income")


class FinancialStatementDocument(_BaseSchema):
    """Complete financial statement extraction model."""

    statement_type: str = Field(..., description="Type: balance_sheet, income_statement, cash_flow, trial_balance")
    company: CompanyInfo = Field(..., description="Company information")
    period: StatementPeriod = Field(..., description="Reporting period")
    accounts: List[AccountEntry] = Field(..., description="List of accounts with balances")
    totals: Optional[StatementTotals] = Field(None, description="Aggregate totals")
    currency: Optional[str] = Field(None, description="Currency code")
    audited: Optional[bool] = Field(None, description="Whether the statement is audited")
    notes: Optional[List[str]] = Field(None, description="Footnotes and additional information")


# Contract document models
class ContractParty(_BaseSchema):
    """Party to a contract."""

    name: str = Field(..., description="Party name (individual or organization)")
    role: Optional[str] = Field(None, description="Role in the contract (buyer, seller, licensor, tenant, etc.)")
    address: Optional[str] = Field(None, description="Party address")
    representative: Optional[str] = Field(None, description="Authorized representative name")
    title: Optional[str] = Field(None, description="Representative title or position")


class ContractDates(_BaseSchema):
    """Key dates in a contract."""

    effective_date: Optional[str] = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date the contract takes effect (ISO 8601, YYYY-MM-DD)"
    )
    expiration_date: Optional[str] = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date the contract expires (ISO 8601, YYYY-MM-DD)"
    )
    execution_date: Optional[str] = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date the contract was signed (ISO 8601, YYYY-MM-DD)"
    )
    renewal_date: Optional[str] = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Next renewal date if applicable (ISO 8601, YYYY-MM-DD)"
    )


class ContractClause(_BaseSchema):
    """A clause or section within a contract."""

    clause_number: Optional[str] = Field(None, description="Clause or section number")
    title: str = Field(..., description="Clause title or heading")
    summary: Optional[str] = Field(None, description="Brief summary of the clause content")


class ContractDocument(_BaseSchema):
    """Complete contract document extraction model."""

    contract_type: str = Field(
        ..., description="Type of contract: employment, lease, NDA, service, license, purchase, partnership, other"
    )
    title: Optional[str] = Field(None, description="Contract title")
    parties: List[ContractParty] = Field(..., description="Parties to the contract")
    dates: Optional[ContractDates] = Field(None, description="Key contract dates")
    clauses: Optional[List[ContractClause]] = Field(None, description="Key clauses and sections")
    governing_law: Optional[str] = Field(None, description="Governing law or jurisdiction")
    consideration: Optional[str] = Field(None, description="Consideration or payment terms")
    term_duration: Optional[str] = Field(None, description="Duration of the contract")
    termination_conditions: Optional[str] = Field(None, description="Conditions for early termination")
    signatures: Optional[List[str]] = Field(None, description="Names of signatories")
    notes: Optional[str] = Field(None, description="Additional notes or observations")


# Form document models
class FormField(_BaseSchema):
    """Individual field in a form document."""

    field_name: str = Field(..., description="Field label or name")
    field_type: Optional[str] = Field(None, description="Field type: text, number, date, checkbox, signature, other")
    field_value: str = Field(..., description="Value entered in the field")
    is_filled: Optional[bool] = Field(None, description="Whether the field has been filled in")


class FormSignature(_BaseSchema):
    """Signature field in a form document."""

    signer_name: Optional[str] = Field(None, description="Name of the signer")
    title: Optional[str] = Field(None, description="Title or role of the signer")
    date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date signed (ISO 8601, YYYY-MM-DD)")
    is_signed: Optional[bool] = Field(None, description="Whether the field has been signed")


class FormDates(_BaseSchema):
    """Key dates in a form document."""

    submission_date: Optional[str] = Field(None, description="Date submitted")
    effective_date: Optional[str] = Field(None, description="Date effective")
    expiration_date: Optional[str] = Field(None, description="Date of expiration")


class FormDocument(_BaseSchema):
    """Complete form document extraction model."""

    form_type: Optional[str] = Field(None, description="Type of form: application, survey, contract, etc.")
    form_title: str = Field(..., description="Title or name of the form")
    form_number: Optional[str] = Field(None, description="Form number or identifier")
    fields: List[FormField] = Field(..., description="Form fields and their values")
    signatures: Optional[List[FormSignature]] = Field(None, description="Signature fields")
    dates: Optional[FormDates] = Field(None, description="Key dates")


# ============================================================================
# Schema Registry (derived from Pydantic models)
# ============================================================================

_DOCUMENT_DESCRIPTIONS = {
    "invoice": "Structured extraction of invoice data including vendor, line items, and totals",
    "financial_statement": "Structured extraction of financial statements including accounts and balances",
    "contract": "Structured extraction of contract parties, terms, clauses, and dates",
    "form": "Structured extraction of form fields and values",
    "generic": "Structured extraction of general document metadata and structure",
}

_BBOX_DESCRIPTIONS = {
    "structured": "Structured extraction of bounding box content with type and formatting",
}

_DOCUMENT_MODEL_MAP = {
    "generic": GenericDocument,
    "invoice": InvoiceDocument,
    "financial_statement": FinancialStatementDocument,
    "contract": ContractDocument,
    "form": FormDocument,
}

_BBOX_MODEL_MAP = {
    "image": ImageAnnotation,
    "table": TableAnnotation,
    "chart": ChartAnnotation,
    "structured": BBoxStructuredAnnotation,
}


def get_document_schema(schema_type: str = "generic") -> Dict[str, Any]:
    """
    Get a document-level JSON schema by type, derived from Pydantic models.

    Args:
        schema_type: Type of schema (invoice, financial_statement, contract, form, generic)

    Returns:
        Schema definition dictionary with name, schema, and description keys
    """
    if schema_type not in _DOCUMENT_MODEL_MAP:
        logger.warning("Unknown document schema type %r, falling back to 'generic'", schema_type)
    model = _DOCUMENT_MODEL_MAP.get(schema_type, GenericDocument)
    return {
        "name": f"{schema_type}_extraction",
        "schema": model.model_json_schema(),
        "description": _DOCUMENT_DESCRIPTIONS.get(schema_type, _DOCUMENT_DESCRIPTIONS["generic"]),
    }


def get_bbox_schema(schema_type: str = "structured") -> Dict[str, Any]:
    """
    Get a bounding box JSON schema by type, derived from Pydantic models.

    Args:
        schema_type: Type of schema (currently only 'structured' available)

    Returns:
        Schema definition dictionary with name, schema, and description keys
    """
    if schema_type not in _BBOX_MODEL_MAP:
        logger.warning("Unknown bbox schema type %r, falling back to 'structured'", schema_type)
    model = _BBOX_MODEL_MAP.get(schema_type, BBoxStructuredAnnotation)
    return {
        "name": f"bbox_{schema_type}_extraction",
        "schema": model.model_json_schema(),
        "description": _BBOX_DESCRIPTIONS.get(schema_type, _BBOX_DESCRIPTIONS["structured"]),
    }


# ============================================================================
# Pydantic Model Getters
# These functions return Pydantic model classes for use with
# mistralai.extra.response_format_from_pydantic_model()
# ============================================================================


def get_bbox_pydantic_model(annotation_type: str = "structured") -> Type[BaseModel]:
    """
    Get a Pydantic model class for bounding box annotation.

    Args:
        annotation_type: Type of annotation model to return:
            - "image": General image/figure annotation
            - "table": Table-specific annotation with structure info
            - "chart": Chart/graph annotation with axis info
            - "structured": Comprehensive bbox annotation (default)

    Returns:
        Pydantic model class
    """
    if annotation_type not in _BBOX_MODEL_MAP:
        logger.warning("Unknown bbox annotation type %r, falling back to 'structured'", annotation_type)
    return _BBOX_MODEL_MAP.get(annotation_type, BBoxStructuredAnnotation)


def get_document_pydantic_model(doc_type: str = "generic") -> Type[BaseModel]:
    """
    Get a Pydantic model class for document-level annotation.

    Args:
        doc_type: Type of document model to return:
            - "generic": General document structure (default)
            - "invoice": Invoice/receipt extraction
            - "financial_statement": Financial statement extraction
            - "contract": Contract extraction
            - "form": Form extraction

    Returns:
        Pydantic model class
    """
    if doc_type not in _DOCUMENT_MODEL_MAP:
        logger.warning("Unknown document type %r, falling back to 'generic'", doc_type)
    return _DOCUMENT_MODEL_MAP.get(doc_type, GenericDocument)
