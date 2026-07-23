# Configuration Reference

Complete reference for all configuration options in Enhanced Document Converter.

## Table of Contents

- [Configuration File](#configuration-file)
- [API Keys](#api-keys)
- [Mistral OCR Settings](#mistral-ocr-settings)
- [OCR 3 Features](#ocr-3-features)
- [Document QnA](#document-qna)
- [Batch OCR Processing](#batch-ocr-processing)
- [File Upload Management](#file-upload-management)
- [Structured Data Extraction](#structured-data-extraction)
- [Image Processing](#image-processing)
- [Table Extraction](#table-extraction)
- [PDF to Image Conversion](#pdf-to-image-conversion)
- [System Paths](#system-paths-windows-only)
- [Caching](#caching)
- [Logging](#logging)
- [Performance](#performance)
- [API Retry Configuration](#api-retry-configuration)
- [Output Settings](#output-settings)
- [MarkItDown Settings](#markitdown-settings)
- [Complete Example Configuration](#complete-example-configuration)
- [System Status and Diagnostics](#system-status-and-diagnostics)
- [Configuration by Use Case](#configuration-by-use-case)
- [Environment Variable Reference](#environment-variable-reference)

## Configuration File

All settings are configured through a `.env` file in the project root directory.

### Creating Your Configuration

1. Create a file named `.env` in the project root
2. Add your settings (see options below)
3. Restart the converter for changes to take effect

**Example `.env` file:**

```ini
# Minimal configuration
MISTRAL_API_KEY="your_key_from_console.mistral.ai"

# Optional optimizations
CACHE_DURATION_HOURS=24
MAX_CONCURRENT_FILES=5
LOG_LEVEL=INFO
```

---

## API Keys

### MISTRAL_API_KEY (Required for OCR)

- **Type:** String
- **Default:** None (must be set)
- **Required for:** Convert (Smart) (`--mode smart`), Convert (Mistral OCR) (`--mode mistral_ocr`), Document QnA (`--mode qna`), Batch OCR (`--mode batch_ocr`); optional for MarkItDown LLM/plugin features when using Convert (MarkItDown) (`--mode markitdown`)
- **Get it from:** https://console.mistral.ai/api-keys/
- **Important:** A valid API key is enough for single-file OCR and Document QnA, but Batch OCR additionally requires Mistral AI Studio Scale / paid access.

```ini
MISTRAL_API_KEY="your_api_key_here"
```

### MISTRAL_SERVER_URL (Optional)

- **Type:** String (URL without trailing slash)
- **Default:** `""` (official Python SDK default host)
- **Description:** Override the Mistral API base URL for private or regional deployments. Passed to the SDK as `server_url`. When set, MarkItDown LLM image descriptions use `{MISTRAL_SERVER_URL}/v1` as the OpenAI-compatible base instead of `https://api.mistral.ai/v1`.

```ini
# MISTRAL_SERVER_URL="https://api.mistral.ai"
```

---

## Mistral OCR Settings

### MISTRAL_OCR_MODEL

- **Type:** String
- **Default:** `"mistral-ocr-latest"`
- **Options:** `mistral-ocr-latest` (recommended)
- **Description:** OCR model to use

```ini
MISTRAL_OCR_MODEL="mistral-ocr-latest"
```

### MISTRAL_INCLUDE_IMAGES

- **Type:** Boolean
- **Default:** `true`
- **Description:** Extract embedded images from documents
- **Output:** Images saved to `output_images/`

```ini
MISTRAL_INCLUDE_IMAGES=true
```

### SAVE_MISTRAL_JSON

- **Type:** Boolean
- **Default:** `true`
- **Description:** Save detailed OCR metadata for quality assessment
- **Output:** Creates `*_ocr_metadata.json` files

```ini
SAVE_MISTRAL_JSON=true
```

### OCR Quality Assessment Thresholds

Control when OCR results are considered usable based on quality scoring (0-100 scale).

#### OCR_QUALITY_THRESHOLD_EXCELLENT

- **Type:** Integer
- **Default:** `80`
- **Description:** Score above this is considered excellent quality

```ini
OCR_QUALITY_THRESHOLD_EXCELLENT=80
```

#### OCR_QUALITY_THRESHOLD_GOOD

- **Type:** Integer
- **Default:** `60`
- **Description:** Score above this is considered good quality

```ini
OCR_QUALITY_THRESHOLD_GOOD=60
```

#### OCR_QUALITY_THRESHOLD_ACCEPTABLE

- **Type:** Integer
- **Default:** `40`
- **Description:** Minimum score for acceptable quality

```ini
OCR_QUALITY_THRESHOLD_ACCEPTABLE=40
```

#### ENABLE_OCR_QUALITY_ASSESSMENT

- **Type:** Boolean
- **Default:** `true`
- **Description:** Enable/disable OCR quality scoring. Set to `false` to skip scoring entirely.

```ini
ENABLE_OCR_QUALITY_ASSESSMENT=true
```

#### ENABLE_OCR_WEAK_PAGE_IMPROVEMENT

- **Type:** Boolean
- **Default:** `true`
- **Description:** Enable/disable weak-page re-OCR. Requires `ENABLE_OCR_QUALITY_ASSESSMENT=true`.

```ini
ENABLE_OCR_WEAK_PAGE_IMPROVEMENT=true
```

### OCR Quality Detection Thresholds

Fine-tune the heuristics used to detect weak OCR pages.

#### OCR_MIN_TEXT_LENGTH

- **Type:** Integer
- **Default:** `50`
- **Description:** Minimum characters for valid page

```ini
OCR_MIN_TEXT_LENGTH=50
```

#### OCR_MIN_UNIQUENESS_RATIO

- **Type:** Float
- **Default:** `0.3`
- **Description:** Minimum unique token ratio (0-1)

```ini
OCR_MIN_UNIQUENESS_RATIO=0.3
```

#### OCR_MAX_PHRASE_REPETITIONS

- **Type:** Integer
- **Default:** `5`
- **Description:** Maximum repetitions before flagging as artifact

```ini
OCR_MAX_PHRASE_REPETITIONS=5
```

#### OCR_MIN_AVG_LINE_LENGTH

- **Type:** Integer
- **Default:** `10`
- **Description:** Minimum average line length

```ini
OCR_MIN_AVG_LINE_LENGTH=10
```

---

## OCR 3 Features

Advanced features available with the `mistral-ocr-2512` model.

### MISTRAL_TABLE_FORMAT

- **Type:** String
- **Default:** `"markdown"` (separate markdown table blocks)
- **Options:** `""` (omit parameter → API default), `"markdown"` (separate blocks), `"html"` (colspan/rowspan support)
- **Description:** Controls how tables are formatted in OCR output
- **Recommendation:** Use `"html"` for complex tables with merged cells

```ini
MISTRAL_TABLE_FORMAT=markdown
```

### MISTRAL_EXTRACT_HEADER

- **Type:** Boolean
- **Default:** `true`
- **Description:** Extract page headers separately from main content

```ini
MISTRAL_EXTRACT_HEADER=true
```

### MISTRAL_EXTRACT_FOOTER

- **Type:** Boolean
- **Default:** `true`
- **Description:** Extract page footers separately from main content

```ini
MISTRAL_EXTRACT_FOOTER=true
```

### MISTRAL_DOCUMENT_ANNOTATION_PROMPT

- **Type:** String
- **Default:** `""` (no custom prompt)
- **Description:** Custom guidance prompt for the document annotation LLM

```ini
MISTRAL_DOCUMENT_ANNOTATION_PROMPT=""
```

### MISTRAL_IMAGE_LIMIT

- **Type:** Integer
- **Default:** `0` (no limit)
- **Description:** Maximum number of images to extract from a document

```ini
MISTRAL_IMAGE_LIMIT=0
```

### MISTRAL_IMAGE_MIN_SIZE

- **Type:** Integer
- **Default:** `0` (no minimum)
- **Description:** Minimum pixel dimension for extracted images (smaller images are skipped)

```ini
MISTRAL_IMAGE_MIN_SIZE=0
```

### MISTRAL_OCR_MAX_FILE_SIZE_MB

- **Type:** Integer
- **Default:** `200`
- **Description:** Maximum file size (in MB) accepted for Mistral OCR uploads. Files exceeding this limit are rejected before upload to prevent unnecessary API charges.

```ini
MISTRAL_OCR_MAX_FILE_SIZE_MB=200
```

### MISTRAL_SIGNED_URL_EXPIRY

- **Type:** Integer
- **Default:** `1`
- **Description:** Signed URL expiry in hours. Increase for large batch jobs that take longer to process.

```ini
MISTRAL_SIGNED_URL_EXPIRY=1
```

### MISTRAL_CLIENT_TIMEOUT_MS

- **Type:** Integer (milliseconds)
- **Default:** `300000` (5 minutes)
- **Description:** Per-request HTTP timeout for the Mistral Python SDK. This is **not** the same as `RETRY_MAX_ELAPSED_TIME_MS`, which only caps total time spent in retry backoff; reusing the retry value for HTTP timeouts can cancel long-running OCR requests prematurely.

```ini
MISTRAL_CLIENT_TIMEOUT_MS=300000
```

---

## Document QnA

Query documents in natural language using Mistral's chat completion with document_url content type.

Important caveat: treat Document QnA as advisory only for exact-value extraction.
For dates, amounts, invoice numbers, IDs, and compliance-sensitive fields, prefer OCR markdown/metadata as the source of truth and cross-check QnA answers before trusting them.

### MISTRAL_DOCUMENT_QNA_MODEL

- **Type:** String
- **Default:** `"mistral-small-latest"`
- **Options:** Any Mistral chat model supporting document_url content type
- **Description:** Model for natural language document queries

```ini
MISTRAL_DOCUMENT_QNA_MODEL="mistral-small-latest"
```

### MISTRAL_QNA_SYSTEM_PROMPT

- **Type:** String
- **Default:** `""` (no system prompt)
- **Description:** Custom system prompt for Document QnA sessions

```ini
MISTRAL_QNA_SYSTEM_PROMPT=""
```

### MISTRAL_QNA_DOCUMENT_IMAGE_LIMIT

- **Type:** Integer
- **Default:** `0` (API default)
- **Description:** Maximum images from the document to include in QnA context

```ini
MISTRAL_QNA_DOCUMENT_IMAGE_LIMIT=0
```

### MISTRAL_QNA_DOCUMENT_PAGE_LIMIT

- **Type:** Integer
- **Default:** `0` (API default)
- **Description:** Maximum pages from the document to include in QnA context

```ini
MISTRAL_QNA_DOCUMENT_PAGE_LIMIT=0
```

**Note:** Documents are limited to 50 MB for QnA. Larger files will be rejected with a clear error message.

**CLI (non-interactive):** `--mode qna --no-interactive --qna-question "Your question?"` runs a single query without `input()` prompts.

**Usage:** Programmatically query documents:

```python
from mistral_converter import query_document
success, answer, error = query_document(
    "https://example.com/document.pdf",
    "What is the main topic of this document?"
)
```

**Streaming:** For real-time token-by-token output, use `query_document_stream()`:

```python
from mistral_converter import query_document_stream
success, stream, error = query_document_stream(
    "https://example.com/document.pdf",
    "Summarize this document."
)
if success:
    for chunk in stream:
        if chunk.data.choices and chunk.data.choices[0].delta.content:
            print(chunk.data.choices[0].delta.content, end="", flush=True)
```

---

## Batch OCR Processing

Process multiple documents at reduced cost using Mistral's Batch API.

Important: Batch OCR requires Mistral AI Studio Scale / paid access.
A valid API key alone is not enough. If batch submit returns free-trial / 402 messaging even for tiny jobs, check the workspace plan first and, after plan changes, consider creating a fresh API key.

### MISTRAL_BATCH_ENABLED

- **Type:** Boolean
- **Default:** `true`
- **Description:** Enable batch OCR processing

```ini
MISTRAL_BATCH_ENABLED=true
```

### MISTRAL_BATCH_MIN_FILES

- **Type:** Integer
- **Default:** `10`
- **Description:** Minimum files to recommend batch processing
- **Recommendation:** Batch is most cost-effective for 10+ files

```ini
MISTRAL_BATCH_MIN_FILES=10
```

### MISTRAL_BATCH_TIMEOUT_HOURS

- **Type:** Integer
- **Default:** `24`
- **Description:** Maximum hours to wait for a batch job to complete

```ini
MISTRAL_BATCH_TIMEOUT_HOURS=24
```

### MISTRAL_BATCH_STRICT

- **Type:** Boolean
- **Default:** `false`
- **Description:** When `true`, creating a batch JSONL file fails if any input file upload fails (default allows a partial batch with only successfully uploaded files).

```ini
MISTRAL_BATCH_STRICT=false
```

**CLI (non-interactive):** With `--mode batch_ocr --no-interactive`, use `--batch-action submit|status|list|download` and for status/download also `--batch-job-id <id>`. Status, list, and download do not require files in `input/`. Batch JSONL is written under a unique `cache/batch_ocr_*.jsonl` file per submit (signed URLs — keep `cache/` private; on Windows tighten directory ACLs if needed).

**Usage:** Programmatically batch process documents:

```python
from mistral_converter import create_batch_ocr_file, submit_batch_ocr_job
files = [Path("doc1.pdf"), Path("doc2.pdf"), ...]
success, batch_file, _ = create_batch_ocr_file(files, Path("batch.jsonl"))
success, job_id, _ = submit_batch_ocr_job(batch_file)
```

---

## File Upload Management

### CLEANUP_OLD_UPLOADS

- **Type:** Boolean
- **Default:** `true`
- **Description:** Automatically delete old uploads from Mistral API during maintenance
- **Benefit:** Prevents unnecessary storage costs

```ini
CLEANUP_OLD_UPLOADS=true
```

### CLEANUP_UPLOAD_SCOPE

- **Type:** String
- **Default:** `registry`
- **Options:** `registry`, `all`
- **Description:** `registry` deletes only file IDs recorded by this app (safe for shared API keys). `all` deletes every account OCR/batch file older than retention (destructive when keys are shared).

```ini
CLEANUP_UPLOAD_SCOPE=registry
```

### UPLOAD_RETENTION_DAYS

- **Type:** Integer
- **Default:** `7`
- **Description:** Days to keep uploaded files before deletion
- **Recommendation:** 7-14 days for most users

```ini
UPLOAD_RETENTION_DAYS=7
```

---

## Structured Data Extraction

**Important:** The Mistral OCR API expects `bbox_annotation_format` / `document_annotation_format` as **ResponseFormat** envelopes (`type: json_schema` wrapping the schema, name, and strict flag). The converter builds those from Pydantic models, the SDK helper when available, or manual wrapping — not “raw schema only” payloads.

### MISTRAL_ENABLE_STRUCTURED_OUTPUT

- **Type:** Boolean
- **Default:** `true`
- **Description:** Enable JSON schema-based extraction
- **Use for:** Contracts, invoices, financial statements, forms, and any structured document

```ini
MISTRAL_ENABLE_STRUCTURED_OUTPUT=true
```

### MISTRAL_DOCUMENT_SCHEMA_TYPE

- **Type:** String
- **Default:** `"auto"`
- **Options:** `invoice`, `financial_statement`, `contract`, `form`, `generic`, `auto`

```ini
MISTRAL_DOCUMENT_SCHEMA_TYPE="auto"
```

### MISTRAL_ENABLE_BBOX_ANNOTATION

- **Type:** Boolean
- **Default:** `false`
- **Description:** Extract bounding boxes for text regions, tables, figures
- **Note:** Adds processing time

```ini
MISTRAL_ENABLE_BBOX_ANNOTATION=false
```

### MISTRAL_ENABLE_DOCUMENT_ANNOTATION

- **Type:** Boolean
- **Default:** `false`
- **Description:** Extract document structure and metadata
- **Note:** Adds processing time

```ini
MISTRAL_ENABLE_DOCUMENT_ANNOTATION=false
```

---

## Image Processing

### MISTRAL_ENABLE_IMAGE_OPTIMIZATION

- **Type:** Boolean
- **Default:** `true`
- **Description:** Resize and compress images before OCR
- **Note:** Only applies to image files, NOT PDFs

```ini
MISTRAL_ENABLE_IMAGE_OPTIMIZATION=true
```

### MISTRAL_ENABLE_IMAGE_PREPROCESSING

- **Type:** Boolean
- **Default:** `false`
- **Description:** Enhance image contrast and sharpness
- **Use for:** Low-quality scans
- **Note:** Only applies to image files, NOT PDFs

```ini
MISTRAL_ENABLE_IMAGE_PREPROCESSING=false
```

### MISTRAL_MAX_IMAGE_DIMENSION

- **Type:** Integer
- **Default:** `2048`
- **Description:** Maximum width/height in pixels for image resize

```ini
MISTRAL_MAX_IMAGE_DIMENSION=2048
```

### MISTRAL_IMAGE_QUALITY_THRESHOLD

- **Type:** Integer (1-100)
- **Default:** `70`
- **Description:** JPEG quality for compression

```ini
MISTRAL_IMAGE_QUALITY_THRESHOLD=70
```

---

## PDF to Image Conversion

### PDF_IMAGE_FORMAT

- **Type:** String
- **Default:** `"png"`
- **Options:** `png`, `jpeg`, `tiff`, `ppm`
- **PNG:** Best quality, larger files
- **JPEG:** Smaller files, slight quality loss

```ini
PDF_IMAGE_FORMAT="png"
```

### PDF_IMAGE_DPI

- **Type:** Integer
- **Default:** `200`
- **Range:** 72-600
- **Recommendations:**
  - 150 - Screen viewing
  - 200 - General purpose
  - 300 - Print quality

```ini
PDF_IMAGE_DPI=200
```

### PDF_IMAGE_THREAD_COUNT

- **Type:** Integer
- **Default:** `4`
- **Range:** 1-16
- **Description:** Concurrent threads for PDF conversion

```ini
PDF_IMAGE_THREAD_COUNT=4
```

### PDF_IMAGE_USE_PDFTOCAIRO

- **Type:** Boolean
- **Default:** `true`
- **Description:** Use pdftocairo for better rendering quality

```ini
PDF_IMAGE_USE_PDFTOCAIRO=true
```

### PDF_IMAGE_MAX_PAGES

- **Type:** Integer
- **Default:** `100`
- **Description:** Cap pages rendered by pdf2image (`0` disables the cap)

```ini
PDF_IMAGE_MAX_PAGES=100
```

---

## System Paths (Windows Only)

### POPPLER_PATH

- **Type:** String
- **Default:** `""` (empty)
- **Required for:** PDF to image conversion on Windows
- **Download:** https://github.com/oschwartz10612/poppler-windows/releases

```ini
POPPLER_PATH="C:/Program Files/poppler-23.08.0/Library/bin"
```

---

## Caching

### CACHE_DURATION_HOURS

- **Type:** Integer
- **Default:** `24`
- **Description:** Hours to keep cached results
- **Benefit:** Second run = $0 API costs

```ini
CACHE_DURATION_HOURS=24
```

### AUTO_CLEAR_CACHE

- **Type:** Boolean
- **Default:** `true`
- **Description:** Automatically remove expired cache entries

```ini
AUTO_CLEAR_CACHE=true
```

---

## Logging

### LOG_LEVEL

- **Type:** String
- **Default:** `"INFO"`
- **Options:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Recommendation:** Use `DEBUG` for troubleshooting

```ini
LOG_LEVEL="INFO"
```

### SAVE_PROCESSING_LOGS

- **Type:** Boolean
- **Default:** `true`
- **Description:** Save detailed logs to `logs/` directory

```ini
SAVE_PROCESSING_LOGS=true
```

### VERBOSE_PROGRESS

- **Type:** Boolean
- **Default:** `true`
- **Description:** Show progress bars during processing

```ini
VERBOSE_PROGRESS=true
```

---

## Performance

### MAX_CONCURRENT_FILES

- **Type:** Integer
- **Default:** `5`
- **Range:** 1-20
- **Description:** Number of files to process concurrently when running multi-file conversion (e.g. `--mode smart`, `--mode mistral_ocr`, `--mode batch_ocr`)
- **Recommendation:** 3-5 for most systems, 10-15 for powerful systems

```ini
MAX_CONCURRENT_FILES=5
```

### MAX_BATCH_FILES

- **Type:** Integer
- **Default:** `100`
- **Description:** Maximum files per batch

```ini
MAX_BATCH_FILES=100
```

### MAX_PAGES_PER_SESSION

- **Type:** Integer
- **Default:** `1000`
- **Description:** Maximum OCR pages per session

```ini
MAX_PAGES_PER_SESSION=1000
```

---

## API Retry Configuration

### MAX_RETRIES

- **Type:** Integer
- **Default:** `3`
- **Description:** Set to `0` to disable retries. Any positive value enables SDK exponential backoff. This is **not** a max-attempt count; elapsed time is bounded by `RETRY_MAX_ELAPSED_TIME_MS`. See also `ENABLE_RETRIES`.

```ini
MAX_RETRIES=3
```

### RETRY_INITIAL_INTERVAL_MS

- **Type:** Integer
- **Default:** `1000`
- **Description:** Initial wait time (milliseconds) before first retry

```ini
RETRY_INITIAL_INTERVAL_MS=1000
```

### RETRY_MAX_INTERVAL_MS

- **Type:** Integer
- **Default:** `10000`
- **Description:** Maximum wait time between retries

```ini
RETRY_MAX_INTERVAL_MS=10000
```

### RETRY_EXPONENT

- **Type:** Float
- **Default:** `2.0`
- **Description:** Exponential backoff multiplier

```ini
RETRY_EXPONENT=2.0
```

### RETRY_MAX_ELAPSED_TIME_MS

- **Type:** Integer
- **Default:** `60000`
- **Description:** Maximum total time for all retries

```ini
RETRY_MAX_ELAPSED_TIME_MS=60000
```

### RETRY_CONNECTION_ERRORS

- **Type:** Boolean
- **Default:** `true`
- **Description:** Retry on network/connection errors

```ini
RETRY_CONNECTION_ERRORS=true
```

---

## Output Settings

### GENERATE_TXT_OUTPUT

- **Type:** Boolean
- **Default:** `true`
- **Description:** Create `.txt` files alongside `.md` files

```ini
GENERATE_TXT_OUTPUT=true
```

### INCLUDE_METADATA

- **Type:** Boolean
- **Default:** `true`
- **Description:** Include YAML frontmatter in markdown output

```ini
INCLUDE_METADATA=true
```

### TABLE_OUTPUT_FORMATS

- **Type:** Comma-separated string
- **Default:** `"markdown"` (when unset)
- **Options:** `markdown`, `csv` (combine as `markdown,csv`)
- **Disable sidecars:** set to an empty value (`TABLE_OUTPUT_FORMATS=""`); tables remain inline in the main `.md`

```ini
TABLE_OUTPUT_FORMATS="markdown"
```

### ENABLE_BATCH_METADATA

- **Type:** Boolean
- **Default:** `true`
- **Description:** When true, writes `logs/metadata/batch_job_<id>.json` after a successful batch OCR submit.

```ini
ENABLE_BATCH_METADATA=true
```

---

## MarkItDown Settings

### MARKITDOWN_ENABLE_BUILTINS

- **Type:** Boolean
- **Default:** `true`
- **Description:** Enable or disable MarkItDown built-in converters

```ini
MARKITDOWN_ENABLE_BUILTINS=true
```

### MARKITDOWN_KEEP_DATA_URIS

- **Type:** Boolean
- **Default:** `false`
- **Description:** Preserve base64-encoded images in output as data URIs

```ini
MARKITDOWN_KEEP_DATA_URIS=false
```

### MARKITDOWN_ENABLE_LLM_DESCRIPTIONS

- **Type:** Boolean
- **Default:** `false`
- **Description:** Use LLM for enhanced image descriptions via Mistral's OpenAI-compatible endpoint

```ini
MARKITDOWN_ENABLE_LLM_DESCRIPTIONS=false
```

### MARKITDOWN_LLM_MODEL

- **Type:** String
- **Default:** `"pixtral-large-latest"`
- **Description:** Vision model for image descriptions (requires LLM descriptions enabled)

```ini
MARKITDOWN_LLM_MODEL="pixtral-large-latest"
```

### MARKITDOWN_LLM_PROMPT

- **Type:** String
- **Default:** `""` (MarkItDown default)
- **Description:** Custom prompt for LLM image descriptions

```ini
MARKITDOWN_LLM_PROMPT=""
```

### MARKITDOWN_ENABLE_PLUGINS

- **Type:** Boolean
- **Default:** `false`
- **Description:** Enable audio/video transcription and OCR plugins
- **Required for:** Optional media plugins and `markitdown-ocr` when using Convert (MarkItDown) (`--mode markitdown`)
- **Requires:** Install `requirements-optional.txt`
- **Note:** The `markitdown-ocr` package (in `requirements-optional.txt`) provides LLM-powered OCR within the MarkItDown pipeline itself, separate from Mistral OCR.
- **Allowlist vs plugins:** `MARKITDOWN_SUPPORTED` in `config.py` lists extensions accepted in `--mode markitdown`. Some entries (for example `flac`, `rtf`) require optional MarkItDown plugins or extras from `requirements-optional.txt`. If conversion fails with plugins disabled, enable `MARKITDOWN_ENABLE_PLUGINS` and install the matching optional dependencies.

```ini
MARKITDOWN_ENABLE_PLUGINS=false
```

### MARKITDOWN_STYLE_MAP

- **Type:** String
- **Default:** `""` (no custom mapping)
- **Description:** DOCX style mapping for mammoth (e.g., `"p[style-name='Custom Heading'] => h2:fresh"`)

```ini
MARKITDOWN_STYLE_MAP=""
```

### MARKITDOWN_EXIFTOOL_PATH

- **Type:** String
- **Default:** `""` (auto-detect)
- **Description:** Path to ExifTool binary for EXIF metadata extraction

```ini
MARKITDOWN_EXIFTOOL_PATH=""
```

### MARKITDOWN_MAX_FILE_SIZE_MB

- **Type:** Integer
- **Default:** `100`
- **Description:** Maximum file size for MarkItDown processing

```ini
MARKITDOWN_MAX_FILE_SIZE_MB=100
```

### STRICT_INPUT_PATH_RESOLUTION

- **Type:** Boolean
- **Default:** `true`
- **Description:** When `true`, `validate_file()` rejects paths whose resolved location lies outside `input/`. Use this to block symlink escapes from a shared inbox. Set `false` for programmatic callers that intentionally pass arbitrary paths (tests do this via monkeypatch).

```ini
STRICT_INPUT_PATH_RESOLUTION=true
```

### CLEANUP_UPLOAD_ALL_CONFIRM

- **Type:** Boolean
- **Default:** `false`
- **Description:** When `CLEANUP_UPLOAD_SCOPE=all`, maintenance requires interactive confirmation or this flag set to `true` before deleting account-wide Files API objects.

```ini
CLEANUP_UPLOAD_ALL_CONFIRM=false
```

### ENABLE_RETRIES

- **Type:** Boolean
- **Default:** derived from `MAX_RETRIES != 0`
- **Description:** Explicit retry enable flag. `MAX_RETRIES` remains an on/off gate (not a max-attempt count); backoff is bounded by `RETRY_MAX_ELAPSED_TIME_MS`.

```ini
ENABLE_RETRIES=true
```

### Maintainer note: upgrading `mistralai` or `markitdown`

After bumping the `mistralai` or `markitdown` package version, compare OCR, Document QnA, and batch request bodies against upstream release notes and run the full test suite. Batch JSONL fields are kept in sync with synchronous OCR options in code; API changes may require updates in the `mistral_converter` package.

---

## Complete Example Configuration

```ini
# ============================================================================
# Minimal Production Configuration
# ============================================================================

# Required
MISTRAL_API_KEY="your_mistral_api_key_here"

# Recommended
CACHE_DURATION_HOURS=24
CLEANUP_OLD_UPLOADS=true
UPLOAD_RETENTION_DAYS=7
LOG_LEVEL=INFO

# ============================================================================
# Advanced Configuration (Optional)
# ============================================================================

# OCR 3 Features
MISTRAL_TABLE_FORMAT=markdown
MISTRAL_EXTRACT_HEADER=true
MISTRAL_EXTRACT_FOOTER=true

# PDF to Image
PDF_IMAGE_FORMAT=png
PDF_IMAGE_DPI=200
PDF_IMAGE_THREAD_COUNT=4

# Performance
MAX_CONCURRENT_FILES=5

# Output
GENERATE_TXT_OUTPUT=true
INCLUDE_METADATA=true
TABLE_OUTPUT_FORMATS=markdown,csv

# Image Processing (for image files only)
MISTRAL_ENABLE_IMAGE_OPTIMIZATION=true
MISTRAL_ENABLE_IMAGE_PREPROCESSING=false
MISTRAL_IMAGE_QUALITY_THRESHOLD=70

# Structured Extraction
MISTRAL_ENABLE_STRUCTURED_OUTPUT=true
MISTRAL_DOCUMENT_SCHEMA_TYPE=auto
MISTRAL_ENABLE_BBOX_ANNOTATION=false
MISTRAL_ENABLE_DOCUMENT_ANNOTATION=false

# Document QnA (natural language queries)
MISTRAL_DOCUMENT_QNA_MODEL=mistral-small-latest

# Batch OCR (reduced cost)
MISTRAL_BATCH_ENABLED=true
MISTRAL_BATCH_MIN_FILES=10
MISTRAL_BATCH_TIMEOUT_HOURS=24

# ============================================================================
# Windows-Specific Paths
# ============================================================================

# Only needed on Windows
POPPLER_PATH="C:/Program Files/poppler-23.08.0/Library/bin"
```

---

## System Status and Diagnostics

Run `python3 main.py --test` or `python3 main.py --mode status` to display the full system status, including:

- **Configuration** — Current settings for API key, OCR model, cache, concurrency
- **Optional Features** — Runtime availability of ffmpeg, pydub, youtube_transcript_api, and olefile
- **Cache Statistics** — Hit rate, entry count, total size
- **Output Statistics** — Markdown, text, and image file counts
- **Recommendations** — Actionable suggestions (e.g., set API key, clear cache)

Optional features are detected at runtime and do not require configuration -- the status report simply tells you what is available on your system.

---

## Configuration by Use Case

### For Text-Based PDFs (Fast & Free)

```ini
MISTRAL_API_KEY=""  # Leave empty for Convert (MarkItDown) only
LOG_LEVEL=INFO
```

Use **Convert (MarkItDown)** (`--mode markitdown`) — no API key required for local conversion.

### For Scanned Documents (OCR Required)

```ini
MISTRAL_API_KEY="your_key"
MISTRAL_INCLUDE_IMAGES=true
SAVE_MISTRAL_JSON=true
CACHE_DURATION_HOURS=72  # Longer cache for expensive OCR
```

### For Financial Documents (Maximum Table Quality)

```ini
MISTRAL_API_KEY="your_key"
MISTRAL_TABLE_FORMAT=html
MISTRAL_EXTRACT_HEADER=true
MISTRAL_EXTRACT_FOOTER=true
TABLE_OUTPUT_FORMATS=markdown,csv
```

Local PDF table extraction uses **pdfplumber** (not Camelot). Tune OCR/table output via Mistral OCR settings above.

### For Batch Processing (Performance Optimized)

```ini
MISTRAL_API_KEY="your_key"
MISTRAL_BATCH_ENABLED=true
MISTRAL_BATCH_MIN_FILES=10
MAX_CONCURRENT_FILES=10  # More parallel processing
CACHE_DURATION_HOURS=24
GENERATE_TXT_OUTPUT=false  # Skip txt to save time
```

### For Development/Debugging

```ini
MISTRAL_API_KEY="your_key"
LOG_LEVEL=DEBUG  # Detailed logging
SAVE_PROCESSING_LOGS=true
VERBOSE_PROGRESS=true
```

---

## Environment Variable Reference

| Variable                           | Type   | Default              | Required                                                              | Section           |
| ---------------------------------- | ------ | -------------------- | --------------------------------------------------------------------- | ----------------- |
| MISTRAL_API_KEY                    | string | -                    | Yes (for smart, mistral_ocr, qna, batch_ocr; optional for markitdown) | API Keys          |
| MISTRAL_SERVER_URL                 | string | ""                   | No                                                                    | API Keys          |
| STRICT_INPUT_PATH_RESOLUTION       | bool   | true                 | No                                                                    | Security          |
| MISTRAL_OCR_MODEL                  | string | mistral-ocr-latest   | No                                                                    | OCR               |
| MISTRAL_INCLUDE_IMAGES             | bool   | true                 | No                                                                    | OCR               |
| SAVE_MISTRAL_JSON                  | bool   | true                 | No                                                                    | OCR               |
| OCR_QUALITY_THRESHOLD_EXCELLENT    | int    | 80                   | No                                                                    | OCR Quality       |
| OCR_QUALITY_THRESHOLD_GOOD         | int    | 60                   | No                                                                    | OCR Quality       |
| OCR_QUALITY_THRESHOLD_ACCEPTABLE   | int    | 40                   | No                                                                    | OCR Quality       |
| ENABLE_OCR_QUALITY_ASSESSMENT      | bool   | true                 | No                                                                    | OCR Quality       |
| ENABLE_OCR_WEAK_PAGE_IMPROVEMENT   | bool   | true                 | No                                                                    | OCR Quality       |
| OCR_MIN_TEXT_LENGTH                | int    | 50                   | No                                                                    | OCR Quality       |
| OCR_MIN_UNIQUENESS_RATIO           | float  | 0.3                  | No                                                                    | OCR Quality       |
| OCR_MAX_PHRASE_REPETITIONS         | int    | 5                    | No                                                                    | OCR Quality       |
| OCR_MIN_AVG_LINE_LENGTH            | int    | 10                   | No                                                                    | OCR Quality       |
| MISTRAL_TABLE_FORMAT               | string | markdown             | No                                                                    | OCR 3             |
| MISTRAL_EXTRACT_HEADER             | bool   | true                 | No                                                                    | OCR 3             |
| MISTRAL_EXTRACT_FOOTER             | bool   | true                 | No                                                                    | OCR 3             |
| MISTRAL_DOCUMENT_ANNOTATION_PROMPT | string | ""                   | No                                                                    | OCR 3             |
| MISTRAL_IMAGE_LIMIT                | int    | 0                    | No                                                                    | OCR 3             |
| MISTRAL_IMAGE_MIN_SIZE             | int    | 0                    | No                                                                    | OCR 3             |
| MISTRAL_OCR_MAX_FILE_SIZE_MB       | int    | 200                  | No                                                                    | OCR               |
| MISTRAL_SIGNED_URL_EXPIRY          | int    | 1                    | No                                                                    | OCR 3             |
| MISTRAL_CLIENT_TIMEOUT_MS          | int    | 300000               | No                                                                    | Mistral API       |
| MISTRAL_DOCUMENT_QNA_MODEL         | string | mistral-small-latest | No                                                                    | Document QnA      |
| MISTRAL_QNA_SYSTEM_PROMPT          | string | ""                   | No                                                                    | Document QnA      |
| MISTRAL_QNA_DOCUMENT_IMAGE_LIMIT   | int    | 0                    | No                                                                    | Document QnA      |
| MISTRAL_QNA_DOCUMENT_PAGE_LIMIT    | int    | 0                    | No                                                                    | Document QnA      |
| MISTRAL_BATCH_ENABLED              | bool   | true                 | No                                                                    | Batch OCR         |
| MISTRAL_BATCH_MIN_FILES            | int    | 10                   | No                                                                    | Batch OCR         |
| MISTRAL_BATCH_TIMEOUT_HOURS        | int    | 24                   | No                                                                    | Batch OCR         |
| MISTRAL_BATCH_STRICT               | bool   | false                | No                                                                    | Batch OCR         |
| CLEANUP_OLD_UPLOADS                | bool   | true                 | No                                                                    | File Management   |
| CLEANUP_UPLOAD_SCOPE               | string | registry             | No                                                                    | File Management   |
| UPLOAD_RETENTION_DAYS              | int    | 7                    | No                                                                    | File Management   |
| MISTRAL_ENABLE_LLM_DOC_CLASSIFICATION | bool | false              | No                                                                    | Structured Data   |
| OCR_OFFICE_PAGE_ESTIMATE_DEFAULT   | int    | 32                   | No                                                                    | Cost Guardrails   |
| ALLOW_INSECURE_MISTRAL_SERVER      | bool   | false                | No                                                                    | Security          |
| SCHEMA_STRICT_UNKNOWN_TYPES        | bool   | false                | No                                                                    | Structured Data   |
| MISTRAL_ENABLE_STRUCTURED_OUTPUT   | bool   | true                 | No                                                                    | Structured Data   |
| MISTRAL_DOCUMENT_SCHEMA_TYPE       | string | auto                 | No                                                                    | Structured Data   |
| MISTRAL_ENABLE_BBOX_ANNOTATION     | bool   | false                | No                                                                    | Structured Data   |
| MISTRAL_ENABLE_DOCUMENT_ANNOTATION | bool   | false                | No                                                                    | Structured Data   |
| MISTRAL_ENABLE_IMAGE_OPTIMIZATION  | bool   | true                 | No                                                                    | Image Processing  |
| MISTRAL_ENABLE_IMAGE_PREPROCESSING | bool   | false                | No                                                                    | Image Processing  |
| MISTRAL_MAX_IMAGE_DIMENSION        | int    | 2048                 | No                                                                    | Image Processing  |
| MISTRAL_IMAGE_QUALITY_THRESHOLD    | int    | 70                   | No                                                                    | Image Processing  |
| PDF_IMAGE_FORMAT                   | string | png                  | No                                                                    | PDF to Image      |
| PDF_IMAGE_DPI                      | int    | 200                  | No                                                                    | PDF to Image      |
| PDF_IMAGE_MAX_PAGES                | int    | 100                  | No                                                                    | PDF to Image      |
| CLEANUP_UPLOAD_ALL_CONFIRM         | bool   | false                | No                                                                    | Maintenance       |
| ENABLE_RETRIES                     | bool   | derived              | No                                                                    | Retry             |
| PDF_IMAGE_THREAD_COUNT             | int    | 4                    | No                                                                    | PDF to Image      |
| PDF_IMAGE_USE_PDFTOCAIRO           | bool   | true                 | No                                                                    | PDF to Image      |
| POPPLER_PATH                       | string | ""                   | No                                                                    | System Paths      |
| CACHE_DURATION_HOURS               | int    | 24                   | No                                                                    | Caching           |
| AUTO_CLEAR_CACHE                   | bool   | true                 | No                                                                    | Caching           |
| LOG_LEVEL                          | string | INFO                 | No                                                                    | Logging           |
| SAVE_PROCESSING_LOGS               | bool   | true                 | No                                                                    | Logging           |
| VERBOSE_PROGRESS                   | bool   | true                 | No                                                                    | Logging           |
| MAX_CONCURRENT_FILES               | int    | 5                    | No                                                                    | Performance       |
| MAX_BATCH_FILES                    | int    | 100                  | No                                                                    | Performance       |
| MAX_PAGES_PER_SESSION              | int    | 1000                 | No                                                                    | Performance       |
| MAX_RETRIES                        | int    | 3                    | No                                                                    | Retry             |
| RETRY_INITIAL_INTERVAL_MS          | int    | 1000                 | No                                                                    | Retry             |
| RETRY_MAX_INTERVAL_MS              | int    | 10000                | No                                                                    | Retry             |
| RETRY_EXPONENT                     | float  | 2.0                  | No                                                                    | Retry             |
| RETRY_MAX_ELAPSED_TIME_MS          | int    | 60000                | No                                                                    | Retry             |
| RETRY_CONNECTION_ERRORS            | bool   | true                 | No                                                                    | Retry             |
| GENERATE_TXT_OUTPUT                | bool   | true                 | No                                                                    | Output            |
| INCLUDE_METADATA                   | bool   | true                 | No                                                                    | Output            |
| TABLE_OUTPUT_FORMATS               | string | markdown             | No                                                                    | Output            |
| ENABLE_BATCH_METADATA              | bool   | true                 | No                                                                    | Output            |
| MARKITDOWN_ENABLE_BUILTINS         | bool   | true                 | No                                                                    | MarkItDown        |
| MARKITDOWN_KEEP_DATA_URIS          | bool   | false                | No                                                                    | MarkItDown        |
| MARKITDOWN_ENABLE_LLM_DESCRIPTIONS | bool   | false                | No                                                                    | MarkItDown        |
| MARKITDOWN_LLM_MODEL               | string | pixtral-large-latest | No                                                                    | MarkItDown        |
| MARKITDOWN_LLM_PROMPT              | string | ""                   | No                                                                    | MarkItDown        |
| MARKITDOWN_ENABLE_PLUGINS          | bool   | false                | No                                                                    | MarkItDown        |
| MARKITDOWN_STYLE_MAP               | string | ""                   | No                                                                    | MarkItDown        |
| MARKITDOWN_EXIFTOOL_PATH           | string | ""                   | No                                                                    | MarkItDown        |
| MARKITDOWN_MAX_FILE_SIZE_MB        | int    | 100                  | No                                                                    | MarkItDown        |

See README.md for complete feature documentation.

---

**Last Updated:** 2026-03-25

**Version:** 3.0.0

**Related Documentation:**

- **[README.md](README.md)** - Complete feature documentation
- **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)** - Troubleshooting guide
