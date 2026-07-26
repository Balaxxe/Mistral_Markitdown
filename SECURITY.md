# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 3.x     | Yes       |
| < 3.0   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities.
2. Email the maintainers or use [GitHub Security Advisories](https://github.com/Balaxxe/Mistral_Markitdown/security/advisories/new) to report privately.
3. Include steps to reproduce, impact assessment, and any suggested fixes.
4. You will receive an acknowledgment within 48 hours and a detailed response within 7 days.

---

## Threat Model

### Trust Boundaries

| Boundary                        | Examples                                                                                                                                                                      |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Attacker-controlled inputs**  | Document files in `input/` or passed programmatically; document URLs for QnA; content returned by Mistral OCR/QnA; streams passed to `convert_stream_with_markitdown`.        |
| **Operator-controlled inputs**  | `.env` configuration, API keys, feature flags (plugins, structured output, image extraction), external binary paths (Poppler), batch/job IDs, and file selection.             |
| **Developer-controlled inputs** | Source code, tests, build scripts, dependency versions.                                                                                                                       |

### Assumptions

- Typical deployment is a **single-user CLI** on a workstation, not a multi-tenant service.
- OS file permissions protect `input/`, `output_*/`, `cache/`, and `.env`.
- Mistral APIs are trusted infrastructure, but data returned by OCR/QnA must be treated as untrusted.
- The tool does **not** sandbox untrusted file parsing. If used as a backend service, it should run in a container or restricted-user environment.

### Untrusted documents and parsers

Document conversion depends on MarkItDown, pdfplumber, Poppler (pdf2image), and optionally third-party MarkItDown plugins. **Hostile files are untrusted inputs** to that entire parsing stack.

- Prefer running conversions in a **sandbox** or dedicated OS user when the input is not fully trusted.
- Keep `MARKITDOWN_ENABLE_PLUGINS=false` unless you trust every installed plugin (see `config.validate_configuration()` and the `SECURITY:` warnings it emits).
- Keep `MARKITDOWN_KEEP_DATA_URIS=false` when generated Markdown may be **rendered in a browser** without sanitization; embedded data URIs widen the XSS review surface.
- Review `validate_configuration()` output in CI or at startup for plugin, data-URI, and signed-URL expiry guidance.

---

## Security Controls

### API Key Handling

- **Never** commit API keys to version control.
- Store your `MISTRAL_API_KEY` in a `.env` file (already in `.gitignore`).
- Use environment variables or a secrets manager in production/CI environments.
- The application loads keys via `python-dotenv` and never logs or prints them.

### File Input Validation

- File paths are validated before processing (`utils.validate_file`): must exist, be a regular file, be non-empty, have an extension in the relevant allowlist, and meet a **mode-specific size cap** (e.g. MarkItDown vs Mistral OCR vs QnA vs smart-mode union cap).
- The same check guards the library entry points, not just the CLI: `process_with_ocr`, `convert_with_mistral_ocr`, `query_document_file`, the Document QnA mode, and `upload_file_for_ocr` each validate the caller's path before any upload. On those Mistral-bound paths FIFOs and device files are rejected whatever `STRICT_INPUT_PATH_RESOLUTION` is set to, because the check requires a regular file. The local MarkItDown path (`convert_with_markitdown`) does not run this check itself and relies on CLI admission, so a library caller using the local API keeps that responsibility.
- **MarkItDown path:** Files exceeding `MARKITDOWN_MAX_FILE_SIZE_MB` (default: 100 MB) are rejected.
- **Mistral OCR path:** Files exceeding `MISTRAL_OCR_MAX_FILE_SIZE_MB` (default: 200 MB) are rejected before upload.
- **Document QnA:** Files exceeding `MISTRAL_QNA_MAX_FILE_SIZE_MB` (default: 50 MB) are rejected.
- **PDF table extraction and PDF-to-images:** In the CLI pipeline, skipped when the PDF exceeds `max(MARKITDOWN_MAX_FILE_SIZE_MB, MISTRAL_OCR_MAX_FILE_SIZE_MB)` (see `config.pdf_heavy_work_max_file_size_mb()` / `utils.pdf_exceeds_heavy_work_limit`) to avoid expensive local work on files that would fail size checks on the conversion path.
- **ZIP and EPUB:** Local conversion is disabled until MarkItDown archive traversal can enforce shared decompressed-byte, member-count, and nesting-depth budgets.
- **OOXML (`.docx`, `.pptx`, `.xlsx`):** Before MarkItDown, the ZIP central directory is checked for package shape, safe member paths, encryption, nested archives, at most 2,000 members, 64 MiB per member, 256 MiB aggregate declared size, and a maximum 100:1 compression ratio.

### File Upload Security

- Documents uploaded for OCR are sent to Mistral's servers via their Files API.
- Uploaded files are subject to Mistral's [data retention policy](https://mistral.ai/terms/).
- The application auto-deletes uploaded files after a configurable retention period (`UPLOAD_RETENTION_DAYS`, default: 7 days).
- By default cleanup is scoped to a local upload registry (`CLEANUP_UPLOAD_SCOPE=registry`) so shared API keys do not lose unrelated Files API objects. Set `CLEANUP_UPLOAD_SCOPE=all` only when this key is exclusive to this tool.
- Registry updates take an exclusive lock on `cache/mistral_upload_registry.lock` (mode `0o600`), so two processes sharing a checkout cannot drop each other's entries. Cleanup reads a snapshot, releases the lock while it deletes remote files, then removes only the ids it deleted, so uploads registered meanwhile survive. The lock wait is capped at 30 seconds on both POSIX and Windows; when it expires the update proceeds without cross-process protection rather than blocking. Platforms without `fcntl` or `msvcrt` log a debug line and run without the lock.
- Review Mistral's terms of service before processing sensitive or regulated documents.

### URL Validation (SSRF Protection)

All document URLs (for QnA and streaming) are validated before use:

- Only HTTPS URLs are accepted.
- URLs with embedded credentials are rejected.
- Private/internal network addresses are blocked (RFC 1918, link-local, loopback, cloud metadata endpoints including `169.254.169.254`).
- IPv6-mapped IPv4 addresses are checked for private ranges.
- Carrier-grade NAT/shared space (`100.64.0.0/10`) is treated as internal and blocked.
- DNS resolution is verified with a 5-second timeout to prevent DNS rebinding stalling.
- **`MISTRAL_DOCUMENT_URL_STRICT_DNS`:** Default `true` — user-supplied document URLs that fail local DNS resolution or time out are rejected (fail closed). Post-upload Mistral signed URLs (file QnA) relax this check so local DNS hiccups do not break the upload→query path. Set to `false` only if you need fail-open for unresolved public hostnames.
- **`ALLOW_INSECURE_MISTRAL_SERVER`:** Default `false`. Cleartext `http://` values for `MISTRAL_SERVER_URL` are rejected unless this is explicitly enabled (API keys must not travel in cleartext).
- **`MISTRAL_QNA_ALLOW_URL_WITH_CUSTOM_SERVER`:** Default `false`. Arbitrary URL QnA is disabled when `MISTRAL_SERVER_URL` is configured because that server performs the fetch from a network the client cannot police. Uploaded-file QnA remains available.

**Known limitation (server-side fetch):** The local DNS resolution check cannot fully prevent DNS rebinding or
redirect-to-private-network attacks because Mistral's servers independently resolve and fetch the document URL.
Mistral's public Document QnA documentation requires a public, API-accessible URL but does not document DNS
pinning, redirect-target revalidation, metadata-address filtering, or egress controls. The local check remains
valuable as a first-pass filter against obvious internal targets. For high-security deployments, restrict QnA to
pre-uploaded files (via `query_document_file`) rather than arbitrary URLs: that path takes a local file, validates
it, and never fetches a caller-supplied URL.

### Batch Job ID Validation

Batch job IDs entered interactively are validated against `^[a-zA-Z0-9_\-]{1,128}$` to prevent injection.

### Output Filename Safety

`utils.safe_output_stem` derives output filenames to prevent path traversal and collisions. Files from outside the standard input directory receive a SHA-256-based hash suffix.

### Input Path Confinement

`STRICT_INPUT_PATH_RESOLUTION` defaults to `true`, so `utils.validate_file` rejects paths (including symlink escapes) that resolve outside `input/`. The OCR, QnA, and upload entry points all run this check, so a library caller cannot pass an out-of-tree path straight to the API. `convert_with_mistral_ocr` validates before it looks in the cache, so a cached result cannot serve a path that would be refused. The check is not atomic: an attacker who can already write to `input/` may replace a validated file between the check and the upload, so treat write access to `input/` as trusted. Setting the flag to `false` is the deliberate opt-out for callers that support arbitrary paths, and should follow only after they enforce an equivalent trust boundary.

### Account-wide upload cleanup

`CLEANUP_UPLOAD_SCOPE=all` requires interactive confirmation or `CLEANUP_UPLOAD_ALL_CONFIRM=true` before maintenance deletes Files API objects account-wide.

### Repository / IDE configuration

If `.cursor/` (or similar IDE automation) is committed for team sharing, review changes like any other config: avoid machine-specific paths, and keep secrets in `.env` or other ignored files (for example `.env.mcp.local`).

---

## Resource Limits and Cost Guardrails

The following limits prevent runaway API spend and resource exhaustion:

| Setting                        | Default | Enforcement                                     |
| ------------------------------ | ------- | ----------------------------------------------- |
| `MARKITDOWN_MAX_FILE_SIZE_MB`  | 100     | Hard reject before local conversion             |
| `MISTRAL_OCR_MAX_FILE_SIZE_MB` | 200     | Hard reject before Mistral upload               |
| `MISTRAL_QNA_MAX_FILE_SIZE_MB` | 50      | Hard reject before Document QnA upload          |
| `MAX_BATCH_FILES`              | 100     | Positive-only hard reject in smart, MarkItDown, OCR, PDF→images, and batch modes |
| `MAX_PAGES_PER_SESSION`        | 1000    | Two uses of one number: a running process-wide budget for Mistral OCR pages and batch admission, and a per-document page ceiling for PDF rendering and table extraction, which never draw the budget down (`0`/negative values fall back to 1000) |
| `PDF_IMAGE_MAX_PAGES`          | 100     | Additional PDF-rendering cap; combined with the session page limit (`0` defers to the session limit) |
| `PDF_IMAGE_MAX_DPI`            | 600     | Ceiling on the render resolution, whoever supplies it (minimum 72, no upper bound); the DPI reaching the render call is clamped into `[72, PDF_IMAGE_MAX_DPI]` and logged. A `PDF_IMAGE_DPI` environment value outside the fixed range `[72, 600]` falls back to the default 200 instead |
| `PDF_IMAGE_MAX_PIXELS_PER_PAGE`| 178956970 | Largest page Poppler may rasterize, checked before the render call (`0` disables the check) |
| `MAX_CONCURRENT_FILES`         | 5       | Thread pool cap for parallel processing         |
| `MISTRAL_BATCH_TIMEOUT_HOURS`  | 24      | Batch job auto-cancellation                     |
| `UPLOAD_RETENTION_DAYS`        | 7       | Auto-cleanup of uploaded files on Mistral       |

Additional fixed safety ceilings bound OCR page text (10 MiB per page and 50 MiB aggregate), tables (256 per page,
4,096 replacements per page, 8 MiB content per table, and 10 MiB aggregate table content), and extracted images
(100 entries, including payload-less metadata; 10 MiB encoded and 7 MiB decoded per image; 50 MiB aggregate decoded
data). OCR hyperlinks are capped at 4,096 per page, headers/footers at 256 KiB each, bounding-box annotations at
4,096, and all structured OCR fields share a 10 MiB / 100,000-node response budget with depth and string-size limits.
Tables, response metadata, and usage info draw on that same shared budget, and a response `model` name is kept only
when it is text of at most 128 characters. These checks run before parser-owned copies are retained. Batch input is
capped at 1 GiB aggregate and result downloads at 512 MiB. Batch downloads require an unconsumed streaming SDK response; eager compatibility payloads
are rejected because they cannot be bounded before allocation. OCR text is revalidated after weak-page improvement,
and all limit violations fail before the affected output set is published.

---

## Output Safety

### Generated Markdown May Contain Untrusted Content

OCR output and QnA answers are derived from document content that may include:

- Arbitrary HTML tags or fragments
- Data URIs (`data:image/...`) when `MARKITDOWN_KEEP_DATA_URIS=true`
- JavaScript or event handlers embedded in HTML-like content

**If you render output Markdown in a web browser, you must sanitize it first** (e.g., using a library like [DOMPurify](https://github.com/cure53/DOMPurify) or [bleach](https://github.com/mozilla/bleach)). Failing to do so may result in XSS vulnerabilities.

### YAML Frontmatter

Metadata strings in YAML frontmatter are escaped via `json.dumps` to prevent injection of arbitrary YAML.

### Extracted Tables

CSV and Markdown table sidecars prefix formula-like cells with an apostrophe. Admission ignores leading whitespace
and byte-order marks and recognizes both ASCII and fullwidth equals signs, preventing common spreadsheet-formula
injection bypasses when sidecars are opened or copied into spreadsheet software.

### Terminal Output

QnA answers and operational log records are sanitized to strip ANSI escape sequences and non-printable control
characters before terminal display. QnA answer lines are visibly prefixed, carriage returns are rendered literally,
and embedded newlines in log values are escaped so one attacker-controlled value cannot forge additional log
records. Common signed-URL credential query values are redacted from terminal/log output and returned cloud-error
messages.

---

## Cache Security

- Cache entries are keyed by SHA-256 hash of file contents, making collisions impractical.
- Cache writes are atomic (write to temp file, then `os.replace`) to prevent partial/corrupt entries under concurrency.
- Cache reads validate the JSON schema: required keys (`timestamp`, `type`, `data`) must be present and the `type` must match. Corrupt or tampered entries are automatically removed.
- The in-memory hash memo is bounded (1000 entries) to prevent memory exhaustion in long-running processes.

**Recommendation:** Protect the `cache/` directory with filesystem permissions (mode `0o700` on POSIX, which every run enforces: new directories are created `0o700` and existing ones are re-tightened). A local attacker with write access to cache files could inject tampered OCR results, causing integrity failures in downstream processing.

---

## Signed URL Security

- Signed URLs are generated with a configurable TTL (`MISTRAL_SIGNED_URL_EXPIRY`, default: 1 hour).
- Anyone with a signed URL can access the corresponding document until the URL expires.
- Batch JSONL files contain signed URLs and are written with restrictive permissions (`0o600` on POSIX).
- **Do not share output directories** or batch JSONL files with untrusted parties.
- For batch jobs, the signed URL expiry is automatically extended to exceed the batch timeout to prevent mid-job expiration.

---

## LLM and AI-Specific Risks

### Prompt Injection

Documents processed via QnA may contain text that attempts to manipulate the LLM's behavior (prompt injection). Mitigations:

- A default system prompt instructs the model to answer only from document content and to ignore embedded instructions.
- Operators can override this via `MISTRAL_QNA_SYSTEM_PROMPT`, but should preserve the anti-injection guidance.
- Structured outputs use strict Pydantic/JSON schemas (`schemas.py`) to constrain the response format.

**QnA answers should not be trusted as authoritative.** Always verify critical information from LLM responses.

### Cost Abuse

Crafted inputs could trigger excessive API calls (e.g., documents with many weak pages triggering re-OCR). Every weak page is retried at most once, each retry reserves against the same process-wide page budget, and batch admission reserves its aggregate estimate against that budget. The batch reservation is charged when the job is submitted and returned on every failed submit; the CLI batch mode also returns it in a `finally` block. A library caller that abandons a created JSONL should call `discard_batch_page_reservation(path)`, and any credit left parked is drained the next time `reset_session_page_counter()` runs. Re-creating a batch file at the same path replaces the earlier reservation and releases it. `MAX_PAGES_PER_SESSION` and the positive-only `MAX_BATCH_FILES` cap bound the work.

---

## Deployment Guidance

### Single-User CLI (Default)

No additional hardening is required beyond standard workstation hygiene:

- Keep `.env` readable only by your user.
- Do not place untrusted files in `input/` without review.
- Run `pip-audit` periodically to check for vulnerable dependencies.

### Backend Service / Multi-Tenant Use

If you wrap this tool in a web service or process untrusted uploads:

1. **Run in a container** (Docker, Podman) or as a restricted OS user with no network access beyond Mistral's API.
2. **Apply resource limits** (CPU time, memory, disk quota) at the OS or container level to guard against decompression bombs and parser exploits.
3. **Add authentication and authorization** -- the tool itself has none.
4. **Sanitize all output** before serving to browsers.
5. **Disable plugins** (`MARKITDOWN_ENABLE_PLUGINS=false`) and data URI preservation (`MARKITDOWN_KEEP_DATA_URIS=false`) to minimize attack surface.
6. **Set tight signed URL expiry** (1 hour or less) and enable upload cleanup.
7. **Restrict QnA to file uploads** (`query_document_file`) rather than arbitrary URLs to eliminate SSRF risk entirely.

### Filesystem Permissions

On POSIX systems, every run creates each missing level of its directories with mode `0o700` (owner-only), including intermediate parents, and re-tightens a directory that already exists, so a checkout that ships placeholder directories still ends up owner-only. Ancestors that were already there, such as the project root, keep the mode their owner chose, and the `chmod` is best-effort so filesystems without POSIX mode bits still work. The processing log file is set to `0o600`. Output directories passed in by a caller keep whatever permissions they already have. On Windows, administrators should configure NTFS ACLs to restrict access to:

- `.env` -- API keys
- `cache/` -- OCR results
- `output_md/`, `output_txt/`, `output_images/` -- converted documents
- `logs/` -- operational metadata and file names

---

## Dependency Security

The CI pipeline runs `pip-audit` on every push and weekly. Bandit (static analysis for Python security issues) runs as a blocking CI check and as a pre-commit hook.

Run periodic dependency audits locally:

```bash
# Using pip-audit
pip install pip-audit
pip-audit

# Using safety
pip install safety
safety check
```

### Parser Libraries

The following libraries process untrusted file content and are potential vectors for exploitation:

- **MarkItDown** (Office, PDF, HTML, archives, audio)
- **pdfplumber** (PDF table extraction)
- **pdf2image** + **Poppler** (PDF rendering)
- **Pillow** (image processing)

Keep these libraries up to date. In containerized deployments, use read-only filesystem mounts for binaries and restrict system calls with seccomp or AppArmor.

---

## Best Practices

- Keep dependencies up to date (`pip install --upgrade -r requirements.txt`).
- Run `make check` before deploying changes (includes linting and tests).
- Use the principle of least privilege for API keys -- only grant OCR access if that is all you need.
- Monitor the [GitHub Security Advisories](https://github.com/Balaxxe/Mistral_Markitdown/security) page for updates.
- Review the `validate_configuration()` output at startup for security warnings (plugins enabled, data URIs preserved, long signed URL expiry).
