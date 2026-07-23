# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 3.x     | Yes       |
| < 3.0   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities.
2. Email the maintainers or use [GitHub Security Advisories](https://github.com/jaim12005/Mistral_Markitdown/security/advisories/new) to report privately.
3. Include steps to reproduce, impact assessment, and any suggested fixes.
4. You will receive an acknowledgment within 48 hours and a detailed response within 7 days.

---

## Threat Model

### Trust Boundaries

| Boundary                        | Examples                                                                                                                                                                      |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Attacker-controlled inputs**  | Document files in `input/` or passed programmatically; document URLs for QnA; content returned by Mistral OCR/QnA; streams passed to `convert_stream_with_markitdown`.        |
| **Operator-controlled inputs**  | `.env` configuration, API keys, feature flags (plugins, structured output, image extraction), external binary paths (Poppler/Ghostscript), batch/job IDs, and file selection. |
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

- File paths are validated before processing (`utils.validate_file`): must exist, be a file, be non-empty, have an extension in the relevant allowlist, and meet a **mode-specific size cap** (e.g. MarkItDown vs Mistral OCR vs QnA vs smart-mode union cap).
- **MarkItDown path:** Files exceeding `MARKITDOWN_MAX_FILE_SIZE_MB` (default: 100 MB) are rejected.
- **Mistral OCR path:** Files exceeding `MISTRAL_OCR_MAX_FILE_SIZE_MB` (default: 200 MB) are rejected before upload.
- **Document QnA:** Files exceeding `MISTRAL_QNA_MAX_FILE_SIZE_MB` (default: 50 MB) are rejected.
- **PDF table extraction and PDF-to-images:** Skipped when the PDF exceeds `max(MARKITDOWN_MAX_FILE_SIZE_MB, MISTRAL_OCR_MAX_FILE_SIZE_MB)` (see `config.pdf_heavy_work_max_file_size_mb()` / `utils.pdf_exceeds_heavy_work_limit`) to avoid expensive local work on files that would fail size checks on the conversion path.

### File Upload Security

- Documents uploaded for OCR are sent to Mistral's servers via their Files API.
- Uploaded files are subject to Mistral's [data retention policy](https://mistral.ai/terms/).
- The application auto-deletes uploaded files after a configurable retention period (`UPLOAD_RETENTION_DAYS`, default: 7 days).
- By default cleanup is scoped to a local upload registry (`CLEANUP_UPLOAD_SCOPE=registry`) so shared API keys do not lose unrelated Files API objects. Set `CLEANUP_UPLOAD_SCOPE=all` only when this key is exclusive to this tool.
- Review Mistral's terms of service before processing sensitive or regulated documents.

### URL Validation (SSRF Protection)

All document URLs (for QnA and streaming) are validated before use:

- Only HTTPS URLs are accepted.
- URLs with embedded credentials are rejected.
- Private/internal network addresses are blocked (RFC 1918, link-local, loopback, cloud metadata endpoints including `169.254.169.254`).
- IPv6-mapped IPv4 addresses are checked for private ranges.
- DNS resolution is verified with a 5-second timeout to prevent DNS rebinding stalling.
- **`MISTRAL_DOCUMENT_URL_STRICT_DNS`:** Default `true` — user-supplied document URLs that fail local DNS resolution or time out are rejected (fail closed). Post-upload Mistral signed URLs (file QnA) relax this check so local DNS hiccups do not break the upload→query path. Set to `false` only if you need fail-open for unresolved public hostnames.
- **`ALLOW_INSECURE_MISTRAL_SERVER`:** Default `false`. Cleartext `http://` values for `MISTRAL_SERVER_URL` are rejected unless this is explicitly enabled (API keys must not travel in cleartext).

**Known limitation (TOCTOU):** The local DNS resolution check cannot fully prevent DNS rebinding attacks because Mistral's servers independently resolve the hostname when fetching the document. The local check remains valuable as a first-pass filter against obvious internal targets. For high-security deployments, restrict QnA to pre-uploaded files (via `query_document_file`) rather than arbitrary URLs.

### Batch Job ID Validation

Batch job IDs entered interactively are validated against `^[a-zA-Z0-9_\-]{1,128}$` to prevent injection.

### Output Filename Safety

`utils.safe_output_stem` derives output filenames to prevent path traversal and collisions. Files from outside the standard input directory receive a SHA-256-based hash suffix.

### Input Path Confinement

`STRICT_INPUT_PATH_RESOLUTION` defaults to `true`: `utils.validate_file` rejects paths (including symlink escapes) that resolve outside `input/`. Set it to `false` only for programmatic callers that intentionally pass arbitrary paths.

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
| `MAX_BATCH_FILES`              | 100     | Hard reject in smart, MarkItDown, OCR, PDF→images, and batch modes |
| `MAX_PAGES_PER_SESSION`        | 1000    | Hard reject (refuses further OCR after limit)   |
| `PDF_IMAGE_MAX_PAGES`          | 100     | Cap pages rendered by local pdf2image (`0` = unlimited) |
| `MAX_CONCURRENT_FILES`         | 5       | Thread pool cap for parallel processing         |
| `MISTRAL_BATCH_TIMEOUT_HOURS`  | 24      | Batch job auto-cancellation                     |
| `UPLOAD_RETENTION_DAYS`        | 7       | Auto-cleanup of uploaded files on Mistral       |

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

### Terminal Output

QnA answers and other untrusted text are sanitized to strip ANSI escape sequences and non-printable control characters before display in the terminal (`utils.sanitize_for_terminal`).

---

## Cache Security

- Cache entries are keyed by SHA-256 hash of file contents, making collisions impractical.
- Cache writes are atomic (write to temp file, then `os.replace`) to prevent partial/corrupt entries under concurrency.
- Cache reads validate the JSON schema: required keys (`timestamp`, `type`, `data`) must be present and the `type` must match. Corrupt or tampered entries are automatically removed.
- The in-memory hash memo is bounded (1000 entries) to prevent memory exhaustion in long-running processes.

**Recommendation:** Protect the `cache/` directory with filesystem permissions (mode `0o700` on POSIX, which is set automatically). A local attacker with write access to cache files could inject tampered OCR results, causing integrity failures in downstream processing.

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

Crafted inputs could trigger excessive API calls (e.g., documents with many weak pages triggering re-OCR). The session page limit (`MAX_PAGES_PER_SESSION`) and batch file limit (`MAX_BATCH_FILES`) mitigate this.

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

On POSIX systems, the application creates directories with mode `0o700` (owner-only). On Windows, administrators should configure NTFS ACLs to restrict access to:

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
- Monitor the [GitHub Security Advisories](https://github.com/jaim12005/Mistral_Markitdown/security) page for updates.
- Review the `validate_configuration()` output at startup for security warnings (plugins enabled, data URIs preserved, long signed URL expiry).
