# Known Issues and Troubleshooting

## Known Issues

### Image preprocessing does not apply to PDFs

The `MISTRAL_ENABLE_IMAGE_PREPROCESSING`, `MISTRAL_ENABLE_IMAGE_OPTIMIZATION`, and `MISTRAL_MAX_IMAGE_DIMENSION` settings only affect standalone image files (PNG, JPG, etc.). PDFs are sent to Mistral OCR as complete documents via the Files API and bypass image preprocessing entirely.

**Workaround:** Convert PDF pages to images first (mode 4), preprocess the images, then OCR the images (mode 3).

---

### Text-based PDFs may receive low OCR quality scores

Mistral OCR works on all PDFs (scanned and text-based), but text-based PDFs sometimes score below 40 on the quality heuristics. This does not mean OCR failed -- it indicates that simpler extraction may produce better results.

**Recommendation:**

- Text-based PDFs: use Convert (MarkItDown) -- faster, free, often more accurate
- Scanned documents: use Convert (Mistral OCR) or Convert (Smart)
- When unsure: use Convert (Smart) which auto-routes by file type

---

### Windows requires manual Poppler and Ghostscript paths

On macOS/Linux these are auto-detected via PATH. On Windows you must configure them in `.env`:

```ini
POPPLER_PATH="C:/Program Files/poppler-23.08.0/Library/bin"
```

- **Poppler** is required for PDF to Images mode

Download links:

- Poppler: https://github.com/oschwartz10612/poppler-windows/releases

---

### MarkItDown on scanned/image inputs produces minimal output

When MarkItDown (mode 2) is used on scanned images or image-only PDFs, the conversion succeeds but the output contains little or no extracted text (typically just metadata frontmatter). A warning is now logged when this happens:

```
WARNING: Conversion of scan.png completed but no meaningful text was extracted.
For scanned or image-based content, consider using Mistral OCR mode.
```

**Recommendation:** Use Convert (Smart) or Convert (Mistral OCR) for scanned documents and images. Smart mode auto-routes image inputs to Mistral OCR.

---

### Audio/video transcription requires extra setup

MarkItDown plugins for audio/video are not installed by default:

1. `pip install -r requirements-optional.txt`
2. Install ffmpeg: `brew install ffmpeg` / `apt install ffmpeg` / [Windows builds](https://www.gyan.dev/ffmpeg/builds/)
3. Set `MARKITDOWN_ENABLE_PLUGINS=true` in `.env`

Run `python3 main.py --test` to check optional feature readiness (ffmpeg, pydub, etc.).

---

### Batch OCR may still fail with free-trial / 402 messaging until the workspace is on Scale

Batch OCR is gated differently from single-file OCR. It is possible for auth, single-file OCR, and Document QnA to work while Batch OCR still fails with Mistral free-trial / 402 wording.

**What to check:**

- Confirm the Mistral workspace is on AI Studio Scale / paid access
- If the plan was just changed, create a fresh API key and retry
- Treat this as a plan gate first, not a local batch-wrapper bug

---

### Batch job IDs are validated for safe characters

When supplying `--batch-job-id` for batch status or download in non-interactive mode, the ID must contain only alphanumeric characters, hyphens, and underscores, and be at most 128 characters. Invalid IDs are rejected with a descriptive error before processing begins.

---

### Document QnA can return plausible-but-wrong exact values

Document QnA is useful for summaries and exploratory follow-up questions, but it is not yet a safe source of truth for exact dates, amounts, invoice numbers, IDs, or compliance-sensitive fields.

**Recommendation:**

- Use OCR markdown/metadata as the source of truth for exact values
- Treat QnA as advisory only
- Cross-check exact-value answers before trusting them

---

### Document QnA URL validation is not a full SSRF barrier

HTTPS and optional DNS checks filter obvious private-host URLs before calling Mistral, but they cannot prevent all rebinding or server-side fetch attacks. Treat URL mode as convenience-only unless you also constrain egress or use local file upload for QnA.

---

### Mistral OCR size limits vs generic Files API

The Mistral Files API allows large uploads, but the **OCR** product enforces stricter document limits (see Mistral docs, e.g. on the order of tens of MB per document). Very large PDFs may fail at OCR time even when upload succeeds.

---

### Table header merging may skip ambiguous cases

The split-header repair heuristic (`_fix_split_headers`) intentionally skips merging when a standalone row already forms a plausible header (e.g., a single word that matches a known pattern). This conservative approach avoids false-positive merges but may leave some legitimately split headers unmerged in rare cases.

---

## Troubleshooting

### "MISTRAL_API_KEY not set"

1. Create a `.env` file in the project root
2. Get an API key from https://console.mistral.ai/api-keys/
3. Add `MISTRAL_API_KEY="your_key_here"`
4. Restart the converter

---

### "401 Unauthorized" or "403 Forbidden"

- Verify your API key at https://console.mistral.ai/
- Check that your plan includes OCR access
- Fallback: use Convert (MarkItDown) which is free and works offline

---

### "Mistral OCR returned empty text"

- The error message now includes parse error details when available -- read the full message.
- Verify your API key has OCR access
- Check that the document is valid and not corrupted
- Try Convert (MarkItDown) as an alternative
- Check `logs/` for detailed error messages

---

### "pdf2image: Unable to get page count" (Windows)

Poppler is not installed or its path is not configured. See the Windows paths section above.

---

### Low OCR quality scores

Quality score < 40 with many "weak pages":

- **Text-based PDFs:** use Convert (MarkItDown) instead
- **Scanned documents:** use Convert (Smart) for best results
- **Poor scans:** ensure source has good DPI and contrast

---

### Cache not working

1. Check `CACHE_DURATION_HOURS` in `.env` (default: 24)
2. Verify the `cache/` directory exists and is writable
3. Run System Status (interactive menu option 7, or `python3 main.py --mode status`) to see cache statistics
4. Set `AUTO_CLEAR_CACHE=true` to auto-expire old entries

---

## Reporting Issues

1. Check `logs/` for detailed error messages
2. Run System Status (interactive menu option 7, or `python3 main.py --mode status`) for diagnostics
3. Review `.env` against [CONFIGURATION.md](CONFIGURATION.md)
4. Open a GitHub issue with: OS, Python version, sanitized config, steps to reproduce, and error output
