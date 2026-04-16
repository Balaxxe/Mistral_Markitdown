"""System Status and Maintenance CLI modes.

Split out of ``main.py`` to keep the entry point focused on argparse and
interactive menu wiring. The two handlers here are read-only from the user's
point of view (status) or run housekeeping (maintenance); neither produces
converter output.
"""

from __future__ import annotations

import shutil
from typing import Tuple

import config
import mistral_converter
import utils

logger = utils.logger


def mode_system_status() -> Tuple[bool, str]:
    """Display cache statistics and system info (read-only, no side effects)."""
    logger.info("SYSTEM STATUS MODE")

    out = utils.ui_print

    out("\n" + "=" * 60)
    out(f"  ENHANCED DOCUMENT CONVERTER v{config.VERSION} - SYSTEM STATUS")
    out("=" * 60 + "\n")

    out("Configuration:")
    out(f"  * Mistral API Key: {'Set' if config.MISTRAL_API_KEY else 'NOT SET'}")
    out(
        f"  * Mistral API base: {config.MISTRAL_SERVER_URL or 'default (api.mistral.ai)'}",
    )
    llm_status = f"Enabled ({config.MARKITDOWN_LLM_MODEL})" if config.MARKITDOWN_ENABLE_LLM_DESCRIPTIONS else "Disabled"
    out(f"  * LLM Descriptions: {llm_status}")
    out(f"  * Cache Duration: {config.CACHE_DURATION_HOURS} hours")
    out(f"  * Max Concurrent Files: {config.MAX_CONCURRENT_FILES}")
    out(f"  * Mistral OCR Model: {config.get_ocr_model()}")
    out(f"  * Table Format: {config.MISTRAL_TABLE_FORMAT or 'API default (unset)'}")
    out(f"  * Extract Headers/Footers: {config.MISTRAL_EXTRACT_HEADER}/{config.MISTRAL_EXTRACT_FOOTER}")
    out(f"  * ExifTool: {'Set' if config.MARKITDOWN_EXIFTOOL_PATH else 'Not configured'}")
    out(f"  * Style Map: {'Set' if config.MARKITDOWN_STYLE_MAP else 'Not configured'}")
    out()

    # Optional feature readiness
    out("Optional Features:")
    ffmpeg_path = shutil.which("ffmpeg")
    out(f"  * ffmpeg: {'Available' if ffmpeg_path else 'Not found (needed for audio conversion)'}")
    _optional_pkgs = [
        ("pydub", "audio conversion"),
        ("youtube_transcript_api", "YouTube transcripts"),
        ("olefile", "Outlook .msg conversion"),
    ]
    for pkg_name, purpose in _optional_pkgs:
        try:
            __import__(pkg_name)
            out(f"  * {pkg_name}: Available")
        except ImportError:
            out(f"  * {pkg_name}: Not installed (needed for {purpose})")
    out()

    cache_stats = utils.cache.get_statistics()
    out("Cache Statistics:")
    out(f"  Total Entries: {cache_stats['total_entries']}")
    out(f"  Total Size: {cache_stats['total_size_mb']:.2f} MB")
    out(f"  Cache Hits: {cache_stats['cache_hits']}")
    out(f"  Cache Misses: {cache_stats['cache_misses']}")
    out(f"  Hit Rate: {cache_stats['hit_rate']:.1f}%")
    out()

    out("Output Statistics:")
    md_files = list(config.OUTPUT_MD_DIR.glob("*.md"))
    txt_files = list(config.OUTPUT_TXT_DIR.glob("*.txt"))
    image_dirs = list(config.OUTPUT_IMAGES_DIR.glob("*"))
    out(f"  Markdown Files: {len(md_files)}")
    out(f"  Text Files: {len(txt_files)}")
    out(f"  Image Directories: {len(image_dirs)}")
    out()

    input_files = list(config.INPUT_DIR.glob("*.*"))
    out(f"Input Directory: {len([f for f in input_files if f.is_file()])} files ready")
    out()

    out("Bundled model reference (verify current IDs on https://docs.mistral.ai):")
    key_models = ["mistral-ocr-latest", "pixtral-large-latest", "ministral-8b-latest"]
    for model_id in key_models:
        if model_id in config.MISTRAL_MODELS:
            model_info = config.MISTRAL_MODELS[model_id]
            out(f"  * {model_info['name']}: {model_info['description']}")
    out()

    out("System Recommendations:")
    recommendations = []

    if not config.MISTRAL_API_KEY:
        recommendations.append("! Set MISTRAL_API_KEY to enable OCR features")

    if cache_stats["total_entries"] > 100:
        recommendations.append("* Consider running Maintenance (option 8) to clear old cache entries")

    if not recommendations:
        recommendations.append("  All systems operational")

    for rec in recommendations:
        out(f"  {rec}")

    out("\n" + "=" * 60 + "\n")

    return True, "System status displayed"


def mode_maintenance() -> Tuple[bool, str]:
    """Run maintenance tasks: clear expired cache entries and old uploaded files."""
    logger.info("MAINTENANCE MODE")

    out = utils.ui_print

    out("\n" + "=" * 60)
    out("  MAINTENANCE")
    out("=" * 60 + "\n")

    actions_taken = []

    # 1. Clear expired cache entries
    if config.AUTO_CLEAR_CACHE:
        cleared = utils.cache.clear_old_entries()
        if cleared > 0:
            msg = f"Cleared {cleared} expired cache entries"
            out(f"  ✓ {msg}")
            actions_taken.append(msg)
        else:
            out("  - No expired cache entries to clear")
    else:
        out("  - Cache auto-clear is disabled (AUTO_CLEAR_CACHE=false)")

    # 2. Clean up old uploaded files from Mistral
    if config.CLEANUP_OLD_UPLOADS and config.MISTRAL_API_KEY:
        try:
            client = mistral_converter.get_mistral_client()
            if client:
                deleted = mistral_converter.cleanup_uploaded_files(client)
                if deleted > 0:
                    msg = f"Cleaned up {deleted} old uploaded files from Mistral (>{config.UPLOAD_RETENTION_DAYS} days)"
                    out(f"  ✓ {msg}")
                    actions_taken.append(msg)
                else:
                    out("  - No old uploaded files to clean up")
            else:
                out("  - Mistral client not available")
        except Exception as e:
            logger.debug("Could not clean up uploads: %s", e)
            out(f"  ! Upload cleanup failed: {e}")
    elif not config.MISTRAL_API_KEY:
        out("  - Skipping upload cleanup (no API key)")
    else:
        out("  - Upload cleanup is disabled (CLEANUP_OLD_UPLOADS=false)")

    if not actions_taken:
        out("\n  No maintenance actions were needed.")

    out("\n" + "=" * 60 + "\n")

    summary = "; ".join(actions_taken) if actions_taken else "No actions needed"
    return True, f"Maintenance complete: {summary}"
