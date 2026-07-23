"""Shared CLI helpers for listing, validating, and selecting input files.

Extracted from :mod:`main` so :mod:`modes.batch` (and tests) can reuse file
selection without a circular import on the CLI entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import config
import utils

logger = utils.logger


def list_input_files() -> List[Path]:
    """Return sorted list of files in the input directory (includes extensionless files).

    Dotfiles (e.g. ``.gitkeep``, ``.DS_Store``) are silently excluded to avoid
    spurious "File is empty" warnings during non-interactive runs.
    """
    return sorted(
        (f for f in config.INPUT_DIR.iterdir() if f.is_file() and not f.name.startswith(".")),
        key=lambda p: p.name.lower(),
    )


def filter_valid_files(files: List[Path], mode: Optional[str] = None) -> List[Path]:
    """Return only valid files, logging warnings for invalid ones.

    Args:
        files: List of file paths to validate.
        mode: Conversion mode (``"markitdown"``, ``"mistral_ocr"``, etc.).
              Passed through to ``utils.validate_file`` for per-mode extension checks.
    """
    valid_files: List[Path] = []
    for file_path in files:
        is_valid, error = utils.validate_file(file_path, mode=mode)
        if is_valid:
            valid_files.append(file_path)
        else:
            logger.warning(error)
    return valid_files


def select_files() -> List[Path]:  # pragma: no cover
    """Prompt user to select files from input directory."""
    input_files = list_input_files()

    out = utils.ui_print

    if not input_files:
        logger.warning("No files found in %s", config.INPUT_DIR)
        out(f"\nNo files found in '{config.INPUT_DIR}'")
        out("Please add files to the input directory and try again.\n")
        return []

    out(f"\nFound {len(input_files)} file(s) in input directory:\n")

    for i, file_path in enumerate(input_files, 1):
        try:
            file_size = file_path.stat().st_size / 1024
            size_str = f"({file_size:.1f} KB)"
        except OSError:
            size_str = "(size unavailable)"
        out(f"  {i}. {utils.sanitize_for_terminal(file_path.name)} {size_str}")

    out(f"\n  {len(input_files) + 1}. Process ALL files")
    out("  0. Cancel\n")

    while True:
        try:
            choice = input("Select file(s) to process (comma-separated or single number): ").strip()

            if choice == "0":
                return []

            if choice == str(len(input_files) + 1):
                return input_files

            indices = [int(c.strip()) for c in choice.split(",")]

            selected: List[Path] = []
            seen_idx = set()
            for idx in indices:
                if 1 <= idx <= len(input_files):
                    if idx not in seen_idx:
                        seen_idx.add(idx)
                        selected.append(input_files[idx - 1])
                else:
                    utils.ui_print(f"Invalid selection: {idx}")
                    selected = []
                    break

            if selected:
                return selected

        except (ValueError, IndexError):
            utils.ui_print("Invalid input. Please enter numbers separated by commas.\n")
        except (KeyboardInterrupt, EOFError):
            return []
