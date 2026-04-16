"""CLI mode handlers for the Enhanced Document Converter.

Each module in this package implements one or more of the top-level
``mode_*`` functions that ``main.py`` wires up to the interactive menu and
the ``--mode`` CLI flag. Keeping them here lets the entry point stay focused
on argparse and menu orchestration.

Public re-exports match the legacy ``main.mode_*`` names so tests and
external callers (e.g. ``from main import mode_document_qna``) keep working
without change.
"""

from modes.batch import mode_batch_ocr
from modes.qna import mode_document_qna
from modes.system import mode_maintenance, mode_system_status

__all__ = [
    "mode_batch_ocr",
    "mode_document_qna",
    "mode_maintenance",
    "mode_system_status",
]
