"""Integration boundaries added for the security review.

These tests deliberately keep the package/module wiring real.  They replace
only the external MarkItDown engine or Mistral service boundary.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import config
import local_converter
import main
import mistral_converter
import mistral_converter.ocr as ocr_module


def _reset_page_budget():
    """Return all test-owned reservations before resetting shared state."""
    mistral_converter._release_session_pages_reservation(mistral_converter._session_pages_inflight)
    mistral_converter.reset_session_page_counter()


class TestMarkItDownStdinBoundary:
    def test_stdin_uses_real_stream_converter_and_writes_frontmatter(self, tmp_path, monkeypatch):
        """Exercise main -> local converter, stubbing just MarkItDown itself."""
        monkeypatch.setattr(config, "OUTPUT_MD_DIR", tmp_path / "markdown")
        monkeypatch.setattr(config, "OUTPUT_TXT_DIR", tmp_path / "text")
        monkeypatch.setattr(config, "GENERATE_TXT_OUTPUT", False)
        config.OUTPUT_MD_DIR.mkdir()

        engine = MagicMock()
        engine.convert_stream.side_effect = lambda stream, **_kwargs: SimpleNamespace(
            markdown=stream.read().decode("utf-8")
        )

        with patch.object(local_converter, "get_markitdown_instance", return_value=engine):
            ok, message = main.mode_markitdown_stdin(b"stdin integration body", "note.txt")

        assert ok is True
        output = config.OUTPUT_MD_DIR / "note.md"
        assert output.exists()
        rendered = output.read_text(encoding="utf-8")
        assert 'source: "stdin"' in rendered
        assert "stdin integration body" in rendered
        assert str(output) in message
        engine.convert_stream.assert_called_once()


class TestSharedBatchAndSyncPageBudget:
    def test_batch_commit_consumes_budget_seen_by_sync_public_api(self, tmp_path, monkeypatch):
        """A batch created via the facade leaves only its remaining budget to OCR."""
        _reset_page_budget()
        monkeypatch.setattr(config, "MAX_PAGES_PER_SESSION", 2)
        monkeypatch.setattr(config, "MAX_BATCH_FILES", 5)
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "test-key")
        monkeypatch.setattr(config, "MISTRAL_BATCH_STRICT", True)

        batch_input = tmp_path / "batch.png"
        sync_input = tmp_path / "sync.png"
        batch_input.write_bytes(b"not-decoded-by-this-boundary-test")
        sync_input.write_bytes(b"not-decoded-by-this-boundary-test")
        client = MagicMock()
        client.ocr.process.return_value = {"pages": [{"markdown": "sync text"}]}

        def estimate(path, pages=None):
            assert pages is None
            return 1

        with (
            patch.object(mistral_converter, "_estimate_session_pages_for_ocr", side_effect=estimate),
            patch.object(mistral_converter, "get_mistral_client", return_value=client),
            patch.object(mistral_converter, "_upload_file_for_ocr_pair", return_value=("https://signed", "ocr-file")),
            patch.object(mistral_converter, "build_ocr_process_kwargs", return_value={"document": {}}),
            patch.object(ocr_module, "_prepare_ocr_document", return_value=({"type": "image_url"}, "https://signed")),
        ):
            batch_ok, _path, batch_error = mistral_converter.create_batch_ocr_file(
                [batch_input], tmp_path / "batch.jsonl"
            )
            sync_ok, _result, sync_error = mistral_converter.process_with_ocr(client, sync_input)
            denied_ok, _result, denied_error = mistral_converter.process_with_ocr(client, sync_input)

        assert (batch_ok, batch_error) == (True, None)
        assert (sync_ok, sync_error) == (True, None)
        assert denied_ok is False
        assert "limit" in (denied_error or "").lower()
        _reset_page_budget()

    def test_weak_page_reocr_cannot_spend_after_the_original_session_budget(self, tmp_path, monkeypatch):
        """Weak-page retries share the same reservation gate as the first OCR pass."""
        _reset_page_budget()
        monkeypatch.setattr(config, "MAX_PAGES_PER_SESSION", 2)
        monkeypatch.setattr(config, "OCR_MAX_WEAK_PAGE_WORKERS", 2)
        source = tmp_path / "document.pdf"
        source.write_bytes(b"%PDF")
        client = MagicMock()
        original = {"pages": [{"text": "short"}, {"text": "short"}]}

        assert mistral_converter._reserve_session_pages(2) is True
        mistral_converter._commit_session_pages(2, 2)
        assert mistral_converter._session_pages_processed == 2
        with patch.object(mistral_converter, "upload_file_for_ocr", return_value="https://signed"):
            result = mistral_converter.improve_weak_pages(client, source, original, "model")

        assert [page["text"] for page in result["pages"]] == ["short", "short"]
        client.ocr.process.assert_not_called()
        _reset_page_budget()


class TestConcurrentBoundaries:
    def test_reservations_are_atomic_when_requests_start_together(self, monkeypatch):
        _reset_page_budget()
        monkeypatch.setattr(config, "MAX_PAGES_PER_SESSION", 1)
        start = threading.Barrier(2)
        results = []

        def reserve():
            start.wait(timeout=2)
            results.append(mistral_converter._reserve_session_pages(1))

        threads = [threading.Thread(target=reserve) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        assert not any(thread.is_alive() for thread in threads)
        assert sorted(results) == [False, True]
        _reset_page_budget()

    def test_client_singleton_constructor_is_called_once_under_real_contention(self, monkeypatch):
        mistral_converter.reset_mistral_client()
        monkeypatch.setattr(config, "MISTRAL_API_KEY", "test-key")
        start = threading.Barrier(2)
        constructor_entered = threading.Event()
        release_constructor = threading.Event()
        calls = []
        client = MagicMock()
        results = [None, None]

        def constructor(**_kwargs):
            calls.append(1)
            constructor_entered.set()
            assert release_constructor.wait(timeout=2)
            return client

        def get_client(index):
            start.wait(timeout=2)
            results[index] = mistral_converter.get_mistral_client()

        with (
            patch.object(mistral_converter, "Mistral", side_effect=constructor),
            patch.object(mistral_converter, "get_retry_config", return_value=None),
        ):
            threads = [threading.Thread(target=get_client, args=(index,)) for index in range(2)]
            for thread in threads:
                thread.start()
            assert constructor_entered.wait(timeout=2)
            release_constructor.set()
            for thread in threads:
                thread.join(timeout=2)

        assert not any(thread.is_alive() for thread in threads)
        assert calls == [1]
        assert results == [client, client]
        mistral_converter.reset_mistral_client()
