"""Tests for Mistral Batch OCR operations (submit/status/download/list).
Split out of test_mistral_converter.py for navigability."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

import config

# Initialize config dirs so imports work
config.ensure_directories()

import mistral_converter

# ============================================================================
# _validate_document_url Tests
# ============================================================================


class TestGetBatchJobStatus:
    """Test batch job status retrieval."""

    def test_successful_status(self):
        mock_job = MagicMock()
        mock_job.status = "RUNNING"
        mock_job.total_requests = 10
        mock_job.succeeded_requests = 5
        mock_job.failed_requests = 1
        mock_job.output_file = None
        mock_job.error_file = None

        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.batch.jobs.get.return_value = mock_job
            mock_get.return_value = mock_client

            ok, status, err = mistral_converter.get_batch_job_status("job_123")

        assert ok is True
        assert status["status"] == "RUNNING"
        assert status["progress_percent"] == 60.0

    def test_no_client(self):
        with patch.object(mistral_converter, "get_mistral_client", return_value=None):
            ok, status, err = mistral_converter.get_batch_job_status("job_123")
        assert ok is False

    def test_api_error(self):
        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.batch.jobs.get.side_effect = Exception("not found")
            mock_get.return_value = mock_client

            ok, status, err = mistral_converter.get_batch_job_status("job_123")
        assert ok is False


# ============================================================================
# download_batch_results Tests
# ============================================================================


class TestDownloadBatchResults:
    """Test batch result downloading."""

    def test_successful_download(self, tmp_path):
        mock_job = MagicMock()
        mock_job.status = "SUCCESS"
        mock_job.output_file = "output_file_id"

        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.batch.jobs.get.return_value = mock_job
            mock_client.files.download.return_value = b'{"result": "data"}\n'
            mock_get.return_value = mock_client

            ok, path, err = mistral_converter.download_batch_results("job_ok", output_dir=tmp_path)

        assert ok is True
        assert path.exists()

    def test_successful_download_httpx_response(self, tmp_path):
        mock_job = MagicMock()
        mock_job.status = "SUCCESS"
        mock_job.output_file = "output_file_id"

        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.batch.jobs.get.return_value = mock_job
            mock_client.files.download.return_value = httpx.Response(
                200,
                content=b'{"result": "streamed"}\n',
                headers={"content-type": "application/octet-stream"},
            )
            mock_get.return_value = mock_client

            ok, path, err = mistral_converter.download_batch_results("job_ok_httpx", output_dir=tmp_path)

        assert ok is True
        assert path.exists()
        assert path.read_text() == '{"result": "streamed"}\n'

    def test_job_not_complete(self):
        mock_job = MagicMock()
        mock_job.status = "RUNNING"

        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.batch.jobs.get.return_value = mock_job
            mock_get.return_value = mock_client

            ok, path, err = mistral_converter.download_batch_results("job_run")
        assert ok is False
        assert "not complete" in err.lower() or "RUNNING" in err

    def test_no_output_file(self):
        mock_job = MagicMock()
        mock_job.status = "SUCCESS"
        mock_job.output_file = None

        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.batch.jobs.get.return_value = mock_job
            mock_get.return_value = mock_client

            ok, path, err = mistral_converter.download_batch_results("job_no_out")
        assert ok is False


# ============================================================================
# list_batch_jobs Tests
# ============================================================================


class TestListBatchJobs:
    """Test batch jobs listing."""

    def test_successful_listing(self):
        mock_job = MagicMock()
        mock_job.id = "job_1"
        mock_job.status = "SUCCESS"
        mock_job.model = "mistral-ocr-latest"
        mock_job.total_requests = 5
        mock_job.succeeded_requests = 5
        mock_job.failed_requests = 0
        mock_job.created_at = "2024-01-01T00:00:00Z"

        jobs_response = MagicMock()
        jobs_response.data = [mock_job]

        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.batch.jobs.list.return_value = jobs_response
            mock_get.return_value = mock_client

            ok, jobs, err = mistral_converter.list_batch_jobs()

        assert ok is True
        assert len(jobs) == 1
        assert jobs[0]["id"] == "job_1"

    def test_filter_by_status(self):
        job1 = MagicMock()
        job1.id = "j1"
        job1.status = "SUCCESS"
        job1.model = "m"
        job1.total_requests = 1
        job1.succeeded_requests = 1
        job1.failed_requests = 0
        job1.created_at = ""

        job2 = MagicMock()
        job2.id = "j2"
        job2.status = "FAILED"
        job2.model = "m"
        job2.total_requests = 1
        job2.succeeded_requests = 0
        job2.failed_requests = 1
        job2.created_at = ""

        jobs_response = MagicMock()
        jobs_response.data = [job1, job2]

        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.batch.jobs.list.return_value = jobs_response
            mock_get.return_value = mock_client

            ok, jobs, err = mistral_converter.list_batch_jobs(status="SUCCESS")

        assert ok is True
        assert len(jobs) == 1
        assert jobs[0]["status"] == "SUCCESS"

    def test_no_client(self):
        with patch.object(mistral_converter, "get_mistral_client", return_value=None):
            ok, jobs, err = mistral_converter.list_batch_jobs()
        assert ok is False


# ============================================================================
# convert_with_mistral_ocr Tests
# ============================================================================


class TestSubmitBatchOcrJob:
    """Test batch job submission."""

    def test_no_client(self, tmp_path):
        with patch.object(mistral_converter, "get_mistral_client", return_value=None):
            ok, job_id, err = mistral_converter.submit_batch_ocr_job(tmp_path / "batch.jsonl")
        assert ok is False

    def test_successful_submission(self, tmp_path):
        batch_file = tmp_path / "batch.jsonl"
        batch_file.write_text('{"body": {}}\n')

        mock_upload = MagicMock(id="batch_file_id")
        mock_job = MagicMock(id="job_abc")

        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.files.upload.return_value = mock_upload
            mock_client.batch.jobs.create.return_value = mock_job
            mock_get.return_value = mock_client

            with patch.object(config, "MISTRAL_BATCH_TIMEOUT_HOURS", 48):
                ok, job_id, err = mistral_converter.submit_batch_ocr_job(batch_file)

        assert ok is True
        assert job_id == "job_abc"

    def test_api_error(self, tmp_path):
        batch_file = tmp_path / "batch.jsonl"
        batch_file.write_text('{"body": {}}\n')

        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.files.upload.side_effect = Exception("upload error")
            mock_get.return_value = mock_client

            ok, job_id, err = mistral_converter.submit_batch_ocr_job(batch_file)
        assert ok is False

    def test_deletes_remote_upload_and_local_file_when_job_create_fails(self, tmp_path):
        batch_file = tmp_path / "batch.jsonl"
        batch_file.write_text('{"body": {}}\n')

        mock_upload = MagicMock(id="orphan_upload_id")

        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.files.upload.return_value = mock_upload
            mock_client.batch.jobs.create.side_effect = RuntimeError("batch create failed")
            mock_get.return_value = mock_client

            ok, job_id, err = mistral_converter.submit_batch_ocr_job(batch_file)

        assert ok is False
        assert job_id is None
        mock_client.files.delete.assert_called_once_with(file_id="orphan_upload_id")
        assert not batch_file.exists()


# ============================================================================
# _process_ocr_result_pipeline Tests
# ============================================================================


class TestCreateBatchOcrFileFull:
    """Test batch file creation with all paths."""

    def test_successful_creation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)
        monkeypatch.setattr(config, "MISTRAL_BATCH_TIMEOUT_HOURS", 24)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")

        pdf1 = tmp_path / "doc1.pdf"
        pdf1.write_bytes(b"%PDF")
        pdf2 = tmp_path / "doc2.pdf"
        pdf2.write_bytes(b"%PDF")
        output = tmp_path / "batch.jsonl"

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch.object(
                mistral_converter,
                "_upload_file_for_ocr_pair",
                side_effect=[
                    ("https://signed.url", "id-a"),
                    ("https://signed.url", "id-b"),
                ],
            ):
                with patch.object(
                    mistral_converter,
                    "get_bbox_annotation_format",
                    return_value=None,
                ):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        ok, path, err = mistral_converter.create_batch_ocr_file([pdf1, pdf2], output)

        assert ok is True
        assert path == output
        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2
        import json

        row0 = json.loads(lines[0])
        assert row0["body"]["id"] == row0["custom_id"]
        assert row0["body"]["extract_header"] == config.MISTRAL_EXTRACT_HEADER
        assert row0["body"]["extract_footer"] == config.MISTRAL_EXTRACT_FOOTER
        assert row0["body"]["table_format"] == config.MISTRAL_TABLE_FORMAT
        assert row0["body"]["document"]["document_name"] == "doc1.pdf"

    def test_with_image_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", True)
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)
        monkeypatch.setattr(config, "MISTRAL_BATCH_TIMEOUT_HOURS", 24)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "Extract all")

        img = tmp_path / "image.png"
        img.write_bytes(b"fake png")
        output = tmp_path / "batch.jsonl"

        mock_bbox = {"type": "json_schema", "json_schema": {}}
        mock_doc = {"type": "json_schema", "json_schema": {}}

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch.object(
                mistral_converter,
                "_upload_file_for_ocr_pair",
                return_value=("https://signed.url", "id-img"),
            ):
                with patch.object(
                    mistral_converter,
                    "get_bbox_annotation_format",
                    return_value=mock_bbox,
                ):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=mock_doc,
                    ):
                        ok, path, err = mistral_converter.create_batch_ocr_file([img], output)

        assert ok is True
        import json

        entry = json.loads(output.read_text().strip())
        assert entry["body"]["document"]["type"] == "image_url"
        assert "include_image_base64" in entry["body"]

    def test_upload_failure_skips_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)
        monkeypatch.setattr(config, "MISTRAL_BATCH_TIMEOUT_HOURS", 24)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        output = tmp_path / "batch.jsonl"

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch.object(
                mistral_converter,
                "_upload_file_for_ocr_pair",
                return_value=None,
            ):
                with patch.object(
                    mistral_converter,
                    "get_bbox_annotation_format",
                    return_value=None,
                ):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        ok, path, err = mistral_converter.create_batch_ocr_file([pdf], output)

        assert ok is False
        assert "no files" in err.lower()

    def test_no_client(self, tmp_path):
        with patch.object(mistral_converter, "get_mistral_client", return_value=None):
            ok, path, err = mistral_converter.create_batch_ocr_file([tmp_path / "doc.pdf"], tmp_path / "batch.jsonl")
        assert ok is False

    def test_exception(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png"})
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)
        monkeypatch.setattr(config, "MISTRAL_BATCH_TIMEOUT_HOURS", 24)

        with patch.object(mistral_converter, "get_mistral_client", return_value=MagicMock()):
            with patch.object(
                mistral_converter,
                "_upload_file_for_ocr_pair",
                side_effect=Exception("boom"),
            ):
                ok, path, err = mistral_converter.create_batch_ocr_file(
                    [tmp_path / "doc.pdf"], tmp_path / "batch.jsonl"
                )
        assert ok is False

    def test_strict_mode_deletes_partial_uploads(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MISTRAL_INCLUDE_IMAGES", False)
        monkeypatch.setattr(config, "IMAGE_EXTENSIONS", {"png", "jpg", "jpeg"})
        monkeypatch.setattr(config, "MISTRAL_SIGNED_URL_EXPIRY", 24)
        monkeypatch.setattr(config, "MISTRAL_BATCH_TIMEOUT_HOURS", 24)
        monkeypatch.setattr(config, "MISTRAL_BATCH_STRICT", True)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_ANNOTATION_PROMPT", "")

        pdf1 = tmp_path / "ok.pdf"
        pdf1.write_bytes(b"%PDF")
        pdf2 = tmp_path / "bad.pdf"
        pdf2.write_bytes(b"%PDF")
        output = tmp_path / "batch.jsonl"

        mock_client = MagicMock()

        def _pair_side_effect(client, path, expiry_hours=None):
            if path.name == "ok.pdf":
                return ("https://signed/1", "file-1")
            return None

        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch.object(
                mistral_converter,
                "_upload_file_for_ocr_pair",
                side_effect=_pair_side_effect,
            ):
                with patch.object(
                    mistral_converter,
                    "get_bbox_annotation_format",
                    return_value=None,
                ):
                    with patch.object(
                        mistral_converter,
                        "get_document_annotation_format",
                        return_value=None,
                    ):
                        ok, path_out, err = mistral_converter.create_batch_ocr_file([pdf1, pdf2], output)

        assert ok is False
        assert path_out is None
        assert "strict" in err.lower()
        mock_client.files.delete.assert_called_once_with(file_id="file-1")
        assert not output.exists()


# ============================================================================
# Additional batch operations coverage
# ============================================================================


class TestBatchOperationsAdditional:
    """Additional tests for batch operations covering edge cases."""

    def test_submit_batch_default_timeout(self, tmp_path):
        batch_file = tmp_path / "batch.jsonl"
        batch_file.write_text('{"body": {}}\n')

        mock_upload = MagicMock(id="batch_file_id")
        mock_job = MagicMock(id="job_default")

        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.files.upload.return_value = mock_upload
            mock_client.batch.jobs.create.return_value = mock_job
            mock_get.return_value = mock_client

            with patch.object(config, "MISTRAL_BATCH_TIMEOUT_HOURS", 24):
                ok, job_id, err = mistral_converter.submit_batch_ocr_job(batch_file, metadata={"type": "test"})

        assert ok is True
        create_kwargs = mock_client.batch.jobs.create.call_args[1]
        assert create_kwargs["metadata"] == {"type": "test"}

    def test_download_results_api_error(self):
        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.batch.jobs.get.side_effect = Exception("not found")
            mock_get.return_value = mock_client

            ok, path, err = mistral_converter.download_batch_results("job_bad")
        assert ok is False

    def test_list_batch_jobs_with_pagination(self):
        mock_job = MagicMock()
        mock_job.id = "j1"
        mock_job.status = "SUCCESS"
        mock_job.model = "m"
        mock_job.total_requests = 1
        mock_job.succeeded_requests = 1
        mock_job.failed_requests = 0
        mock_job.created_at = ""

        jobs_response = MagicMock()
        jobs_response.data = [mock_job]

        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.batch.jobs.list.return_value = jobs_response
            mock_get.return_value = mock_client

            ok, jobs, err = mistral_converter.list_batch_jobs(page=2, page_size=50)

        assert ok is True
        call_kwargs = mock_client.batch.jobs.list.call_args[1]
        assert call_kwargs["page"] == 2
        assert call_kwargs["page_size"] == 50

    def test_download_results_no_client(self):
        with patch.object(mistral_converter, "get_mistral_client", return_value=None):
            ok, path, err = mistral_converter.download_batch_results("job_x")
        assert ok is False

    def test_batch_status_zero_total(self):
        mock_job = MagicMock()
        mock_job.status = "QUEUED"
        mock_job.total_requests = 0
        mock_job.succeeded_requests = 0
        mock_job.failed_requests = 0
        mock_job.output_file = None
        mock_job.error_file = None

        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.batch.jobs.get.return_value = mock_job
            mock_get.return_value = mock_client

            ok, status, err = mistral_converter.get_batch_job_status("job_queued")

        assert ok is True
        assert status["progress_percent"] == 0


# ============================================================================
# convert_with_mistral_ocr - cache miss path
# ============================================================================


class TestListBatchJobsError:
    """Lines 2292-2295: list_batch_jobs exception handler."""

    def test_list_batch_jobs_api_error(self):
        with patch.object(mistral_converter, "get_mistral_client") as mock_get:
            mock_client = MagicMock()
            mock_client.batch.jobs.list.side_effect = Exception("API error")
            mock_get.return_value = mock_client

            ok, jobs, err = mistral_converter.list_batch_jobs()

        assert ok is False
        assert jobs is None
        assert "Error listing batch jobs" in err
