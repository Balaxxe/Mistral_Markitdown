"""Cross-process safety tests for the local Mistral upload registry."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import config
import mistral_converter.upload as upload_module

# Initialize config dirs so registry writes have somewhere to land.
config.ensure_directories()

REPO_ROOT = str(Path(__file__).resolve().parent.parent)

_CHILD_SNIPPET = """
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])

import config

config.CACHE_DIR = Path(sys.argv[2])

import mistral_converter.upload as upload

prefix = sys.argv[3]
start = float(sys.argv[4])
while time.time() < start:
    time.sleep(0.005)
for index in range(20):
    if not upload._register_uploaded_file(prefix + str(index), "ocr"):
        raise SystemExit("registration failed")
    time.sleep(0.01)
"""

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not upload_module._registry_file_locking_available(),
        reason="No file locking module on this platform; the registry lock degrades to a no-op",
    ),
]


class TestUploadRegistryMultiprocess:
    """Concurrent writers must not drop each other's registry entries."""

    def test_concurrent_processes_keep_both_registrations(self, tmp_path, monkeypatch):
        start = time.time() + 1.0
        children = [
            subprocess.Popen(
                [sys.executable, "-c", _CHILD_SNIPPET, REPO_ROOT, str(tmp_path), prefix, str(start)],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for prefix in ("alpha_", "beta_")
        ]
        for child in children:
            stdout, stderr = child.communicate(timeout=60)
            assert child.returncode == 0, stderr.decode(errors="replace")

        monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
        registered = {entry["id"] for entry in upload_module._load_upload_registry()}

        expected = {f"alpha_{i}" for i in range(20)} | {f"beta_{i}" for i in range(20)}
        assert registered == expected


@pytest.mark.skipif(upload_module.fcntl is None, reason="POSIX flock timeout behavior")
class TestRegistryLockTimeout:
    """A stalled peer must not hold every thread behind the registry lock."""

    def test_acquisition_gives_up_at_the_deadline(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(upload_module, "_UPLOAD_REGISTRY_LOCK_TIMEOUT", 0.3)
        lock_path = str(upload_module._upload_registry_lock_path())

        holder = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        contender = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            upload_module.fcntl.flock(holder, upload_module.fcntl.LOCK_EX)
            started = time.monotonic()
            assert upload_module._acquire_registry_file_lock(contender) is False
            elapsed = time.monotonic() - started
            assert 0.3 <= elapsed < 5.0
        finally:
            upload_module.fcntl.flock(holder, upload_module.fcntl.LOCK_UN)
            os.close(contender)
            os.close(holder)

    def test_acquisition_succeeds_once_the_peer_releases(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
        # Generous deadline, fast retries: the assertion is about eventual
        # acquisition, not elapsed time, and CI runners stall unpredictably.
        monkeypatch.setattr(upload_module, "_UPLOAD_REGISTRY_LOCK_TIMEOUT", 30.0)
        monkeypatch.setattr(upload_module, "_UPLOAD_REGISTRY_LOCK_RETRY_INTERVAL", 0.02)
        lock_path = str(upload_module._upload_registry_lock_path())

        holder = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        contender = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        release_errors = []

        def release_soon():
            try:
                time.sleep(0.1)
                upload_module.fcntl.flock(holder, upload_module.fcntl.LOCK_UN)
            except BaseException as e:  # a silent release failure would look like a timeout
                release_errors.append(e)

        try:
            upload_module.fcntl.flock(holder, upload_module.fcntl.LOCK_EX)
            waiter = threading.Thread(target=release_soon)
            waiter.start()
            acquired = upload_module._acquire_registry_file_lock(contender)
            waiter.join(timeout=10)
            assert not release_errors, f"peer release failed: {release_errors[0]!r}"
            assert acquired is True
            upload_module._release_registry_file_lock(contender)
        finally:
            os.close(contender)
            os.close(holder)
