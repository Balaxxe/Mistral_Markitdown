#!/usr/bin/env python3
"""Bootstrap ./env with dev dependencies if needed, then run pytest.

Same role as ``scripts/test-safe.sh`` for environments without bash or when
tools invoke ``python …`` instead of the repo virtualenv.

Usage:
    python3 run_tests.py [pytest arguments]

When no arguments are given, runs ``tests/`` (matches ``scripts/test-safe.sh``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / "env"
IS_WIN = sys.platform == "win32"
VENV_PYTHON = (VENV_DIR / "Scripts" / "python.exe") if IS_WIN else (VENV_DIR / "bin" / "python")
REQ_MAIN = ROOT / "requirements.txt"
REQ_DEV = ROOT / "requirements-dev.txt"


def _py_ok(python: Path) -> bool:
    try:
        r = subprocess.run(
            [str(python), "-c", "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"],
            capture_output=True,
        )
    except OSError:
        return False
    return r.returncode == 0


def _pick_base_python() -> Path | None:
    env_first = os.environ.get("MISTRAL_MARKITDOWN_PYTHON")
    names: list[str] = []
    if env_first:
        names.append(env_first)
    names.extend(["python3.12", "python3.11", "python3.10", "python3", "python"])

    seen: set[str] = set()
    for name in names:
        resolved = shutil.which(name)
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        path = Path(resolved)
        if _py_ok(path):
            return path
    return None


def _ensure_venv(base: Path) -> None:
    """Create ``env/`` using the supplied base Python.

    On Debian/Ubuntu the stock ``python3.N`` package ships without
    ``ensurepip``, so ``python -m venv env`` fails with a message directing
    the user to ``apt install python3-venv``. We catch that failure, wipe the
    half-populated ``env/`` directory, and retry with ``--without-pip``;
    :func:`_bootstrap_pip_in_venv` can then fetch pip via ``get-pip.py``.
    """
    if VENV_PYTHON.is_file():
        return
    try:
        subprocess.check_call([str(base), "-m", "venv", str(VENV_DIR)])
    except subprocess.CalledProcessError:
        if VENV_DIR.is_dir():
            shutil.rmtree(VENV_DIR, ignore_errors=True)
        print(
            "INFO: venv with ensurepip failed; retrying with --without-pip "
            "(will bootstrap pip via get-pip.py).",
            file=sys.stderr,
        )
        subprocess.check_call([str(base), "-m", "venv", "--without-pip", str(VENV_DIR)])


def _venv_has_pytest() -> bool:
    if not VENV_PYTHON.is_file():
        return False
    r = subprocess.run([str(VENV_PYTHON), "-c", "import pytest"], capture_output=True)
    return r.returncode == 0


def _venv_has_pip() -> bool:
    if not VENV_PYTHON.is_file():
        return False
    r = subprocess.run([str(VENV_PYTHON), "-m", "pip", "--version"], capture_output=True)
    return r.returncode == 0


def _bootstrap_pip_in_venv() -> None:
    if _venv_has_pip():
        return
    print("INFO: venv has no pip; trying ensurepip...", file=sys.stderr)
    r = subprocess.run([str(VENV_PYTHON), "-m", "ensurepip", "--upgrade"], capture_output=True)
    if r.returncode == 0 and _venv_has_pip():
        return
    print("INFO: ensurepip failed; fetching get-pip.py...", file=sys.stderr)
    import tempfile
    import urllib.error
    import urllib.request

    with tempfile.NamedTemporaryFile(suffix="-get-pip.py", delete=False) as f:
        path = Path(f.name)
    try:
        try:
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", path)
        except urllib.error.URLError as e:
            print(f"ERROR: could not download get-pip.py: {e}", file=sys.stderr)
            raise SystemExit(1) from e
        subprocess.check_call([str(VENV_PYTHON), str(path), "--no-warn-script-location"])
    finally:
        path.unlink(missing_ok=True)
    if not _venv_has_pip():
        print(f"ERROR: pip still missing; remove {VENV_DIR} and retry.", file=sys.stderr)
        raise SystemExit(1)


def _install_deps() -> None:
    _bootstrap_pip_in_venv()
    subprocess.check_call([str(VENV_PYTHON), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    subprocess.check_call([str(VENV_PYTHON), "-m", "pip", "install", "-q", "-r", str(REQ_MAIN), "-r", str(REQ_DEV)])


def main() -> int:
    pytest_args = sys.argv[1:]
    if not pytest_args:
        pytest_args = ["tests/"]

    if not VENV_PYTHON.is_file() or not _venv_has_pytest():
        base = _pick_base_python()
        if base is None:
            print("ERROR: need Python >= 3.10 on PATH (or set MISTRAL_MARKITDOWN_PYTHON).", file=sys.stderr)
            return 2
        _ensure_venv(base)
        if not _venv_has_pytest():
            print("INFO: installing dependencies into env/ ...", file=sys.stderr)
            _install_deps()
        if not _venv_has_pytest():
            print("ERROR: pytest missing after install; check requirements-dev.txt.", file=sys.stderr)
            return 2

    return subprocess.call([str(VENV_PYTHON), "-m", "pytest", *pytest_args], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
