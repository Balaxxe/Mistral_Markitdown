#!/usr/bin/env bash
# .cursor/hooks/stop-scoped-tests.sh
# Runs pytest scoped to changed files at end of agent run.
# Returns followup_message on failure so the agent can attempt a fix (up to loop_limit).
# Output is written to a temp file to avoid bloating agent context.
# No jq dependency — uses python stdlib for JSON output (proper escaping).
# Cross-platform: works on macOS, Linux, Windows (Git Bash).

set -euo pipefail

PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "")

# Cross-platform mktemp (/tmp/ doesn't exist on Windows Git Bash)
RESULTS_FILE=$(mktemp /tmp/cursor-test-results-XXXXXX.txt 2>/dev/null || mktemp)
FAILED=0
RAN_SOMETHING=0

# Get changed files (staged + unstaged, relative to HEAD or index)
CHANGED_FILES=$(git diff --name-only HEAD 2>/dev/null || git diff --name-only 2>/dev/null || echo "")

if [ -z "$CHANGED_FILES" ]; then
  echo '{}'
  exit 0
fi

# --- Python ---
PY_FILES=$(echo "$CHANGED_FILES" | grep -E '\.py$' || true)
if [ -n "$PY_FILES" ]; then
  PY_TEST_FILES=""
  for f in $PY_FILES; do
    dir=$(dirname "$f")
    base=$(basename "$f" .py)
    for candidate in \
      "${dir}/test_${base}.py" \
      "${dir}/tests/test_${base}.py" \
      "tests/test_${base}.py" \
      "tests/${dir}/test_${base}.py"; do
      if [ -f "$candidate" ]; then
        PY_TEST_FILES="$PY_TEST_FILES $candidate"
      fi
    done
    # Also include changed test files directly
    if echo "$f" | grep -qE '(^|/)test_.*\.py$'; then
      PY_TEST_FILES="$PY_TEST_FILES $f"
    fi
  done

  if [ -n "$PY_TEST_FILES" ]; then
    RAN_SOMETHING=1
    ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -z "$ROOT_DIR" ]; then
      ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
    TEST_SAFE="$ROOT_DIR/scripts/test-safe.sh"
    # Prefer test-safe.sh: creates ./env and installs pytest + dev deps (avoids bare /usr/bin/python3).
    if [ -f "$TEST_SAFE" ] && command -v bash >/dev/null 2>&1; then
      bash "$TEST_SAFE" -q $PY_TEST_FILES >> "$RESULTS_FILE" 2>&1 || FAILED=1
    elif [ -x "$ROOT_DIR/env/bin/python" ]; then
      "$ROOT_DIR/env/bin/python" -m pytest -q $PY_TEST_FILES >> "$RESULTS_FILE" 2>&1 || FAILED=1
    elif [ -f "$ROOT_DIR/env/Scripts/python.exe" ]; then
      "$ROOT_DIR/env/Scripts/python.exe" -m pytest -q $PY_TEST_FILES >> "$RESULTS_FILE" 2>&1 || FAILED=1
    elif command -v python3 >/dev/null 2>&1 && [ -f "$ROOT_DIR/run_tests.py" ]; then
      (cd "$ROOT_DIR" && python3 run_tests.py -q $PY_TEST_FILES) >> "$RESULTS_FILE" 2>&1 || FAILED=1
    elif command -v python3 >/dev/null 2>&1; then
      (cd "$ROOT_DIR" && python3 -m pytest -q $PY_TEST_FILES) >> "$RESULTS_FILE" 2>&1 || FAILED=1
    elif command -v pytest >/dev/null 2>&1; then
      (cd "$ROOT_DIR" && pytest -q $PY_TEST_FILES) >> "$RESULTS_FILE" 2>&1 || FAILED=1
    fi
  fi
fi

# --- Report ---
if [ "$FAILED" -eq 1 ]; then
  TAIL=$(tail -80 "$RESULTS_FILE")
  # Use python json.dumps for proper escaping of control chars, ANSI codes, etc.
  if [ -n "$PY" ]; then
    $PY -c "
import json, sys
tail = sys.stdin.read()
msg = 'Tests failed for changed files. Results in ${RESULTS_FILE}.\n\n' + tail + '\n\nFix the failures and re-run.'
print(json.dumps({'followup_message': msg}))
" <<< "$TAIL"
  else
    # Bare fallback if python is somehow gone mid-run
    echo '{"followup_message":"Tests failed. Check '"$RESULTS_FILE"' for details."}'
  fi
  exit 0
fi

# All passed or nothing to run
echo '{}'
exit 0
