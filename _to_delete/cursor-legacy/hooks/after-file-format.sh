#!/usr/bin/env bash
# .cursor/hooks/after-file-format.sh
# Runs black + isort after agent file edits on .py files.
# Note: does NOT fire on "Accept All" — pair with editor format-on-save for final state.
# No jq dependency — uses python stdlib for JSON parsing.
# Cross-platform: works on macOS, Linux, Windows (Git Bash).

set -euo pipefail

PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "")
if [ -z "$PY" ]; then
  exit 0
fi

INPUT=$(cat)
FILE=$(echo "$INPUT" | $PY -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('file_edit', {}).get('path', ''))
except:
    pass
" 2>/dev/null || echo "")

if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
  exit 0
fi

EXT="${FILE##*.}"

case "$EXT" in
  py)
    if command -v black &>/dev/null; then
      black --quiet "$FILE" 2>/dev/null || true
    fi
    if command -v isort &>/dev/null; then
      isort --profile black --line-length 120 "$FILE" 2>/dev/null || true
    fi
    ;;
esac

exit 0
