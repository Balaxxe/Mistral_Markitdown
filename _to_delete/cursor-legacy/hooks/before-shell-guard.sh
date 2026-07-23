#!/usr/bin/env bash
# .cursor/hooks/before-shell-guard.sh
# Blocks destructive shell commands. Exit 0 + JSON = allow/deny decision.
# Exit 2 = hard block. failClosed: true in hooks.json means any other error also blocks.
# No jq dependency — uses python stdlib for JSON parsing.
# Cross-platform: works on macOS, Linux, Windows (Git Bash).

set -euo pipefail

PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "")
if [ -z "$PY" ]; then
  echo '{"permission":"allow"}'
  exit 0
fi

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | $PY -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('shell_execution', {}).get('command', ''))
except:
    pass
" 2>/dev/null || echo "")

if [ -z "$COMMAND" ]; then
  echo '{"permission":"allow"}'
  exit 0
fi

# Normalize for matching
CMD_LOWER=$(echo "$COMMAND" | tr '[:upper:]' '[:lower:]')

# --- Destructive filesystem ---
if echo "$CMD_LOWER" | grep -qE '(^|\s|;|&&|\|\|)rm\s+(-[a-z]*f[a-z]*\s+)?-[a-z]*r|rm\s+-rf\b'; then
  echo '{"permission":"deny","agent_message":"Blocked: recursive rm. Run manually if intended."}'
  exit 0
fi

if echo "$CMD_LOWER" | grep -qE 'find\s.*-delete'; then
  echo '{"permission":"deny","agent_message":"Blocked: find -delete. Run manually if intended."}'
  exit 0
fi

if echo "$CMD_LOWER" | grep -qE 'mkfs\b|dd\s.*of=/dev/'; then
  echo '{"permission":"deny","agent_message":"Blocked: disk-level write operation."}'
  exit 0
fi

# --- Git force/destructive ---
if echo "$CMD_LOWER" | grep -qE 'git\s+push\s.*--force|git\s+push\s.*-f\b|git\s+push\s.*--delete'; then
  echo '{"permission":"deny","agent_message":"Blocked: force push or branch delete. Run manually if intended."}'
  exit 0
fi

if echo "$CMD_LOWER" | grep -qE 'git\s+reset\s+--hard|git\s+clean\s+-[a-z]*f'; then
  echo '{"permission":"deny","agent_message":"Blocked: git reset --hard or git clean -f. Run manually if intended."}'
  exit 0
fi

if echo "$CMD_LOWER" | grep -qE 'git\s+checkout\s+-f\b'; then
  echo '{"permission":"deny","agent_message":"Blocked: git checkout -f. Run manually if intended."}'
  exit 0
fi

# --- Destructive SQL ---
if echo "$CMD_LOWER" | grep -qE 'drop\s+(table|database|schema|index)|truncate\s+table|delete\s+from\s+\S+\s*(;|$)'; then
  echo '{"permission":"deny","agent_message":"Blocked: destructive SQL. Run manually if intended."}'
  exit 0
fi

# --- Infrastructure ---
if echo "$CMD_LOWER" | grep -qE 'terraform\s+destroy|terraform\s+apply\s+-auto-approve'; then
  echo '{"permission":"deny","agent_message":"Blocked: terraform destroy/auto-approve. Run manually if intended."}'
  exit 0
fi

if echo "$CMD_LOWER" | grep -qE 'docker\s+system\s+prune\s+-a|docker\s+volume\s+prune\s+-f'; then
  echo '{"permission":"deny","agent_message":"Blocked: aggressive Docker prune. Run manually if intended."}'
  exit 0
fi

# --- Pipe-to-shell (supply chain risk) ---
if echo "$CMD_LOWER" | grep -qE '(curl|wget)\s.*\|\s*(bash|sh|zsh|sudo\s+bash|sudo\s+sh)'; then
  echo '{"permission":"deny","agent_message":"Blocked: pipe-to-shell install. Download and inspect the script first."}'
  exit 0
fi

# --- System ---
if echo "$CMD_LOWER" | grep -qE '(^|\s|;)(reboot|shutdown|halt|poweroff)\b'; then
  echo '{"permission":"deny","agent_message":"Blocked: system power command."}'
  exit 0
fi

# --- Secrets exposure ---
if echo "$CMD_LOWER" | grep -qE '(^|\s|;)(cat|less|more|head|tail|bat)\s+.*\.(env|pem|key|p12|pfx)\b'; then
  echo '{"permission":"deny","agent_message":"Blocked: reading potential secret file via shell. Use Cursor file access instead, or run manually."}'
  exit 0
fi

# Default: allow
echo '{"permission":"allow"}'
exit 0
