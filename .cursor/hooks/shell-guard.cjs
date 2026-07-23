#!/usr/bin/env node
// .cursor/hooks/shell-guard.js — project-level destructive-command guard for Cursor agents.
// Cross-platform: Windows, macOS, Linux (requires Node on PATH).
//
// Invoked by beforeShellExecution. The matcher in hooks.json gates the spawn and is
// deliberately PRECISE: only command forms this script would plausibly deny ever spawn
// it (plain `git push` / `git checkout` / `terraform plan` never invoke the hook), so a
// hook failure under failClosed can never block routine commands. The matcher is a
// machine-generated case-insensitive expansion (matchers compile with NO regex flags);
// flag letters with case-sensitive meaning (git -f, -D) are kept verbatim.
//
// Input  (stdin JSON):  { command, cwd, sandbox, conversation_id, ... }
// Output (stdout JSON): { permission: "allow" | "deny", agent_message?, user_message? }
// Docs: https://cursor.com/docs/agent/hooks
//
// Hardening against real-world Cursor spawn behavior (learned in production):
//   - stdout is written with fs.writeSync(1, ...) — process.stdout.write followed by
//     process.exit() can be truncated on Windows pipes ("hook returned no output").
//   - stdin is read async with early-JSON-parse and a 3s deadline — Cursor may not
//     close the pipe promptly, and a sync read-to-EOF would hang into the timeout.
//   - unreadable/empty input returns an explicit deny WITH a retry hint (valid JSON,
//     exit 0) rather than silent failure, keeping fail-closed semantics legible.
// Only "deny" is enforced by Cursor as of July 2026; "ask"/"allow" are advisory.
// Denials are logged to ~/.cursor/logs/shell-guard.jsonl.

'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');

function emit(obj) { fs.writeSync(1, JSON.stringify(obj)); }

function readInput(deadlineMs) {
  return new Promise((resolve) => {
    let buf = '';
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(buf); } };
    const timer = setTimeout(finish, deadlineMs);
    if (timer.unref) timer.unref();
    try {
      process.stdin.setEncoding('utf8');
      process.stdin.on('data', (c) => {
        buf += c;
        try { JSON.parse(buf); clearTimeout(timer); finish(); } catch { /* incomplete — keep reading */ }
      });
      process.stdin.on('end', () => { clearTimeout(timer); finish(); });
      process.stdin.on('error', () => { clearTimeout(timer); finish(); });
    } catch { clearTimeout(timer); finish(); }
  });
}

// ---------------------------------------------------------------------------
// PROJECT_RULES - extend with project-specific dangers, e.g.:
//   { re: /npm.run.deploy/i,        why: 'production deploy (run manually)' },
//   { re: /prod(uction)?_db/i,      why: 'references the production database' },
// ---------------------------------------------------------------------------
const PROJECT_RULES = [
  // Mistral_Markitdown-specific dangers
  { re: /\btwine\s+upload\b|\bmake\s+publish\b/i, why: 'PyPI publish (run manually)' },
  { re: /\bpip\s+uninstall\b[^|;]*(-y\b|--yes\b)/i, why: 'non-interactive pip uninstall' },
];

const RULES = [
  ...PROJECT_RULES,
  { re: /\brm\s+(-[a-z]*r|--recursive)/i, why: 'recursive rm' },
  { re: /\b(del|erase)\b[^|;]*\/s\b/i, why: 'recursive Windows delete (del /s)' },
  { re: /\b(rd|rmdir)\b[^|;]*\/s\b/i, why: 'recursive rmdir (/s)' },
  { re: /remove-item\b[^|;]*(-recurse|-force)/i, why: 'Remove-Item -Recurse/-Force' },
  { re: /\bfind\s+[^|;]*-delete\b/i, why: 'find -delete' },
  { re: /\bmkfs|\bdd\s+[^|;]*\bof=(\/dev\/|\\\\\.\\)|\bdiskpart\b|\bformat\s+[a-z]:/i, why: 'disk-level destructive operation' },
  { re: /\bgit\s+push\b[^|;]*(--force(-with-lease)?\b|\s-f\b|--delete\b)/i, why: 'git force push / remote branch delete' },
  { re: /\bgit\s+reset\s+--hard\b/i, why: 'git reset --hard' },
  { re: /\bgit\s+clean\b[^|;]*\s-[a-z]*f/i, why: 'git clean -f' },
  { re: /\bgit\s+checkout\b[^|;]*(\s-f\b|--force\b)/i, why: 'git checkout --force' },
  { re: /\bgit\s+branch\b[^|;]*(\s-D\b|--delete\s+--force)/, why: 'force branch delete' },
  { re: /\bgit\s+stash\s+(drop|clear)\b/i, why: 'git stash drop/clear' },
  { re: /\bdrop\s+(table|database|schema|index)\b/i, why: 'destructive SQL (DROP)' },
  { re: /\btruncate\s+table\b/i, why: 'destructive SQL (TRUNCATE)' },
  { re: /\bdelete\s+from\s+\S+\s*(;|"|'|$)/i, why: 'unfiltered SQL DELETE (no WHERE clause)' },
  { re: /\b(terraform|tofu)\s+destroy\b/i, why: 'terraform/tofu destroy' },
  { re: /\b(terraform|tofu)\s+apply\b[^|;]*-auto-approve/i, why: 'terraform apply -auto-approve' },
  { re: /\bdocker\s+system\s+prune\b[^|;]*-a|\bdocker\s+volume\s+prune\b[^|;]*-f/i, why: 'aggressive Docker prune' },
  { re: /\bkubectl\s+delete\b/i, why: 'kubectl delete' },
  { re: /\bhelm\s+(uninstall|delete)\b/i, why: 'helm uninstall' },
  { re: /\b(curl|wget|iwr|irm|invoke-webrequest|invoke-restmethod)\b[^|;]*\|\s*(bash|sh|zsh|pwsh|powershell|iex)\b/i, why: 'pipe-to-shell install (inspect the script first)' },
  { re: /\binvoke-expression\b|\biex\s*\(/i, why: 'Invoke-Expression of dynamic content' },
  { re: /(^|[\s;|&])(shutdown|reboot|poweroff|halt)\b/i, why: 'system power command' },
  { re: /\breg\s+delete\b/i, why: 'registry delete' },
  { re: /\bsetx\s+/i, why: 'persistent environment variable change (setx)' },
  { re: /\bset-executionpolicy\b/i, why: 'PowerShell execution policy change' },
  { re: /\bschtasks\b[^|;]*\/delete/i, why: 'scheduled task delete' },
];

// Secret-file reads via shell — the terminal bypasses .cursorignore, so this is the
// real exfiltration path. Exemptions: example/template files and SSH PUBLIC keys.
const SECRET_READ = /\b(cat|type|gc(?=\s+[^-\s])|get-content|more|less|head|tail|bat)\b[^|;&]*(\.(env|pem|p12|pfx)\b|\.key\b|id_rsa(?!\.pub)|id_ed25519(?!\.pub))/i;
const SAFE_SECRET = /\.(env|tfvars)\.(example|template|sample)\b/i;

(async () => {
  const rawInput = await readInput(5000);

  let input = null;
  try { input = JSON.parse(rawInput); } catch { /* handled below */ }

  if (!input || typeof input !== 'object') {
    // Matched commands are dangerous-looking by construction — failing closed is
    // correct, but do it legibly: a real response Cursor can display, not silence.
    emit({
      permission: 'deny',
      agent_message: 'shell-guard could not read its hook input (transient Cursor stdin issue, not a policy decision). The command was blocked as a precaution. Retry it once; if this repeats, tell the user "shell-guard stdin flake" so they can investigate.',
      user_message: 'Shell guard: hook input unreadable — blocked as a precaution (agent will retry).',
    });
    return;
  }

  const cmd = String(input.command || '');
  if (!cmd) { emit({ permission: 'allow' }); return; }

  let denial = null;
  for (const r of RULES) {
    try { if (r.re.test(cmd)) { denial = r.why; break; } }
    catch { /* one bad rule must not disable the guard */ }
  }
  if (!denial && SECRET_READ.test(cmd) && !SAFE_SECRET.test(cmd)) {
    denial = 'shell read of a potential secret file (terminal bypasses .cursorignore)';
  }

  if (denial) {
    try {
      const dir = path.join(os.homedir(), '.cursor', 'logs');
      fs.mkdirSync(dir, { recursive: true });
      fs.appendFileSync(path.join(dir, 'shell-guard.jsonl'),
        JSON.stringify({ ts: new Date().toISOString(), cwd: input.cwd || '', reason: denial, command: cmd.slice(0, 500) }) + '\n');
    } catch { /* logging must never affect the verdict */ }
    emit({
      permission: 'deny',
      agent_message: 'Blocked by the project shell guard: ' + denial + '. Do not retry variations of this command. If the operation is genuinely required, ask the user to run it manually.',
      user_message: 'Shell guard blocked: ' + denial,
    });
    return;
  }

  emit({ permission: 'allow' });
})();
