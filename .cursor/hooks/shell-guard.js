#!/usr/bin/env node
// .cursor/hooks/shell-guard.js — project-level destructive-command guard for Cursor agents.
// Cross-platform: Windows, macOS, Linux (requires Node on PATH).
//
// Invoked by beforeShellExecution. The matcher in hooks.json gates the spawn (a
// machine-generated case-insensitive expansion — matchers compile with NO regex
// flags, so [rR][mM]-style classes are required for case robustness). The precise
// rules below make the final call; matcher overmatch is intentional.
//
// Input  (stdin JSON):  { command, cwd, sandbox, conversation_id, ... }
// Output (stdout JSON): { permission: "allow" | "deny", agent_message?, user_message? }
// Docs: https://cursor.com/docs/agent/hooks
//
// Failure policy (this is a GUARD, paired with failClosed:true in hooks.json):
//   - unreadable/unparseable stdin  -> exit 1, no output  => Cursor blocks the matched command
//   - internal rule error           -> that rule is skipped, others still run
// Only "deny" is enforced by Cursor as of July 2026; "ask"/"allow" are advisory.
// Denials are logged to ~/.cursor/logs/shell-guard.jsonl.

'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');

function out(obj) { process.stdout.write(JSON.stringify(obj)); }

// Windows named pipes can throw transient EAGAIN on sync stdin reads — retry
// briefly before declaring stdin unreadable.
function readStdin() {
  for (let i = 0; i < 5; i++) {
    try { return fs.readFileSync(0, 'utf8'); }
    catch (e) {
      if (e && e.code === 'EAGAIN') {
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 40);
        continue;
      }
      throw e;
    }
  }
  throw new Error('stdin unavailable');
}

let input;
try { input = JSON.parse(readStdin()); }
catch { process.exit(1); } // guard cannot decide -> fail closed via hooks.json failClosed

const cmd = String(input.command || '');
if (!cmd) { out({ permission: 'allow' }); process.exit(0); }

// ---------------------------------------------------------------------------
// PROJECT_RULES — extend with project-specific dangers, e.g.:
//   { re: /npm\s+run\s+deploy/i,      why: 'production deploy (run manually)' },
//   { re: /prisma\s+migrate\s+reset/i, why: 'database reset' },
//   { re: /prod(uction)?_db/i,               why: 'command references the production database' },
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
// `gc` (PowerShell Get-Content alias) requires a non-flag operand so the common
// `gc -m "..."` git-commit alias never false-positives.
const SECRET_READ = /\b(cat|type|gc(?=\s+[^-\s])|get-content|more|less|head|tail|bat)\b[^|;&]*(\.(env|pem|p12|pfx)\b|\.key\b|id_rsa(?!\.pub)|id_ed25519(?!\.pub))/i;
const SAFE_SECRET = /\.(env|tfvars)\.(example|template|sample)\b/i;

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
  out({
    permission: 'deny',
    agent_message: 'Blocked by the project shell guard: ' + denial + '. Do not retry variations of this command. If the operation is genuinely required, ask the user to run it manually.',
    user_message: 'Shell guard blocked: ' + denial,
  });
  process.exit(0);
}

out({ permission: 'allow' });
