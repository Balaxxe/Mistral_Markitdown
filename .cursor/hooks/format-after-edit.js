#!/usr/bin/env node
// .cursor/hooks/format-after-edit.js — run black + isort on Python files the agent edits.
//
// Invoked by afterFileEdit.  Input (stdin JSON): { file_path, edits: [{old_string,new_string}], ... }
// Output (stdout JSON): {} — observational hook; always exits 0.
//
// SECURITY: never uses shell:true. Formatters run via the repo's Python
// interpreter with argv arrays only, so paths with spaces and hostile file
// names are inert argv entries, never shell input.
//
// Known limitation (July 2026): afterFileEdit misses events on batch edits /
// "Accept All". Keep editor format-on-save enabled as the backstop.
//
// Project: Mistral_Markitdown — black (line-length 120) + isort (profile
// "black"), both configured in pyproject.toml. Prefers the repo's ./env (or
// .venv) virtualenv, falling back to python3/python on PATH, matching the
// repo convention of `python3 -m <tool>`.

'use strict';
const fs = require('fs');
const path = require('path');
const { execFileSync, spawnSync } = require('child_process');

function done() { process.stdout.write('{}'); process.exit(0); }

let input;
try { input = JSON.parse(fs.readFileSync(0, 'utf8')); }
catch { done(); }

// Cursor sends Unix-style drive paths on Windows ("/c:/..."): normalize.
let file = String(input.file_path || '');
if (/^\/[a-zA-Z]:/.test(file)) file = file.slice(1);
if (!file || !fs.existsSync(file)) done();

if (path.extname(file).toLowerCase() !== '.py') done();

const cwd = process.cwd(); // project hooks run from the project root

function has(cmd) {
  const probe = spawnSync(process.platform === 'win32' ? 'where' : 'which', [cmd], { stdio: 'ignore' });
  return probe.status === 0;
}

// Prefer the repo virtualenv (./env per Makefile and scripts/, or ./.venv),
// then PATH. Checking all layouts keeps this file identical on every OS.
function resolvePython() {
  for (const rel of [
    ['env', 'Scripts', 'python.exe'],
    ['env', 'bin', 'python'],
    ['.venv', 'Scripts', 'python.exe'],
    ['.venv', 'bin', 'python'],
  ]) {
    const p = path.join(cwd, ...rel);
    if (fs.existsSync(p)) return p;
  }
  if (has('python3')) return 'python3';
  if (has('python')) return 'python';
  return null;
}

const py = resolvePython();
if (!py) done();

// No shell, argv array only: spaces and metacharacters in `file` are harmless.
function run(args) {
  try { execFileSync(py, args, { stdio: 'ignore', timeout: 20000 }); }
  catch { /* formatting is best-effort; missing module = no-op */ }
}

run(['-m', 'black', '--quiet', file]);
run(['-m', 'isort', file]);

done();
