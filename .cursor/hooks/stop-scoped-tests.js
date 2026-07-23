#!/usr/bin/env node
// .cursor/hooks/stop-scoped-tests.js — run pytest scoped to changed files when the
// agent finishes; feed failures back for a self-correcting loop.
//
// Invoked by stop.  Input: { status, loop_count, ... }
// Output: { followup_message } on failure (auto-submitted, capped by loop_limit),
//         {} otherwise.
//
// SECURITY: never uses shell:true — argv arrays only, so paths with spaces or
// metacharacters are inert argv entries.
//
// ⚠ WINDOWS CAVEAT (July 2026): Cursor has an open bug where stop-hook stdout is
// sometimes dropped on Windows. The tests still run and results land in the log
// file below; only the auto-continue message may be lost. CI covers the same
// checks. Reliable on macOS/Linux.
//
// Project: Mistral_Markitdown — maps <module>.py -> tests/test_<module>.py and
// runs pytest via the repo virtualenv (./env), falling back to run_tests.py
// (which bootstraps ./env + dev deps) or bare pytest. `-o addopts=` neutralizes
// the verbose addopts in pyproject.toml, mirroring scripts/test-safe.sh usage.

'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync, spawnSync } = require('child_process');

function out(obj) { process.stdout.write(JSON.stringify(obj)); }

let input = {};
try { input = JSON.parse(fs.readFileSync(0, 'utf8')); } catch { /* proceed */ }

// Only react to completed runs — never pile onto aborts/errors.
if (input.status && input.status !== 'completed') { out({}); process.exit(0); }

let changed = [];
try {
  changed = execFileSync('git', ['diff', '--name-only', 'HEAD'], { encoding: 'utf8', timeout: 15000 })
    .split('\n').map(s => s.trim()).filter(Boolean);
} catch { /* not a git repo or no HEAD */ }

const py = changed.filter(f => /\.py$/.test(f));
if (!py.length) { out({}); process.exit(0); }

const cwd = process.cwd();
const resultsFile = path.join(os.tmpdir(), 'cursor-scoped-tests-' + process.pid + '.log');
let failed = false;
let ran = false;

function has(cmd) {
  const probe = spawnSync(process.platform === 'win32' ? 'where' : 'which', [cmd], { stdio: 'ignore' });
  return probe.status === 0;
}

// Prefer the repo virtualenv (./env per Makefile and scripts/, or ./.venv).
// Checking all layouts keeps this file identical on every OS.
function resolveVenvPython() {
  for (const rel of [
    ['env', 'Scripts', 'python.exe'],
    ['env', 'bin', 'python'],
    ['.venv', 'Scripts', 'python.exe'],
    ['.venv', 'bin', 'python'],
  ]) {
    const p = path.join(cwd, ...rel);
    if (fs.existsSync(p)) return p;
  }
  return null;
}

// No shell anywhere: argv arrays only. Generous timeout: run_tests.py may
// bootstrap ./env with dev deps on first run; the suite itself is ~15s.
function runTests(cmd, args) {
  ran = true;
  const r = spawnSync(cmd, args, { encoding: 'utf8', timeout: 300000 });
  fs.appendFileSync(resultsFile, '\n$ ' + cmd + ' ' + args.join(' ') + '\n' + (r.stdout || '') + (r.stderr || ''));
  if (r.status !== 0) failed = true;
}

// Map changed source files to test files (repo convention: tests/test_<module>.py).
const candidates = new Set();
for (const f of py) {
  const dir = path.dirname(f);
  const base = path.basename(f, '.py');
  for (const c of [
    path.join(dir, 'test_' + base + '.py'),
    path.join(dir, 'tests', 'test_' + base + '.py'),
    path.join('tests', 'test_' + base + '.py'),
  ]) if (fs.existsSync(c)) candidates.add(c);
  if (/(^|[\\/])test_[^\\/]+\.py$/.test(f)) candidates.add(f);
}

if (candidates.size) {
  const pytestArgs = ['-m', 'pytest', '-q', '-o', 'addopts=', ...candidates];
  const venvPy = resolveVenvPython();
  const pathPy = has('python3') ? 'python3' : (has('python') ? 'python' : null);
  if (venvPy) runTests(venvPy, pytestArgs);
  else if (pathPy && fs.existsSync(path.join(cwd, 'run_tests.py'))) {
    runTests(pathPy, ['run_tests.py', '-q', '-o', 'addopts=', ...candidates]);
  }
  else if (pathPy) runTests(pathPy, pytestArgs);
  else if (has('pytest')) runTests('pytest', ['-q', '-o', 'addopts=', ...candidates]);
}

if (failed) {
  let tail = '';
  try { tail = fs.readFileSync(resultsFile, 'utf8').split('\n').slice(-60).join('\n'); } catch { }
  out({
    followup_message:
      'Scoped tests failed for the files you changed. Full output: ' + resultsFile +
      '\n\n' + tail + '\n\nFix the failures, then re-run only the affected tests.',
  });
} else {
  if (ran) { try { fs.unlinkSync(resultsFile); } catch { } }
  out({});
}
