#!/usr/bin/env node
// .cursor/hooks/stop-scoped-tests.js — run tests scoped to changed files when the
// agent finishes; feed failures back for a self-correcting loop (OPT-IN template).
//
// Invoked by stop.  Input: { status, loop_count, ... }
// Output: { followup_message } on failure (auto-submitted, capped by loop_limit),
//         {} otherwise.
//
// SECURITY: never uses shell:true. Local Node test runners are resolved to their
// real JS entrypoints via package.json "bin" and run through process.execPath, so
// paths with spaces or metacharacters are inert argv entries.
//
// Timing: per-runner timeout is 120s; pair with "timeout": 600 in hooks.json so
// multi-ecosystem changes (JS + Python + Go) fit inside the outer budget.
//
// ⚠ WINDOWS CAVEAT (July 2026): Cursor has an open bug where stop-hook stdout is
// sometimes dropped on Windows. On Windows teams, prefer CI or the verifier
// subagent for the fix loop; this hook still runs tests, it just may not
// auto-continue. Reliable on macOS/Linux.
//
// CUSTOMIZE: the generator trims the runners below to the project's actual stack
// and exact commands from AGENTS.md.

'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync, spawnSync } = require('child_process');

// Async stdin read with early-JSON-parse + deadline: Cursor may not close the pipe,
// and sync reads flake with EAGAIN on Windows named pipes.
function readInput(deadlineMs) {
  return new Promise((resolve) => {
    let buf = '';
    let done0 = false;
    const finish = () => { if (!done0) { done0 = true; resolve(buf); } };
    const timer = setTimeout(finish, deadlineMs);
    if (timer.unref) timer.unref();
    try {
      process.stdin.setEncoding('utf8');
      process.stdin.on('data', (c) => {
        buf += c;
        try { JSON.parse(buf); clearTimeout(timer); finish(); } catch { /* keep reading */ }
      });
      process.stdin.on('end', () => { clearTimeout(timer); finish(); });
      process.stdin.on('error', () => { clearTimeout(timer); finish(); });
    } catch { clearTimeout(timer); finish(); }
  });
}


// fs.writeSync: process.stdout.write + process.exit can be truncated on Windows pipes.
function out(obj) { fs.writeSync(1, JSON.stringify(obj)); }

(async () => {
const raw = await readInput(3000);
let input = {};
try { input = JSON.parse(raw); } catch { /* run nothing blind */ out({}); return; }

// Only react to completed runs — never pile onto aborts/errors.
if (input.status && input.status !== 'completed') { out({}); process.exit(0); }

let changed = [];
try {
  changed = execFileSync('git', ['diff', '--name-only', 'HEAD'], { encoding: 'utf8', timeout: 15000 })
    .split('\n').map(s => s.trim()).filter(Boolean);
} catch { /* not a git repo or no HEAD */ }
if (!changed.length) { out({}); process.exit(0); }

const cwd = process.cwd();
const resultsFile = path.join(os.tmpdir(), 'cursor-scoped-tests-' + process.pid + '.log');
let failed = false;
let ran = false;

function nodePkgBin(pkgName, binName) {
  try {
    const pkgDir = path.join(cwd, 'node_modules', pkgName);
    const pkg = JSON.parse(fs.readFileSync(path.join(pkgDir, 'package.json'), 'utf8'));
    let rel = pkg.bin;
    if (rel && typeof rel === 'object') rel = rel[binName || pkgName] || Object.values(rel)[0];
    if (!rel) return null;
    const abs = path.join(pkgDir, rel);
    return fs.existsSync(abs) ? abs : null;
  } catch { return null; }
}

// No shell anywhere: argv arrays only.
function runTests(cmd, args) {
  ran = true;
  const r = spawnSync(cmd, args, { encoding: 'utf8', timeout: 120000 });
  fs.appendFileSync(resultsFile, '\n$ ' + cmd + ' ' + args.join(' ') + '\n' + (r.stdout || '') + (r.stderr || ''));
  if (r.status !== 0) failed = true;
}

const js = changed.filter(f => /\.(js|jsx|ts|tsx)$/.test(f));
const py = changed.filter(f => /\.py$/.test(f));
const go = changed.filter(f => /\.go$/.test(f));
const rs = changed.filter(f => /\.rs$/.test(f));

if (js.length) {
  const vitest = nodePkgBin('vitest');
  const jest = nodePkgBin('jest-cli', 'jest') || nodePkgBin('jest');
  if (vitest) runTests(process.execPath, [vitest, 'related', '--run', ...js]);
  else if (jest) runTests(process.execPath, [jest, '--findRelatedTests', ...js]);
}

if (py.length) {
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
  if (candidates.size) runTests('pytest', ['-q', ...candidates]);
}

if (go.length) {
  const dirs = [...new Set(go.map(f => './' + path.dirname(f)))];
  runTests('go', ['test', ...dirs]);
}

if (rs.length && fs.existsSync('Cargo.toml')) {
  runTests('cargo', ['test', '--lib', '--quiet']);
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
})();
