#!/usr/bin/env node
// .cursor/hooks/format-after-edit.js — run the project's formatter on files the agent edits.
//
// Invoked by afterFileEdit.  Input (stdin JSON): { file_path, edits: [{old_string,new_string}], ... }
// Output (stdout JSON): {} — observational hook; always exits 0.
//
// SECURITY: never uses shell:true. Local Node formatters are resolved to their real
// JS entrypoints via package.json "bin" and run through process.execPath, so paths
// with spaces and hostile file names are inert argv entries, never shell input.
//
// Known limitation (July 2026): afterFileEdit misses events on batch edits /
// "Accept All". Keep editor format-on-save enabled as the backstop.
//
// CUSTOMIZE: the generator trims the switch below to the formatters this project
// actually uses, preferring repo-local binaries over globals.

'use strict';
const fs = require('fs');
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
function done() { fs.writeSync(1, '{}'); process.exit(0); }

(async () => {
const raw = await readInput(3000);
let input;
try { input = JSON.parse(raw); }
catch { done(); }

// Cursor sends Unix-style drive paths on Windows ("/c:/..."): normalize.
let file = String(input.file_path || '');
if (/^\/[a-zA-Z]:/.test(file)) file = file.slice(1);
if (!file || !fs.existsSync(file)) done();

const ext = path.extname(file).toLowerCase().replace('.', '');
const cwd = process.cwd(); // project hooks run from the project root

// Resolve a local Node package's executable to its real JS entrypoint
// (never the .cmd shim — those require a shell and invite injection).
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

function has(cmd) {
  const probe = spawnSync(process.platform === 'win32' ? 'where' : 'which', [cmd], { stdio: 'ignore' });
  return probe.status === 0;
}

// No shell, argv array only: spaces and metacharacters in `file` are harmless.
function run(cmd, args) {
  try { execFileSync(cmd, args, { stdio: 'ignore', timeout: 20000 }); }
  catch { /* formatting is best-effort */ }
}

switch (ext) {
  case 'js': case 'jsx': case 'ts': case 'tsx': case 'json': case 'css':
  case 'scss': case 'html': case 'md': case 'yaml': case 'yml': {
    const biome = nodePkgBin('@biomejs/biome', 'biome') || nodePkgBin('biome');
    const prettier = nodePkgBin('prettier');
    if (biome) run(process.execPath, [biome, 'format', '--write', file]);
    else if (prettier) run(process.execPath, [prettier, '--write', file]);
    break;
  }
  case 'py': {
    if (has('ruff')) { run('ruff', ['format', file]); run('ruff', ['check', '--fix', file]); }
    else if (has('black')) run('black', ['--quiet', file]);
    break;
  }
  case 'tf': case 'tfvars': {
    if (has('terraform')) run('terraform', ['fmt', file]);
    else if (has('tofu')) run('tofu', ['fmt', file]);
    break;
  }
  case 'go': {
    if (has('gofumpt')) run('gofumpt', ['-w', file]);
    else if (has('gofmt')) run('gofmt', ['-w', file]);
    break;
  }
  case 'rs': if (has('rustfmt')) run('rustfmt', [file]); break;
  case 'rb': if (has('rubocop')) run('rubocop', ['-a', '--fail-level', 'error', file]); break;
  case 'dart': if (has('dart')) run('dart', ['format', file]); break;
  case 'kt': case 'kts': if (has('ktlint')) run('ktlint', ['-F', file]); break;
  case 'swift': if (has('swift-format')) run('swift-format', ['-i', file]); break;
}

done();
})();
