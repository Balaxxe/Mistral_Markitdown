# Mistral MarkItDown guide

This repository contains a Python document converter. Make the smallest safe
change that completes the request, preserve public behavior unless requested,
and validate the changed behavior proportionately.

## Always preserve

- Do not commit API keys, tokens, credentials, or generated/vendor artifacts.
- Preserve `.env`; copy `.env.example` only when the file is absent.
- Keep API validation and SSRF protections intact. Use `python3 -m <tool>`
  when no virtual environment is active.
- An explicit request authorizes the needed local edit and relevant validation.
  Ask before an unrequested external, destructive, irreversible, or material
  scope expansion.

## Route by task

The detailed original commands, compatibility notes, and Cursor configuration
are split in `docs/agent-guidance/index.md`. Load only the reference for the
affected boundary.

- **Source or tests:** inspect the affected module and its tests; use focused
  pytest, lint, or type checks as warranted.
- **Configuration, dependency, security, or CI behavior:** inspect the named
  configuration and its consumers before editing. Update docs/contracts when
  the requested behavior changes.
- **GUI, Cursor config, hooks, or Cursor slash skills:** read the relevant
  `.cursor/` file. Cursor-specific skills/policies are not generic Codex rules.
- **PR or release:** use `make check` when repository policy or requested
  delivery requires it; do not run the full battery for an isolated prose edit.

Keep provider-specific subagent policy in `.cursor/agents/` or `.cursor/rules/`.
An explicit target takes priority over editor selection in Cursor slash skills.
