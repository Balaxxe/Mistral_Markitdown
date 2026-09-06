# Subagents

Three custom subagents are defined in `.cursor/agents/`:

- **verifier** (read-only) — reviews changes for correctness, coverage gaps, and rule violations. Use after edits to catch issues before committing.
- **test-runner** — runs scoped pytest for changed files and reports results. Does not fix failures.
- **researcher** (read-only) — explores the codebase for usage sites, patterns, and context before making changes in unfamiliar areas.

**Critical rules subagents must follow** (they do NOT inherit User Rules):

- Use `python3 -m <tool>` instead of bare commands (no virtualenv assumed).
- Never hardcode secrets. `MISTRAL_API_KEY` is loaded via `python-dotenv` and `config.py`.
- Tests mock API calls — they work without a Mistral key.
- Lint: `python3 -m flake8 .` | Format: `python3 -m black . && python3 -m isort .`
- Test: `python3 -m pytest tests/test_<name>.py -v` (scoped) or `python3 -m pytest tests/` (full)
- Subagents must not commit, push, publish, or run destructive operations.
