# Cursor config

- Slash skills live in `.cursor/skills/`: `/explain`, `/review`, `/pr-description`, `/test-plan`, `/refactor-plan`, `/migration-plan`.
- Hooks (`.cursor/hooks.json`, Node scripts in `.cursor/hooks/`): a destructive-command shell guard, black+isort auto-format after agent edits, and a scoped pytest run when the agent stops.
- Do not edit `.cursor/hooks*`, `.cursor/agents/`, `.cursor/skills/`, or `.cursor/rules/` unless the user explicitly asks for Cursor config changes.
