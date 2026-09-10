# Rules

- Make the smallest safe change.
- Preserve public APIs unless the task says otherwise.
- Reuse existing patterns/utilities before adding abstractions.
- Add/update tests for behavior changes.
- An explicit request authorizes the local edits and relevant validation needed to complete it. Ask before materially expanding the scope or causing an external or irreversible effect.
- Ask first before changing schema, auth, CI, infra, or dependencies when the request does not explicitly cover that change.
- Never commit secrets or edit generated/vendor files casually.
- Use `python3 -m <tool>` instead of bare commands when not in a virtualenv.
