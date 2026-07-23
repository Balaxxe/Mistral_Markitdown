---
description: Generate a practical, prioritized test plan for the current selection, function, class, or file.
---

## User Input

```text
$ARGUMENTS
```

You MUST consider the user input before proceeding if it is not empty.

## Goal

Produce a concrete, prioritized test plan for the selected code, current file, or user-specified target.

Do not modify files or write the final tests unless the user explicitly asks for implementation after the plan.

## Target Selection

Use this order:

1. Selected code, if present
2. Current active file
3. A file, function, class, or behavior named in `$ARGUMENTS`
4. Inspect nearby tests, fixtures, helpers, and public callers only as needed

If no target is available, say `No selection or active file was available for a test plan.` and stop.

## Instructions

- Infer the intended behavior from code and nearby tests.
- Prefer concrete scenarios over generic testing advice.
- Cover the most important happy paths, edge cases, invalid inputs, state transitions, and error paths.
- Include integration or contract tests only when they materially matter.
- Call out concurrency, idempotency, retries, time boundaries, serialization, locale, permissions, and data-shape issues when relevant.
- Mention fixtures, mocks, stubs, factories, and environment setup that the tests will need.
- If existing tests already cover a scenario, mark it as existing coverage instead of proposing duplicate work.
- Honor `$ARGUMENTS` for focus, such as `only unit tests`, `API contract coverage`, or `be exhaustive`.

## Output Format

## Test Target
- What you are testing and at what level.

## Existing Coverage
- Relevant existing tests or `Not found`.

## Priority 1 Tests
For each test, include:
- Test name
  - Scenario:
  - Setup:
  - Assertions:

## Priority 2 Tests
For each test, include:
- Test name
  - Scenario:
  - Setup:
  - Assertions:

## Integration / Contract Tests
- Only include if relevant.
- Otherwise say `None needed beyond unit coverage based on current evidence`.

## Fixtures / Mocks Needed
- Data builders, mocks, seeded state, or fake services.

## Regression Risks To Lock Down
- The changes most likely to break later.

## Suggested Execution Order
- The order that would give the fastest confidence signal.
