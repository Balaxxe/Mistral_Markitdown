---
name: review
description: Review the current selection or file and produce an actionable code review with severity and evidence. Slash-invoked; does not modify files.
disable-model-invocation: true
---


## User Input

```text
$ARGUMENTS
```

You MUST consider the user input before proceeding if it is not empty.

## Goal

Review the user-specified target, selected code, or current file and produce a concise, actionable review.

Do not modify files or apply fixes. This command reviews only.

## Target Selection

Use this order:

1. If `$ARGUMENTS` identifies an explicit file, symbol, or concrete code scope, use that.
2. Otherwise, review a code selection if present.
3. Otherwise, review the current active file.
4. Read neighboring definitions, callers, and tests only as needed.

Focus modifiers such as `security`, `performance`, `API design`, or `be strict` apply to whichever target this order selects. When they are the only arguments, use the selected code or active file; they do not identify a target on their own.

If no selection, active file, or identifiable target is available, say `No selection or active file was available to review.` and stop.

## Review Criteria

Evaluate only what is relevant to the target:

- Correctness and logic errors
- Edge cases and failure handling
- Data validation and security concerns
- Performance and scalability risks
- State management, concurrency, and idempotency issues
- API and type contract mismatches
- Maintainability and readability problems that affect future changes
- Test coverage gaps

## Instructions

- Focus on defects and meaningful risks, not praise.
- Avoid style-only nits unless they materially affect readability or maintenance.
- For every issue, include severity, why it matters, evidence, and a suggested fix.
- Use the strongest evidence available: file path, symbol name, line range if available, or concrete code behavior.
- If no major issues are found, say so clearly and still note residual risks or missing tests.
- Honor `$ARGUMENTS` as a focus modifier, for example `security`, `performance`, `API design`, or `be strict`.

## Output Format

## Scope Reviewed
- State whether you reviewed a selection, current file, or user-specified target.
- Mention any adjacent files or tests you inspected.

## Findings
For each finding, use this structure:

- `[severity] Short title`
  - Why it matters:
  - Evidence:
  - Suggested fix:

Use these severity labels only: `blocker`, `high`, `medium`, `low`, `nit`.

## Missing or Weak Tests
- List the most important missing test scenarios.
- If coverage looks adequate, say `No major test gaps noted`.

## Review Outcome
For a complete change or PR review, choose exactly one:
- `Ready to merge`
- `Mergeable with follow-ups`
- `Needs changes before merge`

Then add 2 to 4 bullets explaining the decision.

For a selection, file, or other partial scope, use:
- `Scope reviewed; merge readiness not assessed`

Then state whether the reviewed scope has findings and which broader checks were not performed.
