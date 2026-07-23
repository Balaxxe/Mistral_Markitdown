---
description: Review the current selection or file and produce an actionable code review with severity and evidence.
---

## User Input

```text
$ARGUMENTS
```

You MUST consider the user input before proceeding if it is not empty.

## Goal

Review the selected code, current file, or user-specified target and produce a concise, actionable review.

Do not modify files or apply fixes. This command reviews only.

## Target Selection

Use this order:

1. If a code selection is present, review that selection first.
2. Otherwise review the current active file.
3. Otherwise, if `$ARGUMENTS` names a file, symbol, or review focus, use that.
4. Read neighboring definitions, callers, and tests only as needed.

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

## Approval Summary
Choose exactly one:
- `Ready to merge`
- `Mergeable with follow-ups`
- `Needs changes before merge`

Then add 2 to 4 bullets explaining the decision.
