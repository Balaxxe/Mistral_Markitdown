---
description: Produce a scoped refactor plan with affected files, sequencing, risks, and test strategy.
---

## User Input

```text
$ARGUMENTS
```

You MUST consider the user input before proceeding if it is not empty.

## Goal

Design a behavior-preserving refactor plan for the selected code, current file, or user-specified target.

Do not modify files. This command plans only.

## Target Selection

Use this order:

1. Selected code, if present
2. Current active file
3. A file, module, symbol, or refactor goal named in `$ARGUMENTS`
4. Relevant callers, dependencies, and tests only as needed to plan safely

If the target cannot be inferred, say `No refactor target was available.` and stop.

## Planning Principles

- Keep the scope tight and explicit.
- Prefer small, reviewable, behavior-preserving steps.
- Avoid big-bang rewrites unless the user explicitly asks for one.
- Separate mechanical changes from semantic changes.
- Include a file list and explain why each file is likely to change.
- Sequence the work so each step leaves the codebase in a valid, testable state.
- Call out what is intentionally out of scope.
- Prefer 3 to 7 ordered steps unless the codebase complexity clearly requires more.

## Output Format

## Refactor Goal
- One short paragraph.

## Scope / Non-goals
- What is in scope
- What is out of scope

## Current Constraints
- Public APIs, compatibility requirements, generated code, ownership boundaries, performance constraints, or migration concerns.

## Files Likely to Change
For each file, include:
- Path
- Why it changes
- Whether the change is mechanical, structural, or behavioral-risky

## Proposed Sequence
Provide a numbered sequence. For each step include:
- What changes
- Why this order is safest
- What should remain behaviorally unchanged
- A checkpoint to verify before continuing

## Safety Checks
- The invariants and guardrails to preserve during the refactor.

## Test Strategy
- Tests to run before, during, and after the refactor
- Any new tests needed to lock in behavior

## Risks / Trade-offs
- Main refactor risks, alternatives, and why this plan is preferred.

## Suggested PR Breakdown
- One PR or multiple PRs, with rationale.

## Done Criteria
- Clear exit conditions for considering the refactor complete.
