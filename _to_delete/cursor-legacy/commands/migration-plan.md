---
description: Plan a safe data or schema migration with sequencing, validation, rollout, and rollback steps.
---

## User Input

```text
$ARGUMENTS
```

You MUST consider the user input before proceeding if it is not empty.

## Goal

Produce a migration plan for a schema change, data migration, or storage transition.

Do not modify files or generate the final migration code unless the user explicitly asks for implementation after the plan.

## Context Gathering

Inspect only what is needed to plan safely:

1. Current schema, models, and migration framework
2. Existing migration patterns in the repo
3. Read and write paths affected by the change
4. Downstream consumers, jobs, reports, APIs, and analytics dependencies
5. Data volume, backfill risk, locking risk, and deployment topology when these can be inferred
6. Existing observability, feature flags, backup strategy, and rollback conventions

If key details are missing, continue with clearly labeled assumptions instead of blocking.

## Planning Principles

- Prefer expand-contract or another low-downtime strategy when feasible.
- Separate additive schema changes, application rollout, backfill, cutover, and cleanup.
- Call out risks for long transactions, table rewrites, locks, index creation, nullability changes, uniqueness changes, enum changes, and large backfills.
- Distinguish reversible steps from irreversible ones.
- If a full rollback is not possible, say exactly what can be rolled back and what cannot.
- Include preflight checks, abort criteria, observability, and validation gates.
- Keep the plan specific to this codebase instead of giving generic database advice.

## Output Format

## Objective
- What is changing and why.

## Assumptions
- List assumptions explicitly.

## Impacted Assets
- Tables, collections, models, services, jobs, APIs, dashboards, or other consumers.

## Migration Strategy
Break this into phases as needed:
- Phase 0: Prechecks
- Phase 1: Additive changes
- Phase 2: Application compatibility or dual write/read changes
- Phase 3: Backfill or data movement
- Phase 4: Cutover
- Phase 5: Cleanup

Compress phases only if the migration is truly simple.

## Rollback Plan
For each phase, include:
- Rollback point
- What is reversible
- What is not reversible
- Data recovery or backup requirements
- Decision gate for aborting or continuing

## Validation Plan
- Before migration
- During migration
- After migration

## Test / Dry-Run Plan
- Local or ephemeral environment checks
- Staging plan
- Production safeguards

## Operational Risks
- Highest-risk failure modes and mitigations.

## Suggested Execution Order
- Numbered, end-to-end runbook style steps.

## Open Questions
- Only questions that materially affect safety or sequencing.
