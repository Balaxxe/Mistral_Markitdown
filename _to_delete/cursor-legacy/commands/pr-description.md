---
description: Generate a copy-paste-ready PR description from staged changes, working tree changes, or recent commits.
---

## User Input

```text
$ARGUMENTS
```

You MUST consider the user input before proceeding if it is not empty.

## Goal

Generate a reviewer-friendly pull request description in GitHub-flavored Markdown.

Do not create commits, push branches, open a PR, or modify files. This command only produces the PR text.

## Context Gathering

1. Prefer staged changes first.
   - Inspect the staged diff summary and full staged diff.
2. If nothing is staged, inspect unstaged working tree changes.
3. If the working tree is clean, summarize recent branch changes.
   - Prefer the diff from the current branch to its likely base branch.
   - If that is unavailable, summarize the most recent relevant commits.
4. Read nearby tests, docs, config, and migration files only when needed to understand intent.
5. Treat commit messages as hints, not ground truth.

## Instructions

- Infer the user-visible purpose, key implementation changes, test evidence, risks, and rollout notes from the actual diff.
- Do not invent issue numbers, screenshots, benchmarks, migrations, deployment steps, or tests.
- If evidence is missing, say `Not verified`.
- Group noisy or mixed changes into coherent themes.
- Call out breaking changes, API contract changes, config changes, data migrations, feature flags, permissions changes, and operational steps whenever present.
- Keep the description compact and scannable.
- Honor extra guidance in `$ARGUMENTS`, such as tone, word limit, or reviewer emphasis.

## Output Format

Return exactly these sections in Markdown:

## Summary
- 2 to 4 bullets focused on user-visible or reviewer-relevant outcomes.

## What Changed
- Grouped bullets by theme.

## Testing
- Bullets based only on observed evidence.
- If testing cannot be verified, say `Not verified`.

## Risks / Reviewer Focus
- Bullets for risky areas, edge cases, migrations, or follow-up attention.
- If nothing notable is found, say `None noted`.

## Rollout Notes
- Deployment, config, backfill, flags, or cleanup notes.
- If none are found, say `None noted`.

## Suggested PR Title
- One line only.

## Copy/Paste Version
- Provide the final PR body inside a single fenced Markdown block.

## Quality Bar

- Prefer precise verbs like `adds`, `removes`, `renames`, `tightens`, `fixes`, and `deprecates`.
- Be specific without dumping raw diff details.
- Mention migrations, config changes, and test coverage changes when present.
