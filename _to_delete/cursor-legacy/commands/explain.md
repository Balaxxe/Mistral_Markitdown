---
description: Explain the selected code or current file, including purpose, control flow, dependencies, edge cases, and gotchas.
---

## User Input

```text
$ARGUMENTS
```

You MUST consider the user input before proceeding if it is not empty.

## Goal

Explain the target code clearly for an engineer who is new to this part of the system.

Do not modify files. This command explains only.

## Target Selection

Use this order:

1. Selected code, if present
2. Current active file
3. A file, symbol, or topic named in `$ARGUMENTS`
4. Read referenced symbols, imports, callers, and tests only as needed

If no target is available, say `No selection or active file was available to explain.` and stop.

## Instructions

- Start with the practical purpose of the code: what problem it solves.
- Then explain the main control flow and the important symbols involved.
- Describe inputs, outputs, side effects, state changes, and external dependencies.
- Call out invariants, hidden assumptions, tricky branches, and failure modes.
- Distinguish direct observations from inferred intent.
- Avoid generic textbook explanations. Anchor everything to the actual code.
- If several functions or classes are involved, organize the explanation by symbol.
- Honor `$ARGUMENTS` for emphasis, for example `focus on caching`, `explain like onboarding notes`, or `compare with callers`.

## Output Format

## What This Code Does
- One short, concrete explanation.

## How It Works
- Step-by-step control flow or symbol-by-symbol explanation.

## Key Inputs / Outputs / Side Effects
- Important parameters, return values, mutations, I/O, network calls, storage writes, emitted events, or logging.

## Dependencies
- Internal modules, external services, config, feature flags, or environment assumptions.

## Edge Cases and Failure Modes
- Tricky branches, null states, retries, timeouts, races, or validation paths.

## Hidden Assumptions / Gotchas
- Things that are easy to miss when changing this code.

## Where To Change It Safely
- The safest extension points or the places most likely to require coordinated changes.

## 30-Second Summary
- A brief recap in plain English.
