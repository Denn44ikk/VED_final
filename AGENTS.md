# AGENTS.md

This file defines the working rules for the coding agent in this project.
If the user does not provide a newer direct instruction, follow the rules below.

## 1. Purpose

The agent should help the project grow predictably:

- keep the directory layout consistent;
- create new files only in appropriate places;
- avoid mixing business logic, infrastructure, and utility code;
- keep `PLAN.md` up to date;
- record important architectural decisions inside the repository.

## 2. Default project structure

Until the user approves a different architecture, use this Python-oriented structure:

```text
src/
  app/
    main.py
    core/
    services/
    models/
    schemas/
    utils/
tests/
docs/
scripts/
PLAN.md
AGENTS.md
```

## 3. File placement rules

- The main application entry point should be `src/app/main.py`.
- Business logic should live in `src/app/services/`.
- Domain models should live in `src/app/models/`.
- Validation schemas, DTOs, and request/response contracts should live in `src/app/schemas/`.
- Shared infrastructure and configuration code should live in `src/app/core/`.
- Small reusable helpers may live in `src/app/utils/`.
- Tests should live in `tests/` and should mirror the source structure when practical.
- Development, migration, import, and maintenance scripts should live in `scripts/`.
- Architecture and process documentation should live in `docs/` once it outgrows `AGENTS.md` or `PLAN.md`.

## 4. Rules for creating new files

- Do not create new files in the repository root unless there is a clear reason.
- Before creating a file, check whether the work belongs in an existing module.
- Every new file should have one clear responsibility.
- File names should describe responsibility and should not be vague names like `helper.py`, `temp.py`, or `misc.py`.
- If a new module is created, a matching test should also be added or planned.
- If the task introduces a new subsystem, record it in `PLAN.md` before or during implementation.

## 5. Rules for changing structure

- Do not change the directory structure silently.
- If the current structure is no longer enough, update `PLAN.md` first in `Decisions` or `Next`.
- If the change affects architecture, add a short explanation for why it is needed.
- If a new application layer is introduced, document its purpose in `AGENTS.md` or `docs/architecture.md`.

## 6. Quality rules

- Prefer small cohesive modules over large mixed-responsibility files.
- Avoid duplicated logic.
- Follow consistent naming conventions.
- Do not add temporary files, drafts, or one-off artifacts to the repository without a clear need.
- Add or update tests together with code whenever practical.

## 7. Mandatory PLAN.md workflow

`PLAN.md` is a live working document and must stay current.

Before meaningful work starts, the agent should:

- read `PLAN.md`;
- review the current task context;
- update `In Progress` if a new task has started.

After meaningful work is finished, the agent should:

- move completed items to `Done`;
- update `Next`;
- add newly discovered follow-up tasks to `Backlog`;
- record important decisions and assumptions.

## 8. Required PLAN.md sections

`PLAN.md` should contain:

- `Goal` - what we are building now;
- `Architecture / Structure` - current structure and architectural agreements;
- `Backlog` - known tasks that are not started yet;
- `In Progress` - what is actively being worked on;
- `Done` - completed work;
- `Decisions` - important architectural and process decisions;
- `Next` - the nearest next steps.

## 9. Agent behavior rules

- If the user request conflicts with the agreed structure, first propose the correct placement.
- If the structure is still undefined, use the default structure from this file.
- If the user approves a new structure, update `AGENTS.md` and follow the new version.
- If the task does not fit the current architecture, do not improvise silently; record the change in `PLAN.md`.

## 10. Instruction priority

Priority order:

1. Direct user instruction.
2. This file `AGENTS.md`.
3. Current `PLAN.md`.
4. Standard engineering judgment.
5. Отмечай в `PLAN.md` что уже реализовано, а что ещё нет.
