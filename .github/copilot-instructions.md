# GitHub Copilot Repository Instructions

Read and follow `/AGENTS.md` as the authoritative repository instructions.

Before proposing or implementing changes:

- identify the active Jira issue;
- keep changes within that issue's scope;
- inspect existing code and tests;
- preserve the architecture documented in `/docs/architecture.md`;
- apply all safeguards in `/docs/research-integrity.md`.

When generating code:

- use precise type annotations;
- keep market-data, strategy, execution, accounting, optimization, and reporting concerns separate;
- make timestamp and price-adjustment semantics explicit;
- never introduce look-ahead bias;
- write or update tests for behavior changes;
- avoid broad refactors unrelated to the active ticket.

Before presenting work as complete, run the checks documented in `/docs/development.md`. Never claim a check passed unless it was executed successfully.

Use Jira-aware branches, commits, and PRs:

- branch: `qf/QF-###-short-description`
- commit: `<type>(<scope>): <description> [QF-###]`
- PR title: `QF-###: Clear imperative summary`

Do not merge pull requests.
