## Jira ticket

- QF-___

## Summary

Describe what changed and why.

## Implementation notes

Explain the important design choices, boundaries, and assumptions.

## Validation

List the exact commands that were executed and their results.

- [ ] Formatting passed
- [ ] Linting passed
- [ ] Type checking passed
- [ ] Relevant tests passed
- [ ] Full test suite passed, or reason documented

Commands:

```text
Add exact commands here
```

## Research-integrity review

- [ ] No future information is used by features, signals, orders, or fills
- [ ] Signal and execution timestamps are explicit
- [ ] Price-adjustment policy is explicit
- [ ] Commission, fee, and slippage assumptions are preserved or documented
- [ ] Train, validation, walk-forward, and holdout boundaries remain isolated
- [ ] Results remain deterministic when expected
- [ ] New feature relationships are treated as hypotheses until out-of-sample validation

Explain any item that is not applicable or any assumption that changed.

## Data and schema impact

Describe changes to:

- canonical schemas;
- persisted artifacts;
- data fingerprints;
- migrations;
- backward compatibility.

Write `None` when there is no impact.

## Risks and limitations

Describe known edge cases, performance concerns, statistical limitations, or operational risks.

## Documentation

- [ ] Relevant documentation was updated
- [ ] An ADR was added or updated for a durable architectural decision
- [ ] No documentation change was required

## Follow-up work

List deferred work or related Jira tickets. Write `None` when complete.
