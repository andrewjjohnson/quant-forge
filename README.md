# QuantForge

QuantForge is a quantitative research and systematic-trading platform focused on
reproducible, auditable experiments. The project is currently establishing the
**MVP Quantitative Research Foundation (QF-1)**. Provider-agnostic adjusted daily
market-data ingestion and reusable causal indicator and engine-neutral strategy
contracts are available. Deterministic single-symbol, long-only daily-bar
backtesting now provides explicit next-open execution, separate commission and
transaction-fee policies, slippage, whole-share accounting, typed metrics, a
buy-and-hold benchmark, and structured result export. Optimization, paper
trading, and live trading are not yet implemented.

## Prerequisites

- Git
- [`uv`](https://docs.astral.sh/uv/)

The project pins Python 3.13 in `.python-version`. `uv` installs the matching
interpreter and locked development dependencies when needed.

## Setup

Clone the repository and run the one-command clean-checkout installation:

```bash
git clone <repository-url>
cd quant-forge
uv sync --all-extras
```

The committed `uv.lock` makes dependency resolution reproducible. Do not commit
the generated `.venv` directory.

## Quality checks

```bash
# Format source files (mutating)
uv run ruff format .

# Verify formatting and lint rules
uv run ruff format --check .
uv run ruff check .

# Type-check the package and tests
uv run pyright

# Run all local tests
uv run pytest

# Select future optional integration tests explicitly
uv run pytest -m integration
```

Ordinary tests are deterministic and must not require network access. See
[`docs/market-data.md`](docs/market-data.md) for adjusted daily data usage and
the explicitly opted-in Alpha Vantage SPY verification. See
[`docs/strategy-contracts.md`](docs/strategy-contracts.md) for indicator,
strategy, signal-timing, and sizing-intent usage. See
[`docs/backtesting.md`](docs/backtesting.md) for execution, accounting, metric,
benchmark, and export semantics.

## Repository layout

```text
src/quantforge/    Importable, typed Python package
tests/unit/        Fast isolated tests
tests/integration/ Optional component-boundary tests
scripts/           Maintained developer and operational scripts
data/              Local downloaded data (ignored by Git)
reports/           Local generated reports (ignored by Git)
docs/              Architecture, development, and integrity guidance
```

## Environment variables

Create a local settings file only when an integration requires it:

```bash
cp .env.example .env
```

The example contains names and empty placeholders only. `.env` files are ignored;
never commit API keys, broker credentials, account identifiers, or market data.
The initial package does not read environment variables at runtime.

## Contributing

1. Start from the latest `main` branch.
2. Work on one Jira ticket using `qf/QF-###-short-description`.
3. Run every quality check listed above.
4. Commit with Conventional Commits and the ticket key, for example
   `chore(repo): initialize tooling [QF-2]`.
5. Push the branch and open a draft pull request targeting `main` using the
   repository template.

Read `AGENTS.md`, `docs/architecture.md`, and `docs/research-integrity.md` before
making changes. Preserve time-series correctness and reproducibility even in
early research work.

## Disclaimer

QuantForge is research software. It does not provide financial advice, promise
investment performance, or currently execute trades. Historical or hypothetical
results must not be interpreted as guarantees of future performance.
