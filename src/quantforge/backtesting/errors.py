"""Backtesting domain exceptions."""


class BacktestError(Exception):
    """Base class for public backtesting failures."""


class InvalidBacktestConfigurationError(BacktestError, ValueError):
    """Raised when execution or cost assumptions are invalid."""


class InvalidMarketDataError(BacktestError, ValueError):
    """Raised when a dataset violates the QF-3 backtesting contract."""


class InvalidSignalError(BacktestError, ValueError):
    """Raised when strategy decisions cannot be consumed safely."""


class ExecutionError(BacktestError):
    """Raised when an eligible order cannot be processed consistently."""


class PortfolioAccountingError(BacktestError):
    """Raised when a cash or position invariant would be violated."""


class ResultExportError(BacktestError):
    """Raised when immutable structured export cannot be completed."""
