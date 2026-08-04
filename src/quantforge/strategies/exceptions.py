"""Stable public exceptions for strategy contracts and decisions."""


class StrategyError(Exception):
    """Base strategy-contract failure."""


class InvalidStrategyParametersError(StrategyError):
    """A strategy parameter set is invalid."""


class MissingRequiredMarketFieldError(StrategyError):
    """Strategy input does not expose a declared market field."""


class UnorderedStrategyInputError(StrategyError):
    """Strategy input sessions are duplicated or not chronological."""


class DuplicateStrategyDecisionError(StrategyError):
    """More than one decision exists for a strategy, symbol, and session."""


class InvalidTargetWeightError(StrategyError):
    """A requested target weight violates the long-only sizing contract."""


class UnsupportedTimingConventionError(StrategyError):
    """A timing rule or exchange calendar cannot be resolved safely."""


class InvalidStrategyOutputError(StrategyError):
    """Strategy output violates ordering, identity, or timing constraints."""
