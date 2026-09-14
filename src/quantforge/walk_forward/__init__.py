"""Deterministic rolling/expanding studies ending at typed per-fold OOS artifacts."""

from quantforge.walk_forward.backtest import BacktestEvaluator
from quantforge.walk_forward.models import (
    BacktestOOSArtifact,
    CandidateConfiguration,
    CandidateUniverse,
    FoldResult,
    FoldStatus,
    FrozenSelection,
    OOSArtifact,
    PredictionOOSArtifact,
    SelectionEvidence,
    SelectionPolicy,
    WalkForwardConfig,
    WalkForwardError,
    WalkForwardEvaluator,
    WalkForwardPersistenceError,
    WalkForwardResult,
)
from quantforge.walk_forward.prediction import PredictionEvaluator
from quantforge.walk_forward.study import WalkForwardStudy

__all__ = [
    "BacktestEvaluator",
    "BacktestOOSArtifact",
    "CandidateConfiguration",
    "CandidateUniverse",
    "FoldResult",
    "FoldStatus",
    "FrozenSelection",
    "OOSArtifact",
    "PredictionEvaluator",
    "PredictionOOSArtifact",
    "SelectionEvidence",
    "SelectionPolicy",
    "WalkForwardConfig",
    "WalkForwardError",
    "WalkForwardEvaluator",
    "WalkForwardPersistenceError",
    "WalkForwardResult",
    "WalkForwardStudy",
]
