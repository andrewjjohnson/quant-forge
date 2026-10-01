"""Non-authoritative rapid exploratory strategy scans (QF-72).

EXPLORATORY / NON-AUTHORITATIVE. Rapid scans screen strategy hypotheses over a
permitted QF-8 development or selection window using the authoritative
canonical data, calendar, warm-up, prepared causal features, shared rule kernel
and outcome definitions, while skipping identity, persistence, resume, audit and
publication machinery. Results must be reproduced through the authoritative
QuantForge pipeline before any research conclusion. Rapid scans can never read
a reserved final holdout. See ``docs/rapid-strategy-scan.md``.
"""

from quantforge.rapid.export import RAPID_EXPORT_SUFFIX, export_rapid_scan
from quantforge.rapid.models import (
    NON_AUTHORITATIVE_NOTICE,
    RAPID_SCAN_MODE,
    RAPID_SCAN_SCHEMA_VERSION,
    PromotedStrategyConfiguration,
    RapidAdmissionError,
    RapidBarInput,
    RapidDecisionWindow,
    RapidEvent,
    RapidHoldoutError,
    RapidIndicatorInput,
    RapidOutcomeConfiguration,
    RapidOutcomeSummary,
    RapidOutcomeValues,
    RapidPeriodSummary,
    RapidResearchWindow,
    RapidRuleSpecification,
    RapidScanError,
    RapidScanProfile,
    RapidScanResult,
    RapidScopeError,
    RapidSourceReference,
    RapidStrategyConfiguration,
    promote_strategy_configuration,
)
from quantforge.rapid.session import RapidResearchSession, rapid_research_session

__all__ = [
    "NON_AUTHORITATIVE_NOTICE",
    "RAPID_EXPORT_SUFFIX",
    "RAPID_SCAN_MODE",
    "RAPID_SCAN_SCHEMA_VERSION",
    "PromotedStrategyConfiguration",
    "RapidAdmissionError",
    "RapidBarInput",
    "RapidDecisionWindow",
    "RapidEvent",
    "RapidHoldoutError",
    "RapidIndicatorInput",
    "RapidOutcomeConfiguration",
    "RapidOutcomeSummary",
    "RapidOutcomeValues",
    "RapidPeriodSummary",
    "RapidResearchSession",
    "RapidResearchWindow",
    "RapidRuleSpecification",
    "RapidScanError",
    "RapidScanProfile",
    "RapidScanResult",
    "RapidScopeError",
    "RapidSourceReference",
    "RapidStrategyConfiguration",
    "export_rapid_scan",
    "promote_strategy_configuration",
    "rapid_research_session",
]
