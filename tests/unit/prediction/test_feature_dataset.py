import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import MarketDataset
from quantforge.indicators import Indicator
from quantforge.prediction import (
    AtrPercentageContext,
    ContextualFeature,
    ForwardReturnEvaluator,
    ForwardReturnOutcomeLabeler,
    ForwardReturnValues,
    InvalidPredictionOutputError,
    OutcomeLabel,
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
    OvernightGapSignalFeatureRule,
    PredictionDirection,
    PredictionOutcome,
    PredictionRule,
    PredictionStudy,
    PredictionStudyOutcome,
    SchemaField,
    SchemaFieldCategory,
    SignalDisposition,
    SignalFeatureCandidate,
    SignalFeatureCandidateOutput,
    SignalFeatureDatasetError,
    SignalFeatureDatasetResult,
    SignalFeaturePersistenceError,
    SignalFeatureRow,
    SignalFeatureValue,
    TrendDistanceContext,
    VolumeRatioContext,
    build_signal_feature_dataset,
    excursion_outcome,
    forward_return_outcome,
    target_stop_outcome,
)
from quantforge.prediction import feature_dataset as feature_dataset_module

from ..helpers import make_dataset

_FIXTURE_SOURCE_CONFIG_ID = configuration_identity(
    {
        "component_name": "fixture_source_rule",
        "implementation_version": "1",
    }
)


@dataclass(frozen=True, slots=True)
class FixtureParameters:
    mode: str = "fixture"

    def to_primitive(self) -> PrimitiveMapping:
        return {"mode": self.mode}


class FixtureCandidateRule:
    name = "fixture_candidates"
    implementation_version = "1"
    required_indicators: tuple[Indicator, ...] = ()
    warm_up_observations = 1

    def __init__(self, dispositions: tuple[SignalDisposition, ...]) -> None:
        self._dispositions = dispositions
        self._parameters = FixtureParameters()
        self.generate_calls = 0

    @property
    def parameters(self) -> FixtureParameters:
        return self._parameters

    @property
    def strategy_feature_definitions(self) -> tuple[SchemaField, ...]:
        return (
            SchemaField(
                "decision_close",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "decimal",
                "price_per_share",
                False,
                "fixture completed close",
                "after the signal-session close",
            ),
        )

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_strategy",
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": self.parameters.to_primitive(),
            "required_indicators": [],
            "warm_up_observations": self.warm_up_observations,
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def generate(self, dataset: MarketDataset) -> SignalFeatureCandidateOutput:
        self.generate_calls += 1
        signals = tuple(
            SignalFeatureCandidate(
                symbol=dataset.metadata.canonical_symbol,
                signal_session=bar.session_date,
                strategy_id=self.name,
                strategy_implementation_version=self.implementation_version,
                strategy_configuration_id=self.configuration_id,
                source_rule_id="fixture_source_rule",
                source_rule_implementation_version="1",
                source_rule_configuration_id=_FIXTURE_SOURCE_CONFIG_ID,
                strategy_parameters=PrimitiveMappingSnapshot.capture(
                    self.parameters.to_primitive()
                ),
                disposition=disposition,
                reason_codes=(f"{disposition.value}_reason",),
                explanation=f"fixture {disposition.value}",
                direction=(
                    None
                    if disposition is SignalDisposition.REJECTED
                    else PredictionDirection.UP
                ),
                selected_rule_reason=(
                    None if disposition is SignalDisposition.REJECTED else "fixture_up"
                ),
                matched_rule_reasons=(
                    () if disposition is SignalDisposition.REJECTED else ("fixture_up",)
                ),
                strategy_features=(SignalFeatureValue("decision_close", bar.close),),
            )
            for bar, disposition in zip(dataset.bars, self._dispositions, strict=False)
        )
        return SignalFeatureCandidateOutput(
            self.name,
            self.configuration_id,
            dataset.metadata.dataset_id,
            signals,
        )


@dataclass(frozen=True, slots=True)
class LastCloseContext:
    version: int = 1
    unit: str = "price_per_share"
    name: str = "last_close_context"

    @property
    def definition(self) -> SchemaField:
        return SchemaField(
            self.name,
            SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
            "decimal",
            self.unit,
            False,
            "last close in the supplied causal history",
            "after the signal-session close",
        )

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "signal_contextual_feature",
            "implementation_version": str(self.version),
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def value_from_history(self, history: MarketDataset) -> Decimal:
        assert history.metadata.actual_last_session == history.bars[-1].session_date
        assert history.metadata.requested_end == history.bars[-1].session_date
        return history.bars[-1].close


class AlternatingDefinitionContext:
    name = "alternating_definition_context"

    def __init__(self) -> None:
        self.definition_reads = 0

    @property
    def definition(self) -> SchemaField:
        self.definition_reads += 1
        unit = "price_per_share" if self.definition_reads == 1 else "ratio"
        return SchemaField(
            self.name,
            SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
            "decimal",
            unit,
            False,
            "last close with a definition that changes on repeated access",
            "after the signal-session close",
        )

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "signal_contextual_feature",
            "implementation_version": "1",
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def value_from_history(self, history: MarketDataset) -> Decimal:
        return history.bars[-1].close


class ProvenanceReadingContext:
    name = "provenance_reading_context"

    @property
    def definition(self) -> SchemaField:
        return SchemaField(
            self.name,
            SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
            "decimal",
            "deterministic_test_signature",
            False,
            "signature of provenance visible to a custom causal callback",
            "after the signal-session close",
        )

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "signal_contextual_feature",
            "implementation_version": "1",
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def value_from_history(self, history: MarketDataset) -> Decimal:
        metadata = history.metadata
        provenance = (
            metadata.retrieved_at.isoformat(),
            metadata.raw_location,
            metadata.normalized_location,
            metadata.corporate_actions_location,
            metadata.raw_sha256,
            metadata.data_sha256,
            metadata.dataset_id,
            metadata.corporate_action_snapshot_id,
            str(metadata.corporate_actions_complete),
            metadata.adapter_version,
            *(
                value
                for action in history.corporate_actions
                for value in (action.action_id, action.source_dataset_id)
            ),
        )
        encoded = "|".join(provenance)
        return Decimal(
            sum((index + 1) * ord(character) for index, character in enumerate(encoded))
        )


@dataclass(frozen=True, slots=True)
class FutureReadingContext(LastCloseContext):
    name: str = "future_reading_context"

    def value_from_history(self, history: MarketDataset) -> Decimal:
        return history.bars[1].close


@dataclass(frozen=True, slots=True)
class FutureAlignedContext(LastCloseContext):
    name: str = "future_aligned_context"

    def values_for_dataset(self, dataset: MarketDataset) -> tuple[Decimal, ...]:
        return tuple(
            dataset.bars[min(index + 1, len(dataset.bars) - 1)].close
            for index in range(len(dataset.bars))
        )


class RestoringConfigurationContext:
    name = "restoring_configuration_context"

    def __init__(self) -> None:
        self.version = 1
        self.calculation_count = 0

    @property
    def definition(self) -> SchemaField:
        return SchemaField(
            self.name,
            SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
            "decimal",
            "price_per_share",
            False,
            "last close after a mutable configuration callback",
            "after the signal-session close",
        )

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "signal_contextual_feature",
            "implementation_version": str(self.version),
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def value_from_history(self, history: MarketDataset) -> Decimal:
        self.calculation_count += 1
        self.version = 2 if self.calculation_count == 1 else 1
        return history.bars[-1].close


class CachedConfigurationContext(LastCloseContext):
    __slots__ = ("_cached_configuration_id",)

    def __init__(self) -> None:
        object.__setattr__(self, "version", 1)
        object.__setattr__(self, "unit", "price_per_share")
        object.__setattr__(self, "name", "cached_configuration_context")
        object.__setattr__(
            self,
            "_cached_configuration_id",
            configuration_identity(self.configuration()),
        )

    @property
    def configuration_id(self) -> str:
        return self._cached_configuration_id

    def value_from_history(self, history: MarketDataset) -> Decimal:
        object.__setattr__(self, "version", 2)
        return history.bars[-1].close


@dataclass(frozen=True, slots=True)
class MiscategorizedContext(LastCloseContext):
    name: str = "miscategorized_context"

    @property
    def definition(self) -> SchemaField:
        return replace(
            super().definition,
            category=SchemaFieldCategory.FUTURE_OUTCOME,
        )


@dataclass(frozen=True, slots=True)
class NullNonNullableContext:
    name: str = "null_non_nullable_context"

    @property
    def definition(self) -> SchemaField:
        return SchemaField(
            self.name,
            SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
            "decimal",
            "ratio",
            False,
            "fixture null contextual feature",
            "after the signal-session close",
        )

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "signal_contextual_feature",
            "implementation_version": "1",
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def value_from_history(self, history: MarketDataset) -> Decimal | None:
        del history
        return None


class ExtraFeatureCandidateRule(FixtureCandidateRule):
    def generate(self, dataset: MarketDataset) -> SignalFeatureCandidateOutput:
        output = super().generate(dataset)
        signals = tuple(
            replace(
                signal,
                strategy_features=(
                    *signal.strategy_features,
                    SignalFeatureValue("undeclared_feature", Decimal(1)),
                ),
            )
            for signal in output.signals
        )
        return replace(output, signals=signals)


class MiscategorizedFeatureRule(FixtureCandidateRule):
    @property
    def strategy_feature_definitions(self) -> tuple[SchemaField, ...]:
        return tuple(
            replace(field, category=SchemaFieldCategory.FUTURE_OUTCOME)
            for field in super().strategy_feature_definitions
        )


class InvalidFeatureCandidateRule(FixtureCandidateRule):
    def __init__(
        self,
        dispositions: tuple[SignalDisposition, ...],
        invalid_value: str | None,
    ) -> None:
        super().__init__(dispositions)
        self._invalid_value = invalid_value

    def generate(self, dataset: MarketDataset) -> SignalFeatureCandidateOutput:
        output = super().generate(dataset)
        signals = tuple(
            replace(
                signal,
                strategy_features=(
                    SignalFeatureValue("decision_close", self._invalid_value),
                ),
            )
            for signal in output.signals
        )
        return replace(output, signals=signals)


class EmptyNullableStringFeatureRule(FixtureCandidateRule):
    @property
    def strategy_feature_definitions(self) -> tuple[SchemaField, ...]:
        return (
            *super().strategy_feature_definitions,
            SchemaField(
                "optional_note",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "string",
                "text",
                True,
                "optional fixture note",
                "after the signal-session close",
            ),
        )

    def generate(self, dataset: MarketDataset) -> SignalFeatureCandidateOutput:
        output = super().generate(dataset)
        signals = tuple(
            replace(
                signal,
                strategy_features=(
                    *signal.strategy_features,
                    SignalFeatureValue("optional_note", ""),
                ),
            )
            for signal in output.signals
        )
        return replace(output, signals=signals)


class PrepopulatedContextCandidateRule(FixtureCandidateRule):
    def generate(self, dataset: MarketDataset) -> SignalFeatureCandidateOutput:
        output = super().generate(dataset)
        signals = tuple(
            replace(
                signal,
                contextual_features=(
                    SignalFeatureValue("undeclared_context", Decimal("1")),
                ),
            )
            for signal in output.signals
        )
        return replace(output, signals=signals)


class InvalidSourceConfigurationCandidateRule(FixtureCandidateRule):
    def generate(self, dataset: MarketDataset) -> SignalFeatureCandidateOutput:
        output = super().generate(dataset)
        signals = tuple(
            replace(signal, source_rule_configuration_id="not-a-sha256")
            for signal in output.signals
        )
        return replace(output, signals=signals)


class InvalidDispositionMetadataRule(FixtureCandidateRule):
    def generate(self, dataset: MarketDataset) -> SignalFeatureCandidateOutput:
        output = super().generate(dataset)
        signals = tuple(
            replace(signal, explanation=cast(str, 123)) for signal in output.signals
        )
        return replace(output, signals=signals)


class RegeneratingCandidateRule(FixtureCandidateRule):
    regenerate_differently = False

    def generate(self, dataset: MarketDataset) -> SignalFeatureCandidateOutput:
        output = super().generate(dataset)
        if not self.regenerate_differently or not output.signals:
            return output
        first = replace(
            output.signals[0],
            strategy_features=(SignalFeatureValue("decision_close", Decimal("999")),),
        )
        return replace(output, signals=(first, *output.signals[1:]))


class AlternatingConfigurationCandidateRule(FixtureCandidateRule):
    def __init__(self, dispositions: tuple[SignalDisposition, ...]) -> None:
        super().__init__(dispositions)
        self.configuration_reads = 0

    def _configuration_for_version(self, version: str) -> PrimitiveMapping:
        configuration = super().configuration()
        configuration["fixture_configuration_version"] = version
        return configuration

    @property
    def configuration_id(self) -> str:
        version = "1" if self.configuration_reads == 0 else "2"
        return configuration_identity(self._configuration_for_version(version))

    def configuration(self) -> PrimitiveMapping:
        self.configuration_reads += 1
        version = "1" if self.configuration_reads == 1 else "2"
        return self._configuration_for_version(version)


class CountingForwardReturnOutcomeLabeler(ForwardReturnOutcomeLabeler):
    name = "counting_forward_close_return"

    def __init__(self, horizon_sessions: int) -> None:
        super().__init__(horizon_sessions)
        self.label_calls = 0
        self.validate_calls = 0

    def validate_dataset(self, dataset: MarketDataset) -> None:
        self.validate_calls += 1
        super().validate_dataset(dataset)

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[ForwardReturnValues] | None:
        self.label_calls += 1
        return super().label(dataset, signal_session)


class DirectConfiguredOutcome:
    def __init__(
        self,
        delegate: PredictionStudyOutcome[ForwardReturnValues, ForwardReturnValues],
        malformed_source: str,
    ) -> None:
        self._delegate = delegate
        self._malformed_source = malformed_source
        self._implementation_version = 1
        self._run_count = 0

    @property
    def namespace(self) -> str:
        return self._delegate.namespace

    @property
    def fields(self) -> tuple[SchemaField, ...]:
        return self._delegate.fields

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    @property
    def labeler_configuration_id(self) -> str:
        return self._delegate.labeler_configuration_id

    @property
    def evaluator_configuration_id(self) -> str:
        return self._delegate.evaluator_configuration_id

    def configuration(self) -> PrimitiveMapping:
        return {
            "component": "direct_configured_outcome_fixture",
            "implementation_version": str(self._implementation_version),
            "namespace": self.namespace,
        }

    def unavailable_row(self) -> PrimitiveMapping:
        values = self._delegate.unavailable_row()
        if self._malformed_source == "unavailable":
            values["raw_return"] = "not-a-decimal"
        if self._malformed_source == "availability":
            values["available"] = True
        return values

    def run(
        self,
        dataset: MarketDataset,
        strategy: PredictionRule[SignalFeatureCandidate],
        feature_configuration: PrimitiveMapping,
    ) -> feature_dataset_module.OutcomeRun:
        outcome_run = self._delegate.run(dataset, strategy, feature_configuration)
        values_by_session = dict(outcome_run.values_by_session)
        for signal in strategy.generate(dataset).signals:
            values_by_session.setdefault(signal.signal_session, self.unavailable_row())
        outcome_run = feature_dataset_module.OutcomeRun(
            outcome_run.study_id, values_by_session
        )
        if self._malformed_source == "configuration":
            self._run_count += 1
            self._implementation_version = 2 if self._run_count == 1 else 1
        if self._malformed_source == "resume_configuration":
            self._run_count += 1
            if self._run_count == 2:
                self._implementation_version = 2
        if self._malformed_source == "feature_configuration":
            feature_configuration["feature_schema_version"] = "mutated"
        if self._malformed_source == "study_id":
            return feature_dataset_module.OutcomeRun(
                "not-a-canonical-study-id", outcome_run.values_by_session
            )
        if self._malformed_source == "canonical_study_id":
            return feature_dataset_module.OutcomeRun(
                "0" * 64, outcome_run.values_by_session
            )
        if self._malformed_source == "session" and outcome_run.values_by_session:
            values_by_session = dict(outcome_run.values_by_session)
            first_session = next(iter(values_by_session))
            first_values = values_by_session.pop(first_session)
            values_by_session[first_session + timedelta(days=1)] = first_values
            return feature_dataset_module.OutcomeRun(
                outcome_run.study_id, values_by_session
            )
        if self._malformed_source == "missing_session" and values_by_session:
            values_by_session.pop(next(iter(values_by_session)))
            return feature_dataset_module.OutcomeRun(
                outcome_run.study_id, values_by_session
            )
        if self._malformed_source != "run" or not outcome_run.values_by_session:
            return outcome_run
        values_by_session = {
            signal_session: dict(values)
            for signal_session, values in outcome_run.values_by_session.items()
        }
        first_values = values_by_session[next(iter(values_by_session))]
        first_values["raw_return"] = "not-a-decimal"
        return feature_dataset_module.OutcomeRun(
            outcome_run.study_id, values_by_session
        )


class MiscategorizedDirectOutcome(DirectConfiguredOutcome):
    @property
    def fields(self) -> tuple[SchemaField, ...]:
        return tuple(
            replace(field, category=SchemaFieldCategory.CONTEMPORANEOUS_FEATURE)
            for field in super().fields
        )


class AlternatingFieldsDirectOutcome(DirectConfiguredOutcome):
    def __init__(
        self,
        delegate: PredictionStudyOutcome[ForwardReturnValues, ForwardReturnValues],
    ) -> None:
        super().__init__(delegate, "valid")
        self.field_reads = 0

    @property
    def fields(self) -> tuple[SchemaField, ...]:
        self.field_reads += 1
        fields = super().fields
        if self.field_reads == 1:
            return fields
        return tuple(
            replace(field, unit="percentage_points")
            if field.name == "raw_return"
            else field
            for field in fields
        )


class AlternatingConfigurationDirectOutcome(DirectConfiguredOutcome):
    def __init__(
        self,
        delegate: PredictionStudyOutcome[ForwardReturnValues, ForwardReturnValues],
    ) -> None:
        super().__init__(delegate, "valid")
        self.configuration_reads = 0
        self._declared_configuration: PrimitiveMapping = {
            "component": "alternating_configuration_outcome_fixture",
            "implementation_version": "1",
            "namespace": self.namespace,
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self._declared_configuration)

    def configuration(self) -> PrimitiveMapping:
        self.configuration_reads += 1
        configuration = dict(self._declared_configuration)
        if self.configuration_reads == 2:
            configuration["implementation_version"] = "2"
        return configuration


@dataclass(frozen=True, slots=True)
class NullRawReturnValues:
    source: ForwardReturnValues

    def to_primitive(self) -> PrimitiveMapping:
        values = self.source.to_primitive()
        values["raw_return"] = None
        return values


@dataclass(frozen=True, slots=True)
class InvalidRawReturnValues:
    source: ForwardReturnValues

    def to_primitive(self) -> PrimitiveMapping:
        values = self.source.to_primitive()
        values["raw_return"] = "not-a-decimal"
        return values


class NullRawReturnEvaluator:
    name = "null_raw_return_fixture_evaluator"
    implementation_version = "1"
    result_schema_version = "1"

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_evaluator",
            "implementation_version": self.implementation_version,
            "result_schema_version": self.result_schema_version,
        }

    def evaluate(
        self,
        signal: SignalFeatureCandidate,
        outcome: PredictionOutcome[ForwardReturnValues],
    ) -> NullRawReturnValues:
        del signal
        return NullRawReturnValues(outcome.values)


class InvalidRawReturnEvaluator:
    name = "invalid_raw_return_fixture_evaluator"
    implementation_version = "1"
    result_schema_version = "1"

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_evaluator",
            "implementation_version": self.implementation_version,
            "result_schema_version": self.result_schema_version,
        }

    def evaluate(
        self,
        signal: SignalFeatureCandidate,
        outcome: PredictionOutcome[ForwardReturnValues],
    ) -> InvalidRawReturnValues:
        del signal
        return InvalidRawReturnValues(outcome.values)


def _fixture_study(
    rule: FixtureCandidateRule,
) -> tuple[
    PredictionStudy[SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues],
    PredictionStudyOutcome[ForwardReturnValues, ForwardReturnValues],
]:
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(
        rule,
        primary.labeler,
        primary.evaluator,
        feature_configuration={"feature_schema_version": "1"},
    )
    return study, primary


def _build_fixture(
    dataset: MarketDataset,
    rule: FixtureCandidateRule,
    output_root: Path,
    *,
    context: ContextualFeature | None = None,
) -> SignalFeatureDatasetResult:
    study, primary = _fixture_study(rule)
    return build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=() if context is None else (context,),
        outcomes=(primary,),
        output_root=output_root,
        chunk_size=2,
    )


def test_dataset_preserves_all_dispositions_and_labels_rejected_rows(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102", "101", "104"))
    rule = FixtureCandidateRule(
        (
            SignalDisposition.ACCEPTED,
            SignalDisposition.REJECTED,
            SignalDisposition.BLOCKED,
            SignalDisposition.OVERLAPPING,
        )
    )

    result = _build_fixture(dataset, rule, tmp_path)
    rows = tuple(row.to_primitive() for row in result.rows)

    assert result.summary.to_primitive() == {
        "accepted_count": 1,
        "blocked_count": 1,
        "candidate_count": 4,
        "overlapping_count": 1,
        "rejected_count": 1,
    }
    row_candidate_ids = tuple(row["candidate_id"] for row in rows)
    assert all(isinstance(value, str) for value in row_candidate_ids)
    assert len(set(cast(tuple[str, ...], row_candidate_ids))) == 4
    assert rows[1]["signal_disposition"] == "rejected"
    assert rows[1]["outcome_forward_return_1_available"] is True
    assert rows[1]["outcome_forward_return_1_raw_return"] == (
        "-0.0098039215686274509803921568627451"
    )
    assert rows[-1]["outcome_forward_return_1_available"] is False
    assert rows[-1]["outcome_forward_return_1_raw_return"] is None


def test_schema_documents_every_column_and_exports_flat_features(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tuple(str(100 + index % 3) for index in range(15)))
    parameters = OvernightGapPredictionParameters(excluded_weekdays=(4,))
    rule = OvernightGapSignalFeatureRule(parameters)
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)

    result = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(
            AtrPercentageContext(2),
            TrendDistanceContext(2),
            VolumeRatioContext(2),
        ),
        outcomes=(
            primary,
            excursion_outcome(2),
            target_stop_outcome(2, Decimal("0.01"), Decimal("0.005")),
        ),
        output_root=tmp_path,
        chunk_size=2,
    )
    destination = tmp_path / result.dataset_id

    assert all(
        field.name
        and field.unit
        and field.calculation_or_source
        and field.temporal_availability
        for field in result.schema.fields
    )
    assert result.schema.column_names == tuple(
        field.name for field in result.schema.fields
    )
    assert {
        "feature_atr_percentage_of_close",
        "feature_volume_ratio",
        "feature_trend_distance_percentage",
        "feature_rsi",
        "feature_previous_rsi",
        "feature_adx",
        "feature_previous_adx",
        "feature_plus_di",
        "feature_previous_plus_di",
        "feature_minus_di",
        "feature_previous_minus_di",
        "feature_signal_weekday",
        "feature_open",
        "feature_close",
    }.issubset(result.schema.column_names)
    assert (destination / "manifest.json").is_file()
    assert (destination / "features.csv").is_file()
    assert (destination / "schema.json").is_file()
    assert (destination / "summary.json").is_file()
    assert (destination / "rows").is_dir()


def test_outcome_study_setup_is_reused_across_checkpoint_chunks(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102", "101", "104"))
    rule = FixtureCandidateRule(
        (
            SignalDisposition.ACCEPTED,
            SignalDisposition.REJECTED,
            SignalDisposition.BLOCKED,
            SignalDisposition.OVERLAPPING,
        )
    )
    template = forward_return_outcome(1)
    labeler = CountingForwardReturnOutcomeLabeler(1)
    primary = PredictionStudyOutcome[ForwardReturnValues, ForwardReturnValues].create(
        "forward_return_1",
        labeler,
        template.evaluator,
        template.fields,
        unavailable_values=template.unavailable_row(),
    )
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, labeler, template.evaluator)

    result = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(primary,),
        output_root=tmp_path,
        chunk_size=1,
    )

    assert len(result.rows) == 4
    assert labeler.validate_calls == 1
    assert labeler.label_calls == 4


def test_fixed_candidate_replay_identity_binds_full_population(tmp_path: Path) -> None:
    dataset = make_dataset(("100", "102", "101"))
    first_rule = FixtureCandidateRule(
        (SignalDisposition.ACCEPTED, SignalDisposition.REJECTED)
    )
    second_rule = FixtureCandidateRule(
        (SignalDisposition.ACCEPTED, SignalDisposition.BLOCKED)
    )
    first_study, first_outcome = _fixture_study(first_rule)
    second_study, second_outcome = _fixture_study(second_rule)

    first = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=first_study,
        contextual_features=(),
        outcomes=(first_outcome,),
        output_root=tmp_path / "first",
        chunk_size=1,
    )
    second = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=second_study,
        contextual_features=(),
        outcomes=(second_outcome,),
        output_root=tmp_path / "second",
        chunk_size=1,
    )

    assert first_rule.configuration_id == second_rule.configuration_id
    assert first.prediction_study_ids != second.prediction_study_ids
    assert all(
        row.to_primitive()["prediction_study_ids"]
        == {"forward_return_1": first.prediction_study_ids[0]}
        for row in first.rows
    )


def test_complete_resume_performs_no_new_generation(tmp_path: Path) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = FixtureCandidateRule(
        (SignalDisposition.ACCEPTED, SignalDisposition.REJECTED)
    )

    first = _build_fixture(dataset, rule, tmp_path, context=LastCloseContext())
    calls_after_first = rule.generate_calls
    second = _build_fixture(dataset, rule, tmp_path, context=LastCloseContext())

    assert rule.generate_calls == calls_after_first
    assert first.to_primitive() == second.to_primitive()


def test_empty_candidate_dataset_exports_and_resumes(tmp_path: Path) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = FixtureCandidateRule(())

    first = _build_fixture(dataset, rule, tmp_path)
    calls_after_first = rule.generate_calls
    second = _build_fixture(dataset, rule, tmp_path)

    assert not first.rows
    assert first.summary.candidate_count == 0
    assert first.prediction_study_ids
    assert rule.generate_calls == calls_after_first
    assert second.to_primitive() == first.to_primitive()


def test_empty_completed_resume_rechecks_direct_outcome_configuration(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = FixtureCandidateRule(())
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    configured_outcome = DirectConfiguredOutcome(primary, "resume_configuration")

    first = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(configured_outcome,),
        output_root=tmp_path,
    )
    assert not first.rows

    with pytest.raises(
        InvalidPredictionOutputError,
        match="configured outcome configuration changed during execution",
    ):
        build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(configured_outcome,),
            output_root=tmp_path,
        )


def test_empty_candidate_resume_rejects_corrupt_manifest_study_ids(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = FixtureCandidateRule(())
    result = _build_fixture(dataset, rule, tmp_path)
    manifest_path = tmp_path / result.dataset_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prediction_study_ids"] = ["corrupt-study-id"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(SignalFeaturePersistenceError, match="metadata does not match"):
        _build_fixture(dataset, rule, tmp_path)


def test_interrupted_generation_resumes_to_uninterrupted_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(("100", "102", "101", "104"))
    dispositions = (
        SignalDisposition.ACCEPTED,
        SignalDisposition.REJECTED,
        SignalDisposition.BLOCKED,
        SignalDisposition.OVERLAPPING,
    )
    rule = FixtureCandidateRule(dispositions)
    original_persist = cast(
        Callable[[Path, SignalFeatureRow], None],
        getattr(feature_dataset_module, "_persist_progress_row"),
    )
    persisted = 0

    def interrupt_after_first(destination: Path, row: SignalFeatureRow) -> None:
        nonlocal persisted
        original_persist(destination, row)
        persisted += 1
        if persisted == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        feature_dataset_module, "_persist_progress_row", interrupt_after_first
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _build_fixture(dataset, rule, tmp_path / "resumed")
    monkeypatch.setattr(
        feature_dataset_module, "_persist_progress_row", original_persist
    )

    resumed = _build_fixture(dataset, rule, tmp_path / "resumed")
    uninterrupted = _build_fixture(
        dataset, FixtureCandidateRule(dispositions), tmp_path / "uninterrupted"
    )

    assert resumed.to_primitive() == uninterrupted.to_primitive()
    assert (
        tmp_path / "resumed" / resumed.dataset_id / "features.csv"
    ).read_bytes() == (
        tmp_path / "uninterrupted" / uninterrupted.dataset_id / "features.csv"
    ).read_bytes()


def test_interrupted_startup_without_manifest_restarts_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = FixtureCandidateRule(
        (SignalDisposition.ACCEPTED, SignalDisposition.REJECTED)
    )
    original_atomic_json = cast(
        Callable[[Path, PrimitiveMapping], None],
        getattr(feature_dataset_module, "_atomic_json"),
    )

    def interrupt_before_manifest(path: Path, values: PrimitiveMapping) -> None:
        if path.name == "manifest.json" and values.get("status") == "in_progress":
            raise RuntimeError("simulated startup interruption")
        original_atomic_json(path, values)

    monkeypatch.setattr(
        feature_dataset_module, "_atomic_json", interrupt_before_manifest
    )
    with pytest.raises(RuntimeError, match="simulated startup interruption"):
        _build_fixture(dataset, rule, tmp_path)
    monkeypatch.setattr(feature_dataset_module, "_atomic_json", original_atomic_json)

    result = _build_fixture(dataset, rule, tmp_path)
    destination = tmp_path / result.dataset_id

    assert len(result.rows) == 2
    assert (
        json.loads((destination / "manifest.json").read_text(encoding="utf-8"))[
            "status"
        ]
        == "complete"
    )


def test_resume_rejects_a_different_regenerated_causal_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = RegeneratingCandidateRule(
        (SignalDisposition.ACCEPTED, SignalDisposition.REJECTED)
    )
    original_persist = cast(
        Callable[[Path, SignalFeatureRow], None],
        getattr(feature_dataset_module, "_persist_progress_row"),
    )

    def interrupt_after_first(destination: Path, row: SignalFeatureRow) -> None:
        original_persist(destination, row)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        feature_dataset_module, "_persist_progress_row", interrupt_after_first
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _build_fixture(dataset, rule, tmp_path)
    monkeypatch.setattr(
        feature_dataset_module, "_persist_progress_row", original_persist
    )
    rule.regenerate_differently = True

    with pytest.raises(
        SignalFeaturePersistenceError, match="regenerated causal candidate"
    ):
        _build_fixture(dataset, rule, tmp_path)


def test_builder_rejects_strategy_configuration_changed_after_snapshot(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102"))
    rule = AlternatingConfigurationCandidateRule((SignalDisposition.ACCEPTED,))

    with pytest.raises(
        SignalFeatureDatasetError,
        match="prediction strategy configuration identity is invalid",
    ):
        _build_fixture(dataset, rule, tmp_path)

    assert rule.configuration_reads == 1
    assert rule.generate_calls == 0
    assert not tuple(tmp_path.iterdir())


def test_partial_resume_evaluates_outcomes_only_for_missing_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(("100", "102", "101", "104"))
    rule = FixtureCandidateRule(
        (
            SignalDisposition.ACCEPTED,
            SignalDisposition.REJECTED,
            SignalDisposition.BLOCKED,
            SignalDisposition.OVERLAPPING,
        )
    )
    template = forward_return_outcome(1)
    labeler = CountingForwardReturnOutcomeLabeler(1)
    primary = PredictionStudyOutcome[ForwardReturnValues, ForwardReturnValues].create(
        "forward_return_1",
        labeler,
        template.evaluator,
        template.fields,
        unavailable_values={"available": False, "horizon_sessions": 1},
    )
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, labeler, template.evaluator)

    def build() -> SignalFeatureDatasetResult:
        return build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(primary,),
            output_root=tmp_path,
            chunk_size=2,
        )

    original_persist = cast(
        Callable[[Path, SignalFeatureRow], None],
        getattr(feature_dataset_module, "_persist_progress_row"),
    )

    def interrupt_after_first(destination: Path, row: SignalFeatureRow) -> None:
        original_persist(destination, row)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        feature_dataset_module, "_persist_progress_row", interrupt_after_first
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build()
    monkeypatch.setattr(
        feature_dataset_module, "_persist_progress_row", original_persist
    )
    calls_after_interruption = labeler.label_calls

    result = build()

    assert calls_after_interruption == 2
    assert labeler.label_calls - calls_after_interruption == 3
    assert len(result.rows) == 4


def test_resume_rejects_valid_json_checkpoint_payload_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = FixtureCandidateRule(
        (SignalDisposition.ACCEPTED, SignalDisposition.REJECTED)
    )
    original_persist = cast(
        Callable[[Path, SignalFeatureRow], None],
        getattr(feature_dataset_module, "_persist_progress_row"),
    )

    def interrupt_after_first(destination: Path, row: SignalFeatureRow) -> None:
        original_persist(destination, row)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        feature_dataset_module, "_persist_progress_row", interrupt_after_first
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _build_fixture(dataset, rule, tmp_path)
    monkeypatch.setattr(
        feature_dataset_module, "_persist_progress_row", original_persist
    )

    destination = next(path for path in tmp_path.iterdir() if path.is_dir())
    checkpoint_path = next((destination / "rows").glob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["feature_decision_close"] = "999"
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(SignalFeaturePersistenceError, match="identity or schema"):
        _build_fixture(dataset, rule, tmp_path)


@pytest.mark.parametrize(
    "unavailable_values",
    [
        {"available": False},
        {"available": False, "horizon_sessions": None},
    ],
)
def test_outcome_requires_non_null_defaults_for_non_nullable_fields(
    unavailable_values: PrimitiveMapping,
) -> None:
    primary = forward_return_outcome(1)

    with pytest.raises(SignalFeatureDatasetError, match="non-nullable outcome"):
        PredictionStudyOutcome[ForwardReturnValues, ForwardReturnValues].create(
            "invalid_defaults",
            primary.labeler,
            primary.evaluator,
            primary.fields,
            unavailable_values=unavailable_values,
        )


def test_outcome_availability_field_requires_false_unavailable_default() -> None:
    primary = forward_return_outcome(1)
    nullable_availability_fields = tuple(
        replace(field, nullable=True) if field.name == "available" else field
        for field in primary.fields
    )
    invalid_configurations = (
        (nullable_availability_fields, False),
        (primary.fields, True),
    )

    for fields, unavailable_default in invalid_configurations:
        unavailable_values = primary.unavailable_row()
        unavailable_values["available"] = unavailable_default
        with pytest.raises(
            SignalFeatureDatasetError,
            match=r"availability fields must be non-nullable booleans.*false",
        ):
            PredictionStudyOutcome[ForwardReturnValues, ForwardReturnValues].create(
                "invalid_availability",
                primary.labeler,
                primary.evaluator,
                fields,
                unavailable_values=unavailable_values,
            )


def test_available_outcome_values_obey_declared_nullability(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102"))
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))
    primary = forward_return_outcome(1)
    fields = tuple(
        replace(field, nullable=False) if field.name == "raw_return" else field
        for field in primary.fields
    )
    outcome = PredictionStudyOutcome[ForwardReturnValues, NullRawReturnValues].create(
        "null_raw_return",
        primary.labeler,
        NullRawReturnEvaluator(),
        fields,
        unavailable_values={
            "available": False,
            "horizon_sessions": 1,
            "raw_return": "0",
        },
    )

    with pytest.raises(InvalidPredictionOutputError, match=r"null.*non-nullable"):
        outcome.run(dataset, rule, {"feature_schema_version": "1"})


def test_outcome_values_obey_declared_field_types(tmp_path: Path) -> None:
    dataset = make_dataset(("100", "102"))
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))
    primary = forward_return_outcome(1)
    outcome = PredictionStudyOutcome[
        ForwardReturnValues, InvalidRawReturnValues
    ].create(
        "invalid_raw_return",
        primary.labeler,
        InvalidRawReturnEvaluator(),
        primary.fields,
        unavailable_values={"available": False, "horizon_sessions": 1},
    )

    with pytest.raises(InvalidPredictionOutputError, match="declared field types"):
        outcome.run(dataset, rule, {"feature_schema_version": "1"})


@pytest.mark.parametrize("malformed_source", ["run", "unavailable"])
def test_builder_validates_custom_flattened_outcome_values(
    malformed_source: str,
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102"))
    rule = FixtureCandidateRule(
        (SignalDisposition.ACCEPTED, SignalDisposition.REJECTED)
    )
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    configured_outcome = DirectConfiguredOutcome(primary, malformed_source)

    with pytest.raises(
        InvalidPredictionOutputError,
        match=r"flattened values.*field types or nullability",
    ):
        build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(configured_outcome,),
            output_root=tmp_path,
        )


def test_builder_requires_false_direct_outcome_unavailable_flag(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102"))
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    configured_outcome = DirectConfiguredOutcome(primary, "availability")

    with pytest.raises(
        SignalFeatureDatasetError,
        match=r"availability fields must be non-nullable booleans.*false",
    ):
        build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(configured_outcome,),
            output_root=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())


def test_builder_rejects_noncanonical_direct_outcome_study_id(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102"))
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    configured_outcome = DirectConfiguredOutcome(primary, "study_id")

    with pytest.raises(
        InvalidPredictionOutputError, match=r"non-canonical QF-11 study ID"
    ):
        build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(configured_outcome,),
            output_root=tmp_path,
        )

    assert not tuple(tmp_path.rglob("rows/*.json"))


def test_builder_binds_canonical_direct_outcome_study_id_to_composition(
    tmp_path: Path,
) -> None:
    first_dataset = make_dataset(("100", "102"), dataset_id="first")
    second_dataset = make_dataset(("100", "103"), dataset_id="second")
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    configured_outcome = DirectConfiguredOutcome(primary, "canonical_study_id")

    first = build_signal_feature_dataset(
        dataset=first_dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(configured_outcome,),
        output_root=tmp_path / "first",
    )
    second = build_signal_feature_dataset(
        dataset=second_dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(configured_outcome,),
        output_root=tmp_path / "second",
    )

    assert first.prediction_study_ids[0] != "0" * 64
    assert first.prediction_study_ids != second.prediction_study_ids


def test_builder_rejects_direct_outcome_sessions_outside_chunk(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102"))
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    configured_outcome = DirectConfiguredOutcome(primary, "session")

    with pytest.raises(
        InvalidPredictionOutputError,
        match=r"sessions outside the current candidate chunk",
    ):
        build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(configured_outcome,),
            output_root=tmp_path,
        )

    assert not tuple(tmp_path.rglob("rows/*.json"))


def test_builder_rejects_missing_direct_outcome_sessions(tmp_path: Path) -> None:
    dataset = make_dataset(("100", "102"))
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    configured_outcome = DirectConfiguredOutcome(primary, "missing_session")

    with pytest.raises(
        InvalidPredictionOutputError,
        match=r"direct outcome.*omitted candidate sessions.*explicit availability",
    ):
        build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(configured_outcome,),
            output_root=tmp_path,
        )

    assert not tuple(tmp_path.rglob("rows/*.json"))


def test_builder_normalizes_direct_configured_outcome_metadata(tmp_path: Path) -> None:
    dataset = make_dataset(("100", "102"))
    rule = FixtureCandidateRule(
        (SignalDisposition.ACCEPTED, SignalDisposition.REJECTED)
    )
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    configured_outcome = DirectConfiguredOutcome(primary, "valid")

    result = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(configured_outcome,),
        output_root=tmp_path,
    )
    configured_outcomes = result.configuration["outcomes"]
    assert isinstance(configured_outcomes, list)
    normalized = configured_outcomes[0]
    assert isinstance(normalized, dict)

    assert "fields" not in configured_outcome.configuration()
    assert normalized["component_configuration"] == configured_outcome.configuration()
    assert normalized["fields"]
    assert normalized["unavailable_values"] == configured_outcome.unavailable_row()


def test_direct_outcome_fields_are_snapshotted_once_for_identity_and_schema(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102"))
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    configured_outcome = AlternatingFieldsDirectOutcome(primary)

    result = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(configured_outcome,),
        output_root=tmp_path,
    )

    configured_outcomes = result.configuration["outcomes"]
    assert isinstance(configured_outcomes, list)
    normalized = configured_outcomes[0]
    assert isinstance(normalized, dict)
    normalized_fields = normalized["fields"]
    assert isinstance(normalized_fields, list)
    normalized_raw_return = next(
        field
        for field in normalized_fields
        if isinstance(field, dict) and field["field_name"] == "raw_return"
    )
    schema_raw_return = next(
        field
        for field in result.schema.fields
        if field.name == "outcome_forward_return_1_raw_return"
    )

    assert configured_outcome.field_reads == 1
    assert normalized_raw_return["unit"] == schema_raw_return.unit == "ratio"


def test_builder_rechecks_direct_outcome_configuration_after_each_run(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102"))
    rule = FixtureCandidateRule(
        (SignalDisposition.ACCEPTED, SignalDisposition.REJECTED)
    )
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    configured_outcome = DirectConfiguredOutcome(primary, "configuration")

    with pytest.raises(
        InvalidPredictionOutputError,
        match="configured outcome configuration changed during execution",
    ):
        build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(configured_outcome,),
            output_root=tmp_path,
            chunk_size=1,
        )

    destination = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert not tuple((destination / "rows").glob("*.json"))


def test_builder_rejects_direct_outcome_feature_configuration_mutation(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102"))
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    configured_outcome = DirectConfiguredOutcome(primary, "feature_configuration")

    with pytest.raises(
        InvalidPredictionOutputError,
        match="configured outcome mutated feature configuration during execution",
    ):
        build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(configured_outcome,),
            output_root=tmp_path,
        )

    destination = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert not tuple((destination / "rows").glob("*.json"))


def test_builder_snapshots_outcome_configuration_before_dataset_identity(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102"))
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    configured_outcome = AlternatingConfigurationDirectOutcome(primary)

    with pytest.raises(
        InvalidPredictionOutputError,
        match="configured outcome configuration changed during execution",
    ):
        build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(configured_outcome,),
            output_root=tmp_path,
        )

    assert configured_outcome.configuration_reads == 2
    destination = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert not tuple((destination / "rows").glob("*.json"))


def test_builder_requires_direct_outcome_fields_to_be_future_outcomes(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102"))
    rule = FixtureCandidateRule(())
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    configured_outcome = MiscategorizedDirectOutcome(primary, "valid")

    with pytest.raises(
        SignalFeatureDatasetError,
        match="sorted, unique future-outcome definitions",
    ):
        build_signal_feature_dataset(
            dataset=dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(configured_outcome,),
            output_root=tmp_path,
        )

    assert not tuple(tmp_path.iterdir())


def test_unavailable_outcome_defaults_obey_declared_field_types() -> None:
    primary = forward_return_outcome(1)

    with pytest.raises(SignalFeatureDatasetError, match="declared field types"):
        PredictionStudyOutcome[ForwardReturnValues, ForwardReturnValues].create(
            "invalid_unavailable_type",
            primary.labeler,
            primary.evaluator,
            primary.fields,
            unavailable_values={"available": "false", "horizon_sessions": 1},
        )


def test_schema_fields_reject_unsupported_data_types() -> None:
    with pytest.raises(InvalidPredictionOutputError, match="unsupported"):
        SchemaField(
            "invalid_field",
            SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
            "number",
            "fixture_unit",
            False,
            "fixture calculation",
            "after the signal-session close",
        )


def test_accepted_candidate_requires_direction_and_selected_rule_reason() -> None:
    dataset = make_dataset(("100", "102"))
    candidate = (
        FixtureCandidateRule((SignalDisposition.ACCEPTED,)).generate(dataset).signals[0]
    )

    with pytest.raises(InvalidPredictionOutputError, match="require a direction"):
        replace(candidate, direction=None)
    with pytest.raises(InvalidPredictionOutputError, match="require a direction"):
        replace(candidate, selected_rule_reason=None)
    with pytest.raises(InvalidPredictionOutputError, match="first in the matched"):
        replace(candidate, matched_rule_reasons=("different_rule",))
    with pytest.raises(InvalidPredictionOutputError, match="first in the matched"):
        replace(
            candidate,
            matched_rule_reasons=("lower_priority_rule", "fixture_up"),
        )


@pytest.mark.parametrize(
    "disposition",
    [SignalDisposition.BLOCKED, SignalDisposition.OVERLAPPING],
)
def test_nonaccepted_selected_reason_must_lead_matched_trace(
    disposition: SignalDisposition,
) -> None:
    dataset = make_dataset(("100",))
    candidate = FixtureCandidateRule((disposition,)).generate(dataset).signals[0]

    with pytest.raises(InvalidPredictionOutputError, match="first in the matched"):
        replace(
            candidate,
            matched_rule_reasons=("lower_priority_rule", "fixture_up"),
        )

    directionless = replace(
        candidate,
        direction=None,
        selected_rule_reason=None,
        matched_rule_reasons=(),
    )
    assert directionless.direction is None


def test_future_changes_affect_outcomes_but_not_earlier_features(
    tmp_path: Path,
) -> None:
    first_dataset = make_dataset(("100", "102", "101"), dataset_id="first")
    changed_future = make_dataset(("100", "150", "101"), dataset_id="changed")
    dispositions = (SignalDisposition.ACCEPTED,)

    first = (
        _build_fixture(
            first_dataset,
            FixtureCandidateRule(dispositions),
            tmp_path / "first",
            context=LastCloseContext(),
        )
        .rows[0]
        .to_primitive()
    )
    changed = (
        _build_fixture(
            changed_future,
            FixtureCandidateRule(dispositions),
            tmp_path / "changed",
            context=LastCloseContext(),
        )
        .rows[0]
        .to_primitive()
    )

    assert first["feature_decision_close"] == changed["feature_decision_close"] == "100"
    assert (
        first["feature_last_close_context"]
        == changed["feature_last_close_context"]
        == "100"
    )
    assert (
        first["outcome_forward_return_1_raw_return"]
        != changed["outcome_forward_return_1_raw_return"]
    )


def test_causal_history_scrubs_full_dataset_provenance(tmp_path: Path) -> None:
    dividend = ((date(2024, 7, 1), "1"),)
    first_dataset = make_dataset(
        ("100", "102"),
        dataset_id="first-provider-response",
        dividends=dividend,
    )
    appended_dataset = make_dataset(
        ("100", "102", "104"),
        dataset_id="appended-provider-response",
        dividends=dividend,
    )
    dispositions = (SignalDisposition.ACCEPTED,)

    first = _build_fixture(
        first_dataset,
        FixtureCandidateRule(dispositions),
        tmp_path / "first",
        context=ProvenanceReadingContext(),
    ).rows[0]
    appended = _build_fixture(
        appended_dataset,
        FixtureCandidateRule(dispositions),
        tmp_path / "appended",
        context=ProvenanceReadingContext(),
    ).rows[0]

    assert (
        first.to_primitive()["feature_provenance_reading_context"]
        == (appended.to_primitive()["feature_provenance_reading_context"])
    )


def test_builtin_contextual_feature_calculates_one_aligned_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(("100", "102", "101", "104"))
    original_values = AtrPercentageContext.values_for_dataset
    calculation_count = 0

    def counting_values(
        context: AtrPercentageContext, dataset_to_calculate: MarketDataset
    ) -> tuple[Decimal | None, ...]:
        nonlocal calculation_count
        calculation_count += 1
        return original_values(context, dataset_to_calculate)

    monkeypatch.setattr(AtrPercentageContext, "values_for_dataset", counting_values)

    _build_fixture(
        dataset,
        FixtureCandidateRule(
            (
                SignalDisposition.ACCEPTED,
                SignalDisposition.REJECTED,
                SignalDisposition.BLOCKED,
            )
        ),
        tmp_path,
        context=AtrPercentageContext(2),
    )

    assert calculation_count == 1


@pytest.mark.parametrize(
    "context",
    [AtrPercentageContext(2), TrendDistanceContext(2), VolumeRatioContext(2)],
    ids=("atr", "trend", "volume"),
)
def test_aligned_contextual_series_do_not_change_prior_values(
    context: AtrPercentageContext | TrendDistanceContext | VolumeRatioContext,
) -> None:
    first = make_dataset(("100", "102", "101"), dataset_id="first")
    changed_future = make_dataset(("100", "102", "150"), dataset_id="changed")

    assert (
        context.values_for_dataset(first)[:2]
        == context.values_for_dataset(changed_future)[:2]
    )


def test_custom_aligned_callback_cannot_read_a_future_bar(tmp_path: Path) -> None:
    dataset = make_dataset(("100", "102", "101"))
    context = FutureAlignedContext()

    assert context.values_for_dataset(dataset)[0] == Decimal("102")

    result = _build_fixture(
        dataset,
        FixtureCandidateRule((SignalDisposition.ACCEPTED,)),
        tmp_path,
        context=context,
    )

    assert result.rows[0].to_primitive()["feature_future_aligned_context"] == "100"


def test_contextual_feature_cannot_read_a_future_bar(tmp_path: Path) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))

    with pytest.raises(IndexError):
        _build_fixture(dataset, rule, tmp_path, context=FutureReadingContext())


def test_contextual_configuration_is_checked_after_each_candidate(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = FixtureCandidateRule(
        (SignalDisposition.ACCEPTED, SignalDisposition.REJECTED)
    )

    with pytest.raises(
        InvalidPredictionOutputError,
        match="contextual feature configuration changed during calculation",
    ):
        _build_fixture(
            dataset,
            rule,
            tmp_path,
            context=RestoringConfigurationContext(),
        )


def test_contextual_configuration_is_rehashed_after_each_callback(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))

    with pytest.raises(
        InvalidPredictionOutputError,
        match="contextual feature configuration changed during calculation",
    ):
        _build_fixture(
            dataset,
            rule,
            tmp_path,
            context=CachedConfigurationContext(),
        )

    assert not tuple(tmp_path.rglob("rows/*.json"))


def test_non_nullable_contextual_feature_rejects_null_value(tmp_path: Path) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))

    with pytest.raises(
        InvalidPredictionOutputError, match=r"contextual feature.*non-nullable"
    ):
        _build_fixture(dataset, rule, tmp_path, context=NullNonNullableContext())


def test_contextual_fields_must_be_contemporaneous_with_no_candidates(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = FixtureCandidateRule(())

    with pytest.raises(SignalFeatureDatasetError, match="contemporaneous features"):
        _build_fixture(dataset, rule, tmp_path, context=MiscategorizedContext())


def test_candidate_feature_names_must_match_declared_schema(tmp_path: Path) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = ExtraFeatureCandidateRule((SignalDisposition.ACCEPTED,))

    with pytest.raises(InvalidPredictionOutputError, match="candidate feature names"):
        _build_fixture(dataset, rule, tmp_path)


def test_candidate_rules_must_not_prepopulate_contextual_features(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = PrepopulatedContextCandidateRule((SignalDisposition.ACCEPTED,))

    with pytest.raises(
        InvalidPredictionOutputError,
        match=r"must not pre-populate builder-owned contextual features",
    ):
        _build_fixture(dataset, rule, tmp_path)

    assert not tuple(tmp_path.rglob("rows/*.json"))


def test_source_rule_configuration_id_must_be_canonical_sha256(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = InvalidSourceConfigurationCandidateRule((SignalDisposition.ACCEPTED,))

    with pytest.raises(
        InvalidPredictionOutputError,
        match=r"source rule configuration ID.*canonical SHA-256",
    ):
        _build_fixture(dataset, rule, tmp_path)

    assert not tuple(tmp_path.rglob("rows/*.json"))


def test_strategy_fields_must_be_contemporaneous_with_no_candidates(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = MiscategorizedFeatureRule(())

    with pytest.raises(SignalFeatureDatasetError, match="contemporaneous features"):
        _build_fixture(dataset, rule, tmp_path)


@pytest.mark.parametrize("invalid_value", [None, "not-a-decimal"])
def test_candidate_feature_values_must_match_declared_schema(
    tmp_path: Path, invalid_value: str | None
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = InvalidFeatureCandidateRule((SignalDisposition.ACCEPTED,), invalid_value)

    with pytest.raises(InvalidPredictionOutputError, match="candidate feature values"):
        _build_fixture(dataset, rule, tmp_path)


def test_nullable_string_feature_rejects_empty_string_for_csv_fidelity(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = EmptyNullableStringFeatureRule((SignalDisposition.ACCEPTED,))

    with pytest.raises(
        InvalidPredictionOutputError,
        match=r"candidate feature values.*feature_optional_note",
    ):
        _build_fixture(dataset, rule, tmp_path)

    assert not tuple(tmp_path.rglob("rows/*.json"))


def test_complete_row_values_must_match_declared_schema(tmp_path: Path) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = InvalidDispositionMetadataRule((SignalDisposition.ACCEPTED,))

    with pytest.raises(
        InvalidPredictionOutputError,
        match=r"signal-feature row values.*disposition_explanation",
    ):
        _build_fixture(dataset, rule, tmp_path)

    assert not tuple(tmp_path.rglob("rows/*.json"))


def test_material_configuration_changes_create_distinct_dataset_ids(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    dispositions = (SignalDisposition.ACCEPTED, SignalDisposition.REJECTED)
    first = _build_fixture(
        dataset,
        FixtureCandidateRule(dispositions),
        tmp_path,
        context=LastCloseContext(1),
    )
    changed_feature = _build_fixture(
        dataset,
        FixtureCandidateRule(dispositions),
        tmp_path,
        context=LastCloseContext(2),
    )
    rule = FixtureCandidateRule(dispositions)
    primary = forward_return_outcome(2)
    changed_outcome = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=PredictionStudy[
            SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
        ].create(rule, primary.labeler, primary.evaluator),
        contextual_features=(LastCloseContext(1),),
        outcomes=(primary,),
        output_root=tmp_path,
    )

    assert (
        len({first.dataset_id, changed_feature.dataset_id, changed_outcome.dataset_id})
        == 3
    )


def test_contextual_field_definition_changes_dataset_identity(tmp_path: Path) -> None:
    dataset = make_dataset(("100", "102", "101"))
    dispositions = (SignalDisposition.ACCEPTED, SignalDisposition.REJECTED)
    price_feature = LastCloseContext(unit="price_per_share")
    ratio_feature = LastCloseContext(unit="ratio")

    assert price_feature.configuration_id == ratio_feature.configuration_id

    price_result = _build_fixture(
        dataset,
        FixtureCandidateRule(dispositions),
        tmp_path,
        context=price_feature,
    )
    ratio_result = _build_fixture(
        dataset,
        FixtureCandidateRule(dispositions),
        tmp_path,
        context=ratio_feature,
    )

    assert price_result.dataset_id != ratio_result.dataset_id
    assert (tmp_path / price_result.dataset_id).is_dir()
    assert (tmp_path / ratio_result.dataset_id).is_dir()


def test_contextual_definition_is_snapshotted_once_for_identity_and_schema(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    context = AlternatingDefinitionContext()

    result = _build_fixture(
        dataset,
        FixtureCandidateRule((SignalDisposition.ACCEPTED,)),
        tmp_path,
        context=context,
    )

    field = next(
        field
        for field in result.schema.fields
        if field.name == "feature_alternating_definition_context"
    )
    contextual_configuration = result.configuration["contextual_features"]
    assert isinstance(contextual_configuration, list)
    configured_feature = contextual_configuration[0]
    assert isinstance(configured_feature, dict)
    configured_definition = configured_feature["definition"]
    assert isinstance(configured_definition, dict)

    assert context.definition_reads == 1
    assert field.unit == "price_per_share"
    assert configured_definition["unit"] == field.unit


def test_exact_qf3_dataset_identity_changes_feature_dataset_identity(
    tmp_path: Path,
) -> None:
    original = make_dataset(("100", "102", "101"), dataset_id="original-cache")
    refreshed = make_dataset(("100", "102", "101"), dataset_id="refreshed-cache")

    assert original.metadata.data_sha256 == refreshed.metadata.data_sha256
    assert original.metadata.dataset_id != refreshed.metadata.dataset_id

    original_result = _build_fixture(
        original,
        FixtureCandidateRule((SignalDisposition.ACCEPTED,)),
        tmp_path,
    )
    refreshed_result = _build_fixture(
        refreshed,
        FixtureCandidateRule((SignalDisposition.ACCEPTED,)),
        tmp_path,
    )

    assert original_result.dataset_id != refreshed_result.dataset_id
    assert (tmp_path / original_result.dataset_id).is_dir()
    assert (tmp_path / refreshed_result.dataset_id).is_dir()


def test_incompatible_progress_manifest_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = FixtureCandidateRule(
        (SignalDisposition.ACCEPTED, SignalDisposition.REJECTED)
    )
    original_persist = cast(
        Callable[[Path, SignalFeatureRow], None],
        getattr(feature_dataset_module, "_persist_progress_row"),
    )

    def interrupt(destination: Path, row: SignalFeatureRow) -> None:
        original_persist(destination, row)
        raise RuntimeError("interrupt")

    monkeypatch.setattr(feature_dataset_module, "_persist_progress_row", interrupt)
    with pytest.raises(RuntimeError):
        _build_fixture(dataset, rule, tmp_path)
    monkeypatch.setattr(
        feature_dataset_module, "_persist_progress_row", original_persist
    )
    destination = next(path for path in tmp_path.iterdir() if path.is_dir())
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["market_data"]["dataset_id"] = "incompatible-source"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(SignalFeaturePersistenceError, match="incompatible"):
        _build_fixture(dataset, rule, tmp_path)


def test_candidate_adapter_preserves_original_qf11_accepted_predictions() -> None:
    dataset = make_dataset(tuple(str(100 + index % 4) for index in range(15)))
    parameters = OvernightGapPredictionParameters(excluded_weekdays=(4,))

    original = OvernightGapPredictionStrategy(parameters).generate(dataset)
    candidates = OvernightGapSignalFeatureRule(parameters).generate(dataset)
    accepted = tuple(
        candidate
        for candidate in candidates.signals
        if candidate.disposition is SignalDisposition.ACCEPTED
    )

    assert tuple(
        (signal.signal_session, signal.direction, signal.reason)
        for signal in original.signals
    ) == tuple(
        (
            candidate.signal_session,
            candidate.direction,
            candidate.selected_rule_reason,
        )
        for candidate in accepted
    )


def test_qf11_components_are_used_for_dataset_outcomes(tmp_path: Path) -> None:
    dataset = make_dataset(("100", "102", "101"))
    rule = FixtureCandidateRule((SignalDisposition.ACCEPTED,))
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, ForwardReturnOutcomeLabeler(1), ForwardReturnEvaluator())

    result = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(primary,),
        output_root=tmp_path,
    )

    row = result.rows[0].to_primitive()
    assert result.prediction_study_ids
    assert row["prediction_study_ids"] == {
        "forward_return_1": result.prediction_study_ids[0]
    }
