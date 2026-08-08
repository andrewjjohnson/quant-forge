"""QF-7 candidate adapter for the unchanged QF-11 overnight-gap baseline."""

from decimal import Decimal

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.models import MarketDataset
from quantforge.indicators import (
    AVERAGE_DIRECTIONAL_INDEX_OUTPUT,
    NEGATIVE_DIRECTIONAL_INDICATOR_OUTPUT,
    POSITIVE_DIRECTIONAL_INDICATOR_OUTPUT,
    WILDER_RSI_OUTPUT,
    Indicator,
)
from quantforge.prediction.overnight_gap import (
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
    evaluate_overnight_gap_rules,
)
from quantforge.prediction.signal_feature_models import (
    SchemaField,
    SchemaFieldCategory,
    SignalDisposition,
    SignalFeatureCandidate,
    SignalFeatureCandidateOutput,
    SignalFeatureValue,
)


class OvernightGapSignalFeatureRule:
    """Emit every identifiable baseline candidate with a fixed disposition."""

    name = "overnight_gap_direction_signal_candidates"
    implementation_version = "1"

    def __init__(self, parameters: OvernightGapPredictionParameters) -> None:
        self._parameters = parameters
        self._source = OvernightGapPredictionStrategy(parameters)

    @property
    def parameters(self) -> OvernightGapPredictionParameters:
        return self._parameters

    @property
    def required_indicators(self) -> tuple[Indicator, ...]:
        return self._source.required_indicators

    @property
    def warm_up_observations(self) -> int:
        return self._source.warm_up_observations

    @property
    def strategy_feature_definitions(self) -> tuple[SchemaField, ...]:
        timing = "available after the completed signal-session close"
        definitions = (
            SchemaField(
                "adx",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "decimal",
                "index_points",
                False,
                "QF-4 Wilder ADX using the configured period",
                timing,
            ),
            SchemaField(
                "close",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "decimal",
                "price_per_share",
                False,
                "QF-3 completed-session close",
                timing,
            ),
            SchemaField(
                "minus_di",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "decimal",
                "index_points",
                False,
                "QF-4 Wilder negative directional indicator",
                timing,
            ),
            SchemaField(
                "open",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "decimal",
                "price_per_share",
                False,
                "QF-3 signal-session open",
                timing,
            ),
            SchemaField(
                "plus_di",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "decimal",
                "index_points",
                False,
                "QF-4 Wilder positive directional indicator",
                timing,
            ),
            SchemaField(
                "previous_adx",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "decimal",
                "index_points",
                False,
                "prior completed-session QF-4 Wilder ADX",
                timing,
            ),
            SchemaField(
                "previous_minus_di",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "decimal",
                "index_points",
                False,
                "prior completed-session negative directional indicator",
                timing,
            ),
            SchemaField(
                "previous_plus_di",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "decimal",
                "index_points",
                False,
                "prior completed-session positive directional indicator",
                timing,
            ),
            SchemaField(
                "previous_rsi",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "decimal",
                "index_points",
                False,
                "prior completed-session QF-4 Wilder RSI",
                timing,
            ),
            SchemaField(
                "rsi",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "decimal",
                "index_points",
                False,
                "QF-4 Wilder RSI using the configured period",
                timing,
            ),
            SchemaField(
                "signal_weekday",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "integer",
                "python_weekday_monday_zero",
                False,
                "QF-3 signal session weekday",
                timing,
            ),
        )
        return tuple(sorted(definitions, key=lambda field: field.name))

    def configuration(self) -> PrimitiveMapping:
        return {
            "candidate_semantics": {
                "blocked": "configured weekday exclusion",
                "overlapping": "not_identifiable_for_stateless_prediction_rule",
                "rejected": "baseline veto or no directional rule match",
            },
            "component_name": self.name,
            "component_type": "prediction_strategy",
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": self.parameters.to_primitive(),
            "required_indicators": [
                indicator.configuration() for indicator in self.required_indicators
            ],
            "source_prediction_rule": self._source.configuration(),
            "warm_up_observations": self.warm_up_observations,
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def generate(self, dataset: MarketDataset) -> SignalFeatureCandidateOutput:
        rsi_values = (
            self.required_indicators[0].calculate(dataset).values_for(WILDER_RSI_OUTPUT)
        )
        dmi_output = self.required_indicators[1].calculate(dataset)
        positive_di_values = dmi_output.values_for(
            POSITIVE_DIRECTIONAL_INDICATOR_OUTPUT
        )
        negative_di_values = dmi_output.values_for(
            NEGATIVE_DIRECTIONAL_INDICATOR_OUTPUT
        )
        adx_values = dmi_output.values_for(AVERAGE_DIRECTIONAL_INDEX_OUTPUT)
        parameter_snapshot = PrimitiveMappingSnapshot.capture(
            self.parameters.to_primitive()
        )
        candidates: list[SignalFeatureCandidate] = []

        for index in range(1, len(dataset.bars)):
            values = (
                rsi_values[index - 1],
                rsi_values[index],
                positive_di_values[index - 1],
                positive_di_values[index],
                negative_di_values[index - 1],
                negative_di_values[index],
                adx_values[index - 1],
                adx_values[index],
            )
            if any(value is None for value in values):
                continue
            (
                previous_rsi,
                current_rsi,
                previous_positive_di,
                current_positive_di,
                previous_negative_di,
                current_negative_di,
                previous_adx,
                current_adx,
            ) = (value for value in values if value is not None)
            bar = dataset.bars[index]
            evaluation = evaluate_overnight_gap_rules(
                self.parameters,
                previous_rsi=previous_rsi,
                current_rsi=current_rsi,
                previous_positive_di=previous_positive_di,
                current_positive_di=current_positive_di,
                previous_negative_di=previous_negative_di,
                current_negative_di=current_negative_di,
                previous_adx=previous_adx,
                current_adx=current_adx,
                session_open=bar.open,
                session_close=bar.close,
            )
            if bar.session_date.weekday() in self.parameters.excluded_weekdays:
                disposition = SignalDisposition.BLOCKED
                reason_codes = ("weekday_excluded",)
                explanation = "signal session weekday is excluded by configuration"
            elif evaluation.veto_reason is not None:
                disposition = SignalDisposition.REJECTED
                reason_codes = (evaluation.veto_reason,)
                explanation = "the baseline ADX veto rejected this candidate"
            elif evaluation.selected_reason is None:
                disposition = SignalDisposition.REJECTED
                reason_codes = ("no_directional_rule_matched",)
                explanation = "no baseline directional rule matched this candidate"
            else:
                disposition = SignalDisposition.ACCEPTED
                reason_codes = (evaluation.selected_reason,)
                explanation = "the highest-priority matching baseline rule accepted"

            feature_values: dict[str, Decimal | int] = {
                "adx": current_adx,
                "close": bar.close,
                "minus_di": current_negative_di,
                "open": bar.open,
                "plus_di": current_positive_di,
                "previous_adx": previous_adx,
                "previous_minus_di": previous_negative_di,
                "previous_plus_di": previous_positive_di,
                "previous_rsi": previous_rsi,
                "rsi": current_rsi,
                "signal_weekday": bar.session_date.weekday(),
            }
            candidates.append(
                SignalFeatureCandidate(
                    symbol=dataset.metadata.canonical_symbol,
                    signal_session=bar.session_date,
                    strategy_id=self.name,
                    strategy_implementation_version=self.implementation_version,
                    strategy_configuration_id=self.configuration_id,
                    source_rule_id=self._source.name,
                    source_rule_implementation_version=(
                        self._source.implementation_version
                    ),
                    source_rule_configuration_id=self._source.configuration_id,
                    strategy_parameters=parameter_snapshot,
                    disposition=disposition,
                    reason_codes=reason_codes,
                    explanation=explanation,
                    direction=evaluation.direction,
                    selected_rule_reason=evaluation.selected_reason,
                    matched_rule_reasons=evaluation.matched_reasons,
                    strategy_features=tuple(
                        SignalFeatureValue(name, value)
                        for name, value in sorted(feature_values.items())
                    ),
                )
            )

        return SignalFeatureCandidateOutput(
            self.name,
            self.configuration_id,
            dataset.metadata.dataset_id,
            tuple(candidates),
        )
