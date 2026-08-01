"""Pure provider-record normalization."""

from datetime import date
from decimal import Decimal, InvalidOperation

from quantforge.data.exceptions import ValidationError
from quantforge.data.models import AdjustmentMode, DailyBar, ProviderResponse

_REQUIRED = ("session_date", "open", "high", "low", "close", "volume")


def normalize_symbol(symbol: str) -> str:
    """Return a canonical stock/ETF symbol."""
    canonical = symbol.strip().upper()
    if not canonical or not all(
        character.isalnum() or character in ".-" for character in canonical
    ):
        raise ValidationError(f"unsupported symbol: {symbol!r}")
    return canonical


def normalize_response(
    response: ProviderResponse, canonical_symbol: str
) -> tuple[DailyBar, ...]:
    """Convert lossless adapter records and apply a coherent split basis.

    ``split_coefficient`` is the shares-after/shares-before ratio effective on a
    session. Each earlier price is divided by all later coefficients, while its
    volume is multiplied by the same cumulative factor. No dividend factor is
    inferred here.
    """
    symbol = normalize_symbol(canonical_symbol)
    if response.adjustment_mode is AdjustmentMode.SPLIT_AND_DIVIDEND_ADJUSTED:
        raise ValidationError(
            "local split factors cannot produce dividend-adjusted OHLCV"
        )
    parsed: list[tuple[date, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]] = []
    for index, record in enumerate(response.records):
        missing = [field for field in _REQUIRED if field not in record]
        if missing:
            raise ValidationError(
                f"record {index} missing fields: {', '.join(missing)}"
            )
        try:
            session = date.fromisoformat(str(record["session_date"]))
            open_price = Decimal(str(record["open"]))
            high = Decimal(str(record["high"]))
            low = Decimal(str(record["low"]))
            close = Decimal(str(record["close"]))
            volume = Decimal(str(record["volume"]))
            split = Decimal(str(record.get("split_coefficient", "1")))
        except (ValueError, InvalidOperation) as error:
            raise ValidationError(
                f"record {index} contains an invalid date or number"
            ) from error
        if split <= 0:
            raise ValidationError("split coefficient must be positive")
        parsed.append((session, open_price, high, low, close, volume, split))
    parsed.sort(key=lambda item: item[0])
    factor = Decimal(1)
    adjusted_reversed: list[DailyBar] = []
    for session, open_price, high, low, close, volume, split in reversed(parsed):
        if response.adjustment_mode is AdjustmentMode.SPLIT_ADJUSTED:
            adjusted_reversed.append(
                DailyBar(
                    symbol,
                    session,
                    open_price / factor,
                    high / factor,
                    low / factor,
                    close / factor,
                    volume * factor,
                )
            )
            factor *= split
        else:
            adjusted_reversed.append(
                DailyBar(symbol, session, open_price, high, low, close, volume)
            )
    return tuple(reversed(adjusted_reversed))
