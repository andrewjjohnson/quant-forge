# Timeframe and exchange-session semantics

QF-13 defines the provider-neutral temporal vocabulary used by future market
data, aggregation, alignment, indicator, prediction, and strategy consumers. It
defines configuration and boundary validation only. It does not retrieve data,
aggregate OHLCV, align multiple timeframes, or change indicators.

The public implementation is `quantforge.timeframes`.

## Interval types

The three interval categories are structurally different:

- `IntradayInterval` is a positive elapsed duration shorter than one day. The
  duration is serialized as exact integer microseconds, never a floating-point
  number.
- `SessionInterval` is one or more complete exchange sessions. One session is
  the canonical meaning of daily; it is not 24 elapsed hours.
- `TradingWeekInterval` is one or more Monday-Sunday exchange trading weeks.
  The observations in a week are the sessions published by the configured
  calendar; a week is not seven elapsed days or five assumed weekdays.

This separation prevents code from converting daily or weekly research into a
minute count and silently losing holiday, early-close, or daylight-saving
semantics.

```python
from datetime import timedelta

from quantforge.timeframes import (
    IntradayInterval,
    SessionInterval,
    Timeframe,
    TradingWeekInterval,
)

four_hour = Timeframe.us_equity(IntradayInterval(timedelta(hours=4)))
daily = Timeframe.us_equity(SessionInterval())
weekly = Timeframe.us_equity(TradingWeekInterval())
```

Positive counts greater than one represent multi-session and multi-week
intervals without pretending either is a timedelta.

## Default U.S. equity policy

`DEFAULT_US_EQUITY_TIMEFRAME` and `Timeframe.us_equity()` use:

| Policy | Default |
| --- | --- |
| Exchange calendar | `XNYS` |
| Exchange timezone | `America/New_York` |
| Session scope | regular trading hours |
| Intraday anchor | actual exchange-session open |
| Cross-session bars | prohibited |
| Bar label | bar start |
| Consumer exposure | completed bars only |

Both start and end timestamps remain present regardless of the selected label.
The label is an indexing convention only. A completed bar becomes available at
its end timestamp; selecting a start label never makes its eventual high, low,
close, or volume available at the start.

Regular session boundaries come from the installed `exchange-calendars`
schedule. Canonical boundary records are stored in UTC and convert to exchange
local time for chart interpretation. As a result, XNYS opens remain 09:30 local
across daylight-saving transitions while their UTC offsets change. Early closes
use the schedule's actual close. Weekends and exchange holidays are rejected as
sessions rather than classified as missing observations.

Extended-hours scope is a separate identity-bearing policy. Because extended
hours are not uniform across venues and feeds, it requires explicit same-day
local start and end times. QF-13 does not choose an extended-hours default or
claim provider support for one.

## Intraday anchoring and terminal bars

Session-open anchoring measures consecutive nominal durations from the actual
session open. For a regular XNYS session and a four-hour interval, the canonical
windows are:

```text
09:30-13:30  completed full-duration bar
13:30-16:00  completed partial-duration terminal bar
```

On the 2024-11-29 early close, the session is 09:30-13:00. Its only four-hour
window is a completed 3.5-hour partial-duration terminal bar. The nominal
duration remains four hours and the actual duration is 3.5 hours.

`IntradayBarWindow` validates boundaries without calculating OHLCV. Its
`BarCompletion` value is one of:

- `COMPLETED`: actual duration equals nominal duration;
- `DEVELOPING`: the observed boundary has not reached its full or session-close
  terminal boundary and the timeframe explicitly opts into developing bars;
- `COMPLETED_PARTIAL_DURATION_TERMINAL`: actual duration is shorter than nominal
  and ends at the actual session close.

A partial terminal bar is completed and causally usable after its end. It is
not a developing bar. A developing bar that reaches the terminal boundary is
invalid and must be reclassified as completed.

With the default `CrossSessionPolicy.PROHIBITED`, an intraday boundary outside
its resolved session fails validation. `PERMITTED` exists only as an explicit,
identity-bearing future policy; no QF-13 code aggregates cross-session OHLCV.

`IntradayAnchor.CLOCK` is also explicit and requires an exchange-local clock
origin. It supports future consumers that need clock-aligned candles without
changing the session-open default. Clock anchoring does not implicitly permit a
bar to cross the configured session.

## Stable serialization and identity

`Timeframe.to_primitive()` records the schema version, typed interval, exact
duration or count, calendar, timezone, session scope and explicit extended-hours
bounds, anchor and clock origin, cross-session policy, label policy, and
developing-bar exposure. `Timeframe.configuration_id` is the repository's
canonical sorted-JSON SHA-256 identity of that complete primitive mapping.

Changing any material policy therefore changes the configuration ID. Consumers
must persist both the primitive configuration and ID, then verify the ID before
trusting cached or reportable results. Future intraday dataset and aggregation
stories must bind this ID into their own manifests rather than copying only a
display name such as `4h`.

The schema is version `1`. Adding a new semantic policy or changing an existing
meaning requires a schema-version and compatibility review; field-order changes
alone do not affect the canonical identity.

## Existing daily-bar compatibility

QF-3 `DailyBar` remains a date-labeled, completed exchange-session record and
its schema stays version 4. Existing provider APIs, immutable cache artifacts,
dataset IDs, backtests, predictions, indicators, and reports are unchanged by
QF-13.

For the maintained U.S. equity path, an existing QF-3 daily bar corresponds to
the default `SessionInterval(1)` conceptually. QF-13 does not retrofit a new
field into existing manifests because that would rewrite scientific identities
and prematurely implement later QF-12 ingestion scope. New consumers should use
the canonical configuration directly; later schema work must define an explicit
migration or adapter rather than mutating old artifacts.

## Deliberate limitations

QF-13 does not provide:

- provider retrieval or provider interval mapping;
- OHLCV aggregation or missing-source-bar policy;
- multi-timeframe as-of alignment;
- indicator generalization;
- overnight extended-hours windows;
- proprietary chart-platform candle replication.

Provider or chart-platform compatibility must be an explicit adapter from the
platform's documented candle semantics to this model. If a platform needs a
new anchor or boundary policy, add it explicitly and version the configuration;
never silently reinterpret the canonical defaults.
