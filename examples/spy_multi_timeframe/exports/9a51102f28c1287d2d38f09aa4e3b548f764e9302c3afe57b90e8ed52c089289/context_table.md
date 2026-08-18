# QF-30 synchronized SPY context table

This table contains OHLCV context only. The fixture is synthetic, and no
prediction, signal, order, fill, P&L, or future-performance claim is made.

| scenario | policy | as_of_utc | timeframe | availability | completion | start_utc | end_utc | open | high | low | close | volume |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normal_session | completed | 2024-07-01T15:00:00+00:00 | 5m | available | completed | 2024-07-01T14:55:00+00:00 | 2024-07-01T15:00:00+00:00 | 504.07 | 504.12 | 504.04 | 504.09 | 1407 |
| normal_session | completed | 2024-07-01T15:00:00+00:00 | daily | available | completed | 2024-06-28T13:30:00+00:00 | 2024-06-28T20:00:00+00:00 | 503.12 | 503.94 | 503.09 | 503.91 | 105339 |
| normal_session | completed | 2024-07-01T15:00:00+00:00 | 4h | available | completed_partial_duration_terminal | 2024-06-28T17:30:00+00:00 | 2024-06-28T20:00:00+00:00 | 503.6 | 503.94 | 503.57 | 503.91 | 41235 |
| normal_session | completed | 2024-07-01T15:00:00+00:00 | weekly | available | completed | 2024-06-24T13:30:00+00:00 | 2024-06-28T20:00:00+00:00 | 500 | 503.94 | 499.97 | 503.91 | 465855 |
| early_close_near_close | completed | 2024-07-03T16:55:00+00:00 | 5m | available | completed | 2024-07-03T16:50:00+00:00 | 2024-07-03T16:55:00+00:00 | 505.86 | 505.91 | 505.83 | 505.88 | 1586 |
| early_close_near_close | completed | 2024-07-03T16:55:00+00:00 | daily | available | completed | 2024-07-02T13:30:00+00:00 | 2024-07-02T20:00:00+00:00 | 504.68 | 505.5 | 504.65 | 505.47 | 117507 |
| early_close_near_close | completed | 2024-07-03T16:55:00+00:00 | 4h | available | completed_partial_duration_terminal | 2024-07-02T17:30:00+00:00 | 2024-07-02T20:00:00+00:00 | 505.16 | 505.5 | 505.13 | 505.47 | 45915 |
| early_close_near_close | completed | 2024-07-03T16:55:00+00:00 | weekly | available | completed | 2024-06-24T13:30:00+00:00 | 2024-06-28T20:00:00+00:00 | 500 | 503.94 | 499.97 | 503.91 | 465855 |
| midweek | completed | 2024-07-10T16:00:00+00:00 | 5m | available | completed | 2024-07-10T15:55:00+00:00 | 2024-07-10T16:00:00+00:00 | 508.51 | 508.56 | 508.48 | 508.53 | 1851 |
| midweek | completed | 2024-07-10T16:00:00+00:00 | daily | available | completed | 2024-07-09T13:30:00+00:00 | 2024-07-09T20:00:00+00:00 | 507.44 | 508.26 | 507.41 | 508.23 | 139035 |
| midweek | completed | 2024-07-10T16:00:00+00:00 | 4h | available | completed_partial_duration_terminal | 2024-07-09T17:30:00+00:00 | 2024-07-09T20:00:00+00:00 | 507.92 | 508.26 | 507.89 | 508.23 | 54195 |
| midweek | completed | 2024-07-10T16:00:00+00:00 | weekly | available | completed | 2024-07-01T13:30:00+00:00 | 2024-07-05T20:00:00+00:00 | 503.9 | 506.7 | 503.87 | 506.67 | 421590 |
| normal_session_near_close | completed | 2024-07-12T19:55:00+00:00 | 5m | available | completed | 2024-07-12T19:50:00+00:00 | 2024-07-12T19:55:00+00:00 | 510.54 | 510.59 | 510.51 | 510.56 | 2054 |
| normal_session_near_close | completed | 2024-07-12T19:55:00+00:00 | daily | available | completed | 2024-07-11T13:30:00+00:00 | 2024-07-11T20:00:00+00:00 | 509 | 509.82 | 508.97 | 509.79 | 151203 |
| normal_session_near_close | completed | 2024-07-12T19:55:00+00:00 | 4h | available | completed | 2024-07-12T13:30:00+00:00 | 2024-07-12T17:30:00+00:00 | 509.78 | 510.3 | 509.75 | 510.27 | 96072 |
| normal_session_near_close | completed | 2024-07-12T19:55:00+00:00 | weekly | available | completed | 2024-07-01T13:30:00+00:00 | 2024-07-05T20:00:00+00:00 | 503.9 | 506.7 | 503.87 | 506.67 | 421590 |
| normal_session | developing | 2024-07-01T15:00:00+00:00 | 5m | available | completed | 2024-07-01T14:55:00+00:00 | 2024-07-01T15:00:00+00:00 | 504.07 | 504.12 | 504.04 | 504.09 | 1407 |
| normal_session | developing | 2024-07-01T15:00:00+00:00 | daily | available | developing | 2024-07-01T13:30:00+00:00 | 2024-07-01T15:00:00+00:00 | 503.9 | 504.12 | 503.87 | 504.09 | 25173 |
| normal_session | developing | 2024-07-01T15:00:00+00:00 | 4h | available | developing | 2024-07-01T13:30:00+00:00 | 2024-07-01T15:00:00+00:00 | 503.9 | 504.12 | 503.87 | 504.09 | 25173 |
| normal_session | developing | 2024-07-01T15:00:00+00:00 | weekly | available | developing | 2024-07-01T13:30:00+00:00 | 2024-07-01T15:00:00+00:00 | 503.9 | 504.12 | 503.87 | 504.09 | 25173 |
| early_close_near_close | developing | 2024-07-03T16:55:00+00:00 | 5m | available | completed | 2024-07-03T16:50:00+00:00 | 2024-07-03T16:55:00+00:00 | 505.86 | 505.91 | 505.83 | 505.88 | 1586 |
| early_close_near_close | developing | 2024-07-03T16:55:00+00:00 | daily | available | developing | 2024-07-03T13:30:00+00:00 | 2024-07-03T16:55:00+00:00 | 505.46 | 505.91 | 505.43 | 505.88 | 64206 |
| early_close_near_close | developing | 2024-07-03T16:55:00+00:00 | 4h | available | developing | 2024-07-03T13:30:00+00:00 | 2024-07-03T16:55:00+00:00 | 505.46 | 505.91 | 505.43 | 505.88 | 64206 |
| early_close_near_close | developing | 2024-07-03T16:55:00+00:00 | weekly | available | developing | 2024-07-01T13:30:00+00:00 | 2024-07-03T16:55:00+00:00 | 503.9 | 505.91 | 503.87 | 505.88 | 293136 |
| midweek | developing | 2024-07-10T16:00:00+00:00 | 5m | available | completed | 2024-07-10T15:55:00+00:00 | 2024-07-10T16:00:00+00:00 | 508.51 | 508.56 | 508.48 | 508.53 | 1851 |
| midweek | developing | 2024-07-10T16:00:00+00:00 | daily | available | developing | 2024-07-10T13:30:00+00:00 | 2024-07-10T16:00:00+00:00 | 508.22 | 508.56 | 508.19 | 508.53 | 55095 |
| midweek | developing | 2024-07-10T16:00:00+00:00 | 4h | available | developing | 2024-07-10T13:30:00+00:00 | 2024-07-10T16:00:00+00:00 | 508.22 | 508.56 | 508.19 | 508.53 | 55095 |
| midweek | developing | 2024-07-10T16:00:00+00:00 | weekly | available | developing | 2024-07-08T13:30:00+00:00 | 2024-07-10T16:00:00+00:00 | 506.66 | 508.56 | 506.63 | 508.53 | 327081 |
| normal_session_near_close | developing | 2024-07-12T19:55:00+00:00 | 5m | available | completed | 2024-07-12T19:50:00+00:00 | 2024-07-12T19:55:00+00:00 | 510.54 | 510.59 | 510.51 | 510.56 | 2054 |
| normal_session_near_close | developing | 2024-07-12T19:55:00+00:00 | daily | available | developing | 2024-07-12T13:30:00+00:00 | 2024-07-12T19:55:00+00:00 | 509.78 | 510.59 | 509.75 | 510.56 | 155232 |
| normal_session_near_close | developing | 2024-07-12T19:55:00+00:00 | 4h | available | developing | 2024-07-12T17:30:00+00:00 | 2024-07-12T19:55:00+00:00 | 510.26 | 510.59 | 510.23 | 510.56 | 59160 |
| normal_session_near_close | developing | 2024-07-12T19:55:00+00:00 | weekly | available | developing | 2024-07-08T13:30:00+00:00 | 2024-07-12T19:55:00+00:00 | 506.66 | 510.59 | 506.63 | 510.56 | 723540 |
