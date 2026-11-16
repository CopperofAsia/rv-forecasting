"""Trading-calendar exclusions (proposal 3.2).

Two documented exclusions:
  1. The incomplete session of 2024-12-23 is dropped entirely.
  2. On early-close days, bars at/after 13:00 ET are dropped (the session ends
     early, so a 30-min forward window there would reach past the close).

We keep the early-close list explicit and dated rather than inferring it, so the
exclusion is auditable. Extend EARLY_CLOSE_DATES as the snapshot's date range
grows; these cover the US half-days within the library's 2022-03-07+ range.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

# Sessions dropped in full.
EXCLUDED_SESSIONS: frozenset[date] = frozenset({date(2024, 12, 23)})

# US half-days (early close 13:00 ET): day before July 4, day after Thanksgiving,
# Christmas Eve when a trading day. Covers 2022..2025 within the snapshot range.
EARLY_CLOSE_DATES: frozenset[date] = frozenset({
    date(2022, 11, 25), date(2022, 12, 23),
    date(2023, 7, 3), date(2023, 11, 24),
    date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24),
    date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
})


def apply_calendar_exclusions(df: pd.DataFrame, dt_col: str,
                              early_close: str = "13:00") -> pd.DataFrame:
    """Return df with excluded sessions removed and early-close tails trimmed.

    df[dt_col] must be tz-aware or naive Eastern-Time timestamps. We operate on
    the calendar date and clock time only, so either works.
    """
    ts = pd.to_datetime(df[dt_col])
    d = ts.dt.date
    t = ts.dt.time
    cutoff = pd.to_datetime(early_close).time()

    drop_session = d.isin(EXCLUDED_SESSIONS)
    drop_early = d.isin(EARLY_CLOSE_DATES) & (t >= cutoff)
    keep = ~(drop_session | drop_early)
    return df.loc[keep].copy()
