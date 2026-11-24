"""Loading 1-minute bars + per-stock liquidity audit (proposal 3.1-3.2).

Design choices:
  * Schema is read from config (SchemaConfig), not hard-coded, so a library
    rename doesn't break the loader.
  * We never gap-fill. Missing minutes stay missing; validity is enforced later
    at the window level.
  * The liquidity audit computes the SAME quality indicators the library
    publishes daily (gap rate, observed bars), so screening is reproducible
    whether or not the precomputed daily file is present.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import pandas as pd

from ..config import DataConfig
from .calendar import apply_calendar_exclusions

REGULAR_SESSION_MINUTES = 390  # 09:30-16:00 inclusive-exclusive, 1-min bars


def discover_tickers(data: DataConfig) -> dict[str, str]:
    """Map ticker symbol -> file path for every file in raw_dir.

    The HF Data Library API names files like ``AAPL_clean.parquet`` /
    ``AAPL_raw.parquet``; a plain website download may be just ``AAPL.parquet``.
    We strip a trailing ``_clean``/``_raw`` so the ticker is the real symbol in
    both cases (otherwise it would read as ``AAPL_CLEAN``).
    """
    paths = sorted(glob.glob(os.path.join(data.raw_dir, data.file_glob)))
    out: dict[str, str] = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        for suffix in ("_clean", "_raw"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        out[stem.upper()] = p
    return out


def load_ticker(path: str, data: DataConfig) -> pd.DataFrame:
    """Load one ticker's bars, normalise the datetime column, sort, and apply
    calendar exclusions. Accepts parquet or csv by extension."""
    s = data.schema
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    missing = {s.datetime, s.open, s.high, s.low, s.close, s.volume} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {sorted(missing)}")
    df[s.datetime] = pd.to_datetime(df[s.datetime])
    df = df.sort_values(s.datetime).reset_index(drop=True)
    df = apply_calendar_exclusions(df, s.datetime, data.window.early_close)
    return df


@dataclass
class DailyQuality:
    """Per-stock-day quality summary used for liquidity screening."""
    ticker: str
    session: object           # datetime.date
    observed_bars: int
    gap_rate: float           # 1 - observed_bars / expected_bars
    dollar_volume: float


def audit_ticker(df: pd.DataFrame, ticker: str, data: DataConfig) -> pd.DataFrame:
    """Compute per-session quality indicators for one ticker."""
    s = data.schema
    ts = pd.to_datetime(df[s.datetime])
    g = df.assign(_session=ts.dt.date)
    rows = []
    for session, part in g.groupby("_session"):
        obs = len(part)
        # Early-close sessions have fewer expected bars; approximate expected as
        # min(observed-implied, REGULAR). Gap rate is clipped to [0, 1].
        expected = REGULAR_SESSION_MINUTES
        gap_rate = max(0.0, 1.0 - obs / expected)
        dvol = float((part[s.close] * part[s.volume]).sum())
        rows.append(DailyQuality(ticker, session, obs, gap_rate, dvol).__dict__)
    return pd.DataFrame(rows)


def screen_tickers(audits: dict[str, pd.DataFrame], data: DataConfig
                   ) -> tuple[list[str], pd.DataFrame]:
    """Apply liquidity thresholds. Returns (eligible_tickers, summary_table).

    A ticker is eligible if it has >= min_valid_days sessions that each pass the
    per-day gap/observed/dollar-volume thresholds.
    """
    lq = data.liquidity
    # Defensive float coercion: YAML parses "1.0e6" as a string, which would make
    # the threshold comparisons raise. Fail loudly with a clear message instead.
    try:
        max_gap = float(lq.max_gap_rate)
        min_obs = float(lq.min_observed_bars)
        min_dvol = float(lq.min_dollar_volume)
        min_days = int(lq.min_valid_days)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Liquidity thresholds must be numeric. A common cause is writing "
            "'1.0e6' in YAML (parsed as a string); use 1000000.0 instead."
        ) from e
    summary_rows = []
    eligible = []
    for ticker, aud in audits.items():
        good = aud[(aud["gap_rate"] <= max_gap)
                   & (aud["observed_bars"] >= min_obs)
                   & (aud["dollar_volume"] >= min_dvol)]
        n_good = len(good)
        passed = n_good >= min_days
        summary_rows.append({
            "ticker": ticker,
            "n_days": len(aud),
            "n_good_days": n_good,
            "median_gap_rate": float(aud["gap_rate"].median()) if len(aud) else 1.0,
            "median_dollar_volume": float(aud["dollar_volume"].median()) if len(aud) else 0.0,
            "eligible": passed,
        })
        if passed:
            eligible.append(ticker)
    return sorted(eligible), pd.DataFrame(summary_rows).sort_values("ticker")
