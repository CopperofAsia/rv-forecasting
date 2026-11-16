"""Window construction + realized-volatility labels (proposal 3.2).

For each eligible stock and decision time t (a 1-minute bar close):
  * FEATURES use only information at or before t (recent returns, OHLCV
    summaries, past realized volatility, observed-bar counts).
  * LABEL uses only the FUTURE window (t, t+H]: the next H minutes of prices.

The feature window and label window share no bar, so no future price can leak
into a feature. Windows never cross a session boundary. We require at least N
valid consecutive-minute returns inside the future window; windows that fail are
discarded and counted (no gap filling).

Label math (proposal 3.2), over consecutive observed future prices P_j:
    r_j   = ln(P_j / P_{j-1})
    RV30  = sum_j r_j^2
    sigma30 = sqrt(RV30)          <- prediction target
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DataConfig


# Feature column order is fixed and documented so every model sees an identical
# vector (proposal 4.2/4.3: inputs held constant across all neural comparisons).
FEATURE_NAMES: tuple[str, ...] = (
    "rv_lookback",        # realized vol over the preceding lookback window
    "ret_mean",           # mean 1-min log return over lookback
    "ret_std",            # std of 1-min log returns over lookback
    "ret_last",           # most recent 1-min log return
    "log_volume",         # log(1 + summed volume over lookback)
    "hl_range",           # mean (High-Low)/Close over lookback
    "observed_frac",      # observed bars / lookback length (data-quality signal)
    "minute_of_day",      # normalised time-of-day of decision point in [0,1]
)


def _session_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.date


def _safe_log_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns of consecutive positive prices; non-finite entries dropped."""
    prices = np.asarray(prices, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(prices))
    return r[np.isfinite(r)]


def build_windows(df: pd.DataFrame, ticker: str, data: DataConfig) -> pd.DataFrame:
    """Return a tidy frame of (decision_time, features..., label) rows.

    Non-overlapping decision points stepped by H minutes keep feature windows
    from containing another window's future label (proposal 3.2).
    """
    s = data.schema
    w = data.window
    H = w.horizon_minutes
    L = w.feature_lookback_minutes
    df = df.copy()
    df[s.datetime] = pd.to_datetime(df[s.datetime])
    df["_session"] = _session_key(df[s.datetime])

    rows = []
    for session, part in df.groupby("_session"):
        part = part.sort_values(s.datetime).reset_index(drop=True)
        close = part[s.close].to_numpy(dtype=float)
        high = part[s.high].to_numpy(dtype=float)
        low = part[s.low].to_numpy(dtype=float)
        vol = part[s.volume].to_numpy(dtype=float)
        times = part[s.datetime].to_numpy()
        n = len(part)
        # Decision index i: features from (i-L, i], label from (i, i+H].
        # Step by H so successive feature windows never overlap a prior label.
        i = L
        while i + H < n:
            feat_close = close[i - L:i + 1]
            lab_close = close[i:i + H + 1]        # includes P at t as P_0 of future
            lab_rets = _safe_log_returns(lab_close)
            if len(lab_rets) < w.min_valid_returns:
                i += H
                continue
            rv30 = float(np.sum(lab_rets ** 2))
            sigma30 = float(np.sqrt(rv30))

            feat_rets = _safe_log_returns(feat_close)
            rv_lb = float(np.sqrt(np.sum(feat_rets ** 2))) if len(feat_rets) else 0.0
            hl = (high[i - L:i + 1] - low[i - L:i + 1]) / np.where(
                feat_close != 0, feat_close, np.nan)
            dec_ts = pd.Timestamp(times[i])
            minute_of_day = (dec_ts.hour * 60 + dec_ts.minute - 9 * 60 - 30) / 390.0

            rows.append({
                "ticker": ticker,
                "session": session,
                "decision_time": dec_ts,
                "rv_lookback": rv_lb,
                "ret_mean": float(np.mean(feat_rets)) if len(feat_rets) else 0.0,
                "ret_std": float(np.std(feat_rets)) if len(feat_rets) else 0.0,
                "ret_last": float(feat_rets[-1]) if len(feat_rets) else 0.0,
                "log_volume": float(np.log1p(np.nansum(vol[i - L:i + 1]))),
                "hl_range": float(np.nanmean(hl)) if np.isfinite(hl).any() else 0.0,
                "observed_frac": len(feat_rets) / L,
                "minute_of_day": float(np.clip(minute_of_day, 0.0, 1.0)),
                "label_sigma30": sigma30,
            })
            i += H
    return pd.DataFrame(rows)


def window_discard_report(df: pd.DataFrame, ticker: str, data: DataConfig) -> dict:
    """Diagnostics on how many candidate windows survived the validity rule."""
    s = data.schema
    w = data.window
    H, L = w.horizon_minutes, w.feature_lookback_minutes
    df = df.copy()
    df["_session"] = _session_key(df[s.datetime])
    candidates = 0
    for _, part in df.groupby("_session"):
        n = len(part)
        if n > L + H:
            candidates += len(range(L, n - H, H))
    built = len(build_windows(df, ticker, data))
    return {
        "ticker": ticker,
        "candidate_windows": candidates,
        "valid_windows": built,
        "discard_rate": 1.0 - (built / candidates) if candidates else 1.0,
    }
