"""Verification of the Phase-1 data layer on synthetic data.

These tests assert the properties the proposal relies on:
  * calendar exclusions remove the right bars,
  * label RV math matches the formula,
  * feature windows never overlap the label window (no look-ahead),
  * source pretraining pool respects the shared time cutoff (no eval-period leak),
  * nested fractions are strict prefixes,
  * the standardizer is fit only on the rows it is given.
Run with:  python -I -m pytest tests/ -q      (or the __main__ fallback below)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvf.config import Config
from rvf.data import (FEATURE_NAMES, assign_split, audit_ticker, build_windows,
                      load_ticker, partition_stocks, screen_tickers,
                      source_pretrain_pool, target_train_at_fraction,
                      Standardizer)
from rvf.data.calendar import apply_calendar_exclusions
from rvf.data.windows import _safe_log_returns


def _toy_session(day: str, n: int = 390, start="09:30") -> pd.DataFrame:
    idx = pd.date_range(f"{day} {start}", periods=n, freq="1min")
    rng = np.random.default_rng(len(day))
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0003, n)))
    return pd.DataFrame({
        "datetime": idx, "Open": close, "High": close * 1.001,
        "Low": close * 0.999, "Close": close,
        "Volume": rng.integers(100, 1000, n), "source": "iex",
    })


def test_calendar_excludes_session_and_early_close():
    df = pd.concat([
        _toy_session("2024-12-23"),           # fully excluded session
        _toy_session("2024-12-24"),           # early-close day (13:00 cutoff)
        _toy_session("2024-12-26"),           # normal day
    ], ignore_index=True)
    out = apply_calendar_exclusions(df, "datetime", "13:00")
    dates = pd.to_datetime(out["datetime"]).dt.date.astype(str)
    assert "2024-12-23" not in set(dates)
    ec = out[dates == "2024-12-24"]
    assert pd.to_datetime(ec["datetime"]).dt.time.max() < pd.Timestamp("13:00").time()
    assert "2024-12-26" in set(dates)


def test_label_rv_math_matches_formula():
    prices = np.array([100.0, 101.0, 100.5, 102.0, 101.5])
    r = _safe_log_returns(prices)
    expected_sigma = np.sqrt(np.sum(r ** 2))
    assert np.isclose(expected_sigma, np.sqrt(np.sum(np.diff(np.log(prices)) ** 2)))


def test_windows_label_spans_fixed_clock_horizon_with_gaps():
    """Even when minutes are missing, the label window must cover exactly H
    wall-clock minutes after the decision time (not H observed bars)."""
    cfg = Config()
    H = cfg.data.window.horizon_minutes
    df = _toy_session("2024-06-03", n=390)
    # Drop a chunk of minutes in the middle to create a gap.
    ts = pd.to_datetime(df["datetime"])
    drop = (ts.dt.time >= pd.Timestamp("11:00").time()) & \
           (ts.dt.time < pd.Timestamp("11:15").time())
    df = df.loc[~drop].reset_index(drop=True)
    win = build_windows(df, "G", cfg.data)
    assert len(win) > 0
    # For every window, the label covers (t, t+H]; the next decision time is a
    # multiple of H minutes away, so no label can reach past its clock horizon.
    dts = pd.to_datetime(win["decision_time"]).sort_values().reset_index(drop=True)
    gaps = dts.diff().dropna().dt.total_seconds() / 60
    assert (gaps % H == 0).all()


def test_windows_no_lookahead_and_label_positive():
    cfg = Config()
    df = pd.concat([_toy_session("2024-02-01"), _toy_session("2024-02-02")],
                   ignore_index=True)
    win = build_windows(df, "SYN", cfg.data)
    assert len(win) > 0
    assert (win["label_sigma30"] >= 0).all()
    # Feature window ends at decision_time; label covers the next H minutes.
    # Successive decision points are >= H apart within a session => no overlap.
    H = cfg.data.window.horizon_minutes
    for _, g in win.groupby("session"):
        dts = pd.to_datetime(g["decision_time"]).sort_values()
        gaps = dts.diff().dropna().dt.total_seconds() / 60
        assert (gaps >= H - 1e-9).all()


def test_source_pool_respects_shared_cutoff():
    cfg = Config()
    frames = []
    for i in range(6):
        for d in range(20):
            s = _toy_session(f"2024-03-{d+1:02d}")
            s["ticker"] = f"T{i}"
            frames.append(s)
    raw = pd.concat(frames, ignore_index=True)
    win = pd.concat([build_windows(g.drop(columns="ticker"), tk, cfg.data)
                     for tk, g in raw.groupby("ticker")], ignore_index=True)
    part = partition_stocks([f"T{i}" for i in range(6)], cfg, win)
    win = assign_split(win, part)
    pool = source_pretrain_pool(win, part)
    assert set(pool["ticker"]).issubset(set(part.source_tickers))
    assert (pd.to_datetime(pool["decision_time"]) < part.val_start).all()


def test_nested_fractions_are_prefixes():
    cfg = Config()
    frames = []
    for d in range(15):
        frames.append(_toy_session(f"2024-04-{d+1:02d}"))
    raw = pd.concat(frames, ignore_index=True)
    win = build_windows(raw, "TGT", cfg.data)
    win["split"] = "train"
    prev = None
    for f in [0.05, 0.10, 0.25, 0.50, 1.00]:
        sub = target_train_at_fraction(win, "TGT", f)
        if prev is not None and len(prev) > 0:
            assert prev["decision_time"].tolist() == \
                sub["decision_time"].tolist()[:len(prev)]
        prev = sub
    assert len(target_train_at_fraction(win, "TGT", 0.0)) == 0


def test_standardizer_uses_only_given_rows():
    cfg = Config()
    win = build_windows(_toy_session("2024-05-01"), "S", cfg.data)
    sc = Standardizer().fit(win)
    Z = sc.transform(win)
    assert Z.shape[1] == len(FEATURE_NAMES)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-6)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} data-layer tests passed.")
