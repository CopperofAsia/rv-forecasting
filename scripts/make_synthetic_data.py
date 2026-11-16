"""Generate synthetic 1-minute OHLCV files matching the HF Data Library schema.

This lets the whole data layer be exercised end-to-end WITHOUT the real IEX
download (which each team member must obtain under their own account). It is a
testing fixture only -- NOT research data. Prices follow a simple GBM with an
intraday volatility smile; some minutes are randomly dropped to mimic IEX
sparsity so the discard/gap-rate logic has something to chew on.

Usage:
    python -I scripts/make_synthetic_data.py --out data/raw --n-tickers 8 --days 90
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def session_minutes(day: pd.Timestamp) -> pd.DatetimeIndex:
    start = day + pd.Timedelta(hours=9, minutes=30)
    return pd.date_range(start, periods=390, freq="1min")


def make_ticker(symbol: str, days: list[pd.Timestamp], seed: int,
                drop_prob: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_vol = rng.uniform(0.00015, 0.0004)  # per-minute sigma, ticker-specific
    price = rng.uniform(20, 400)
    frames = []
    for day in days:
        idx = session_minutes(day)
        m = np.arange(390)
        smile = 1.0 + 0.8 * np.exp(-((m - 0) / 60) ** 2) + 0.8 * np.exp(-((m - 389) / 60) ** 2)
        rets = rng.normal(0, base_vol * smile)
        closes = price * np.exp(np.cumsum(rets))
        price = float(closes[-1])
        opens = np.concatenate([[closes[0]], closes[:-1]])
        spread = np.abs(rng.normal(0, base_vol, 390)) * closes
        highs = np.maximum(opens, closes) + spread
        lows = np.minimum(opens, closes) - spread
        vols = rng.integers(50, 5000, 390)
        df = pd.DataFrame({
            "datetime": idx, "Open": opens, "High": highs, "Low": lows,
            "Close": closes, "Volume": vols, "source": "iex",
        })
        keep = rng.random(390) > drop_prob
        frames.append(df.loc[keep])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--n-tickers", type=int, default=8)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--drop-prob", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                    help="csv avoids a pyarrow dependency for quick testing")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    start = pd.Timestamp("2024-01-02")
    days = list(pd.bdate_range(start, periods=args.days))
    symbols = [f"SYN{i:02d}" for i in range(args.n_tickers)]
    for k, sym in enumerate(symbols):
        df = make_ticker(sym, days, seed=args.seed + k, drop_prob=args.drop_prob)
        path = os.path.join(args.out, f"{sym}.{args.format}")
        if args.format == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        print(f"wrote {path}  rows={len(df)}")


if __name__ == "__main__":
    main()
