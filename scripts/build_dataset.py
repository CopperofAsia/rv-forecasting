"""Phase-1 pipeline driver: raw bars -> audited, windowed, split dataset.

Reads a config (defaults + optional YAML overlay), runs the full data layer, and
writes artifacts to results/ and data/processed/:
  * results/liquidity_audit.csv   per-stock screening summary
  * results/discard_report.csv    per-stock window discard rates
  * results/partition.json        source/target sets + shared cutoffs + snapshot
  * data/processed/windows.parquet (or .csv)  all windows with split tags

Usage:
    python -I scripts/build_dataset.py --config configs/example.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvf.config import load_config
from rvf.data import (assign_split, audit_ticker, build_windows,
                      discover_tickers, load_ticker, partition_stocks,
                      screen_tickers, window_discard_report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-format", choices=["parquet", "csv"], default="parquet")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.data.processed_dir, exist_ok=True)

    tickers = discover_tickers(cfg.data)
    if not tickers:
        raise SystemExit(f"No files in {cfg.data.raw_dir!r} matching "
                         f"{cfg.data.file_glob!r}. See docs/DATA.md to download.")
    print(f"Found {len(tickers)} tickers in {cfg.data.raw_dir}")

    audits, windows, discards = {}, [], []
    for tk, path in tickers.items():
        df = load_ticker(path, cfg.data)
        audits[tk] = audit_ticker(df, tk, cfg.data)
        windows.append(build_windows(df, tk, cfg.data))
        discards.append(window_discard_report(df, tk, cfg.data))

    all_windows = pd.concat(windows, ignore_index=True)
    eligible, summary = screen_tickers(audits, cfg.data)
    summary.to_csv(os.path.join(cfg.results_dir, "liquidity_audit.csv"), index=False)
    pd.DataFrame(discards).to_csv(
        os.path.join(cfg.results_dir, "discard_report.csv"), index=False)
    print(f"Eligible stocks: {len(eligible)} / {len(tickers)}")

    if len(eligible) < 2:
        raise SystemExit("Need >=2 eligible stocks to form source/target sets. "
                         "Loosen LiquidityConfig thresholds or add data.")

    eligible_windows = all_windows[all_windows["ticker"].isin(eligible)].copy()
    part = partition_stocks(eligible, cfg, eligible_windows)
    tagged = assign_split(eligible_windows, part)

    out_path = os.path.join(cfg.data.processed_dir, f"windows.{args.out_format}")
    if args.out_format == "parquet":
        tagged.to_parquet(out_path, index=False)
    else:
        tagged.to_csv(out_path, index=False)

    meta = {
        "snapshot": cfg.data.snapshot.__dict__,
        "source_tickers": list(part.source_tickers),
        "target_tickers": list(part.target_tickers),
        "val_start": str(part.val_start),
        "test_start": str(part.test_start),
        "fractions": list(cfg.fractions),
        "efficiency_tolerance": cfg.efficiency_tolerance,
        "n_windows": int(len(tagged)),
        "split_counts": tagged["split"].value_counts().to_dict(),
    }
    with open(os.path.join(cfg.results_dir, "partition.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f"Wrote {out_path} ({len(tagged)} windows)")
    print(f"Source: {part.source_tickers}")
    print(f"Target: {part.target_tickers}")
    print(f"Shared cutoff val_start={part.val_start}  test_start={part.test_start}")


if __name__ == "__main__":
    main()
