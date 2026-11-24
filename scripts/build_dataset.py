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
    ap.add_argument("--reuse-partition", default=None,
                    help="path to an existing partition.json whose source/target "
                         "assignment + cutoffs to reuse (for apples-to-apples "
                         "robustness runs, e.g. Raw vs Clean)")
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
    for i, (tk, path) in enumerate(tickers.items(), 1):
        df = load_ticker(path, cfg.data)
        audits[tk] = audit_ticker(df, tk, cfg.data)
        w = build_windows(df, tk, cfg.data)          # build once...
        windows.append(w)
        discards.append(window_discard_report(df, tk, cfg.data, built=len(w)))
        if i % 25 == 0:
            print(f"  processed {i}/{len(tickers)} tickers")

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
    if args.reuse_partition:
        # Robustness runs (e.g. Raw vs Clean) must use the SAME source/target
        # assignment and cutoffs as the reference run, or the comparison is not
        # apples-to-apples. Load them from an existing partition.json. Any stock
        # not eligible in this version is dropped from its set (reported).
        import json as _json
        from rvf.data.splits import Partition
        with open(args.reuse_partition) as fh:
            ref = _json.load(fh)
        elig = set(eligible)
        src = tuple(t for t in ref["source_tickers"] if t in elig)
        tgt = tuple(t for t in ref["target_tickers"] if t in elig)
        dropped = (set(ref["source_tickers"]) | set(ref["target_tickers"])) - elig
        part = Partition(src, tgt, pd.Timestamp(ref["val_start"]),
                         pd.Timestamp(ref["test_start"]))
        print(f"Reusing partition from {args.reuse_partition}: "
              f"{len(src)} source, {len(tgt)} target"
              + (f" ({len(dropped)} ref stocks not eligible here: "
                 f"{sorted(dropped)})" if dropped else ""))
        eligible_windows = eligible_windows[
            eligible_windows["ticker"].isin(set(src) | set(tgt))].copy()
    else:
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
