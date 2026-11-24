"""Build report-ready artifacts from experiment_results.csv, and optionally run
the source-stock-count secondary analysis.

    # aggregates + plots from the main sweep (fast):
    python -I scripts/run_analysis.py --config configs/example.yaml
    # also run the source-count study (slower; pretrains several times):
    python -I scripts/run_analysis.py --config configs/example.yaml --source-count

Writes to results/report/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvf.config import load_config
from rvf.data.splits import Partition
from rvf.eval import report as R
from rvf.training.trainer import GridConfig


def _load_partition(results_dir: str) -> Partition:
    with open(os.path.join(results_dir, "partition.json")) as fh:
        meta = json.load(fh)
    return Partition(tuple(meta["source_tickers"]), tuple(meta["target_tickers"]),
                     pd.Timestamp(meta["val_start"]), pd.Timestamp(meta["test_start"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source-count", action="store_true",
                    help="also run the source-stock-count secondary analysis")
    ap.add_argument("--sc-counts", type=int, nargs="+", default=[10, 30, 60, 0],
                    help="source-stock counts to try (0 = all available)")
    ap.add_argument("--sc-targets", type=int, default=6,
                    help="how many target stocks to use in the source-count study")
    ap.add_argument("--sc-fractions", type=float, nargs="+", default=[0.05, 0.25])
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = os.path.join(cfg.results_dir, "report")
    os.makedirs(out, exist_ok=True)

    results = pd.read_csv(os.path.join(cfg.results_dir, "experiment_results.csv"))
    eff = pd.read_csv(os.path.join(cfg.results_dir, "efficiency_summary.csv"))

    smallest = min(f for f in cfg.fractions if f > 0)
    R.summary_by_fraction(results).to_csv(
        os.path.join(out, "summary_by_fraction.csv"), index=False)
    R.efficiency_by_method(eff).to_csv(
        os.path.join(out, "efficiency_by_method.csv"), index=False)
    R.heterogeneity(results, smallest).to_csv(
        os.path.join(out, "heterogeneity.csv"), index=False)
    R.plot_aggregate_curve(results, os.path.join(out, "agg_rmspe_curve.png"))
    R.plot_full_vs_head(results, os.path.join(out, "full_vs_head.png"))
    print(f"Wrote aggregate tables + plots to {out}/")
    print("\nefficiency_by_method:")
    print(R.efficiency_by_method(eff).to_string(index=False))

    if args.source_count:
        from rvf.eval.source_count import run_source_count
        part = _load_partition(cfg.results_dir)
        proc = os.path.join(cfg.data.processed_dir, "windows.parquet")
        windows = pd.read_parquet(proc)
        counts = [c if c > 0 else len(part.source_tickers) for c in args.sc_counts]
        targets = list(part.target_tickers[: args.sc_targets])
        print(f"\nSource-count study: counts={counts}, targets={targets}, "
              f"fractions={args.sc_fractions}")
        sc = run_source_count(windows, part, counts=counts, targets=targets,
                              fractions=args.sc_fractions, seeds=(0,),
                              grid=GridConfig())
        sc.to_csv(os.path.join(out, "source_count.csv"), index=False)
        R.plot_source_count(sc, os.path.join(out, "source_count.png"))
        print("source_count (mean RMSPE by n_source x fraction):")
        print(sc.pivot_table(index="fraction", columns="n_source",
                             values="rmspe", aggfunc="mean").round(4).to_string())


if __name__ == "__main__":
    main()
