"""Phase 2-6 driver: run the controlled comparison from processed windows.

Prereq: scripts/build_dataset.py has produced data/processed/windows.parquet and
results/partition.json. This script reloads them, runs baselines + target-only
MLP + pretrain/fine-tune across the prespecified fractions and seeds, writes the
results table and efficiency summary, and saves RMSPE-vs-fraction plots.

    python -I scripts/run_experiments.py --config configs/example.yaml

Use --quick for a fast smoke run (1 seed, tiny grid) to check the plumbing before
committing to the full multi-seed sweep.
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
from rvf.eval.experiment import _efficiency_summary, run_experiment
from rvf.eval.plots import plot_rmspe_curves
from rvf.training.trainer import GridConfig


def _load_partition(results_dir: str) -> Partition:
    with open(os.path.join(results_dir, "partition.json")) as fh:
        meta = json.load(fh)
    return Partition(
        source_tickers=tuple(meta["source_tickers"]),
        target_tickers=tuple(meta["target_tickers"]),
        val_start=pd.Timestamp(meta["val_start"]),
        test_start=pd.Timestamp(meta["test_start"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--processed", default=None,
                    help="path to windows.parquet (default: <processed_dir>/windows.parquet)")
    ap.add_argument("--quick", action="store_true",
                    help="1 seed + small grid for a fast plumbing check")
    ap.add_argument("--fresh", action="store_true",
                    help="delete any existing results and start over")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N not-yet-done targets this run "
                         "(for chunking a long sweep; re-run to continue)")
    ap.add_argument("--max-pretrain-samples", type=int, default=300000,
                    help="cap source-pool training rows for pretraining (compute; "
                         "a small MLP does not need the full ~1M+). 0 = no cap.")
    args = ap.parse_args()
    if args.max_pretrain_samples == 0:
        args.max_pretrain_samples = None

    cfg = load_config(args.config)
    proc = args.processed or os.path.join(cfg.data.processed_dir, "windows.parquet")
    if not os.path.exists(proc):
        alt = proc.replace(".parquet", ".csv")
        proc = alt if os.path.exists(alt) else proc
    if not os.path.exists(proc):
        raise SystemExit(f"Processed windows not found at {proc}. "
                         f"Run scripts/build_dataset.py first.")
    windows = (pd.read_parquet(proc) if proc.endswith(".parquet")
               else pd.read_csv(proc, parse_dates=["decision_time"]))
    part = _load_partition(cfg.results_dir)

    seeds = (0,) if args.quick else cfg.seeds
    grid = (GridConfig(learning_rates=(1e-2,), max_epochs=40, patience=8)
            if args.quick else GridConfig())

    results_path = os.path.join(cfg.results_dir, "experiment_results.csv")
    # Resume: skip targets already in the results CSV (unless --fresh).
    done: set[str] = set()
    if args.fresh and os.path.exists(results_path):
        os.remove(results_path)
    if os.path.exists(results_path):
        done = set(pd.read_csv(results_path)["target"].unique())

    todo = [t for t in part.target_tickers if t not in done]
    print(f"Targets: {len(part.target_tickers)} total, {len(done)} done, "
          f"{len(todo)} to run | sources: {len(part.source_tickers)} "
          f"| seeds: {seeds} | fractions: {cfg.fractions}")
    if args.limit:
        todo = todo[: args.limit]
        print(f"  (this run limited to {len(todo)} targets)")

    if todo:
        import pickle
        from rvf.eval.experiment import pretrain_all_seeds, run_one_target
        cache = os.path.join(cfg.results_dir, "pretrained.pkl")
        if os.path.exists(cache) and not args.fresh:
            print("  loading cached pretrained source models...")
            with open(cache, "rb") as fh:
                pretrained = pickle.load(fh)
        else:
            print("  pretraining source models (cached for later chunks)...")
            pretrained = pretrain_all_seeds(windows, part, seeds=seeds, grid=grid,
                                            max_samples=args.max_pretrain_samples)
            with open(cache, "wb") as fh:
                pickle.dump(pretrained, fh)
        header_needed = not os.path.exists(results_path)
        for i, ticker in enumerate(todo, 1):
            rows = run_one_target(windows, ticker, pretrained,
                                  fractions=cfg.fractions, seeds=seeds, grid=grid)
            pd.DataFrame(rows).to_csv(results_path, mode="a", index=False,
                                      header=header_needed)
            header_needed = False
            print(f"  [{i}/{len(todo)}] {ticker} done")

    # Recompute summary + plots over everything accumulated so far.
    results = pd.read_csv(results_path)
    summary = _efficiency_summary(results, cfg.fractions, cfg.efficiency_tolerance)
    summary.to_csv(os.path.join(cfg.results_dir, "efficiency_summary.csv"), index=False)
    plots = plot_rmspe_curves(results, cfg.results_dir)

    n_targets = results["target"].nunique()
    print(f"\n=== Efficiency summary over {n_targets} targets (smallest fraction "
          f"within {cfg.efficiency_tolerance}x of scratch@100%) ===")
    print(summary.to_string(index=False))
    print(f"\nWrote experiment_results.csv, efficiency_summary.csv, "
          f"and {len(plots)} plot(s) to {cfg.results_dir}/")


if __name__ == "__main__":
    main()
