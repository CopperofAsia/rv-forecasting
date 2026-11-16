"""Report-ready aggregation of experiment results (for the DDA4220 write-up).

Produces summary tables and figures from results/experiment_results.csv. The team
writes the prose; these are the numbers and plots to drop in.

Artifacts (written to results/report/):
  * summary_by_fraction.csv  -- mean +/- std RMSPE per method x fraction
  * efficiency_by_method.csv -- how often each method meets the 1.05x criterion
  * heterogeneity.csv        -- per-target RMSPE spread (finetune_full vs scratch
                                vs best baseline) at the smallest fraction
  * agg_rmspe_curve.png      -- mean RMSPE vs fraction with +/-1 std bands
  * full_vs_head.png         -- full vs output-layer-only fine-tuning
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURVE = ["scratch", "finetune_full", "finetune_head", "zeroshot"]
FLAT = ["persistence", "linear", "har"]


def summary_by_fraction(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of test RMSPE across targets, per method x fraction."""
    g = results.groupby(["method", "fraction"])["rmspe"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    return out.sort_values(["method", "fraction"])


def efficiency_by_method(summary: pd.DataFrame) -> pd.DataFrame:
    """From the per-target efficiency summary, how often each method meets the
    criterion and the median threshold fraction among those that do."""
    rows = []
    for method, g in summary.groupby("method"):
        met = g["efficiency_threshold"].notna()
        vals = g.loc[met, "efficiency_threshold"]
        rows.append({
            "method": method,
            "n_targets": len(g),
            "n_meeting_criterion": int(met.sum()),
            "median_threshold_fraction": float(vals.median()) if len(vals) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("method")


def heterogeneity(results: pd.DataFrame, smallest_fraction: float) -> pd.DataFrame:
    """Per-target RMSPE at the smallest target-data fraction: finetune_full vs
    scratch vs the best flat baseline. Shows how transfer benefit varies by stock."""
    rows = []
    for ticker, g in results.groupby("target"):
        def at(method, frac):
            sub = g[(g["method"] == method) & (g["fraction"] == frac)]["rmspe"]
            return float(sub.iloc[0]) if len(sub) else np.nan
        best_base = min(at(m, 0.0) for m in FLAT)
        ft = at("finetune_full", smallest_fraction)
        sc = at("scratch", smallest_fraction)
        rows.append({
            "target": ticker,
            "finetune_full": ft,
            "scratch": sc,
            "best_baseline": best_base,
            "transfer_gain_vs_scratch": (sc - ft) / sc if sc else np.nan,
            "beats_best_baseline": bool(ft < best_base) if np.isfinite(ft) else False,
        })
    return pd.DataFrame(rows).sort_values("transfer_gain_vs_scratch", ascending=False)


def plot_aggregate_curve(results: pd.DataFrame, path: str) -> None:
    """Mean RMSPE vs fraction across targets, with +/-1 std bands for the
    fraction-dependent methods and dashed reference lines for flat baselines."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in CURVE:
        g = results[results["method"] == m].groupby("fraction")["rmspe"]
        mean, std = g.mean(), g.std()
        xs = mean.index.to_numpy()
        if len(xs) == 0:
            continue
        ax.plot(xs, mean.to_numpy(), marker="o", label=m)
        ax.fill_between(xs, (mean - std).to_numpy(), (mean + std).to_numpy(),
                        alpha=0.15)
    for m in FLAT:
        g = results[results["method"] == m]["rmspe"]
        if len(g):
            ax.axhline(float(g.mean()), linestyle="--", alpha=0.6, label=f"{m} (ref)")
    ax.set_xlabel("target-data fraction")
    ax.set_ylabel("mean test RMSPE (30 targets)")
    ax.set_title("RMSPE vs target-data fraction (mean +/- 1 std across targets)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_full_vs_head(results: pd.DataFrame, path: str) -> None:
    """Full-parameter vs output-layer-only fine-tuning, mean across targets."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in ["finetune_full", "finetune_head", "scratch"]:
        g = results[results["method"] == m].groupby("fraction")["rmspe"].mean()
        if len(g):
            ax.plot(g.index.to_numpy(), g.to_numpy(), marker="o", label=m)
    ax.set_xlabel("target-data fraction")
    ax.set_ylabel("mean test RMSPE")
    ax.set_title("Fine-tuning scope: full vs output-layer-only")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_source_count(sc_results: pd.DataFrame, path: str) -> None:
    """RMSPE vs number of source stocks, one line per target-data fraction."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for f, g in sc_results.groupby("fraction"):
        m = g.groupby("n_source")["rmspe"].mean()
        ax.plot(m.index.to_numpy(), m.to_numpy(), marker="o", label=f"frac={f}")
    ax.set_xlabel("number of source stocks in pretraining")
    ax.set_ylabel("mean test RMSPE (fine-tune full)")
    ax.set_title("Does broader source pretraining improve data efficiency?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
