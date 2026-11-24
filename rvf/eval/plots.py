"""RMSPE-vs-fraction curves (proposal 5.3). matplotlib only."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless: write files, no display
import matplotlib.pyplot as plt
import pandas as pd

_CURVE_METHODS = ["scratch", "finetune_full", "finetune_head", "zeroshot"]
_FLAT_METHODS = ["persistence", "linear", "har"]


def plot_rmspe_curves(results: pd.DataFrame, out_dir: str) -> list[str]:
    """One figure per target stock: RMSPE vs target-data fraction."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ticker, g in results.groupby("target"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for m in _CURVE_METHODS:
            gm = g[g["method"] == m].sort_values("fraction")
            if len(gm):
                ax.plot(gm["fraction"], gm["rmspe"], marker="o", label=m)
        for m in _FLAT_METHODS:
            gm = g[g["method"] == m]
            if len(gm):
                ax.axhline(float(gm["rmspe"].iloc[0]), linestyle="--",
                           alpha=0.6, label=m)
        ax.set_xlabel("target-data fraction")
        ax.set_ylabel("test RMSPE")
        ax.set_title(f"RMSPE vs target data — {ticker}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(out_dir, f"rmspe_curve_{ticker}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths
