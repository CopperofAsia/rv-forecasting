"""Source/target partition, chronological splits, scaling, nested fractions.

Leakage controls (proposal 3.3):
  * Source and target stock sets are DISJOINT.
  * Splits are chronological, by a SHARED calendar cutoff computed over the
    pooled decision times. train < val_start <= val < test_start <= test.
  * Source pretraining may use only source-stock windows with decision_time
    strictly before val_start (the shared cutoff). No source example from the
    target evaluation period can enter pretraining.
  * Nested target-training fractions take the EARLIEST chronological rows, so
    a smaller fraction is always a prefix of a larger one.
  * The feature standardizer is fit ONLY on the rows permitted to a stage; the
    caller passes those rows in. Nothing is fit on val/test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config, DataConfig
from .windows import FEATURE_NAMES


@dataclass(frozen=True)
class Partition:
    source_tickers: tuple[str, ...]
    target_tickers: tuple[str, ...]
    val_start: pd.Timestamp     # the shared source cutoff
    test_start: pd.Timestamp


def partition_stocks(eligible: list[str], cfg: Config,
                     all_windows: pd.DataFrame) -> Partition:
    """Assign stocks to disjoint source/target sets and compute shared cutoffs.

    Cutoffs are quantiles of the pooled decision_time, so every stock shares the
    same train/val/test calendar boundaries.
    """
    sp = cfg.data.split
    rng = np.random.default_rng(sp.seed)
    tickers = sorted(eligible)
    n_target = max(1, round(len(tickers) * sp.target_stock_fraction))
    perm = rng.permutation(len(tickers))
    target_idx = set(perm[:n_target].tolist())
    target = tuple(sorted(tickers[i] for i in target_idx))
    source = tuple(sorted(tickers[i] for i in range(len(tickers))
                          if i not in target_idx))

    t = pd.to_datetime(all_windows["decision_time"]).sort_values()
    val_start = t.quantile(sp.train_frac)
    test_start = t.quantile(sp.train_frac + sp.val_frac)
    return Partition(source, target,
                     pd.Timestamp(val_start), pd.Timestamp(test_start))


def assign_split(windows: pd.DataFrame, part: Partition) -> pd.DataFrame:
    """Tag each window row with split in {train, val, test} by shared cutoffs."""
    out = windows.copy()
    dt = pd.to_datetime(out["decision_time"])
    split = np.where(dt < part.val_start, "train",
                     np.where(dt < part.test_start, "val", "test"))
    out["split"] = split
    return out


def source_pretrain_pool(windows: pd.DataFrame, part: Partition) -> pd.DataFrame:
    """Source-stock windows strictly before the shared cutoff (val_start)."""
    dt = pd.to_datetime(windows["decision_time"])
    mask = windows["ticker"].isin(part.source_tickers) & (dt < part.val_start)
    return windows.loc[mask].copy()


def target_train_at_fraction(windows: pd.DataFrame, ticker: str,
                             fraction: float) -> pd.DataFrame:
    """Earliest chronological `fraction` of one target stock's TRAIN rows.

    fraction == 0.0 returns an empty frame (the zero-shot reference; a
    target-only model is undefined there and must be handled by the caller).
    """
    rows = windows[(windows["ticker"] == ticker) & (windows["split"] == "train")]
    rows = rows.sort_values("decision_time")
    if fraction <= 0.0:
        return rows.iloc[0:0].copy()
    k = int(np.floor(len(rows) * fraction))
    return rows.iloc[:k].copy()


class Standardizer:
    """z-score standardizer fit on permitted rows only (numpy, no sklearn dep)."""

    def __init__(self, columns: tuple[str, ...] = FEATURE_NAMES):
        self.columns = columns
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "Standardizer":
        X = df[list(self.columns)].to_numpy(dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)  # guard constant columns
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Standardizer used before fit().")
        X = df[list(self.columns)].to_numpy(dtype=float)
        return (X - self.mean_) / self.std_
