"""Assemble (X, y) arrays from processed windows for a given training stage.

This is the bridge between the tidy window frame (data/processed/windows.parquet)
and the models. It enforces the proposal's scaling rule: the Standardizer is fit
ONLY on the rows permitted to the relevant training stage, then applied to val
and test. Nothing is ever fit on evaluation data.

A `Dataset` bundle carries the standardized feature matrices and raw sigma30
labels for train/val/test, plus the fitted scaler so the same transform can be
reused when fine-tuning from a pretrained model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .splits import Standardizer, target_train_at_fraction
from .windows import FEATURE_NAMES


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: Standardizer
    n_train: int

    @property
    def n_features(self) -> int:
        return len(FEATURE_NAMES)


def _xy(df: pd.DataFrame, scaler: Standardizer) -> tuple[np.ndarray, np.ndarray]:
    if len(df) == 0:
        return (np.empty((0, len(FEATURE_NAMES))), np.empty((0,)))
    X = scaler.transform(df)
    y = df["label_sigma30"].to_numpy(dtype=float)
    return X, y


def make_target_dataset(windows: pd.DataFrame, ticker: str, fraction: float
                        ) -> Dataset:
    """Target-only dataset at a given training fraction.

    Scaler is fit on the target stock's permitted train subset only. val/test are
    that stock's full val/test windows (fixed across fractions, per 3.3).
    """
    tgt = windows[windows["ticker"] == ticker]
    train = target_train_at_fraction(windows, ticker, fraction)
    val = tgt[tgt["split"] == "val"]
    test = tgt[tgt["split"] == "test"]

    scaler = Standardizer()
    if len(train) > 0:
        scaler.fit(train)
    else:
        # fraction == 0.0 (zero-shot): no target rows are permitted, so we must
        # NOT fit on val/test (that would touch evaluation data). Use an identity
        # scaler; this Dataset's X is never used to train a model at fraction 0.
        n = len(FEATURE_NAMES)
        scaler.mean_ = np.zeros(n)
        scaler.std_ = np.ones(n)

    Xtr, ytr = _xy(train, scaler)
    Xva, yva = _xy(val, scaler)
    Xte, yte = _xy(test, scaler)
    return Dataset(Xtr, ytr, Xva, yva, Xte, yte, scaler, n_train=len(train))


def make_source_dataset(source_pool: pd.DataFrame) -> Dataset:
    """Pretraining dataset: all source-pool windows (already cut at val_start).

    Scaler is fit on the source pool. A small val split is carved chronologically
    from the pool tail for early stopping; there is no test here (targets own it).
    """
    pool = source_pool.sort_values("decision_time")
    n = len(pool)
    cut = int(n * 0.9)
    train, val = pool.iloc[:cut], pool.iloc[cut:]
    scaler = Standardizer().fit(train)
    Xtr, ytr = _xy(train, scaler)
    Xva, yva = _xy(val, scaler)
    empty = np.empty((0, len(FEATURE_NAMES)))
    return Dataset(Xtr, ytr, Xva, yva, empty, np.empty((0,)), scaler,
                   n_train=len(train))
