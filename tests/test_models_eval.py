"""Verification of metrics, baselines, and the training/transfer wiring.

Pure-numpy checks (metrics, baselines) always run. The torch-dependent smoke
test is skipped automatically if torch is not installed.
Run:  python -I tests/test_models_eval.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvf.config import Config
from rvf.data import build_windows, assign_split, partition_stocks
from rvf.eval.metrics import (efficiency_threshold, relative_error_difference,
                              rmspe)
from rvf.models.baselines import (HARBaseline, LinearRegression,
                                  PersistenceBaseline)
from rvf.data.windows import FEATURE_NAMES


def _toy(day: str, n: int = 390) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:30", periods=n, freq="1min")
    rng = np.random.default_rng(len(day))
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0004, n)))
    return pd.DataFrame({"datetime": idx, "Open": close, "High": close * 1.001,
                         "Low": close * 0.999, "Close": close,
                         "Volume": rng.integers(100, 1000, n), "source": "iex"})


def test_rmspe_excludes_and_counts_zero_labels():
    y = np.array([0.0, 1.0, 2.0, 4.0])
    yhat = np.array([5.0, 1.1, 1.8, 4.4])
    r = rmspe(y, yhat)
    assert r.n_zero_label == 1 and r.n_used == 3 and r.n_total == 4
    # manual: residual%^2 over the 3 positive labels
    exp = np.sqrt(np.mean([(0.1/1)**2, (-0.2/2)**2, (0.4/4)**2]))
    assert np.isclose(r.rmspe, exp)


def test_rmspe_perfect_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert np.isclose(rmspe(y, y).rmspe, 0.0)


def test_efficiency_threshold_no_interpolation():
    e_ref = 1.0
    f2e = {0.0: 2.0, 0.01: 1.5, 0.05: 1.06, 0.10: 1.04, 0.25: 1.02, 1.0: 1.0}
    # tol=1.05 -> first fraction with error <= 1.05 is 0.10 (0.05 is 1.06 > 1.05)
    assert efficiency_threshold(f2e, e_ref, tol=1.05) == 0.10
    # a method that never qualifies -> None
    assert efficiency_threshold({0.1: 2.0, 1.0: 1.9}, e_ref, tol=1.05) is None


def test_relative_error_difference():
    assert np.isclose(relative_error_difference(1.1, 1.0), 0.1)


def test_persistence_predicts_rv_lookback():
    cfg = Config()
    w = build_windows(_toy("2024-02-01"), "S", cfg.data)
    X = w[list(FEATURE_NAMES)].to_numpy(float)
    pred = PersistenceBaseline().predict(X)
    assert np.allclose(pred, np.clip(w["rv_lookback"].to_numpy(), 0, None))


def test_linear_and_har_run_and_are_nonneg():
    cfg = Config()
    frames = [_toy(f"2024-03-{d+1:02d}") for d in range(12)]
    w = build_windows(pd.concat(frames, ignore_index=True), "TGT", cfg.data)
    X = w[list(FEATURE_NAMES)].to_numpy(float)
    y = w["label_sigma30"].to_numpy(float)
    lin = LinearRegression().fit(X, y)
    assert (lin.predict(X) >= 0).all()
    w["split"] = "train"
    w["ticker"] = "TGT"
    har = HARBaseline().fit_frame(w)
    yt, yp = har.predict_frame(w)
    assert (yp >= 0).all() and len(yt) == len(yp)


def test_torch_training_smoke():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP test_torch_training_smoke (torch not installed)")
        return
    from rvf.training.trainer import GridConfig, predict, select_and_train
    cfg = Config()
    frames = [_toy(f"2024-04-{d+1:02d}") for d in range(10)]
    w = build_windows(pd.concat(frames, ignore_index=True), "TGT", cfg.data)
    X = w[list(FEATURE_NAMES)].to_numpy(float)
    y = w["label_sigma30"].to_numpy(float)
    k = int(len(X) * 0.7)
    grid = GridConfig(learning_rates=(1e-2,), max_epochs=15, patience=5)
    fit = select_and_train(X[:k], y[:k], X[k:], y[k:], grid=grid, seed=0)
    p = predict(fit.model, X[k:])
    assert p.shape[0] == len(X) - k
    assert (p >= 0).all()  # softplus output is nonnegative


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} model/eval checks passed.")
