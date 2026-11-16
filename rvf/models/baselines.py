"""Simple forecasting baselines (proposal 4.1).

These are low-complexity references: if neural transfer does not beat them, any
"gain" is not meaningful. All are numpy-only (no torch) and respect the volatility
target by clipping predictions to be nonnegative.

  * Persistence: predict the most recently observed realized volatility
    (the `rv_lookback` feature at the decision point).
  * LinearRegression: ordinary least squares on the fixed feature vector,
    solved with numpy lstsq; predictions clipped at 0.
  * HARBaseline: a HAR-style regression on lagged realized-volatility summaries
    (short / medium / long horizons) built ONLY from windows strictly before
    each decision point, so no current/future-window information leaks in.

Each model exposes fit(X, y) / predict(X) over the standardized feature matrix,
except HAR which needs the tidy window frame to construct its lagged summaries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.windows import FEATURE_NAMES

_RV_IDX = FEATURE_NAMES.index("rv_lookback")


class PersistenceBaseline:
    """yhat = most recently observed realized volatility."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceBaseline":
        return self  # nothing to learn

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from a feature MATRIX. NOTE: only meaningful if X is on the
        RAW (unstandardized) scale, since the prediction is read straight from
        the rv_lookback column and compared against raw sigma30 labels."""
        return np.clip(np.asarray(X, dtype=float)[:, _RV_IDX], 0.0, None)

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Predict from a tidy window frame using the RAW rv_lookback column.
        This is the correct entry point for evaluation (avoids any scaler)."""
        return np.clip(df["rv_lookback"].to_numpy(dtype=float), 0.0, None)


class LinearRegression:
    """OLS via lstsq with an intercept; nonnegative-clipped predictions."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        A = self._design(X)
        self.coef_, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LinearRegression used before fit().")
        return np.clip(self._design(X) @ self.coef_, 0.0, None)


class HARBaseline:
    """HAR-style regression on lagged realized-volatility summaries.

    For each decision point we build three regressors from PAST labels of the
    same stock: the most recent realized vol (short), and rolling means over the
    previous `week`/`month` windows. Because every regressor is strictly lagged
    (shifted by one so the current label is never included), there is no leakage.
    """

    def __init__(self, week: int = 5, month: int = 22) -> None:
        self.week = week
        self.month = month
        self.coef_: np.ndarray | None = None

    def _features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (design matrix, target) from a single ticker's ordered windows."""
        df = df.sort_values("decision_time")
        y = df["label_sigma30"].to_numpy(dtype=float)
        # Lag by one so the current label is excluded from its own regressors.
        past = pd.Series(y).shift(1)
        short = past
        week = past.rolling(self.week, min_periods=1).mean()
        month = past.rolling(self.month, min_periods=1).mean()
        feats = np.column_stack([short.to_numpy(), week.to_numpy(),
                                 month.to_numpy()])
        valid = ~np.isnan(feats).any(axis=1)
        return feats[valid], y[valid]

    def fit_frame(self, df: pd.DataFrame) -> "HARBaseline":
        parts_X, parts_y = [], []
        for _, g in df.groupby("ticker"):
            Xg, yg = self._features(g)
            if Xg.shape[0] == 0:
                continue
            parts_X.append(Xg)
            parts_y.append(yg)
        if not parts_X:
            # No valid lagged rows anywhere -> degenerate fit (predict 0).
            self.coef_ = np.zeros(4)
            return self
        X = np.vstack(parts_X)
        y = np.concatenate(parts_y)
        A = np.hstack([np.ones((X.shape[0], 1)), X])
        self.coef_, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict_frame(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict for a frame; returns (y_true, y_pred) aligned on valid rows.

        Returns empty arrays if no group yields a valid lagged row (e.g. a test
        split shorter than the HAR warmup) -- the caller's RMSPE then reports NaN,
        the correct 'not evaluable' outcome rather than a crash.
        """
        if self.coef_ is None:
            raise RuntimeError("HARBaseline used before fit_frame().")
        preds, trues = [], []
        for _, g in df.groupby("ticker"):
            Xg, yg = self._features(g)
            if Xg.shape[0] == 0:
                continue
            A = np.hstack([np.ones((Xg.shape[0], 1)), Xg])
            preds.append(np.clip(A @ self.coef_, 0.0, None))
            trues.append(yg)
        if not trues:
            return np.empty((0,)), np.empty((0,))
        return np.concatenate(trues), np.concatenate(preds)
