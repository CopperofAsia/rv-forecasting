"""Forecast-accuracy and data-efficiency metrics (proposal 5.1-5.2).

RMSPE is the primary metric. It requires positive labels, so zero-volatility
labels are EXCLUDED and COUNTED separately (never silently dropped or treated as
a tiny positive). Lower is better.

    RMSPE = sqrt( mean_i [ (y_i - yhat_i) / y_i ]^2 )     over y_i > 0

Data efficiency uses a shared reference E_ref (the target-only MLP at 100% target
data) and reports the SMALLEST evaluated fraction f with E_m(f) <= tol * E_ref.
No interpolation between measured fractions; "never meets" is a valid result.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RMSPEResult:
    rmspe: float
    n_used: int           # positive-label windows actually scored
    n_zero_label: int     # zero-volatility labels excluded (reported, per 5.1)
    n_total: int

    def as_dict(self) -> dict:
        return {"rmspe": self.rmspe, "n_used": self.n_used,
                "n_zero_label": self.n_zero_label, "n_total": self.n_total}


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> RMSPEResult:
    """Root mean squared percentage error over strictly-positive labels."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    n_total = y_true.size
    pos = y_true > 0
    n_zero = int(np.sum(y_true == 0))
    yt, yp = y_true[pos], y_pred[pos]
    if yt.size == 0:
        return RMSPEResult(float("nan"), 0, n_zero, n_total)
    val = float(np.sqrt(np.mean(((yt - yp) / yt) ** 2)))
    return RMSPEResult(val, int(yt.size), n_zero, n_total)


def relative_error_difference(e_m: float, e_ref: float) -> float:
    """D_m(f) = (E_m(f) - E_ref) / E_ref  (proposal 5.2)."""
    if e_ref <= 0 or not np.isfinite(e_ref):
        return float("nan")
    return (e_m - e_ref) / e_ref


def efficiency_threshold(fraction_to_error: dict[float, float], e_ref: float,
                         tol: float = 1.05) -> float | None:
    """Smallest evaluated fraction f with E_m(f) <= tol * E_ref.

    Returns None if no measured fraction meets the criterion (a valid outcome).
    Fractions are considered in ascending order; no interpolation is performed.
    """
    if not np.isfinite(e_ref) or e_ref <= 0:
        return None
    target = tol * e_ref
    for f in sorted(fraction_to_error):
        e = fraction_to_error[f]
        if np.isfinite(e) and e <= target:
            return f
    return None
