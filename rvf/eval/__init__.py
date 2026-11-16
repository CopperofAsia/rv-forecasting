from __future__ import annotations

from .metrics import (RMSPEResult, efficiency_threshold,
                      relative_error_difference, rmspe)

__all__ = ["rmspe", "RMSPEResult", "relative_error_difference",
           "efficiency_threshold"]

# experiment/plots import torch & matplotlib lazily; keep this module importable
# without them for the pure-numpy metrics path.
try:
    from .experiment import run_experiment  # noqa: F401
    from .plots import plot_rmspe_curves     # noqa: F401
    __all__ += ["run_experiment", "plot_rmspe_curves"]
except Exception:  # pragma: no cover - optional heavy deps
    pass
