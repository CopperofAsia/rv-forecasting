"""Run the controlled comparison and assemble the results table (proposal 5).

Methods compared per target stock and fraction:
  * persistence, linear       -- fraction-independent baselines (trained on the
    target's full train split; shown as flat references)
  * har                       -- HAR-style baseline (flat reference)
  * scratch                   -- target-only MLP at each fraction
  * finetune_full             -- pretrained + full fine-tune at each fraction
  * finetune_head             -- pretrained + output-layer-only at each fraction
  * zeroshot                  -- pretrained, no adaptation (the 0% point)

For each (method, target, fraction) we record test RMSPE averaged over seeds.
E_ref = scratch at fraction 1.0 for that target. Efficiency threshold per method
is the smallest fraction with mean RMSPE <= tol * E_ref (no interpolation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.datamodule import make_target_dataset
from ..data.splits import Partition, source_pretrain_pool
from ..eval.metrics import efficiency_threshold, rmspe
from ..models.baselines import HARBaseline, LinearRegression, PersistenceBaseline
from ..training.trainer import GridConfig, predict, select_and_train
from ..training.transfer import finetune_target, pretrain_source, zero_shot


def _mean_rmspe_over_seeds(fn, seeds) -> float:
    vals = [fn(s) for s in seeds]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def run_baselines(windows: pd.DataFrame, ticker: str) -> dict[str, float]:
    """Fraction-independent baselines on the target's full train split."""
    ds = make_target_dataset(windows, ticker, 1.0)
    tgt = windows[windows["ticker"] == ticker]
    test = tgt[tgt["split"] == "test"]
    out: dict[str, float] = {}
    # Persistence predicts the most recently observed realized vol. It must use
    # the RAW rv_lookback (same scale as the sigma30 label), NOT the standardized
    # feature matrix -- z-scored inputs would make its RMSPE meaningless.
    y_test = test["label_sigma30"].to_numpy(dtype=float)
    pers_pred = PersistenceBaseline().predict_raw(test)
    out["persistence"] = rmspe(y_test, pers_pred).rmspe
    # Linear regression is scale-equivariant (OLS with intercept), so the
    # standardized matrix is fine here.
    lin = LinearRegression().fit(ds.X_train, ds.y_train)
    out["linear"] = rmspe(ds.y_test, lin.predict(ds.X_test)).rmspe
    har = HARBaseline().fit_frame(tgt[tgt["split"] == "train"])
    yt, yp = har.predict_frame(tgt[tgt["split"] == "test"])
    out["har"] = rmspe(yt, yp).rmspe
    return out


def run_one_target(windows: pd.DataFrame, ticker: str, pretrained: dict, *,
                   fractions, seeds, grid: GridConfig) -> list[dict]:
    """All method results for a single target stock. Pretrained models (one per
    seed) are passed in so the expensive source pretraining is shared across
    targets rather than repeated."""
    rows = []
    base = run_baselines(windows, ticker)
    for name, val in base.items():
        for f in fractions:
            rows.append({"target": ticker, "method": name,
                         "fraction": f, "rmspe": val})

    def zs(s):
        y, p = zero_shot(windows, ticker, pretrained=pretrained[s], grid=grid)
        return rmspe(y, p).rmspe
    rows.append({"target": ticker, "method": "zeroshot",
                 "fraction": 0.0, "rmspe": _mean_rmspe_over_seeds(zs, seeds)})

    for f in fractions:
        if f == 0.0:
            continue  # scratch/finetune undefined with 0 target rows

        def scratch(s):
            ds = make_target_dataset(windows, ticker, f)
            if ds.n_train == 0:
                return float("nan")
            fit = select_and_train(ds.X_train, ds.y_train, ds.X_val,
                                   ds.y_val, grid=grid, seed=s)
            return rmspe(ds.y_test, predict(fit.model, ds.X_test)).rmspe

        def ft(s, output_only):
            fit, ds = finetune_target(windows, ticker, f,
                                      pretrained=pretrained[s], grid=grid,
                                      seed=s, output_only=output_only)
            if ds.n_train == 0:
                return float("nan")
            return rmspe(ds.y_test, predict(fit.model, ds.X_test)).rmspe

        rows.append({"target": ticker, "method": "scratch", "fraction": f,
                     "rmspe": _mean_rmspe_over_seeds(scratch, seeds)})
        rows.append({"target": ticker, "method": "finetune_full", "fraction": f,
                     "rmspe": _mean_rmspe_over_seeds(lambda s: ft(s, False), seeds)})
        rows.append({"target": ticker, "method": "finetune_head", "fraction": f,
                     "rmspe": _mean_rmspe_over_seeds(lambda s: ft(s, True), seeds)})
    return rows


def pretrain_all_seeds(windows: pd.DataFrame, part: Partition, *, seeds,
                       grid: GridConfig, max_samples: int | None = None) -> dict:
    """Pretrain one source model per seed (shared across all targets)."""
    pool = source_pretrain_pool(windows, part)
    return {s: pretrain_source(pool, grid=grid, seed=s, max_samples=max_samples)
            for s in seeds}


def run_experiment(windows: pd.DataFrame, part: Partition, *,
                   fractions, seeds=(0, 1, 2), grid: GridConfig | None = None,
                   tol: float = 1.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (long_results, efficiency_summary) for all targets in one pass."""
    grid = grid or GridConfig()
    pretrained = pretrain_all_seeds(windows, part, seeds=seeds, grid=grid)
    rows = []
    for ticker in part.target_tickers:
        rows.extend(run_one_target(windows, ticker, pretrained,
                                   fractions=fractions, seeds=seeds, grid=grid))

    results = pd.DataFrame(rows)
    summary = _efficiency_summary(results, fractions, tol)
    return results, summary


# The data-efficiency threshold is defined over the TARGET-DATA fraction, so it
# only applies to methods that actually vary with that fraction. The flat
# baselines (persistence/linear/har) train on the full target split regardless of
# fraction, so a "threshold fraction" for them is meaningless -- they are excluded
# and shown only as reference lines in the plots.
_FRACTION_DEPENDENT = ("scratch", "finetune_full", "finetune_head", "zeroshot")


def _efficiency_summary(results: pd.DataFrame, fractions, tol: float
                        ) -> pd.DataFrame:
    rows = []
    for ticker, g in results.groupby("target"):
        scratch = g[g["method"] == "scratch"]
        ref = scratch[scratch["fraction"] == 1.0]["rmspe"]
        e_ref = float(ref.iloc[0]) if len(ref) else float("nan")
        for method, gm in g.groupby("method"):
            if method not in _FRACTION_DEPENDENT:
                continue
            f2e = {float(r.fraction): float(r.rmspe) for r in gm.itertuples()}
            thr = efficiency_threshold(f2e, e_ref, tol=tol)
            rows.append({"target": ticker, "method": method, "e_ref": e_ref,
                         "efficiency_threshold": thr})
    return pd.DataFrame(rows)
