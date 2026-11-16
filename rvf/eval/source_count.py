"""Secondary analysis: does broader source-stock pretraining improve target
data efficiency? (proposal 4.3 / 6 — "source-stock counts will be varied").

We pretrain the same MLP on nested subsets of the source pool (e.g. 10, 30, 60,
all source stocks), then fine-tune (full) on each target at a few representative
fractions. Everything else (architecture, target subsets, splits) is held fixed,
so the only thing varying is how many source stocks the representation saw.

Kept deliberately small (few fractions, a seed or two, a handful of targets) so
it runs on CPU in a reasonable time; widen via the script flags if you have time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.splits import Partition, source_pretrain_pool
from ..eval.metrics import rmspe
from ..training.trainer import GridConfig, predict
from ..training.transfer import finetune_target, pretrain_source


def _subset_pool(pool: pd.DataFrame, source_tickers, k: int, seed: int
                 ) -> pd.DataFrame:
    """Pool restricted to k source stocks, chosen by a seeded permutation."""
    rng = np.random.default_rng(seed)
    chosen = list(rng.permutation(list(source_tickers))[:k])
    return pool[pool["ticker"].isin(chosen)].copy()


def run_source_count(windows: pd.DataFrame, part: Partition, *,
                     counts, targets, fractions, seeds=(0,),
                     grid: GridConfig | None = None,
                     max_pretrain_samples: int | None = 300000) -> pd.DataFrame:
    """Return long results: (n_source, target, fraction, rmspe) for full fine-tune."""
    grid = grid or GridConfig()
    full_pool = source_pretrain_pool(windows, part)
    n_available = len(part.source_tickers)
    counts = sorted({min(c, n_available) for c in counts})

    rows = []
    for k in counts:
        # Pretrain once per (count, seed); fine-tune all targets/fractions on it.
        pre = {}
        for s in seeds:
            pool_k = _subset_pool(full_pool, part.source_tickers, k, seed=s)
            pre[s] = pretrain_source(pool_k, grid=grid, seed=s,
                                     max_samples=max_pretrain_samples)
        for ticker in targets:
            for f in fractions:
                vals = []
                for s in seeds:
                    fit, ds = finetune_target(windows, ticker, f,
                                              pretrained=pre[s], grid=grid, seed=s)
                    if ds.n_train == 0:
                        continue
                    vals.append(rmspe(ds.y_test, predict(fit.model, ds.X_test)).rmspe)
                vals = [v for v in vals if np.isfinite(v)]
                rows.append({"n_source": k, "target": ticker, "fraction": f,
                             "rmspe": float(np.mean(vals)) if vals else float("nan")})
    return pd.DataFrame(rows)
