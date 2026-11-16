"""Cross-stock pretraining and fine-tuning (proposal 4.3).

Flow:
  1. pretrain_source(): train the MLP on the pooled source-stock windows
     (already cut at the shared val_start). Returns a state_dict + the source
     scaler.
  2. finetune_target(): for one target stock at one data fraction, initialise
     from the pretrained state and adapt on the permitted target subset. Supports
     full-parameter and output-layer-only adaptation.
  3. zero_shot(): the 0% reference -- apply the pretrained model to target test
     windows with NO target adaptation.

A subtlety handled here: the pretrained model expects features on the SOURCE
scaler. For zero-shot we must transform target windows with the source scaler.
For fine-tuning we re-fit on the target subset (standard transfer practice); this
is a documented choice, kept consistent across the scratch and fine-tune arms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.datamodule import Dataset, make_source_dataset, make_target_dataset
from ..data.splits import Standardizer
from ..data.windows import FEATURE_NAMES
from .trainer import FitResult, GridConfig, predict, select_and_train


@dataclass
class PretrainResult:
    state_dict: dict
    scaler: Standardizer
    val_rmspe: float
    n_source_windows: int


def pretrain_source(source_pool: pd.DataFrame, *, grid: GridConfig,
                    seed: int = 0) -> PretrainResult:
    ds = make_source_dataset(source_pool)
    fit = select_and_train(ds.X_train, ds.y_train, ds.X_val, ds.y_val,
                           grid=grid, seed=seed)
    return PretrainResult({k: v.clone() for k, v in fit.model.state_dict().items()},
                          ds.scaler, fit.val_rmspe, ds.n_train)


def finetune_target(windows: pd.DataFrame, ticker: str, fraction: float, *,
                    pretrained: PretrainResult, grid: GridConfig, seed: int = 0,
                    output_only: bool = False) -> tuple[FitResult, Dataset]:
    """Fine-tune the pretrained model on a target subset. Returns (fit, dataset)."""
    ds = make_target_dataset(windows, ticker, fraction)
    fit = select_and_train(ds.X_train, ds.y_train, ds.X_val, ds.y_val,
                           grid=grid, seed=seed,
                           init_state=pretrained.state_dict,
                           output_only=output_only)
    return fit, ds


def zero_shot(windows: pd.DataFrame, ticker: str, *, pretrained: PretrainResult,
              grid: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    """0% reference: pretrained model on target TEST windows, no adaptation.

    Target features are transformed with the SOURCE scaler, since the model was
    trained in that feature space and receives no target data to re-fit on.
    """
    from ..models.mlp import VolMLP
    import torch

    tgt = windows[(windows["ticker"] == ticker) & (windows["split"] == "test")]
    model = VolMLP(len(FEATURE_NAMES), hidden=grid.hidden)
    model.load_state_dict(pretrained.state_dict)
    X = pretrained.scaler.transform(tgt) if len(tgt) else np.empty((0, len(FEATURE_NAMES)))
    y = tgt["label_sigma30"].to_numpy(dtype=float)
    return y, predict(model, X)
