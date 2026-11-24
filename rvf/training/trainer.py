"""Training loop + small prespecified hyperparameter grid (proposal 4.3).

Training duration and learning rate are selected from a SMALL fixed grid using
validation loss (MSE on sigma30). The grid is prespecified here, not tuned per
result. `train_model` trains one configuration; `select_and_train` runs the grid
and keeps the configuration with the best validation RMSPE.

Design notes:
  * Loss is MSE on the sigma30 scale (simple, stable); the reported metric is
    RMSPE, computed separately on val/test.
  * Early stopping on validation loss with a patience window.
  * A fine-tuning entry point accepts an initial state_dict (from pretraining)
    and an `output_only` flag to freeze the body.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..eval.metrics import rmspe
from ..models.mlp import VolMLP
from ..utils.seeding import set_seed


@dataclass
class GridConfig:
    learning_rates: tuple[float, ...] = (1e-2, 1e-3)
    max_epochs: int = 200
    patience: int = 20
    batch_size: int = 256
    hidden: tuple[int, ...] = (64, 32)
    weight_decay: float = 1e-5


@dataclass
class FitResult:
    model: VolMLP
    val_rmspe: float
    lr: float
    epochs_run: int
    history: list[float] = field(default_factory=list)


def _to_tensor(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32))


def train_model(X_tr, y_tr, X_va, y_va, *, lr: float, grid: GridConfig,
                seed: int = 0, init_state: dict | None = None,
                output_only: bool = False) -> FitResult:
    """Train one configuration; return the best-by-val-loss checkpoint."""
    set_seed(seed)
    n_features = X_tr.shape[1]
    model = VolMLP(n_features, hidden=grid.hidden)
    if init_state is not None:
        model.load_state_dict(init_state)
    if output_only:
        model.freeze_body()
        params = model.output_only_params()
    else:
        model.unfreeze_all()
        params = model.parameters()

    opt = torch.optim.Adam(params, lr=lr, weight_decay=grid.weight_decay)
    loss_fn = nn.MSELoss()
    Xtr, ytr = _to_tensor(X_tr, y_tr)
    Xva, yva = _to_tensor(X_va, y_va)
    n = Xtr.shape[0]

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    history: list[float] = []
    rng = np.random.default_rng(seed)

    for epoch in range(1, grid.max_epochs + 1):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, grid.batch_size):
            idx = perm[i:i + grid.batch_size]
            opt.zero_grad()
            pred = model(Xtr[idx])
            loss = loss_fn(pred, ytr[idx])
            loss.backward()
            opt.step()
        # validation
        model.eval()
        with torch.no_grad():
            vpred = model(Xva)
            vloss = float(loss_fn(vpred, yva)) if Xva.shape[0] else float("inf")
        history.append(vloss)
        if vloss < best_val - 1e-9:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= grid.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vpred = model(Xva).numpy() if Xva.shape[0] else np.array([])
    val_metric = rmspe(y_va, vpred).rmspe if Xva.shape[0] else float("nan")
    return FitResult(model, val_metric, lr, best_epoch, history)


def select_and_train(X_tr, y_tr, X_va, y_va, *, grid: GridConfig, seed: int = 0,
                     init_state: dict | None = None, output_only: bool = False
                     ) -> FitResult:
    """Run the LR grid; keep the configuration with the best validation RMSPE."""
    best: FitResult | None = None
    for lr in grid.learning_rates:
        res = train_model(X_tr, y_tr, X_va, y_va, lr=lr, grid=grid, seed=seed,
                          init_state=init_state, output_only=output_only)
        if best is None or (np.isfinite(res.val_rmspe)
                            and res.val_rmspe < best.val_rmspe):
            best = res
    assert best is not None
    return best


@torch.no_grad()
def predict(model: VolMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    if X.shape[0] == 0:
        return np.array([])
    return model(torch.as_tensor(X, dtype=torch.float32)).numpy()
