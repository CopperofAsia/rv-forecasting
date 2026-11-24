"""Small MLP for realized-volatility regression (proposal 4.2/4.3).

The same architecture is used everywhere (target-only scratch, source
pretraining, fine-tuning) so neural comparisons differ only in their data and
training, never in capacity. A softplus output makes predictions strictly
nonnegative, respecting the volatility target without post-hoc clipping.

The architecture is deliberately modest: two hidden layers. `output_only_params`
exposes just the final layer so a fine-tuning stage can freeze the body and train
only the head (the proposal's secondary adaptation question).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VolMLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 32),
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = n_features
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # softplus -> strictly positive volatility prediction
        return self.softplus(self.head(self.body(x))).squeeze(-1)

    def output_only_params(self):
        """Parameters of the final layer only (for output-layer-only fine-tuning)."""
        return self.head.parameters()

    def freeze_body(self) -> None:
        for p in self.body.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)
