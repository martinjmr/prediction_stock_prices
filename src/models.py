"""Model definitions: Transformer encoder, TCN, and CatBoost wrappers."""
import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """Transformer encoder for intraday return sequences."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class TCN(nn.Module):
    """Temporal Convolutional Network with dilated causal convolutions."""

    def __init__(self, n_channels: int = 64, kernel_size: int = 3):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


def get_catboost_model(params: dict | None = None):
    """Return a CatBoostClassifier with given params (or sensible defaults)."""
    pass
