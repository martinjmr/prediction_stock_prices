"""Model definitions: LSTM, Transformer encoder, TCN, and CatBoost wrappers."""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM pour classification binaire sur séquences de returns intraday.

    Entrée  : (batch, seq_len, 1)  — séquence de returns bruts
              + optionnel (batch, extra_size) — features engineerées
    Sortie  : (batch,)  — probabilité d'un return positif (après Sigmoid)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        extra_size: int = 0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,        # attend (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        head_in = hidden_size + extra_size
        mid     = max(head_in // 2, 16)
        self.head = nn.Sequential(
            nn.Linear(head_in, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        # x : (batch, seq_len, 1)
        _, (h_n, _) = self.lstm(x)
        last_hidden  = h_n[-1]           # (batch, hidden_size)
        if extra is not None and extra.shape[-1] > 0:
            last_hidden = torch.cat([last_hidden, extra], dim=-1)
        return self.head(last_hidden).squeeze(1)  # (batch,)


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
