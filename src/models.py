"""Neural models. Both read the 71 returns (xr) and the engineered inputs (xf), and return a logit."""
import torch
import torch.nn as nn


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer: every input value becomes a token, prediction from a CLS token."""

    def __init__(self, n_raw, n_feat, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        n_total = n_raw + n_feat
        self.W = nn.Parameter(torch.empty(n_total, d_model))
        self.b = nn.Parameter(torch.zeros(n_total, d_model))
        nn.init.normal_(self.W, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
        )

    def forward(self, xr, xf):
        x = torch.cat([xr, xf], dim=1)
        tokens = x.unsqueeze(2) * self.W + self.b
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return self.head(self.norm(self.transformer(tokens)[:, 0])).squeeze(1)


class LSTMClassifier(nn.Module):
    """LSTM over the 71 returns of the day; its last hidden state is joined to the engineered inputs."""

    def __init__(self, n_feat, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        head_in = hidden_size + n_feat
        mid = max(head_in // 2, 16)
        self.head = nn.Sequential(nn.Linear(head_in, mid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mid, 1))

    def forward(self, xr, xf):
        _, (h_n, _) = self.lstm(xr.unsqueeze(-1))  # (batch, 71, 1): one return per time step
        return self.head(torch.cat([h_n[-1], xf], dim=1)).squeeze(1)
