"""
Step 3: Feature Tokenizer Transformer with Optuna
Implements FT-Transformer: tokenizes features, passes through Transformer encoder.
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import optuna
    from optuna.pruners import MedianPruner
except ImportError:
    print("Install: pip install optuna torch")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import set_seed

warnings.filterwarnings('ignore')
set_seed(42)
torch.manual_seed(42)

print("=" * 70)
print("STEP 3: FEATURE TOKENIZER TRANSFORMER")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load features
print("\n[1/5] Loading feature checkpoint...")
checkpoint = np.load('checkpoints/features.npz', allow_pickle=True)
X_train = pd.DataFrame(checkpoint['X_train_feat'], columns=checkpoint['X_train_cols'])
y_train = checkpoint['y_train']
X_test = pd.DataFrame(checkpoint['X_test_feat'], columns=checkpoint['X_test_cols'])

META_COLS = ['ID', 'date', 'eqt_code']
feature_cols = [c for c in X_train.columns if c not in META_COLS]
y_train_binary = (y_train > 0).astype(int)

print(f"  Loaded {X_train.shape[1]} features, {len(feature_cols)} for training")

# Train-val split by date
print("\n[2/5] Train-val split by date...")
dates = X_train['date'].unique()
np.random.seed(42)
np.random.shuffle(dates)
n_val = max(1, int(len(dates) * 0.2))
val_dates = set(dates[-n_val:])

mask_val = X_train['date'].isin(val_dates)
X_tr = X_train[~mask_val][feature_cols].values.astype(np.float32)
y_tr = y_train_binary[~mask_val].values.astype(np.float32)
X_va = X_train[mask_val][feature_cols].values.astype(np.float32)
y_va = y_train_binary[mask_val].values.astype(np.float32)

# Scale
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
X_va_scaled = scaler.transform(X_va).astype(np.float32)

print(f"  Train: {X_tr_scaled.shape}")
print(f"  Val:   {X_va_scaled.shape}")

# Define FT-Transformer model
class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer: tokens from features + Transformer encoder."""

    def __init__(self, n_features, d_model=128, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)  # Tokenize each feature
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, n_features + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, n_features)
        B = x.shape[0]

        # Tokenize: reshape to (batch, n_features, 1) -> embed -> (batch, n_features, d_model)
        tokens = self.embedding(x.unsqueeze(-1))

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_embed

        # Transformer encoder
        out = self.encoder(tokens)

        # Use CLS token output
        cls_out = out[:, 0, :]

        # Classification head
        return self.head(cls_out).squeeze(-1)

# Optuna tuning
print("\n[3/5] Optuna hyperparameter tuning...")

def objective(trial):
    d_model = trial.suggest_int('d_model', 64, 256, step=32)
    n_heads = trial.suggest_int('n_heads', 4, 16, step=4)
    n_layers = trial.suggest_int('n_layers', 2, 6)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.3)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    model = FTTransformer(X_tr_scaled.shape[1], d_model, n_heads, n_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_ds = TensorDataset(torch.tensor(X_tr_scaled), torch.tensor(y_tr))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = TensorDataset(torch.tensor(X_va_scaled), torch.tensor(y_va))
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    for epoch in range(30):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_list = []
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                y_pred_list.append(model(X_batch).cpu())
            y_pred = torch.cat(y_pred_list).numpy() > 0.5

        val_acc = accuracy_score(y_va, y_pred)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_acc

study = optuna.create_study(
    direction='maximize',
    pruner=MedianPruner(),
)
study.optimize(objective, n_trials=15, show_progress_bar=True)

best_params = study.best_params
print(f"  Best accuracy: {study.best_value:.4f}")
print(f"  Best params: {best_params}")

# Train final model
print("\n[4/5] Training final FT-Transformer model...")
model = FTTransformer(
    X_tr_scaled.shape[1],
    best_params['d_model'],
    best_params['n_heads'],
    best_params['n_layers'],
    best_params['dropout'],
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
criterion = nn.BCELoss()

train_ds = TensorDataset(torch.tensor(X_tr_scaled), torch.tensor(y_tr))
train_loader = DataLoader(train_ds, batch_size=best_params['batch_size'], shuffle=True)

val_ds = TensorDataset(torch.tensor(X_va_scaled), torch.tensor(y_va))
val_loader = DataLoader(val_ds, batch_size=best_params['batch_size'])

best_val_acc = 0
for epoch in range(100):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_list = []
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            y_pred_list.append(model(X_batch).cpu())
        y_pred_proba = torch.cat(y_pred_list).numpy()
        y_pred = y_pred_proba > 0.5

    val_acc = accuracy_score(y_va, y_pred)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'checkpoints/ftt_best.pt')

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}/100  val_acc: {val_acc:.4f}")

print(f"  Final val accuracy: {best_val_acc:.4f}")

# Save model
print("\n[5/5] Saving model...")
os.makedirs('checkpoints', exist_ok=True)
with open('checkpoints/ftt_model.pkl', 'wb') as f:
    pickle.dump({
        'model_state': torch.load('checkpoints/ftt_best.pt'),
        'model_config': {
            'n_features': X_tr_scaled.shape[1],
            'd_model': best_params['d_model'],
            'n_heads': best_params['n_heads'],
            'n_layers': best_params['n_layers'],
            'dropout': best_params['dropout'],
        },
        'scaler': scaler,
    }, f)
print(f"  Saved: checkpoints/ftt_model.pkl")

# Generate submission
print("\nGenerating submission...")
model.eval()
X_test_scaled = scaler.transform(X_test[feature_cols]).astype(np.float32)
with torch.no_grad():
    X_test_t = torch.tensor(X_test_scaled).to(device)
    y_pred = model(X_test_t).cpu().numpy()

submission = pd.DataFrame({
    'ID': X_test['ID'].values,
    'end_of_day_return': y_pred
})
os.makedirs('submissions_output', exist_ok=True)
submission.to_csv('submissions_output/ftt_submission.csv', index=False)
print(f"  Saved: submissions_output/ftt_submission.csv")

print("\n" + "=" * 70)
print(f"FT-Transformer complete! Next: python 4_ensemble.py")
print("=" * 70)
