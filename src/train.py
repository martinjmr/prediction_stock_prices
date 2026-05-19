"""Training loops, cross-validation, and submission generation."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


META_COLS = ['ID', 'eqt_code', 'date']
TARGET    = 'end_of_day_return'
CLIP_VAL  = 20

# Noms de colonnes ajoutés par build_advanced_features — séparés des returns bruts
ENGINEERED_COLS = {
    'return_mean', 'return_std', 'return_max', 'return_min',
    'mean_last_5', 'mean_last_15', 'std_last_15',
    'ewma_5', 'ewma_10', 'ewma_20', 'ewma_40',
    'realized_variance', 'vol_morning', 'vol_afternoon', 'vol_ratio',
    'price_end', 'price_max', 'price_min', 'drawdown', 'zero_crossings',
    'z_price_end', 'z_mean_last_15', 'z_return_std', 'z_ewma_5',
    'rank_price_end', 'rank_mean_last_15', 'rank_ewma_5',
    'z_last_bin',
}

CHECKPOINT_PATH = 'models/lstm_checkpoint.pt'   # dernier état (reprise)
BEST_PATH       = 'models/lstm_best.pt'          # meilleur modèle (soumission)


def preprocess(X: pd.DataFrame, clip: float = CLIP_VAL) -> pd.DataFrame:
    """Winsorise return columns then fill NaN with the row mean."""
    ret_cols = [c for c in X.columns if c not in META_COLS]
    X = X.copy()

    arr = X[ret_cols].clip(-clip, clip).to_numpy(dtype=float)
    row_means = np.nanmean(arr, axis=1, keepdims=True)  # moyenne non-NaN par ligne
    X[ret_cols] = np.where(np.isnan(arr), row_means, arr)
    return X


def split_by_date(
    X: pd.DataFrame,
    y: pd.DataFrame,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split train/val by unique date so no date appears in both sets.

    Splitting on rows would put the same day in train and val — the model
    could then learn patterns that leak information across the two sets.
    Splitting on dates guarantees a clean boundary.
    """
    dates = X['date'].unique()
    rng   = np.random.default_rng(seed)
    rng.shuffle(dates)

    n_val     = max(1, int(len(dates) * val_frac))
    val_dates = set(dates[-n_val:])

    mask_val = X['date'].isin(val_dates)

    X_tr = X[~mask_val].reset_index(drop=True)
    X_va = X[mask_val].reset_index(drop=True)

    # Aligner y sur X via ID
    y_tr = X_tr[['ID']].merge(y, on='ID')[TARGET]
    y_va = X_va[['ID']].merge(y, on='ID')[TARGET]

    print(f"Train : {len(X_tr):,} lignes  ({X_tr['date'].nunique()} dates)")
    print(f"Val   : {len(X_va):,} lignes  ({X_va['date'].nunique()} dates)")

    return X_tr, X_va, y_tr, y_va


def train_logreg(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
) -> tuple:
    """Train a logistic regression and return (model, scaler, val_accuracy)."""
    ret_cols = [c for c in X_tr.columns if c not in META_COLS]

    # Cible binaire : 1 si return positif, 0 sinon
    y_tr_bin = (y_tr.values > 0).astype(int)
    y_va_bin = (y_va.values > 0).astype(int)

    # Standardisation : fit sur train uniquement, appliqué sur train et val
    scaler     = StandardScaler()
    Xtr_scaled = scaler.fit_transform(X_tr[ret_cols])
    Xva_scaled = scaler.transform(X_va[ret_cols])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(Xtr_scaled, y_tr_bin)

    # predict_proba retourne [[p_neg, p_pos], ...] — on prend la colonne p_pos
    val_proba = model.predict_proba(Xva_scaled)[:, 1]
    val_acc   = float(((val_proba > 0.5) == y_va_bin).mean())

    print(f"Logreg — val accuracy : {val_acc:.4f}")
    return model, scaler, val_acc


def _to_sequences(X_arr: np.ndarray, y: pd.Series | None = None):
    """Convertit un array numpy en tenseurs PyTorch de shape (N, seq_len, 1)."""
    X_t = torch.tensor(X_arr).unsqueeze(-1)                # (N, T, 1)
    if y is None:
        return X_t
    y_t = torch.tensor((y.values > 0).astype(np.float32)) # (N,)
    return X_t, y_t


def train_lstm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 5,
) -> tuple:
    """Entraîne le LSTM avec early stopping et deux checkpoints.

    Si X_tr contient des colonnes de ENGINEERED_COLS (ajoutées par
    build_advanced_features), elles sont utilisées comme features
    supplémentaires concaténées au dernier état caché du LSTM.

    Deux fichiers sont sauvegardés :
    - lstm_checkpoint.pt : dernier état (pour reprendre après interruption)
    - lstm_best.pt       : meilleur modèle (utilisé pour la soumission)
    """
    import os
    from tqdm import tqdm
    from models import LSTMClassifier

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # ── Séparation returns bruts / features engineerées ──────────────────────
    all_cols  = [c for c in X_tr.columns if c not in META_COLS]
    feat_cols = [c for c in all_cols if c in ENGINEERED_COLS]
    ret_cols  = [c for c in all_cols if c not in ENGINEERED_COLS]
    n_extra   = len(feat_cols)
    print(f"Returns bruts : {len(ret_cols)}  |  Features engineerées : {n_extra}")

    # ── Scalers ───────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    Xtr_arr = scaler.fit_transform(X_tr[ret_cols]).astype(np.float32)
    Xva_arr = scaler.transform(X_va[ret_cols]).astype(np.float32)

    if n_extra > 0:
        scaler_extra = StandardScaler()
        Xtr_extra = np.nan_to_num(X_tr[feat_cols].values.astype(np.float32))
        Xva_extra = np.nan_to_num(X_va[feat_cols].values.astype(np.float32))
        Xtr_extra = scaler_extra.fit_transform(Xtr_extra).astype(np.float32)
        Xva_extra = scaler_extra.transform(Xva_extra).astype(np.float32)
    else:
        scaler_extra = None
        Xtr_extra = np.zeros((len(X_tr), 0), dtype=np.float32)
        Xva_extra = np.zeros((len(X_va), 0), dtype=np.float32)

    # ── Tenseurs & DataLoaders ────────────────────────────────────────────────
    X_tr_seq, y_tr_t = _to_sequences(Xtr_arr, y_tr)
    X_va_seq, y_va_t = _to_sequences(Xva_arr, y_va)
    X_tr_ext_t = torch.tensor(Xtr_extra)
    X_va_ext_t = torch.tensor(Xva_extra)

    train_loader = DataLoader(
        TensorDataset(X_tr_seq, X_tr_ext_t, y_tr_t), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_va_seq, X_va_ext_t, y_va_t), batch_size=batch_size
    )

    # ── Modèle ────────────────────────────────────────────────────────────────
    model_kwargs = {
        'hidden_size': hidden_size,
        'num_layers' : num_layers,
        'dropout'    : dropout,
        'extra_size' : n_extra,
    }
    model     = LSTMClassifier(**model_kwargs).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history     = {'train_acc': [], 'val_acc': []}
    best_val    = 0.0
    no_improve  = 0
    start_epoch = 1

    # ── Reprise depuis checkpoint si compatible ───────────────────────────────
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            history     = ckpt['history']
            best_val    = ckpt['best_val']
            no_improve  = ckpt['no_improve']
            start_epoch = ckpt['epoch'] + 1
            print(f"Reprise depuis epoch {start_epoch}  (meilleur val : {best_val:.4f})")
        except (RuntimeError, KeyError):
            print("Checkpoint incompatible (architecture changée) — entraînement depuis le début.")

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        correct, total = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch:>2}/{epochs} [train]", leave=False)
        for X_seq, X_ext, y_batch in loop:
            X_seq, X_ext, y_batch = X_seq.to(device), X_ext.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_seq, X_ext if n_extra > 0 else None)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            correct += ((preds > 0.5) == y_batch.bool()).sum().item()
            total   += len(y_batch)
            loop.set_postfix(acc=f"{correct/total:.4f}")
        train_acc = correct / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_seq, X_ext, y_batch in val_loader:
                X_seq, X_ext, y_batch = X_seq.to(device), X_ext.to(device), y_batch.to(device)
                preds = model(X_seq, X_ext if n_extra > 0 else None)
                correct += ((preds > 0.5) == y_batch.bool()).sum().item()
                total   += len(y_batch)
        val_acc = correct / total

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        improved = val_acc > best_val
        if improved:
            best_val   = val_acc
            no_improve = 0
            tag        = "  ✓ meilleur"
            # Sauvegarde du MEILLEUR modèle (pour la soumission)
            torch.save({
                'epoch'       : epoch,
                'model_state' : model.state_dict(),
                'best_val'    : best_val,
                'scaler'      : scaler,
                'scaler_extra': scaler_extra,
                'feat_cols'   : feat_cols,
                'model_kwargs': model_kwargs,
            }, BEST_PATH)
        else:
            no_improve += 1
            tag = f"  (pas d'amélioration : {no_improve}/{patience})"

        print(f"Epoch {epoch:>2}/{epochs}  train={train_acc:.4f}  val={val_acc:.4f}{tag}")

        # Sauvegarde du dernier état (pour reprendre après interruption)
        torch.save({
            'epoch'          : epoch,
            'model_state'    : model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'history'        : history,
            'best_val'       : best_val,
            'no_improve'     : no_improve,
            'scaler'         : scaler,
            'scaler_extra'   : scaler_extra,
            'feat_cols'      : feat_cols,
            'model_kwargs'   : model_kwargs,
        }, CHECKPOINT_PATH)

        if no_improve >= patience:
            print(f"\nEarly stopping : val_acc n'a pas progressé depuis {patience} epochs.")
            break

    return model, scaler, history, best_val


def submit_from_checkpoint(X_test: pd.DataFrame, output_path: str) -> None:
    """Charge le meilleur modèle (lstm_best.pt) et génère le fichier de soumission."""
    import os
    from models import LSTMClassifier

    path = BEST_PATH if os.path.exists(BEST_PATH) else CHECKPOINT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Aucun checkpoint trouvé. Lance d'abord train_lstm().")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(path, map_location=device)

    model  = LSTMClassifier(**ckpt['model_kwargs']).to(device)
    model.load_state_dict(ckpt['model_state'])

    scaler_extra = ckpt.get('scaler_extra', None)
    feat_cols    = ckpt.get('feat_cols', [])

    print(f"Checkpoint chargé ({path}) — epoch {ckpt['epoch']}, meilleure val_acc : {ckpt['best_val']:.4f}")
    make_submission(model, ckpt['scaler'], X_test, output_path,
                    scaler_extra=scaler_extra, feat_cols=feat_cols)


def train_lgbm(X_train, y_train, X_val, y_val, params: dict | None = None):
    """Train a LightGBM classifier and return (model, val_accuracy)."""
    pass


def make_submission(
    model,
    scaler,
    X_test: pd.DataFrame,
    output_path: str,
    scaler_extra=None,
    feat_cols: list | None = None,
) -> None:
    """Generate a submission CSV — compatible sklearn et PyTorch (avec ou sans features engineerées)."""
    feat_cols = feat_cols or []
    feat_set  = set(feat_cols)
    all_cols  = [c for c in X_test.columns if c not in META_COLS]
    ret_cols  = [c for c in all_cols if c not in feat_set]

    X_scaled = scaler.transform(X_test[ret_cols]).astype(np.float32)

    if hasattr(model, 'predict_proba'):
        # Sklearn (logreg)
        predictions = model.predict_proba(X_scaled)[:, 1]
    else:
        # PyTorch (LSTM)
        device = next(model.parameters()).device
        model.eval()

        n_extra = len(feat_cols)
        if n_extra > 0 and scaler_extra is not None:
            X_extra = np.nan_to_num(X_test[feat_cols].values.astype(np.float32))
            X_extra_scaled = scaler_extra.transform(X_extra).astype(np.float32)
        else:
            X_extra_scaled = None

        preds_list = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), 512):
                batch_seq = torch.tensor(X_scaled[i:i+512]).unsqueeze(-1).to(device)
                if X_extra_scaled is not None:
                    batch_ext = torch.tensor(X_extra_scaled[i:i+512]).to(device)
                    preds_list.append(model(batch_seq, batch_ext).cpu().numpy())
                else:
                    preds_list.append(model(batch_seq).cpu().numpy())
        predictions = np.concatenate(preds_list)

    submission = pd.DataFrame({'ID': X_test['ID'], 'end_of_day_return': predictions})
    submission.to_csv(output_path, index=False)
    print(f"Submission sauvegardée : {output_path}  ({len(submission):,} lignes)")
