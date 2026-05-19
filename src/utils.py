"""Shared utilities: data loading, reproducibility, and metrics."""
import glob
import os
import re
import random

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and return (X_train, y_train, X_test)."""
    # Le fichier output a un suffixe aléatoire — on le trouve dynamiquement
    output_files = glob.glob(os.path.join(data_dir, "output_training_*.csv"))
    if not output_files:
        raise FileNotFoundError(f"Aucun fichier output_training_*.csv dans {data_dir}")

    X_train = pd.read_csv(os.path.join(data_dir, "input_training.csv"))
    y_train = pd.read_csv(output_files[0])
    X_test  = pd.read_csv(os.path.join(data_dir, "input_test.csv"))

    return X_train, y_train, X_test


def accuracy_sign(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return fraction of samples where sign(y_pred - 0.5) == sign(y_true)."""
    pred_positive = np.array(y_pred) > 0.5
    true_positive = np.array(y_true) > 0
    return float((pred_positive == true_positive).mean())


def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering pipeline : time-series, volatilité, trajectoire, cross-sectionnel."""
    print("Début de la création des features...")

    META_COLS = ['ID', 'date', 'eqt_code', 'target', 'end_of_day_return']
    df_feat = df.copy()

    return_cols = [c for c in df_feat.columns if c not in META_COLS]
    returns = df_feat[return_cols].apply(pd.to_numeric)

    # NaN → moyenne de la ligne (0 est un return valide, ne pas l'utiliser)
    row_means = returns.mean(axis=1)
    returns = returns.apply(lambda col: col.fillna(row_means), axis=0)
    df_feat[return_cols] = returns

    mid = len(return_cols) // 2
    arr = returns.values  # numpy pour la vitesse

    # ── A) Statistiques time-series ──────────────────────────────────────────
    print(" -> Time-series (A)...")
    df_feat['return_mean']  = arr.mean(axis=1)
    df_feat['return_std']   = arr.std(axis=1)
    df_feat['return_max']   = arr.max(axis=1)
    df_feat['return_min']   = arr.min(axis=1)
    df_feat['mean_last_5']  = arr[:, -5:].mean(axis=1)
    df_feat['mean_last_15'] = arr[:, -15:].mean(axis=1)
    df_feat['std_last_15']  = arr[:, -15:].std(axis=1)

    # EWMA multi-échelle : pondère les returns récents plus fortement
    # alpha = 2/(span+1), poids décroissants vers le passé, normalisés
    for span in [5, 10, 20, 40]:
        alpha   = 2 / (span + 1)
        n       = len(return_cols)
        weights = alpha * (1 - alpha) ** np.arange(n - 1, -1, -1)
        weights /= weights.sum()
        df_feat[f'ewma_{span}'] = arr @ weights

    # ── C) Volatilité ────────────────────────────────────────────────────────
    print(" -> Volatilités (C)...")
    df_feat['realized_variance'] = (arr ** 2).sum(axis=1)
    df_feat['vol_morning']       = arr[:, :mid].std(axis=1)
    df_feat['vol_afternoon']     = arr[:, mid:].std(axis=1)
    df_feat['vol_ratio']         = df_feat['vol_morning'] / (df_feat['vol_afternoon'] + 1e-8)

    # ── D) Forme de la trajectoire ───────────────────────────────────────────
    print(" -> Trajectoires (D)...")
    prices = returns.cumsum(axis=1)
    df_feat['price_end']     = prices.iloc[:, -1].values
    df_feat['price_max']     = prices.max(axis=1).values
    df_feat['price_min']     = prices.min(axis=1).values
    df_feat['drawdown']      = df_feat['price_end'] - df_feat['price_max']
    crossings                = (arr[:, 1:] * arr[:, :-1]) < 0
    df_feat['zero_crossings'] = crossings.sum(axis=1)

    # ── B) Cross-sectionnel (par date) ───────────────────────────────────────
    print(" -> Cross-sectionnel (B)...")

    # Z-scores : (valeur - moyenne_date) / std_date
    for feat in ['price_end', 'mean_last_15', 'return_std', 'ewma_5']:
        df_feat[f'z_{feat}'] = df_feat.groupby('date')[feat].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    # Rangs percentile : plus robuste aux outliers que le z-score
    # rank(pct=True) → valeur entre 0 et 1 selon la position dans la distribution du jour
    for feat in ['price_end', 'mean_last_15', 'ewma_5']:
        df_feat[f'rank_{feat}'] = df_feat.groupby('date')[feat].rank(pct=True)

    # Z-score sur le dernier bin de return brut
    last_col = return_cols[-1]
    grp = df_feat.groupby('date')[last_col]
    df_feat['z_last_bin'] = (df_feat[last_col] - grp.transform('mean')) / (grp.transform('std') + 1e-8)

    # ── E) Nettoyage ─────────────────────────────────────────────────────────
    df_feat['eqt_code'] = df_feat['eqt_code'].astype('category')
    df_feat.columns = [re.sub(r'[\[\]{} ,:]+', '_', str(c)) for c in df_feat.columns]

    print(f" Feature Engineering terminé ! Dimensions : {df_feat.shape}")
    return df_feat
