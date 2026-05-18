"""Training loops, cross-validation, and submission generation."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def make_time_split(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, embargo: int = 5):
    """Yield (train_idx, val_idx) with temporal ordering and an embargo gap."""
    pass


def train_lgbm(X_train, y_train, X_val, y_val, params: dict | None = None):
    """Train a LightGBM model and return (model, val_accuracy)."""
    pass


def make_submission(model, X_test: pd.DataFrame, output_path: str) -> None:
    """Generate a submission CSV from model predictions on the test set."""
    pass
