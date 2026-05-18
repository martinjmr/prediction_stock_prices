"""Shared utilities: data loading, reproducibility, and metrics."""
import random
import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across numpy, random, and torch."""
    pass


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and return (X_train, y_train, X_test)."""
    pass


def accuracy_sign(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return fraction of samples where sign(y_pred - 0.5) == sign(y_true)."""
    pass
