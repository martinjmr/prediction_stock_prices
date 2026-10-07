"""Expanding-window cross-validation over dates, shared by every model."""
import numpy as np


def expanding_window_split(dates: np.ndarray, n_splits: int = 4):
    """Yield (train_idx, val_idx) where training dates always precede validation dates.

    The sorted dates are cut into n_splits + 1 blocks; fold k trains on blocks 1..k
    and validates on block k + 1. The first block is therefore never validated.
    All stocks of a given day fall on the same side of the cut.
    """
    uniq = np.sort(np.unique(dates))
    n, step = len(uniq), len(uniq) // (n_splits + 1)
    for k in range(1, n_splits + 1):
        cut = k * step
        end = (k + 1) * step if k < n_splits else n
        tr = np.where(np.isin(dates, uniq[:cut]))[0]
        val = np.where(np.isin(dates, uniq[cut:end]))[0]
        if len(tr) > 0 and len(val) > 0:
            yield tr, val


def accuracy(y: np.ndarray, proba: np.ndarray) -> float:
    """Accuracy on the rows that received a prediction (NaN elsewhere)."""
    seen = ~np.isnan(proba)
    return float(((proba[seen] > 0.5) == (y[seen] > 0.5)).mean())
