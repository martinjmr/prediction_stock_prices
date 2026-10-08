"""Cross-validation by blocks of days, shared by every model."""
import numpy as np


def date_block_folds(dates: np.ndarray, n_splits: int = 4):
    """Yield (train_idx, val_idx), keeping all the stocks of a day on the same side.

    The sorted date IDs are cut into n_splits + 1 blocks; fold k trains on blocks 1..k and
    validates on block k + 1, so the first block is never validated. In this challenge the
    date IDs are anonymised and not chronological: the blocks are groups of days, and the
    split is a grouped split with a growing training set.
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
