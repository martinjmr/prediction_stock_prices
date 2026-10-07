"""Load the Challenge Data files, build the target and the model inputs, cache them."""
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import META, build_advanced_features

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "data.npz"


def load_data(data_dir: str) -> dict:
    """Return the model inputs as float32 arrays.

    raw:  the 71 five-minute returns, missing values set to 0
    feat: the returns imputed with their row mean, plus the 41 engineered features
    y:    1 if the stock's end-of-day return beats the median of its day, else 0
    """
    x_train = pd.read_csv(os.path.join(data_dir, "input_training.csv"))
    y_files = sorted(glob.glob(os.path.join(data_dir, "output_training*.csv")))
    if not y_files:
        raise FileNotFoundError(f"no output_training*.csv in {data_dir}")
    y_raw = pd.read_csv(y_files[0])
    x_test = pd.read_csv(os.path.join(data_dir, "input_test.csv"))

    train = x_train.merge(y_raw, on="ID")
    daily_median = train.groupby("date")["end_of_day_return"].transform("median")
    train["target"] = (train["end_of_day_return"] > daily_median).astype(int)

    raw_cols = [c for c in x_train.columns if c not in ["ID", "date", "eqt_code"]]
    train_e = build_advanced_features(train)
    test_e = build_advanced_features(x_test)
    feat_cols = sorted(c for c in train_e.columns if c not in META)

    def clean(frame):
        return frame[feat_cols].replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)

    return {
        "raw_train": train[raw_cols].fillna(0).values.astype(np.float32),
        "raw_test": x_test[raw_cols].fillna(0).values.astype(np.float32),
        "feat_train": clean(train_e),
        "feat_test": clean(test_e),
        "y": train["target"].values.astype(np.float32),
        "dates": train["date"].values,
        "test_ids": x_test["ID"].values,
        "feat_cols": np.array(feat_cols),
    }


def load(data_dir: str = "data") -> dict:
    """Load the inputs from outputs/data.npz, building it on the first call."""
    if not CACHE.exists():
        print(f"Building features from {data_dir}/ (a few minutes, done once)...")
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **load_data(data_dir))
    with np.load(CACHE, allow_pickle=False) as cached:
        data = {key: cached[key] for key in cached.files}
    print(f"raw {data['raw_train'].shape} | feat {data['feat_train'].shape} | test {data['raw_test'].shape}")
    return data


def save_predictions(name: str, oof: np.ndarray, test: np.ndarray, test_ids: np.ndarray, info: dict) -> None:
    """Save out-of-fold and test predictions, and the submission file for Challenge Data."""
    out = ROOT / "outputs"
    (out / "submissions").mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}.npz", oof=oof, test=test)
    with open(out / f"{name}.pkl", "wb") as f:
        pickle.dump(info, f)
    pd.DataFrame({"ID": test_ids, "end_of_day_return": test}).to_csv(
        out / "submissions" / f"{name}.csv", index=False)
    print(f"Saved outputs/{name}.npz and outputs/submissions/{name}.csv")
