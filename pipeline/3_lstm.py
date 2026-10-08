"""LSTM over the 71 intraday returns, joined to the engineered inputs, on the same 4 folds.

It reads the same inputs, target and folds as the other models, and uses the
FT-Transformer's training loop with fixed hyperparameters (no search).
Usage: python pipeline/3_lstm.py [--data-dir data]
Output: outputs/lstm.npz, outputs/submissions/lstm.csv
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cv import accuracy, date_block_folds  # noqa: E402
from src.data import load, save_predictions  # noqa: E402
from src.models import LSTMClassifier  # noqa: E402
from src.nn_training import DEVICE, scale_fold, train_fold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    print(f"Device: {DEVICE} | {vars(args)}")
    torch.manual_seed(42)

    d = load(args.data_dir)
    raw, feat, y, dates = d["raw_train"], d["feat_train"], d["y"], d["dates"]

    def make_model():
        return LSTMClassifier(feat.shape[1], args.hidden, args.layers, args.dropout)

    oof = np.full(len(y), np.nan)
    test = np.zeros(len(d["raw_test"]))
    folds = list(date_block_folds(dates, n_splits=4))
    fold_acc = []
    for k, (tr, val) in enumerate(folds):
        print(f"Fold {k + 1}/{len(folds)}: {len(tr):,} training rows, {len(val):,} validation rows", flush=True)
        sc = scale_fold(raw[tr], raw[val], feat[tr], feat[val], d["raw_test"], d["feat_test"])
        acc, oof_p, test_p = train_fold(make_model, sc[0], sc[2], y[tr], sc[1], sc[3], y[val], sc[4], sc[5],
                                        args.lr, 0.0, args.batch_size, args.epochs, args.patience, smooth=0.0,
                                        workers=0)
        oof[val], fold_acc = oof_p, fold_acc + [acc]
        test += test_p / len(folds)
        gc.collect()

    print(f"Out-of-fold accuracy (validated dates): {accuracy(y, oof):.4f}")
    save_predictions("lstm", oof, test, d["test_ids"], {"params": vars(args), "fold_accuracy": fold_acc})


if __name__ == "__main__":
    main()
