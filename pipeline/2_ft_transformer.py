"""FT-Transformer: Optuna search on a 15% subsample, then 4 expanding-window folds.

Needs a GPU (about 1-3 h on a Colab T4). Usage: python pipeline/2_ft_transformer.py [--data-dir data]
Output: outputs/ft_transformer.npz, outputs/submissions/ft_transformer.csv
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cv import accuracy, expanding_window_split  # noqa: E402
from src.data import load, save_predictions  # noqa: E402
from src.models import FTTransformer  # noqa: E402
from src.nn_training import DEVICE, scale_fold, train_fold  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50, help="maximum epochs per fold in the final training")
    args = parser.parse_args()
    print(f"Device: {DEVICE}")

    d = load(args.data_dir)
    raw, feat, y, dates = d["raw_train"], d["feat_train"], d["y"], d["dates"]
    n_raw, n_feat = raw.shape[1], feat.shape[1]

    # Phase 1: hyperparameter search on 15% of the rows, 3 folds, 15 epochs at most
    rng = np.random.default_rng(42)
    sub = np.sort(rng.choice(len(y), size=int(len(y) * 0.15), replace=False))
    r_s, f_s, y_s, d_s = raw[sub], feat[sub], y[sub], dates[sub]
    dummy_r, dummy_f = np.zeros((1, n_raw), np.float32), np.zeros((1, n_feat), np.float32)

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [64, 128])
        nhead = trial.suggest_categorical("nhead", [4, 8])
        kw = dict(d_model=d_model, nhead=nhead if d_model % nhead == 0 else 4,
                  num_layers=trial.suggest_int("num_layers", 1, 3), dropout=trial.suggest_float("dropout", 0.0, 0.3))
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [512, 1024])
        smooth = trial.suggest_float("label_smoothing", 0.0, 0.15)
        accs = []
        for tr, val in expanding_window_split(d_s, n_splits=3):
            sc = scale_fold(r_s[tr], r_s[val], f_s[tr], f_s[val], dummy_r, dummy_f)
            acc, _, _ = train_fold(lambda: FTTransformer(n_raw, n_feat, **kw), sc[0], sc[2], y_s[tr], sc[1], sc[3],
                                   y_s[val], sc[4], sc[5], lr, wd, bs, epochs=15, patience=4, smooth=smooth)
            accs.append(acc)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return float(np.mean(accs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials)
    hp = study.best_params
    print(f"Best search accuracy {study.best_value:.4f} with {hp}")

    # Phase 2: full data, 4 expanding-window folds
    kw = dict(d_model=hp["d_model"], nhead=hp["nhead"] if hp["d_model"] % hp["nhead"] == 0 else 4,
              num_layers=hp["num_layers"], dropout=hp["dropout"])
    oof = np.full(len(y), np.nan)
    test = np.zeros(len(d["raw_test"]))
    folds = list(expanding_window_split(dates, n_splits=4))
    fold_acc = []
    for k, (tr, val) in enumerate(folds):
        print(f"Fold {k + 1}/{len(folds)}")
        sc = scale_fold(raw[tr], raw[val], feat[tr], feat[val], d["raw_test"], d["feat_test"])
        acc, oof_p, test_p = train_fold(lambda: FTTransformer(n_raw, n_feat, **kw), sc[0], sc[2], y[tr], sc[1],
                                        sc[3], y[val], sc[4], sc[5], hp["lr"], hp["weight_decay"],
                                        hp["batch_size"], epochs=args.epochs, patience=8,
                                        smooth=hp["label_smoothing"])
        oof[val], fold_acc = oof_p, fold_acc + [acc]
        test += test_p / len(folds)
        gc.collect()

    print(f"Out-of-fold accuracy (validated dates): {accuracy(y, oof):.4f}")
    save_predictions("ft_transformer", oof, test, d["test_ids"], {"best_params": hp, "fold_accuracy": fold_acc})


if __name__ == "__main__":
    main()
