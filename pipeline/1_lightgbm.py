"""LightGBM: Optuna search on a 15% subsample, then 5 seeds x 4 expanding-window folds.

Usage (from the repository root): python pipeline/1_lightgbm.py [--data-dir data]
Output: outputs/lightgbm.npz (out-of-fold and test probabilities),
        outputs/submissions/lightgbm.csv
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cv import accuracy, expanding_window_split  # noqa: E402
from src.data import load, save_predictions  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def fit(params, X_tr, y_tr, X_val, y_val):
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    d = load(args.data_dir)
    X, y, dates = np.hstack([d["feat_train"], d["raw_train"]]), d["y"], d["dates"]
    X_test = np.hstack([d["feat_test"], d["raw_test"]])
    print(f"LightGBM inputs: train {X.shape}, test {X_test.shape}")

    # Phase 1: hyperparameter search on 15% of the rows, 3 expanding-window folds
    rng = np.random.default_rng(42)
    sub = np.sort(rng.choice(len(y), size=int(len(y) * 0.15), replace=False))
    Xs, ys, ds = X[sub], y[sub], dates[sub]

    def objective(trial):
        params = {
            "objective": "binary", "metric": "binary_error", "verbosity": -1,
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        }
        accs = [accuracy(ys[val], fit(params, Xs[tr], ys[tr], Xs[val], ys[val]).predict_proba(Xs[val])[:, 1])
                for tr, val in expanding_window_split(ds, n_splits=3)]
        print(f"  trial {trial.number:>2}: {np.mean(accs):.4f}")
        return float(np.mean(accs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials)
    print(f"Best search accuracy {study.best_value:.4f} with {study.best_params}")

    # Phase 2: full data, average of several seeds over 4 expanding-window folds
    oof = np.zeros(len(y))
    validated = np.zeros(len(y), dtype=bool)
    test = np.zeros(len(X_test))
    fold_acc = []
    for seed in range(args.seeds):
        params = {**study.best_params, "objective": "binary", "metric": "binary_error",
                  "verbosity": -1, "random_state": seed}
        test_seed, accs = np.zeros(len(X_test)), []
        folds = list(expanding_window_split(dates, n_splits=4))
        for k, (tr, val) in enumerate(folds):
            model = fit(params, X[tr], y[tr], X[val], y[val])
            p = model.predict_proba(X[val])[:, 1]
            oof[val] += p / args.seeds
            validated[val] = True
            test_seed += model.predict_proba(X_test)[:, 1] / len(folds)
            accs.append(accuracy(y[val], p))
            print(f"  seed {seed} fold {k + 1}: {accs[-1]:.4f}")
        test += test_seed / args.seeds
        fold_acc.append(accs)

    oof[~validated] = np.nan
    print(f"Out-of-fold accuracy (validated dates): {accuracy(y, oof):.4f}")
    save_predictions("lightgbm", oof, test, d["test_ids"],
                     {"best_params": study.best_params, "fold_accuracy_by_seed": fold_acc})


if __name__ == "__main__":
    main()
