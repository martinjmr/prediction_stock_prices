"""Compare the models on the same folds and average their probabilities.

Reads the predictions saved by steps 1 to 3 (whichever exist), scores each model and each
50/50 average fold by fold, writes reports/results.md and one submission per average.
Usage: python pipeline/4_ensemble.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cv import accuracy, date_block_folds  # noqa: E402

NAMES = {"lightgbm": "LightGBM", "ft_transformer": "FT-Transformer", "lstm": "LSTM"}


def main() -> None:
    with np.load(ROOT / "outputs" / "data.npz") as cached:
        y, dates, test_ids = cached["y"], cached["dates"], cached["test_ids"]
    folds = [val for _, val in date_block_folds(dates, n_splits=4)]

    preds = {}
    for key in NAMES:
        path = ROOT / "outputs" / f"{key}.npz"
        if path.exists():
            with np.load(path) as p:
                preds[key] = (p["oof"], p["test"])
    for a, b in itertools.combinations(list(preds), 2):
        preds[f"{a}+{b}"] = tuple((preds[a][k] + preds[b][k]) / 2 for k in (0, 1))
        pd.DataFrame({"ID": test_ids, "end_of_day_return": preds[f"{a}+{b}"][1]}).to_csv(
            ROOT / "outputs" / "submissions" / f"{a}+{b}.csv", index=False)

    rows = []
    for key, (oof, _) in preds.items():
        name = " + ".join(NAMES[k] for k in key.split("+"))
        per_fold = [accuracy(y[val], oof[val]) for val in folds]
        rows.append((name, per_fold, accuracy(y, oof)))

    lines = ["# Cross-validation results", "",
             "Accuracy on the validation dates of the 4 folds "
             f"({sum(len(v) for v in folds):,} stock-days), on the training target: 1 if the stock's "
             "end-of-day return beats the median of the day. Averages are 50/50 on probabilities.", "",
             "| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | All folds |", "|---|---|---|---|---|---|"]
    for name, per_fold, overall in rows:
        lines.append(f"| {name} | " + " | ".join(f"{100 * a:.2f}%" for a in per_fold) + f" | **{100 * overall:.2f}%** |")
    out = ROOT / "reports" / "results.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
