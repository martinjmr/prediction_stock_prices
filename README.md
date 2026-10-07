# CFM Challenge: Predicting End-of-Day Stock Returns

Week-long project for the Deep Learning course of the L3 IASO program (Université Paris Dauphine-PSL, May 2026), by Quiterie Guignard and Martin Jomier.

**52.43% accuracy against 51.80% for the CFM benchmark, 3rd place in the course ranking.** The benchmark is itself a LightGBM: the gain comes from our engineered features and tuning, and the choice of model then moves the score by at most 0.03 points.

## Task

For each (stock, day) pair, predict whether the return over the last 30 minutes of the session beats that day's cross-sectional median, using the 71 five-minute returns observed between 9:30 and 15:25. About 700 stocks over about 700 days. The data comes from [Challenge Data](https://challengedata.ens.fr/challenges/16) and is not included here.

## Approach

1. **Features** (`build_advanced_features` in `src/utils.py`): the 71 raw returns become 183 features in six families: temporal statistics, multi-scale EWMA, volatility, cumulative price path, higher-order moments, and per-day cross-sectional ranks and z-scores.
2. **Baseline**: logistic regression.
3. **LightGBM**, tuned with Optuna (20 trials).
4. **FT-Transformer**: each feature becomes a token for a Transformer encoder, which predicts from a CLS token; tuned with Optuna (15 trials).
5. **Ensemble**: average of the LightGBM and FT-Transformer probabilities.

Validation splits by date (expanding window with an embargo), so that no information from later days leaks into training. During the project week we also tried an LSTM on the raw five-minute series; it was not kept.

## Results (leaderboard accuracy)

| Model | Accuracy |
|---|---|
| CFM benchmark (LightGBM) | 51.80% |
| LightGBM, engineered features | 52.40% |
| FT-Transformer | 52.41% |
| Ensemble 50/50 | **52.43%** |

## Run it

Put the Challenge Data files in `data/` (`input_training.csv`, `output_training_*.csv`, `input_test.csv`), then, from the repository root:

```bash
pip install -r requirements.txt
python submissions/0_features.py        # 183 features -> checkpoints/features.npz
python submissions/1_logreg.py          # baseline
python submissions/2_lightgbm.py        # LightGBM + Optuna
python submissions/3_ft_transformer.py  # FT-Transformer + Optuna
python submissions/4_ensemble.py        # -> submissions_output/ensemble_submission.csv
```

The numbered scripts were reorganised after the project week; the accuracies above are the leaderboard scores from May 2026.

## Repository

- `submissions/`: the final pipeline, steps 0 to 4
- `src/utils.py`: data loading, feature engineering, seeding
- `notebooks/01_eda.ipynb`: exploratory analysis
