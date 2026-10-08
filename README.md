# CFM Challenge: Predicting End-of-Day Stock Returns

Week-long project for the Deep Learning course of the L3 IASO program (Université Paris Dauphine-PSL, May 2026), which I did with Quiterie Guignard.

**Our best submission reaches 52.48% accuracy on the public test set against 51.99% for the CFM benchmark, and 52.05% against 51.49% on the private test set. 3rd place in the course ranking.**

## Task

For each stock and day, predict whether the stock's residual return between 15:30 and 16:00 is positive, from its 71 five-minute returns between 9:30 and 15:20. The data covers 680 US stocks: 745,327 stock-days for training (1,511 days) and 319,769 for the test set. It comes from [Challenge Data](https://challengedata.ens.fr/challenges/16) and is not included here.

The models learn a balanced version of the target: 1 if the stock's end-of-day return beats the median of all stocks on the same day.

## Pipeline

**Inputs** (`src/data.py`, `src/features.py`). We engineered 41 features in five families: time-series statistics (means, exponentially weighted means, skewness, kurtosis, lag-1 autocorrelation), volatility, shape of the cumulative price path, position of the stock among the other stocks of the day (z-scores, ranks), and market regime (dispersion, share of rising stocks, intraday beta). The models see 183 inputs: these 41 features and the 71 returns twice, with missing values set to 0 and to the mean of the row.

**Cross-validation** (`src/cv.py`). We sort the 1,511 days by their ID and cut them into five blocks; fold k trains on blocks 1 to k and validates on block k + 1. Every stock of a given day stays on one side, which the cross-sectional features require. The date IDs are anonymised, so the blocks are not in chronological order.

| Step | Model | Tuning | Hardware |
|---|---|---|---|
| `pipeline/1_lightgbm.py` | LightGBM | Optuna, 15 trials on 15% of the rows, then 5 seeds × 4 folds | CPU, about 10 min |
| `pipeline/2_ft_transformer.py` | FT-Transformer: each input becomes a token, the prediction comes from a CLS token | Optuna, 5 trials, then 4 folds | GPU (Colab T4), 1 to 3 h |
| `pipeline/3_lstm.py` | LSTM reading the 71 returns as a sequence; its last hidden state is joined to the other inputs | Fixed: 64 units, up to 10 epochs | CPU, about 40 min |
| `pipeline/4_ensemble.py` | 50/50 averages of the models' probabilities, comparison on the same folds | | |

The LSTM comes from the project week, when we trained it on the sign of the return with its own split. In October 2026, I made it share the target, inputs and folds of the other models, with a smaller network so that it trains on a CPU.

## Results

Submissions to the [challenge](https://challengedata.ens.fr/challenges/16), May 2026, with the method recorded for each. The public test set scores every submission; the private test set gives the final ranking.

| Submission | Public accuracy |
|---|---|
| CFM benchmark (LightGBM) | 51.99% |
| LSTM | 51.39% |
| LightGBM, basic features | 52.04% |
| Logistic regression, engineered features | 52.26% |
| Transformer, engineered features | 52.31% |
| CatBoost | 52.38% |
| LightGBM, engineered features | 52.40% |
| Average of LightGBM and CatBoost | 52.43% |
| Best submission (method not recorded) | **52.48%** |

On the leaderboards, our best submission ranks 82nd of 240 on the public test set and 101st of 240 on the private one (52.05% against 51.49% for the benchmark).

Cross-validation on the 596,185 validated stock-days; I reran steps 1, 3 and 4 in October 2026 ([details by fold](reports/results.md)):

| Model | Accuracy |
|---|---|
| LightGBM | 52.28% |
| LSTM | 52.16% |
| LightGBM + LSTM | 52.24% |

- The LSTM scores 0.12 point below LightGBM, and averaging the two does not beat LightGBM alone. Reading the returns as a sequence does not improve on the engineered features here.
- The FT-Transformer needs a GPU and I did not rerun it, so it has no score in this table.
- Every model stops training on its validation fold, so these scores are slightly optimistic. The leaderboard, computed on unseen days, is the reference.

## Run it

Put the Challenge Data files in `data/` (`input_training.csv`, `output_training*.csv`, `input_test.csv`), then, from the repository root:

```bash
pip install -r requirements.txt
python pipeline/1_lightgbm.py        # the first step also caches the inputs in outputs/data.npz
python pipeline/2_ft_transformer.py  # GPU recommended
python pipeline/3_lstm.py
python pipeline/4_ensemble.py        # reports/results.md and outputs/submissions/*.csv
```

`notebooks/01_eda.ipynb` holds the exploratory analysis (in French).
