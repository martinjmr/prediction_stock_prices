# Submission Pipeline

This directory contains the **clean, final pipeline** for the CFM Challenge prediction task. Each script builds on the previous one.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Execute the scripts in order:

```bash
# Step 0: Build features (creates checkpoints/features.npz)
python 0_features.py

# Step 1: Logistic Regression Baseline
python 1_logreg.py

# Step 2: LightGBM with Optuna Tuning
python 2_lightgbm.py

# Step 3: Feature Tokenizer Transformer
python 3_ft_transformer.py

# Step 4: Ensemble (FINAL SUBMISSION)
python 4_ensemble.py
```

## Output

- **checkpoints/** — Trained model states and preprocessing objects
- **submissions_output/** — Prediction CSVs:
  - `ensemble_submission.csv` — **FINAL SUBMISSION** (50/50 LightGBM + FT-Transformer)
  - `lightgbm_only.csv` — LightGBM individual predictions
  - `ftt_only.csv` — FT-Transformer individual predictions

## Pipeline Overview

| Step | Script | What | Output |
|------|--------|------|--------|
| 0 | `0_features.py` | Feature engineering | `checkpoints/features.npz` |
| 1 | `1_logreg.py` | Logistic Regression baseline | `submissions_output/logreg_submission.csv` |
| 2 | `2_lightgbm.py` | LightGBM + Optuna (20 trials) | `submissions_output/lightgbm_submission.csv` |
| 3 | `3_ft_transformer.py` | FT-Transformer + Optuna (15 trials) | `submissions_output/ftt_submission.csv` |
| 4 | `4_ensemble.py` | 50/50 Ensemble | **`submissions_output/ensemble_submission.csv`** |

## Expected Results

- **Baseline (LogReg):** ~51.0% accuracy
- **LightGBM:** ~52.4% accuracy  
- **FT-Transformer:** ~52.4% accuracy
- **Ensemble:** ~52.4% accuracy

(Exact results vary based on cross-validation splits and randomness)

## Data Requirements

Before running, place the following files in the `data/` directory:

```
data/
├── input_training.csv         (feature rows × ~700 days)
├── output_training_*.csv       (target labels)
└── input_test.csv              (test set)
```

These are not included in the repository as they are competition-specific.

## Notes

- Each step is **independent**; you can run models without prior steps (features are cached)
- **Hyperparameter tuning is quick** (Optuna defaults to limited trials to keep compute time reasonable)
- **GPU is optional** but speeds up FT-Transformer training
- All models use **temporal cross-validation** (no data leakage between train/val)

## Architecture

- **Feature Engineering:** 71 bins → 183 engineered features (time-series, volatility, cross-sectional)
- **LightGBM:** Gradient boosting on raw + engineered features
- **FT-Transformer:** Transformer encoder on feature tokens (each scalar feature is a token)
- **Ensemble:** Simple 50/50 probability average

See [../README.md](../README.md) for full methodology.
