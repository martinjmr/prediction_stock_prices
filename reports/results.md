# Cross-validation results

Accuracy on the validation dates of the 4 folds (596,185 stock-days), on the training target: 1 if the stock's end-of-day return beats the median of the day. Averages are 50/50 on probabilities.

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | All folds |
|---|---|---|---|---|---|
| LightGBM | 52.35% | 52.38% | 52.27% | 52.10% | **52.28%** |
| LSTM | 52.12% | 52.21% | 52.15% | 52.16% | **52.16%** |
| LightGBM + LSTM | 52.30% | 52.22% | 52.28% | 52.15% | **52.24%** |
