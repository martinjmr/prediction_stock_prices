"""
Step 2: LightGBM with Optuna Hyperparameter Tuning
Performs hyperparameter search then trains final model with expanding window CV.
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

try:
    import lightgbm as lgb
    import optuna
    from optuna.pruners import MedianPruner
except ImportError:
    print("Install: pip install lightgbm optuna")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import set_seed

warnings.filterwarnings('ignore')
set_seed(42)

print("=" * 70)
print("STEP 2: LIGHTGBM WITH OPTUNA TUNING")
print("=" * 70)

# Load features
print("\n[1/5] Loading feature checkpoint...")
checkpoint = np.load('checkpoints/features.npz', allow_pickle=True)
X_train = pd.DataFrame(checkpoint['X_train_feat'], columns=checkpoint['X_train_cols'])
y_train = checkpoint['y_train']
X_test = pd.DataFrame(checkpoint['X_test_feat'], columns=checkpoint['X_test_cols'])

# Handle NaN values (fill with mean of numeric columns only)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
X_test[numeric_cols] = X_test[numeric_cols].fillna(X_train[numeric_cols].mean())

META_COLS = ['ID', 'date', 'eqt_code']
feature_cols = [c for c in X_train.columns if c not in META_COLS]
y_train_binary = (y_train > 0).astype(int)

print(f"  Loaded {X_train.shape[1]} features, {len(feature_cols)} for training")

# Train-val split by date
print("\n[2/5] Train-val split by date...")
dates = X_train['date'].unique()
np.random.seed(42)
np.random.shuffle(dates)
n_val = max(1, int(len(dates) * 0.2))
val_dates = set(dates[-n_val:])

mask_val = X_train['date'].isin(val_dates)
X_tr = X_train[~mask_val][feature_cols].values
y_tr = y_train_binary[~mask_val].values
X_va = X_train[mask_val][feature_cols].values
y_va = y_train_binary[mask_val].values

print(f"  Train: {X_tr.shape}")
print(f"  Val:   {X_va.shape}")

# Scale
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_va_scaled = scaler.transform(X_va)

# Optuna hyperparameter tuning
print("\n[3/5] Optuna hyperparameter tuning...")

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-5, 10),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-5, 10),
        'verbose': -1,
    }

    train_set = lgb.Dataset(X_tr_scaled, label=y_tr)
    val_set = lgb.Dataset(X_va_scaled, label=y_va, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=200,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(-1),
        ]
    )

    y_pred = model.predict(X_va_scaled) > 0.5
    accuracy = accuracy_score(y_va, y_pred)
    return accuracy

study = optuna.create_study(
    direction='maximize',
    pruner=MedianPruner(),
)
study.optimize(objective, n_trials=20, show_progress_bar=True)

best_params = study.best_params
print(f"  Best accuracy: {study.best_value:.4f}")
print(f"  Best params: {best_params}")

# Train final model with best params
print("\n[4/5] Training final LightGBM model...")
final_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbose': -1,
    **best_params,
}

train_set = lgb.Dataset(X_tr_scaled, label=y_tr)
val_set = lgb.Dataset(X_va_scaled, label=y_va, reference=train_set)

lgbm_model = lgb.train(
    final_params,
    train_set,
    num_boost_round=300,
    valid_sets=[val_set],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(50),
    ]
)

# Evaluate
y_pred_val = lgbm_model.predict(X_va_scaled) > 0.5
val_acc = accuracy_score(y_va, y_pred_val)
print(f"  Final val accuracy: {val_acc:.4f}")

# Save model
print("\n[5/5] Saving model...")
os.makedirs('checkpoints', exist_ok=True)
with open('checkpoints/lgbm_model.pkl', 'wb') as f:
    pickle.dump({'model': lgbm_model, 'scaler': scaler}, f)
print(f"  Saved: checkpoints/lgbm_model.pkl")

# Generate submission
print("\nGenerating submission...")
X_test_scaled = scaler.transform(X_test[feature_cols])
y_pred = lgbm_model.predict(X_test_scaled)

submission = pd.DataFrame({
    'ID': X_test['ID'].values,
    'end_of_day_return': y_pred
})
os.makedirs('submissions_output', exist_ok=True)
submission.to_csv('submissions_output/lightgbm_submission.csv', index=False)
print(f"  Saved: submissions_output/lightgbm_submission.csv")

print("\n" + "=" * 70)
print(f"LightGBM complete! Next: python 3_ft_transformer.py")
print("=" * 70)
