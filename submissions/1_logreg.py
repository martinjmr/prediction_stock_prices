"""
Step 1: Logistic Regression Baseline
Trains a simple logistic regression on engineered features.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import set_seed

set_seed(42)

print("=" * 70)
print("STEP 1: LOGISTIC REGRESSION BASELINE")
print("=" * 70)

# Load features checkpoint
print("\n[1/4] Loading feature checkpoint...")
checkpoint = np.load('checkpoints/features.npz', allow_pickle=True)
X_train = pd.DataFrame(checkpoint['X_train_feat'], columns=checkpoint['X_train_cols'])
y_train = checkpoint['y_train']
X_test = pd.DataFrame(checkpoint['X_test_feat'], columns=checkpoint['X_test_cols'])

print(f"  Loaded {X_train.shape[1]} features")

# Train-val split by date
print("\n[2/4] Splitting train/val by date...")
META_COLS = ['ID', 'date', 'eqt_code']
feature_cols = [c for c in X_train.columns if c not in META_COLS]
y_train_binary = (y_train > 0).astype(int)

dates = X_train['date'].unique()
np.random.seed(42)
np.random.shuffle(dates)
n_val = max(1, int(len(dates) * 0.2))
val_dates = set(dates[-n_val:])

mask_val = X_train['date'].isin(val_dates)
X_tr, X_va = X_train[~mask_val][feature_cols], X_train[mask_val][feature_cols]
y_tr, y_va = y_train_binary[~mask_val], y_train_binary[mask_val]

print(f"  Train: {len(X_tr)} samples")
print(f"  Val:   {len(X_va)} samples")

# Scale and train
print("\n[3/4] Training logistic regression...")
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_va_scaled = scaler.transform(X_va)

logreg = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
logreg.fit(X_tr_scaled, y_tr)

# Evaluate
train_acc = accuracy_score(y_tr, logreg.predict(X_tr_scaled))
val_acc = accuracy_score(y_va, logreg.predict(X_va_scaled))
print(f"  Train accuracy: {train_acc:.4f}")
print(f"  Val accuracy:   {val_acc:.4f}")

# Save model
print("\n[4/4] Saving model...")
os.makedirs('checkpoints', exist_ok=True)
with open('checkpoints/logreg_model.pkl', 'wb') as f:
    pickle.dump({'model': logreg, 'scaler': scaler}, f)
print(f"  Saved: checkpoints/logreg_model.pkl")

# Generate submission
print("\nGenerating submission...")
X_test_scaled = scaler.transform(X_test[feature_cols])
y_pred = logreg.predict_proba(X_test_scaled)[:, 1]

submission = pd.DataFrame({
    'ID': X_test['ID'].values,
    'end_of_day_return': y_pred
})
os.makedirs('submissions_output', exist_ok=True)
submission.to_csv('submissions_output/logreg_submission.csv', index=False)
print(f"  Saved: submissions_output/logreg_submission.csv")

print("\n" + "=" * 70)
print(f"Baseline complete! Next: python 2_lightgbm.py")
print("=" * 70)
