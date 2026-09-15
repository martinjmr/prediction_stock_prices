"""
Step 0: Feature Engineering & Preprocessing
Loads training data, builds engineered features, and saves as NPZ checkpoint.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_data, build_advanced_features, set_seed

set_seed(42)

# Ensure checkpoint directory exists
os.makedirs('checkpoints', exist_ok=True)

print("=" * 70)
print("STEP 0: FEATURE ENGINEERING")
print("=" * 70)

# Load data
print("\n[1/3] Loading data...")
X_train, y_train, X_test = load_data(data_dir='data')
print(f"  Train shape: {X_train.shape}")
print(f"  Test shape:  {X_test.shape}")

# Build features
print("\n[2/3] Building engineered features...")
X_train_feat = build_advanced_features(X_train)
X_test_feat = build_advanced_features(X_test)
print(f"  Train with features: {X_train_feat.shape}")
print(f"  Test with features:  {X_test_feat.shape}")

# Save checkpoint
print("\n[3/3] Saving checkpoint...")
checkpoint_path = 'checkpoints/features.npz'
np.savez_compressed(
    checkpoint_path,
    X_train_feat=X_train_feat.values,
    X_train_cols=X_train_feat.columns.values,
    y_train=y_train['end_of_day_return'].values,
    X_test_feat=X_test_feat.values,
    X_test_cols=X_test_feat.columns.values,
)
print(f"  Saved: {checkpoint_path}")

print("\n" + "=" * 70)
print(f"Features ready for training! Run: python 1_logreg.py")
print("=" * 70)
