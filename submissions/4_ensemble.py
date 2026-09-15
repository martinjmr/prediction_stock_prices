"""
Step 4: Ensemble Prediction
Combines LightGBM and FT-Transformer predictions with 50/50 weighting.
Final submission file.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("STEP 4: ENSEMBLE (LightGBM + FT-Transformer)")
print("=" * 70)

# Load features
print("\n[1/3] Loading data...")
checkpoint = np.load('checkpoints/features.npz', allow_pickle=True)
X_test = pd.DataFrame(checkpoint['X_test_feat'], columns=checkpoint['X_test_cols'])

META_COLS = ['ID', 'date', 'eqt_code']
feature_cols = [c for c in X_test.columns if c not in META_COLS]

# Load LightGBM
print("\n[2/3] Loading models...")
with open('checkpoints/lgbm_model.pkl', 'rb') as f:
    lgbm_data = pickle.load(f)
    lgbm_model = lgbm_data['model']
    scaler_lgbm = lgbm_data['scaler']

X_test_scaled_lgbm = scaler_lgbm.transform(X_test[feature_cols])
y_pred_lgbm = lgbm_model.predict(X_test_scaled_lgbm)
print(f"  LightGBM predictions: shape {y_pred_lgbm.shape}, range [{y_pred_lgbm.min():.3f}, {y_pred_lgbm.max():.3f}]")

# Load FT-Transformer
with open('checkpoints/ftt_model.pkl', 'rb') as f:
    ftt_data = pickle.load(f)
    model_config = ftt_data['model_config']
    scaler_ftt = ftt_data['scaler']
    model_state = ftt_data['model_state']

# Rebuild FT-Transformer model
sys.path.insert(0, 'submissions')
exec(open('submissions/3_ft_transformer.py').read().split('# Optuna tuning')[0])  # Load FTTransformer class

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ftt_model = FTTransformer(**model_config).to(device)
ftt_model.load_state_dict(model_state)
ftt_model.eval()

X_test_scaled_ftt = scaler_ftt.transform(X_test[feature_cols]).astype(np.float32)
with torch.no_grad():
    X_test_t = torch.tensor(X_test_scaled_ftt).to(device)
    y_pred_ftt = ftt_model(X_test_t).cpu().numpy()

print(f"  FT-Transformer predictions: shape {y_pred_ftt.shape}, range [{y_pred_ftt.min():.3f}, {y_pred_ftt.max():.3f}]")

# Ensemble: 50/50 average
print("\n[3/3] Creating ensemble...")
y_pred_ensemble = 0.5 * y_pred_lgbm + 0.5 * y_pred_ftt
print(f"  Ensemble predictions: range [{y_pred_ensemble.min():.3f}, {y_pred_ensemble.max():.3f}]")

# Save final submission
submission = pd.DataFrame({
    'ID': X_test['ID'].values,
    'end_of_day_return': y_pred_ensemble,
})

os.makedirs('submissions_output', exist_ok=True)
submission.to_csv('submissions_output/ensemble_submission.csv', index=False)
print(f"  Saved: submissions_output/ensemble_submission.csv")

# Also save individual predictions for reference
submission_lgbm = pd.DataFrame({
    'ID': X_test['ID'].values,
    'end_of_day_return': y_pred_lgbm,
})
submission_lgbm.to_csv('submissions_output/lightgbm_only.csv', index=False)

submission_ftt = pd.DataFrame({
    'ID': X_test['ID'].values,
    'end_of_day_return': y_pred_ftt,
})
submission_ftt.to_csv('submissions_output/ftt_only.csv', index=False)

print("\n" + "=" * 70)
print("✓ ENSEMBLE COMPLETE")
print("=" * 70)
print("\nSubmission files generated:")
print("  - submissions_output/ensemble_submission.csv (FINAL)")
print("  - submissions_output/lightgbm_only.csv")
print("  - submissions_output/ftt_only.csv")
print("\nTo submit, use: submissions_output/ensemble_submission.csv")
print("=" * 70)
