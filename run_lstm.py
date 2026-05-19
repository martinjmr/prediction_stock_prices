"""Entraîne le LSTM avec les features engineerées et génère une soumission."""
import sys
sys.path.insert(0, 'src')

from utils import set_seed, load_data, build_advanced_features
from train import preprocess, split_by_date, train_lstm, submit_from_checkpoint

set_seed(42)

print("=== Chargement ===")
X, y, X_test = load_data('data')

print("\n=== Préprocessing ===")
X      = preprocess(X)
X_test = preprocess(X_test)

print("\n=== Split (par date) ===")
X_tr, X_va, y_tr, y_va = split_by_date(X, y)

print("\n=== Feature Engineering ===")
X_feat_tr   = build_advanced_features(X_tr)
X_feat_va   = build_advanced_features(X_va)
X_test_feat = build_advanced_features(X_test)

print("\n=== LSTM + features engineerées (hidden=128, epochs=30, patience=7) ===")
lstm, scaler, history, best_val = train_lstm(
    X_feat_tr, y_tr, X_feat_va, y_va,
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
    epochs=30,
    batch_size=512,
    lr=1e-3,
    patience=7,
)

print(f"\nMeilleure val_acc obtenue : {best_val:.4f}")

print("\n=== Génération de la soumission ===")
submit_from_checkpoint(X_test_feat, 'submissions/lstm_features.csv')
