"""Point d'entrée principal — pipeline complet : chargement, préprocessing, entraînement, soumission."""
import sys
sys.path.insert(0, 'src')

from utils import set_seed, load_data, build_advanced_features
from train import preprocess, split_by_date, train_logreg, train_lstm, train_lgbm, make_submission

# ── Reproductibilité ──────────────────────────────────────────────────────────
set_seed(42)

# ── Chargement des données ────────────────────────────────────────────────────
print("=== Chargement des données ===")
X, y, X_test = load_data('data')

# ── Préprocessing ─────────────────────────────────────────────────────────────
print("\n=== Préprocessing ===")
X      = preprocess(X)
X_test = preprocess(X_test)

# ── Split train / validation ──────────────────────────────────────────────────
# Split d'abord sur les returns bruts — les features cross-sectionnelles
# sont calculées par date, donc le split par date préserve leur cohérence.
print("\n=== Split ===")
X_tr, X_va, y_tr, y_va = split_by_date(X, y)

# ── Feature Engineering ───────────────────────────────────────────────────────
# Appliqué après le split : chaque fold est normalisé indépendamment.
# X_tr / X_va     → returns bruts uniquement  → LSTM / Transformer
# X_feat_*        → returns + features engineerées → LightGBM
print("\n=== Feature Engineering ===")
X_feat_tr   = build_advanced_features(X_tr)
X_feat_va   = build_advanced_features(X_va)
X_test_feat = build_advanced_features(X_test)

# ── Baseline : Régression logistique ─────────────────────────────────────────
print("\n=== Baseline : Régression logistique ===")
logreg, scaler_lr, acc_lr = train_logreg(X_tr, y_tr, X_va, y_va)
make_submission(logreg, scaler_lr, X_test, 'submissions/logreg_baseline.csv')

# ── LightGBM avec features engineerées ───────────────────────────────────────
print("\n=== LightGBM ===")
lgbm, acc_lgbm = train_lgbm(X_feat_tr, y_tr, X_feat_va, y_va)
# make_submission(lgbm, None, X_test_feat, 'submissions/lgbm.csv')

# ── LSTM ──────────────────────────────────────────────────────────────────────
print("\n=== LSTM (20 epochs, early stopping patience=5) ===")
lstm, scaler_lstm, history, acc_lstm = train_lstm(X_tr, y_tr, X_va, y_va)
make_submission(lstm, scaler_lstm, X_test, 'submissions/lstm.csv')

# ── Comparaison ───────────────────────────────────────────────────────────────
print("\n=== Résultats ===")
print(f"Logistic Regression : {acc_lr:.4f}")
print(f"LightGBM            : {acc_lgbm:.4f}")
print(f"LSTM                : {acc_lstm:.4f}")
