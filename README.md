# CFM Challenge — Prédiction du signe du rendement intrajournalier

Projet de semaine (L3 Deep Learning / IASO, 2025-2026) : prédire, pour chaque
couple (action, jour), si le rendement des **30 dernières minutes** de la
séance sera supérieur à la médiane cross-sectionnelle du jour, à partir des
rendements 5 minutes observés entre 9h30 et 15h25 (71 bins).

**Résultat final : Ensemble LightGBM + FT-Transformer → 52.43% (3e place),
vs. benchmark CFM à 51.80%.**

| Modèle | Accuracy (leaderboard) |
|---|---|
| Benchmark CFM (référence) | 51.80% |
| LightGBM (features engineerées) | 52.40% |
| FT-Transformer | 52.41% |
| **Ensemble 50/50** | **52.43%** |

## Données

~700 actions × ~700 jours, séries de rendements 5 min (9h30–15h25).

- `data/input_training.csv`, `data/output_training_*.csv` — entraînement
- `data/input_test.csv` — test (non fourni dans le repo, à placer soi-même dans `data/`)

## Approche

1. **Feature engineering** (`src/features.py`, `submission/src/features.py`) :
   71 bins de rendements bruts → 183 features réparties en 6 familles
   (statistiques temporelles, EWMA multi-échelle, volatilité, chemin de prix
   cumulé, moments d'ordre supérieur, features cross-sectionnelles
   rang/z-score par jour).
2. **Baseline** : régression logistique sur features brutes vs. enrichies
   (`logreg_enriched.py`, `submission/1_logreg.py`).
3. **LightGBM** : recherche d'hyperparamètres par Optuna, puis entraînement
   final multi-seeds sur folds expanding-window (`submission/2_lightgbm.py`).
4. **FT-Transformer** : Feature Tokenizer Transformer — chaque feature
   scalaire est tokenisée puis passée dans un encodeur Transformer, prédiction
   via le token CLS (`submission/3_ft_transformer.py`).
5. **Ensemble** : moyenne 50/50 des probabilités LightGBM et FT-Transformer,
   dont les biais inductifs complémentaires réduisent l'erreur
   (`submission/4_ensemble.py`).

Validation : split temporel par date / expanding-window CV avec embargo pour
éviter toute fuite d'information entre passé et futur.

Un pipeline LSTM alternatif (`src/train.py`, `src/models.py`, `run_lstm.py`,
`main.py`) explore une approche séquentielle sur les rendements bruts.

## Structure du dépôt

```
submission/           Pipeline final, propre et numéroté (0→4)
  0_features.py         construit et sauvegarde les 183 features
  1_logreg.py            baseline régression logistique
  2_lightgbm.py          LightGBM + Optuna
  3_ft_transformer.py    FT-Transformer + Optuna
  4_ensemble.py           ensemble final
  src/features.py         feature engineering (version épurée)

src/                  Code source du pipeline exploratoire (LSTM, utils)
  features.py            feature engineering complet
  models.py               LSTM / Transformer / TCN / CatBoost
  train.py                boucles d'entraînement, CV, soumission
  utils.py                chargement des données, reproductibilité

notebooks/            Exploration (EDA) et visualisations
report/                Rapport (LaTeX + PDF) et notes techniques
figures/               Figures générées pour le rapport/la présentation
checkpoints/           Modèles et features sauvegardés (générés localement)
presentation_cfm*.pptx Supports de présentation
```

## Reproduire les résultats

```bash
pip install -r submission/requirements.txt
cd submission
python 0_features.py       # construit checkpoints/features.npz
python 1_logreg.py         # baseline
python 2_lightgbm.py       # LightGBM (Optuna + entraînement final)
python 3_ft_transformer.py # FT-Transformer (Optuna + entraînement final)
python 4_ensemble.py       # soumission finale (ensemble)
```

Placer au préalable `input_training.csv`, `output_training_*.csv` et
`input_test.csv` dans `data/`.

## Équipe

Quiterie Guignard, Martin Jomier — dans le cadre du cours Deep Learning,
L3 IASO, semaine de projet (mai 2026).
