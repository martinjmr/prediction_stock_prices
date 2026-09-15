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

1. **Feature engineering** (`src/utils.build_advanced_features()`) :
   71 bins de rendements bruts → 183 features réparties en 6 familles
   (statistiques temporelles, EWMA multi-échelle, volatilité, chemin de prix
   cumulé, moments d'ordre supérieur, features cross-sectionnelles
   rang/z-score par jour).
2. **Baseline** : régression logistique sur features enrichies
   (`submissions/1_logreg.py`).
3. **LightGBM** : recherche d'hyperparamètres par Optuna, puis entraînement
   final sur folds temporels (`submissions/2_lightgbm.py`).
4. **FT-Transformer** : Feature Tokenizer Transformer — chaque feature
   scalaire est tokenisée puis passée dans un encodeur Transformer, prédiction
   via le token CLS (`submissions/3_ft_transformer.py`).
5. **Ensemble** : moyenne 50/50 des probabilités LightGBM et FT-Transformer,
   dont les biais inductifs complémentaires réduisent l'erreur
   (`submissions/4_ensemble.py`).

Validation : split temporel par date / expanding-window CV avec embargo pour
éviter toute fuite d'information entre passé et futur.

Un pipeline LSTM alternatif (`src/train.py`, `src/models.py`, `run_lstm.py`,
`main.py`) explore une approche séquentielle sur les rendements bruts.

## Structure du dépôt

```
submissions/          📌 Pipeline final, propre et numéroté (0→4) — UTILISER CELUI-CI
  0_features.py         construit et sauvegarde les 183 features
  1_logreg.py            baseline régression logistique
  2_lightgbm.py          LightGBM + Optuna
  3_ft_transformer.py    FT-Transformer + Optuna
  4_ensemble.py           ensemble final
  requirements.txt       dépendances avec versions épinglées
  README.md              instructions détaillées du pipeline

src/                  Code source exploratoire & utilitaires
  features.py            feature engineering complet
  models.py               architectures (LSTM, Transformer, TCN)
  train.py                boucles d'entraînement, CV, soumission
  utils.py                chargement données, reproductibilité, build_advanced_features

notebooks/            Exploration (EDA) et visualisations
  01_eda.ipynb           analyse exploratoire

data/                 Données (à télécharger)
  input_training.csv    features d'entraînement
  output_training_*.csv cibles d'entraînement
  input_test.csv        features de test
```

## Reproduire les résultats

### Setup

```bash
# 1. Télécharger les données de compétition
# Placer dans data/ :
#   - input_training.csv
#   - output_training_*.csv
#   - input_test.csv

# 2. Installer dépendances
cd submissions
pip install -r requirements.txt
```

### Exécution

```bash
# Exécuter depuis le répertoire du projet (parent de submissions/)
cd submissions

python 0_features.py       # Build features → checkpoints/features.npz
python 1_logreg.py         # Baseline logistic regression
python 2_lightgbm.py       # LightGBM + Optuna (20 trials)
python 3_ft_transformer.py # FT-Transformer + Optuna (15 trials)
python 4_ensemble.py       # FINAL SUBMISSION (50/50 ensemble)
```

Les prédictions finales sont sauvegardées dans `submissions_output/ensemble_submission.csv`.

**Durée estimée :** ~10-30 minutes (dépend du hardware et des données)

## Équipe

Quiterie Guignard, Martin Jomier — dans le cadre du cours Deep Learning,
L3 IASO, semaine de projet (mai 2026).
