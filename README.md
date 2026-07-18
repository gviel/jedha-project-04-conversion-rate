# JEDHA CDSD Certification (Bloc 3) — Projet #4 : Conversion Rate Challenge

Challenge de type Kaggle : prédire si un visiteur du site **www.datascienceweekly.org** va s'inscrire à la newsletter (`converted`), à partir de quelques informations sur sa visite. Les prédictions sur le jeu de test (non labellisé) sont soumises pour être classées sur un leaderboard, avec le **f1-score** comme métrique d'évaluation.

## Objectifs

1. **EDA et préprocessing** — analyser la distribution des 6 variables (`country`, `age`, `source`, `total_pages_visited`, `new_user`, `converted`), traiter les outliers (Z-score sur `age` et `total_pages_visited`), encoder les variables catégorielles
2. **Modèles baseline** — DecisionTreeClassifier (pour l'analyse des features importantes) et LogisticRegression avec cross-validation
3. **Modèles ensemblistes/boostés** — RandomForest, AdaBoost (avec LogisticRegression puis DecisionTreeClassifier), XGBoost, avec recherche d'hyperparamètres (GridSearchCV)
4. **Export des prédictions** — entraînement sur l'intégralité du jeu de données labellisé, puis génération des prédictions sur `conversion_data_test.csv` pour soumission au challenge

## Dataset

- **Fichier train :** `data/conversion_data_train.csv` (284 580 lignes, labellisé — colonne `converted`)
- **Fichier test :** `data/conversion_data_test.csv` (non labellisé, pour soumission au challenge)
- **Variable cible :** `converted` (0/1) — fortement déséquilibrée
- **Features :** `country` (4 valeurs), `age`, `source` (3 valeurs), `total_pages_visited`, `new_user`

## Résultats

| Modèle | f1-score |
|---|---|
| **RandomForestClassifier** | **0.76** (meilleur modèle) |
| AdaBoost + DecisionTreeClassifier | 0.7547 |
| LogisticRegression (toutes features) | 0.7576 |
| XGBoost | 0.753 |
| LogisticRegression (sans `source`) | ~0.75 |
| AdaBoost + LogisticRegression | 0.732 |
| DecisionTreeClassifier (baseline) | 0.70 |

Voir `Graph_compare_models.png` pour la comparaison visuelle des scores.

### Conclusions principales
- `total_pages_visited` est de loin la feature la plus explicative de la conversion
- `new_user` est clivant : les anciens utilisateurs sont déjà inscrits
- `source` apporte peu d'information (sa suppression ne dégrade pas significativement les scores)
- `country` est intéressant hors USA : la Chine convertit moins, l'Allemagne convertit plus
- Le RandomForest l'emporte avec un arbre volontairement simple ; AdaBoost n'apporte pas d'amélioration, probablement faute d'un nombre suffisant de features

## Installation

Aucun fichier d'environnement n'est fourni pour ce projet. Dépendances principales utilisées dans le notebook :

```bash
pip install pandas numpy scipy scikit-learn xgboost matplotlib seaborn jupyter
```

## Exécution

```bash
jupyter notebook 02-Conversion_rate_challenge_GV.ipynb
```

Exécuter toutes les cellules (`Run All`). Les prédictions des différents modèles sont exportées dans `output/`.

## Structure

```
├── 02-Conversion_rate_challenge_GV.ipynb        # notebook principal (EDA, modèles, export)
├── 02-Conversion_rate_challenge.ipynb           # énoncé du challenge (référence, ne pas modifier)
├── 02-Conversion_rate_challenge_template.ipynb  # template fourni (baseline LogisticRegression 1 variable)
├── data/
│   ├── conversion_data_train.csv                # dataset labellisé (entraînement)
│   └── conversion_data_test.csv                 # dataset non labellisé (soumission)
├── output/                                      # prédictions exportées par modèle (CSV) + soumission finale
├── Graph_compare_models.png                     # comparaison des f1-scores par modèle
└── __pycache__/                                 # non versionné
```
