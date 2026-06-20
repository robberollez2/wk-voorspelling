# ⚽ International Football Match Prediction AI

A production-ready machine-learning system that predicts the outcome of
international (national-team) football matches. Given two teams, a host country,
a tournament and a date it returns:

- **Home win / Draw / Away win** probabilities
- **Expected goals** for each side
- The **most likely scoreline** and the **top-5 scorelines**

It combines an **XGBoost** classifier, a **LightGBM** classifier and a
**Poisson** goal model into a weighted ensemble, on top of leak-free temporal
features (Elo ratings, rolling form, head-to-head, tournament importance).

```
Belgium vs Netherlands

Home Win:  47.3%
Draw:      25.1%
Away Win:  27.6%

Expected Goals:
  Belgium:     1.74
  Netherlands: 1.21

Most likely score:
  2-1 (12.4%)

Top 5 outcomes:
  2-1, 1-1, 1-0, 2-0, 1-2
```

---

## Key features

| Area | What it does |
| --- | --- |
| **Name normalization** | Historical team names are mapped to their current name via `countries_names.csv` (e.g. *Dahomey → Benin*, *British Honduras → Belize*). |
| **Automatic neutral detection** | The user never sets the `neutral` flag. It is derived: `neutral = (country != home_team)`. |
| **No data leakage** | Every feature for a match is computed from matches that occurred **strictly before** it. The exact same code path is used for training and inference, so there is zero train/serve skew. |
| **Elo ratings** | Full World-Football-Elo system (start 1500, home-advantage bonus, margin-of-victory and tournament-importance scaling). |
| **Rolling form** | Wins/draws/losses, goals for/against, goal difference and points over the last **5 / 10 / 20** matches per team. |
| **Recency weighting** | Exponentially-decayed weighted form (configurable half-life). |
| **Head-to-head** | Directional H2H record and goal difference over the last **10 years**. |
| **Tournament importance** | Raw tournaments are bucketed (Friendly, Nations League, World Cup (Q), Continental (Q), Confederations Cup, Continental Championship, …) and one-hot encoded. |
| **Scoreline model** | Independent-Poisson scoreline matrix (0-0 … 10-10) → top scorelines and an independent W/D/L estimate. |
| **Hyperparameter tuning** | **Optuna** (default 100 trials per classifier) optimising **log loss**. |
| **Time-aware validation** | No random splits. Train ≤ 2020, validate 2021-2023, test 2024+, plus a `TimeSeriesSplit` report. |
| **Explainability** | **SHAP** global feature importances (saved as PNG + table). |
| **Interfaces** | A **Streamlit** web app and an interactive **CLI**. |
| **Engineering** | Type hints, logging, joblib persistence, parquet caching, vectorized pandas, designed to extend to player-level data later. |

---

## Project structure

```
football_ai/
├── data/
│   ├── all_matches.csv          # raw match history
│   └── countries_names.csv      # historical → current name + colours
├── models/                      # generated artifacts (models, caches, metrics)
├── src/
│   ├── config.py                # paths, constants, logging
│   ├── preprocess.py            # loading, normalization, targets, tournament buckets
│   ├── elo.py                   # Elo rating system
│   ├── features.py              # leak-free FeatureBuilder (training + inference)
│   ├── poisson.py               # Poisson goal model + scoreline maths
│   ├── train.py                 # Optuna tuning, ensemble, metrics, SHAP
│   └── predict.py               # Predictor class + interactive CLI
├── app.py                       # Streamlit web app
├── predict.py                   # root entry point → src.predict
├── requirements.txt
└── README.md
```

---

## Installation

Requires **Python 3.12+** (verified on 3.12 and 3.14).

```bash
cd football_ai

# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

Make sure `data/all_matches.csv` and `data/countries_names.csv` are present
(they ship with the repo).

---

## Usage

### 1. Train the models

```bash
python -m src.train                 # full run: 100 Optuna trials per classifier
python -m src.train --trials 30     # fewer trials (faster)
python -m src.train --quick         # 10 trials, for a quick smoke test
python -m src.train --no-cache      # ignore cached preprocessing/features
```

Training writes everything to `models/`:

```
xgb_model.joblib  lgb_model.joblib  poisson_home.joblib  poisson_away.joblib
ensemble_weights.joblib  feature_builder.joblib  feature_list.joblib
name_map.joblib  metrics.json  shap_summary.png  shap_summary.joblib
preprocessed.parquet  feature_matrix.parquet
```

### 2. Predict from the terminal

Interactive:

```bash
python predict.py
```

```
Home Team [Belgium]: Belgium
Away Team [Netherlands]: Netherlands
Country [Belgium]: Belgium
Tournament [Friendly]: Nations League
Date (YYYY-MM-DD) [2026-09-10]: 2026-09-10
```

One-shot:

```bash
python predict.py --home Belgium --away Netherlands \
    --country Germany --tournament "World Cup" --date 2026-06-20
```

> `--country Germany` for a Belgium "home" game ⇒ automatically `neutral = True`.

### 3. Web app

```bash
streamlit run app.py
```

Pick the teams, host country, tournament and date; the app shows the
probabilities, expected goals, most likely score, top-5 scorelines, plus
win-probability, Elo, recent-form and scoreline-matrix charts (using each
team's colours), and a model-transparency panel with the global SHAP summary.

---

## How it works

### Outcome labels

`0 = away win`, `1 = draw`, `2 = home win` (kept consistent everywhere,
including the model `predict_proba` column order).

### Ensemble

```
P(outcome) = w_xgb · P_xgboost + w_lgb · P_lightgbm + w_poisson · P_poisson
```

The weights are found by a simplex grid-search that minimises **validation**
log loss. The Poisson contribution comes from summing the scoreline matrix into
home/draw/away probabilities, while expected goals and scorelines come directly
from the two Poisson regressions.

### Expected goals & scorelines

The Poisson goal model estimates `λ_home` and `λ_away`. Assuming independent
Poisson scoring, `P(home=i, away=j) = Poisson(i; λ_home) · Poisson(j; λ_away)`.
The matrix yields the most likely score and the top-5 scorelines.

### Validation & metrics

Reported on the **2024+ hold-out test set** for each model and the ensemble:
**Accuracy, Log Loss, Brier Score, ROC AUC (OvR macro)**, plus a
`TimeSeriesSplit` cross-validation report and the Poisson expected-goals MAE.
See `models/metrics.json` after training.

---

## Extending to player data (future-proofing)

The pipeline is built around a single `FeatureBuilder` that owns all state. To
add player-level signals (squad strength, availability, lineups) you would:

1. add the new raw columns to the preprocessing step,
2. extend `FeatureBuilder._compute_features` with the new features (they are
   automatically picked up by `feature_columns_`),
3. retrain.

No other module needs to change — the models, ensemble and interfaces consume
`feature_columns_` generically.

---

## Notes & assumptions

- The raw data is cleaned defensively (malformed rows, duplicates, invalid
  scores and self-matches are dropped). One malformed row in the shipped data
  is skipped automatically.
- Tournament bucketing is heuristic; unrecognised regional competitions fall
  into an `Other` category.
- Teams never seen in training fall back to neutral priors (Elo 1500, empty
  form), with a logged warning.
