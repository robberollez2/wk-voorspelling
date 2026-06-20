# ⚽ International Football Match Prediction AI

A production-ready machine-learning system that predicts the outcome of
international (national-team) football matches. Given two teams, a host country,
a tournament and a date it returns:

- **Home win / Draw / Away win** probabilities
- **Expected goals** for each side
- The **most likely scoreline** and the **top-5 scorelines**

It combines an **XGBoost** classifier, a **LightGBM** classifier and a
**Dixon-Coles** attack/defense goal model through a logistic-regression
**stacking** meta-learner, on top of leak-free temporal features (Elo ratings,
rolling form, head-to-head, rest/fatigue, venue-specific form, Elo momentum,
tournament importance) plus an optional **StatsBomb squad-strength** player
layer.

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
| **Fatigue & momentum** | Rest days since the last match, matches-played experience, venue-specific (home/away) form, win/loss streaks and Elo momentum. |
| **Head-to-head** | Directional H2H record and goal difference over the last **10 years**. |
| **Tournament importance** | Raw tournaments are bucketed (Friendly, Nations League, World Cup (Q), Continental (Q), Confederations Cup, Continental Championship, …) and one-hot encoded. |
| **Dixon-Coles goal model** | Team attack/defense strengths with home advantage, **exponential time decay** and the low-score (`rho`) correction → expected goals and a corrected scoreline matrix (0-0 … 10-10). |
| **Player layer** | Optional **StatsBomb** squad-strength feature built from line-ups of covered men's tournaments (World Cups, Euro 2020/24, Copa 2024, AFCON 2023), with a neutral fallback elsewhere. |
| **Stacking ensemble** | A multinomial logistic-regression meta-learner combines **and calibrates** the three models (beats fixed weights on log loss / Brier). |
| **Hyperparameter tuning** | **Optuna** (default 100 trials per classifier) optimising **log loss**; the Dixon-Coles time-decay is searched too. |
| **Time-aware validation** | No random splits. Train ≤ 2020, validate 2021-2023, test 2024+, plus a `TimeSeriesSplit` report. Every Dixon-Coles fit only sees matches before the split it scores. |
| **Explainability** | **SHAP** global feature importances (saved as PNG + table). |
| **Interfaces** | A **Streamlit** web app and an interactive **CLI**. |
| **Engineering** | Type hints, logging, joblib persistence, parquet caching, vectorized pandas, test suite, modular player-data layer. |

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
│   ├── dixon_coles.py           # Dixon-Coles attack/defense goal model
│   ├── player_data.py           # StatsBomb squad-strength player layer
│   ├── poisson.py               # scoreline maths helpers
│   ├── train.py                 # Optuna tuning, stacking ensemble, metrics, SHAP
│   └── predict.py               # Predictor class + interactive CLI
├── tests/test_basic.py          # unit tests (no trained models needed)
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
xgb_model.joblib  lgb_model.joblib  dixon_coles.joblib  stacker.joblib
feature_builder.joblib  feature_list.joblib  name_map.joblib
metrics.json  shap_summary.png  shap_summary.joblib
preprocessed.parquet  feature_matrix.parquet  squad_strength.parquet
```

> The first run fetches StatsBomb line-ups once (cached to
> `squad_strength.parquet`). Use `python -m src.train --no-player-data` to skip
> the player layer (e.g. offline).

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

### Ensemble (stacking)

The three base models each output `[away, draw, home]` probabilities. A
multinomial **logistic-regression stacker** takes their log-probabilities and
produces the final, calibrated probabilities:

```
P(outcome) = softmax( W · [log P_xgb , log P_lgb , log P_dixoncoles] + b )
```

The stacker is fitted on the **validation** split (which the base models do not
train on), so it learns how much to trust each model and corrects miscalibration
at the same time. It beats fixed weighted-averaging on log loss and Brier; an
equal-weight mean is also reported for comparison.

### Dixon-Coles goal model, expected goals & scorelines

Each team has an *attack* and *defense* strength; on non-neutral ground a home
term is added. Strengths are fitted by a convex, **time-decayed** weighted
Poisson regression (recent matches count more), and the Dixon-Coles `rho`
correction adjusts the 0-0/1-0/0-1/1-1 cells (improving draws). For a fixture
this gives `λ_home`, `λ_away` and a corrected matrix
`P(home=i, away=j)`, which yields the most likely score and the top-5
scorelines. Because the home term vanishes on neutral ground, the model is
naturally **immune to the neutral-labelling leak** described below.

### Player layer (StatsBomb)

`player_data.py` fetches line-ups for covered men's international tournaments
from StatsBomb open-data and maintains a per-player result rating; a team's
**squad strength** for a match is the leak-free mean rating of its starting XI.
FBref is blocked to automated access and Transfermarkt needs fragile scraping,
so StatsBomb is the practical free source here — it covers ≈3% of matches (the
marquee tournaments), and every other match (and any future fixture without a
supplied line-up) falls back to a neutral default plus a `squad_data_available`
flag. The feature therefore sharpens World Cup / Euro / Copa / AFCON predictions
without affecting the rest.

### Validation & metrics

Reported on the **2024+ hold-out test set** for each model and the ensemble:
**Accuracy, Log Loss, Brier Score, ROC AUC (OvR macro)**, plus a
`TimeSeriesSplit` cross-validation report and the Poisson expected-goals MAE.
See `models/metrics.json` after training.

---

## Results

Hold-out **test set (2024+, 3,696 matches incl. mirrored neutral rows)** from a
full 100-trial run:

| Model | Accuracy | Log Loss | Brier | ROC AUC (OvR) |
| --- | --- | --- | --- | --- |
| XGBoost | 0.581 | 0.882 | 0.522 | 0.741 |
| LightGBM | 0.583 | 0.882 | 0.522 | 0.741 |
| Poisson | 0.588 | 0.885 | 0.522 | 0.744 |
| **Ensemble** | **0.581** | **0.881** | **0.521** | **0.743** |

- **Ensemble weights:** XGBoost 0.55 · LightGBM 0.10 · Poisson 0.35
- **TimeSeriesSplit CV:** log loss 0.903 ± 0.014, accuracy 0.579 ± 0.008
- **Poisson expected-goals MAE:** home 1.00, away 0.86
- **Top SHAP features:** `elo_expected_home` ≫ `neutral` > `elo_difference` >
  `tournament_importance` > `away_elo` > recent-form / head-to-head features

> These honest numbers reflect the leak-corrected data. A naive model that keeps
> the raw neutral labelling reports a misleadingly higher ~66% accuracy by
> exploiting the "winner is listed as home" artifact in neutral matches — which
> carries no real predictive value. See *Notes & assumptions* below.

## Player data & extending the layer

The StatsBomb squad-strength layer (`src/player_data.py`) is already wired in:
it builds a leak-free per-match squad strength and merges it as features
(`home_squad_strength`, `away_squad_strength`, `squad_strength_diff`,
`squad_data_available`). Because FBref blocks automated access and StatsBomb
open-data only covers a handful of men's tournaments, the signal is **sparse by
design** — it improves predictions for the covered marquee fixtures and falls
back to a neutral default everywhere else, so it cannot (and is not claimed to)
move the global test accuracy much.

To add richer player signals (full line-ups, availability, market values, xG per
player) the pipeline only needs:

1. a loader that yields a leak-free `(date, {teamA, teamB}) -> {team: value}`
   lookup (mirroring `player_data.squad_lookup`),
2. a few lines in `FeatureBuilder._squad_stats` / `_compute_features` (new
   features are picked up by `feature_columns_` automatically),
3. a retrain.

No model, ensemble or interface code needs to change — everything consumes
`feature_columns_` generically. Supplying a real expected line-up at inference
time would let the squad features fire for future fixtures too.

---

## Notes & assumptions

- **Neutral-match leakage (important).** In the source data, neutral-venue
  matches list the *winner* as the "home" team (≈77% home / ≈0.5% away wins,
  consistently across every era). Left untreated, a model simply learns
  "neutral ⇒ home wins", which is useless for a real neutral fixture where the
  user picks the order arbitrarily. This is corrected in two places:
  (1) **training** mirrors every neutral match (swap teams + score, flip label)
  so the model learns symmetric, strength-based behaviour; (2) **inference**
  averages a neutral prediction over both team orderings, guaranteeing the
  result is order-invariant. Non-neutral matches keep their genuine home
  advantage.
- The raw data is cleaned defensively (malformed rows, duplicates, invalid
  scores and self-matches are dropped). One malformed row in the shipped data
  is skipped automatically.
- Tournament bucketing is heuristic; unrecognised regional competitions fall
  into an `Other` category.
- Teams never seen in training fall back to neutral priors (Elo 1500, empty
  form), with a logged warning.
