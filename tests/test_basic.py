"""Unit tests for the core logic (no trained models required).

Run with::

    pytest -q
    # or, without pytest installed:
    python tests/test_basic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dixon_coles import DixonColesModel  # noqa: E402
from src.elo import EloRatingSystem, goal_difference_multiplier  # noqa: E402
from src.features import FeatureBuilder  # noqa: E402
from src.poisson import (  # noqa: E402
    outcome_probabilities,
    scoreline_matrix,
    top_scorelines,
)
from src.predict import Predictor  # noqa: E402
from src.preprocess import categorize_tournament, tournament_importance_weight  # noqa: E402


# --------------------------------------------------------------------------- #
# Tournament categorization
# --------------------------------------------------------------------------- #
def test_tournament_categories():
    assert categorize_tournament("Friendly") == "Friendly"
    assert categorize_tournament("FIFA World Cup") == "World Cup"
    assert categorize_tournament("World Cup qualifier") == "World Cup Qualification"
    assert categorize_tournament("World Cup and Asian Cup qual") == "World Cup Qualification"
    assert categorize_tournament("European Championship") == "Continental Championship"
    assert categorize_tournament("European Championship qual") == "Continental Qualification"
    assert categorize_tournament("UEFA Nations League A") == "Nations League"
    assert categorize_tournament("Confederations Cup") == "Confederations Cup"
    assert categorize_tournament("CECAFA Cup") == "Other"
    assert tournament_importance_weight("World Cup") > tournament_importance_weight("Friendly")


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #
def test_goal_difference_multiplier():
    assert goal_difference_multiplier(0) == 1.0
    assert goal_difference_multiplier(1) == 1.0
    assert goal_difference_multiplier(2) == 1.5
    assert goal_difference_multiplier(3) == (11 + 3) / 8
    assert goal_difference_multiplier(-4) == (11 + 4) / 8


def test_elo_updates_and_advantage():
    elo = EloRatingSystem()
    # Home advantage raises the expected home score on non-neutral ground.
    exp_home_adv = elo.expected_home("A", "B", neutral=False)
    exp_neutral = elo.expected_home("A", "B", neutral=True)
    assert 0.0 < exp_neutral < 1.0
    assert exp_home_adv > exp_neutral

    home_pre, away_pre = elo.update("A", "B", 3, 0, importance_weight=1.0, neutral=True)
    assert home_pre == 1500.0 and away_pre == 1500.0
    assert elo.rating("A") > 1500.0  # winner gains
    assert elo.rating("B") < 1500.0  # loser loses
    # Ratings are zero-sum around the win/loss delta.
    assert abs((elo.rating("A") - 1500.0) + (elo.rating("B") - 1500.0)) < 1e-9


# --------------------------------------------------------------------------- #
# Poisson scoreline maths
# --------------------------------------------------------------------------- #
def test_scoreline_matrix_and_outcomes():
    matrix = scoreline_matrix(1.6, 1.2)
    assert abs(matrix.sum() - 1.0) < 1e-9
    probs = outcome_probabilities(matrix)  # [away, draw, home]
    assert abs(probs.sum() - 1.0) < 1e-9
    # Higher home rate -> home win more likely than away win.
    assert probs[2] > probs[0]

    tops = top_scorelines(matrix, top_n=5)
    assert len(tops) == 5
    probs_sorted = [p for _, p in tops]
    assert probs_sorted == sorted(probs_sorted, reverse=True)


# --------------------------------------------------------------------------- #
# Feature builder: no leakage shape + mirror symmetry
# --------------------------------------------------------------------------- #
def _toy_frame() -> pd.DataFrame:
    rows = [
        ("2000-01-01", "A", "B", 2, 0, False),
        ("2000-02-01", "B", "C", 1, 1, False),
        ("2000-03-01", "A", "C", 0, 3, True),   # neutral -> should be mirrored
        ("2000-04-01", "C", "A", 2, 2, False),
        ("2000-05-01", "B", "A", 1, 0, True),   # neutral -> should be mirrored
    ]
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "neutral"])
    df["date"] = pd.to_datetime(df["date"])
    df["tournament_category"] = "Friendly"
    df["tournament_importance"] = tournament_importance_weight("Friendly")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["result"] = np.where(df.home_score > df.away_score, 2, np.where(df.home_score == df.away_score, 1, 0))
    return df


def test_feature_builder_no_nan_and_columns():
    df = _toy_frame()
    builder = FeatureBuilder()
    feats = builder.fit_transform(df, use_cache=False, mirror_neutral=True)
    # Two neutral matches -> two extra mirrored rows.
    assert len(feats) == len(df) + 2
    assert not feats[builder.feature_columns_].isna().any().any()
    one = builder.transform_one("A", "B", pd.Timestamp("2000-06-01"), False, "Friendly")
    assert list(one.columns) == builder.feature_columns_
    # Squad features exist and default to neutral when no StatsBomb lookup.
    for col in ("home_squad_strength", "squad_strength_diff", "squad_data_available"):
        assert col in builder.feature_columns_
    assert (feats["squad_data_available"] == 0).all()


def test_dixon_coles_model():
    rng = np.random.default_rng(0)
    strength = {"A": 3, "C": 2, "D": 1, "B": 0}
    teams = list(strength)
    dates = pd.date_range("2015-01-01", periods=80, freq="20D")
    rows = []
    for d in dates:
        h, a = rng.choice(teams, size=2, replace=False)
        hg = int(rng.poisson(0.8 + 0.4 * strength[h]))
        ag = int(rng.poisson(0.8 + 0.4 * strength[a]))
        rows.append((d, h, a, hg, ag, False))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "neutral"])

    dc = DixonColesModel(xi=0.0).fit(df)
    # Stronger attack -> higher attack coefficient.
    assert dc.attack["A"] > dc.attack["B"]
    matrix = dc.scoreline_matrix("A", "B", neutral=False)
    assert abs(matrix.sum() - 1.0) < 1e-9
    probs = dc.outcome_probabilities("A", "B", neutral=False)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert probs[2] > probs[0]  # A should be favoured over B
    # Neutral predictions are order-invariant.
    p_ab = dc.outcome_probabilities("A", "B", neutral=True)
    p_ba = dc.outcome_probabilities("B", "A", neutral=True)
    assert abs(p_ab[2] - p_ba[0]) < 1e-9 and abs(p_ab[0] - p_ba[2]) < 1e-9
    lam, mu = dc.predict_expected("A", "B", neutral=False)
    assert lam > 0 and mu > 0


def test_mirror_symmetry():
    df = _toy_frame()
    builder = FeatureBuilder()
    feats = builder.fit_transform(df, use_cache=False, mirror_neutral=True)
    neutral_pairs = feats[feats["is_mirror"] == 1]
    assert len(neutral_pairs) == 2
    # For every mirrored row there is an original with swapped teams + flipped label.
    for _, mrow in neutral_pairs.iterrows():
        orig = feats[(feats["is_mirror"] == 0) & (feats["home_team"] == mrow["away_team"]) &
                     (feats["away_team"] == mrow["home_team"]) & (feats["date"] == mrow["date"])]
        assert len(orig) == 1
        orig = orig.iloc[0]
        assert mrow["result"] == 2 - orig["result"]
        # home/away Elo features are swapped between the original and its mirror.
        assert abs(mrow["home_elo"] - orig["away_elo"]) < 1e-9
        assert abs(mrow["away_elo"] - orig["home_elo"]) < 1e-9
        assert abs(mrow["elo_difference"] + orig["elo_difference"]) < 1e-9


# --------------------------------------------------------------------------- #
# Neutral detection (static method, no model needed)
# --------------------------------------------------------------------------- #
def test_detect_neutral():
    assert Predictor.detect_neutral("Belgium", "Belgium") is False
    assert Predictor.detect_neutral("Belgium", "Germany") is True
    assert Predictor.detect_neutral("Brazil", "United States") is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll tests passed.")
