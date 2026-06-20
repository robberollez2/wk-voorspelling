"""Poisson goal model and scoreline-probability utilities.

Two :class:`~sklearn.linear_model.PoissonRegressor` models (wrapped in scaling
pipelines) estimate the expected number of goals for the home and away side.
From those two rate parameters we build a full independent-Poisson scoreline
matrix, which yields:

* the most likely scoreline and the top-N scorelines,
* an independent home/draw/away probability estimate (used in the ensemble).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .config import get_logger

logger = get_logger(__name__)


class PoissonGoalModel:
    """Predicts expected home/away goals with two Poisson regressions."""

    def __init__(self, alpha: float = 1e-3, max_iter: int = 1000) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.home_model: Pipeline = self._make_pipeline()
        self.away_model: Pipeline = self._make_pipeline()

    def _make_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "poisson",
                    PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter, tol=1e-7),
                ),
            ]
        )

    def fit(
        self,
        X: pd.DataFrame,
        home_goals: pd.Series | np.ndarray,
        away_goals: pd.Series | np.ndarray,
    ) -> "PoissonGoalModel":
        """Fit both goal regressions.

        Args:
            X: Feature matrix.
            home_goals: Observed home goals (target for the home model).
            away_goals: Observed away goals (target for the away model).
        """
        logger.info("Fitting Poisson goal models on %d rows", len(X))
        self.home_model.fit(X, np.asarray(home_goals, dtype=float))
        self.away_model.fit(X, np.asarray(away_goals, dtype=float))
        return self

    def predict_expected(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return clipped expected ``(home_goals, away_goals)`` arrays."""
        lam_home = self.home_model.predict(X)
        lam_away = self.away_model.predict(X)
        lam_home = np.clip(lam_home, config.MIN_EXPECTED_GOALS, config.MAX_EXPECTED_GOALS)
        lam_away = np.clip(lam_away, config.MIN_EXPECTED_GOALS, config.MAX_EXPECTED_GOALS)
        return lam_home, lam_away


# --------------------------------------------------------------------------- #
# Scoreline maths
# --------------------------------------------------------------------------- #
def scoreline_matrix(
    lam_home: float,
    lam_away: float,
    max_goals: int = config.SCORE_MATRIX_MAX_GOALS,
) -> np.ndarray:
    """Independent-Poisson scoreline probability matrix.

    Args:
        lam_home: Expected home goals.
        lam_away: Expected away goals.
        max_goals: Largest goal count modelled per side.

    Returns:
        A ``(max_goals + 1, max_goals + 1)`` array where ``M[i, j]`` is the
        probability of the scoreline ``home=i, away=j`` (rows/cols normalised
        to sum to 1).
    """
    goals = np.arange(max_goals + 1)
    home_probs = poisson.pmf(goals, lam_home)
    away_probs = poisson.pmf(goals, lam_away)
    matrix = np.outer(home_probs, away_probs)
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def outcome_probabilities(matrix: np.ndarray) -> np.ndarray:
    """Return ``[p_away_win, p_draw, p_home_win]`` from a scoreline matrix."""
    p_home = np.tril(matrix, k=-1).sum()  # home goals > away goals
    p_draw = np.trace(matrix)
    p_away = np.triu(matrix, k=1).sum()   # away goals > home goals
    probs = np.array([p_away, p_draw, p_home], dtype=float)
    total = probs.sum()
    return probs / total if total > 0 else probs


def top_scorelines(
    matrix: np.ndarray,
    top_n: int = 5,
    report_max_goals: int = config.SCORE_REPORT_MAX_GOALS,
) -> list[tuple[tuple[int, int], float]]:
    """Return the ``top_n`` most likely scorelines within the report grid.

    Args:
        matrix: Scoreline probability matrix.
        top_n: Number of scorelines to return.
        report_max_goals: Only consider scorelines up to this many goals/side.

    Returns:
        A list of ``((home_goals, away_goals), probability)`` sorted by
        descending probability.
    """
    limit = min(report_max_goals, matrix.shape[0] - 1)
    sub = matrix[: limit + 1, : limit + 1]
    flat_idx = np.argsort(sub, axis=None)[::-1][:top_n]
    results: list[tuple[tuple[int, int], float]] = []
    for idx in flat_idx:
        i, j = np.unravel_index(idx, sub.shape)
        results.append(((int(i), int(j)), float(sub[i, j])))
    return results


def most_likely_score(matrix: np.ndarray) -> tuple[tuple[int, int], float]:
    """Return the single most likely ``((home, away), probability)``."""
    return top_scorelines(matrix, top_n=1)[0]
