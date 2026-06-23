"""Scoreline-probability utilities (shared by the Dixon-Coles model).

These helpers turn a pair of expected-goal rates into a full scoreline
probability matrix and derive the most likely scoreline, the top-N scorelines
and the home/draw/away outcome probabilities. The goal-rate estimation itself
lives in :mod:`src.dixon_coles`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from . import config


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
