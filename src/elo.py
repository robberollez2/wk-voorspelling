"""Elo rating system for international football teams.

A World-Football-Elo style implementation: ratings start at
:data:`config.ELO_START`, the expected score uses a home-advantage bonus
(suppressed on neutral ground), and the update magnitude scales with the
tournament importance and the goal-difference margin of victory.

The :class:`EloRatingSystem` keeps only a ``dict`` of ratings as state, so it is
cheap to serialize with joblib and reuse at inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .config import get_logger

logger = get_logger(__name__)


def goal_difference_multiplier(goal_diff: int) -> float:
    """Margin-of-victory multiplier ``G`` for the Elo update.

    Args:
        goal_diff: Absolute goal difference of the match.

    Returns:
        ``1.0`` for a 0/1 goal margin, ``1.5`` for two goals, and
        ``(11 + gd) / 8`` for larger margins (standard World-Football-Elo).
    """
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


class EloRatingSystem:
    """Incremental Elo rating tracker.

    Attributes:
        ratings: Mapping ``team -> current rating``.
    """

    def __init__(
        self,
        start: float = config.ELO_START,
        base_k: float = config.ELO_BASE_K,
        home_advantage: float = config.ELO_HOME_ADVANTAGE,
    ) -> None:
        self.start = float(start)
        self.base_k = float(base_k)
        self.home_advantage = float(home_advantage)
        self.ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        """Return the current rating for ``team`` (initialising if unseen)."""
        return self.ratings.get(team, self.start)

    def expected_home(self, home: str, away: str, neutral: bool) -> float:
        """Expected score (win probability proxy) for the home team."""
        home_elo = self.rating(home)
        away_elo = self.rating(away)
        bonus = 0.0 if neutral else self.home_advantage
        return 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + bonus)) / 400.0))

    def pre_match(self, home: str, away: str) -> tuple[float, float]:
        """Return the ``(home_elo, away_elo)`` *before* the match is played."""
        return self.rating(home), self.rating(away)

    def update(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        importance_weight: float,
        neutral: bool,
    ) -> tuple[float, float]:
        """Update ratings after a match and return the pre-match ratings.

        Args:
            home: Home team (current name).
            away: Away team (current name).
            home_score: Goals scored by the home team.
            away_score: Goals scored by the away team.
            importance_weight: Tournament importance weight (scales ``K``).
            neutral: Whether the match was played on neutral ground.

        Returns:
            The ``(home_elo, away_elo)`` ratings as they were *before* the update
            (the leak-free values to use as features for this match).
        """
        home_elo = self.rating(home)
        away_elo = self.rating(away)

        bonus = 0.0 if neutral else self.home_advantage
        expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + bonus)) / 400.0))

        if home_score > away_score:
            actual_home = 1.0
        elif home_score == away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        k = self.base_k * float(importance_weight)
        g = goal_difference_multiplier(home_score - away_score)
        delta = k * g * (actual_home - expected_home)

        self.ratings[home] = home_elo + delta
        self.ratings[away] = away_elo - delta
        return home_elo, away_elo


def attach_elo(df: pd.DataFrame) -> tuple[pd.DataFrame, EloRatingSystem]:
    """Attach leak-free ``home_elo``/``away_elo``/``elo_difference`` columns.

    Processes matches in chronological order (the frame is assumed sorted) and
    records each team's rating *before* the match, then applies the update.

    Args:
        df: Preprocessed match frame (must contain ``tournament_importance``).

    Returns:
        ``(df_with_elo, fitted_system)``.
    """
    logger.info("Computing Elo ratings over %d matches", len(df))
    elo = EloRatingSystem()

    n = len(df)
    home_elos = np.empty(n, dtype=float)
    away_elos = np.empty(n, dtype=float)

    homes = df["home_team"].to_numpy()
    aways = df["away_team"].to_numpy()
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()
    weights = df["tournament_importance"].to_numpy()
    neutrals = df["neutral"].to_numpy()

    for i in range(n):
        home_elos[i], away_elos[i] = elo.update(
            homes[i], aways[i], hs[i], as_[i], weights[i], neutrals[i]
        )

    out = df.copy()
    out["home_elo"] = home_elos
    out["away_elo"] = away_elos
    out["elo_difference"] = home_elos - away_elos
    logger.info("Elo computed for %d teams", len(elo.ratings))
    return out, elo


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from .preprocess import preprocess

    frame, _ = preprocess()
    frame, system = attach_elo(frame)
    top = sorted(system.ratings.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print("Top 15 teams by Elo:")
    for team, rating in top:
        print(f"  {team:25s} {rating:7.1f}")
