"""Leak-free temporal feature engineering.

The :class:`FeatureBuilder` walks every match in chronological order while
maintaining mutable state (Elo ratings, per-team rolling logs, head-to-head
records).  For each match it first *reads* features from the current state
(which by construction only reflects earlier matches -> no data leakage) and
*then* updates the state with the match result.

Crucially, the **same** ``_compute_features`` code path is used both when
building the training matrix and when scoring a hypothetical future match, so
there can be no train/inference skew.

Design notes
------------
* Rolling form is read from bounded per-team deques (O(window) per match).
* Head-to-head uses per-pair logs filtered to the last
  :data:`config.H2H_YEARS` years.
* Recency weighting uses exponential decay with a configurable half-life.
* The fully built feature matrix is cached to parquet; the fitted builder
  (state needed for inference) is serialized separately with joblib.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config
from .config import get_logger
from .elo import EloRatingSystem
from .preprocess import (
    TOURNAMENT_CATEGORIES,
    categorize_tournament,
    tournament_importance_weight,
)

logger = get_logger(__name__)

_MAX_LOG = max(max(config.FORM_WINDOWS), config.RECENCY_WINDOW)
_DECAY_PER_STEP = 0.5 ** (1.0 / config.RECENCY_HALFLIFE)
_H2H_DELTA = pd.Timedelta(days=int(config.H2H_YEARS * 365.25))


@dataclass
class _TeamState:
    """Bounded rolling log of a single team's recent appearances."""

    # Each entry: (timestamp, goals_for, goals_against, outcome) where outcome
    # is 2=win, 1=draw, 0=loss from the team's own perspective.
    log: deque = field(default_factory=lambda: deque(maxlen=_MAX_LOG))


class FeatureBuilder:
    """Builds leak-free features for training and inference.

    After :meth:`fit`, the instance carries all state required to score future
    matches and can be serialized with joblib.

    Attributes:
        feature_columns_: Ordered list of feature column names (set by ``fit``).
        last_date_: Date of the most recent match seen during ``fit``.
        teams_: Sorted list of all teams observed.
    """

    def __init__(self) -> None:
        self.elo = EloRatingSystem()
        self._teams: dict[str, _TeamState] = {}
        self._h2h: dict[tuple[str, str], list[tuple]] = {}
        self.feature_columns_: list[str] = []
        self.last_date_: pd.Timestamp | None = None
        self.teams_: list[str] = []

    # ------------------------------------------------------------------ #
    # State access helpers
    # ------------------------------------------------------------------ #
    def _team(self, name: str) -> _TeamState:
        state = self._teams.get(name)
        if state is None:
            state = _TeamState()
            self._teams[name] = state
        return state

    @staticmethod
    def _pair_key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    # ------------------------------------------------------------------ #
    # Feature computation (shared by training and inference)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _form_stats(log: deque, prefix: str) -> dict[str, float]:
        """Compute rolling form statistics for every configured window."""
        feats: dict[str, float] = {}
        entries = list(log)
        for window in config.FORM_WINDOWS:
            recent = entries[-window:]
            wins = draws = losses = gf = ga = 0
            for _, goals_for, goals_against, outcome in recent:
                gf += goals_for
                ga += goals_against
                if outcome == 2:
                    wins += 1
                elif outcome == 1:
                    draws += 1
                else:
                    losses += 1
            points = wins * 3 + draws
            feats[f"{prefix}_wins_last_{window}"] = float(wins)
            feats[f"{prefix}_draws_last_{window}"] = float(draws)
            feats[f"{prefix}_losses_last_{window}"] = float(losses)
            feats[f"{prefix}_goals_scored_last_{window}"] = float(gf)
            feats[f"{prefix}_goals_conceded_last_{window}"] = float(ga)
            feats[f"{prefix}_goal_difference_last_{window}"] = float(gf - ga)
            feats[f"{prefix}_points_last_{window}"] = float(points)
        return feats

    @staticmethod
    def _recency_stats(log: deque, prefix: str) -> dict[str, float]:
        """Exponentially-decayed weighted average points and goal difference."""
        entries = list(log)[-config.RECENCY_WINDOW:]
        if not entries:
            return {
                f"{prefix}_weighted_points": 0.0,
                f"{prefix}_weighted_goal_diff": 0.0,
            }
        m = len(entries)
        total_w = 0.0
        w_points = 0.0
        w_gd = 0.0
        for idx, (_, goals_for, goals_against, outcome) in enumerate(entries):
            distance = (m - 1) - idx  # 0 == most recent
            weight = _DECAY_PER_STEP ** distance
            points = 3.0 if outcome == 2 else 1.0 if outcome == 1 else 0.0
            total_w += weight
            w_points += weight * points
            w_gd += weight * (goals_for - goals_against)
        return {
            f"{prefix}_weighted_points": w_points / total_w,
            f"{prefix}_weighted_goal_diff": w_gd / total_w,
        }

    def _h2h_stats(self, home: str, away: str, date: pd.Timestamp) -> dict[str, float]:
        """Head-to-head record (last :data:`config.H2H_YEARS` years)."""
        meetings = self._h2h.get(self._pair_key(home, away), ())
        cutoff = date - _H2H_DELTA
        home_wins = draws = away_wins = 0
        goal_diff = 0
        matches = 0
        for ts, m_home, _m_away, m_hs, m_as in meetings:
            if ts < cutoff:
                continue
            matches += 1
            if m_home == home:
                home_goals, away_goals = m_hs, m_as
            else:  # the stored home team was the current away team
                home_goals, away_goals = m_as, m_hs
            goal_diff += home_goals - away_goals
            if home_goals > away_goals:
                home_wins += 1
            elif home_goals == away_goals:
                draws += 1
            else:
                away_wins += 1
        return {
            "h2h_matches": float(matches),
            "h2h_home_wins": float(home_wins),
            "h2h_draws": float(draws),
            "h2h_away_wins": float(away_wins),
            "h2h_goal_difference": float(goal_diff),
        }

    def _compute_features(
        self,
        home: str,
        away: str,
        date: pd.Timestamp,
        neutral: bool,
        category: str,
        importance: float,
        month: int,
        day_of_week: int,
    ) -> dict[str, float]:
        """Assemble the full feature dict from the current state (no update)."""
        home_elo, away_elo = self.elo.pre_match(home, away)

        feats: dict[str, float] = {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": home_elo - away_elo,
            "elo_expected_home": self.elo.expected_home(home, away, neutral),
        }

        home_form = self._form_stats(self._team(home).log, "home")
        away_form = self._form_stats(self._team(away).log, "away")
        feats.update(home_form)
        feats.update(away_form)

        feats.update(self._recency_stats(self._team(home).log, "home"))
        feats.update(self._recency_stats(self._team(away).log, "away"))

        feats.update(self._h2h_stats(home, away, date))

        # Explicit difference features (help linear models & readability).
        feats["form_points_diff_10"] = (
            home_form["home_points_last_10"] - away_form["away_points_last_10"]
        )
        feats["form_goal_diff_diff_10"] = (
            home_form["home_goal_difference_last_10"]
            - away_form["away_goal_difference_last_10"]
        )
        feats["weighted_points_diff"] = (
            feats["home_weighted_points"] - feats["away_weighted_points"]
        )

        # Context features.
        feats["neutral"] = 1.0 if neutral else 0.0
        feats["tournament_importance"] = float(importance)
        for cat in TOURNAMENT_CATEGORIES:
            feats[f"tournament_is_{cat.lower().replace(' ', '_')}"] = (
                1.0 if cat == category else 0.0
            )
        feats["month"] = float(month)
        feats["day_of_week"] = float(day_of_week)
        return feats

    # ------------------------------------------------------------------ #
    # State update
    # ------------------------------------------------------------------ #
    def _update_state(
        self,
        home: str,
        away: str,
        date: pd.Timestamp,
        home_score: int,
        away_score: int,
        importance: float,
        neutral: bool,
    ) -> None:
        """Apply a match result to the Elo, rolling logs and h2h state."""
        if home_score > away_score:
            home_outcome, away_outcome = 2, 0
        elif home_score == away_score:
            home_outcome, away_outcome = 1, 1
        else:
            home_outcome, away_outcome = 0, 2

        self._team(home).log.append((date, home_score, away_score, home_outcome))
        self._team(away).log.append((date, away_score, home_score, away_outcome))
        self._h2h.setdefault(self._pair_key(home, away), []).append(
            (date, home, away, home_score, away_score)
        )
        self.elo.update(home, away, home_score, away_score, importance, neutral)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit_transform(
        self,
        df: pd.DataFrame,
        *,
        use_cache: bool = True,
        cache_path: Path | str = config.FEATURES_PARQUET,
        mirror_neutral: bool = True,
    ) -> pd.DataFrame:
        """Fit the builder on the full history and return the feature matrix.

        Args:
            df: Preprocessed match frame (chronologically sorted).
            use_cache: Reuse a cached feature matrix if present *and* refit the
                builder state (state is cheap and always recomputed).
            cache_path: Parquet cache location for the feature matrix.
            mirror_neutral: When True, every neutral match is emitted twice -
                once as recorded and once with the teams (and score) swapped and
                the label flipped. This removes the home/away ordering leak that
                exists for neutral matches in the source data (where the winner
                tends to be listed as the "home" team), so the models learn
                symmetric, strength-based behaviour for neutral fixtures. The
                synthetic mirror rows never affect the Elo / form / h2h state.

        Returns:
            A DataFrame with the engineered features plus the meta columns
            ``date``, ``year``, ``result``, ``home_team``, ``away_team``,
            ``home_score``, ``away_score`` and ``is_mirror``.
        """
        cache_path = Path(cache_path)
        # State is always (re)built because it is required for inference and is
        # cheap; the expensive-to-recompute matrix itself is what we cache.
        rows: list[dict[str, float]] = []
        meta: list[dict] = []

        homes = df["home_team"].to_numpy()
        aways = df["away_team"].to_numpy()
        hs = df["home_score"].to_numpy()
        as_ = df["away_score"].to_numpy()
        results = df["result"].to_numpy()
        weights = df["tournament_importance"].to_numpy()
        neutrals = df["neutral"].to_numpy()
        cats = df["tournament_category"].astype(str).to_numpy()
        months = df["month"].to_numpy()
        dows = df["day_of_week"].to_numpy()
        years = df["year"].to_numpy()
        dates_ts = df["date"].reset_index(drop=True)

        logger.info("Building features for %d matches (mirror_neutral=%s)", len(df), mirror_neutral)
        for i in range(len(df)):
            date = dates_ts.iloc[i]
            neutral = bool(neutrals[i])
            weight = float(weights[i])
            month = int(months[i])
            dow = int(dows[i])

            feats = self._compute_features(homes[i], aways[i], date, neutral, cats[i], weight, month, dow)
            rows.append(feats)
            meta.append({
                "date": date, "year": int(years[i]), "result": int(results[i]),
                "home_team": homes[i], "away_team": aways[i],
                "home_score": int(hs[i]), "away_score": int(as_[i]), "is_mirror": 0,
            })

            if mirror_neutral and neutral:
                # Mirror computed from the SAME pre-update state (no leakage).
                feats_m = self._compute_features(aways[i], homes[i], date, neutral, cats[i], weight, month, dow)
                rows.append(feats_m)
                meta.append({
                    "date": date, "year": int(years[i]), "result": 2 - int(results[i]),
                    "home_team": aways[i], "away_team": homes[i],
                    "home_score": int(as_[i]), "away_score": int(hs[i]), "is_mirror": 1,
                })

            self._update_state(homes[i], aways[i], date, int(hs[i]), int(as_[i]), weight, neutral)
            if (i + 1) % 10000 == 0:
                logger.info("  ... %d / %d matches", i + 1, len(df))

        features = pd.DataFrame(rows)
        self.feature_columns_ = list(features.columns)

        # Attach meta columns used downstream for splitting / Poisson targets.
        meta_df = pd.DataFrame(meta)
        for col in meta_df.columns:
            features[col] = meta_df[col].to_numpy()

        self.last_date_ = pd.Timestamp(df["date"].max())
        self.teams_ = sorted(set(homes.tolist()) | set(aways.tolist()))
        logger.info("Feature matrix: %d rows (%d original + %d mirrored neutral)",
                    len(features), len(df), len(features) - len(df))

        if use_cache:
            features.to_parquet(cache_path, index=False)
            logger.info("Cached feature matrix to %s", cache_path)
        return features

    def transform_one(
        self,
        home: str,
        away: str,
        date: pd.Timestamp,
        neutral: bool,
        tournament_category: str,
        importance: float | None = None,
    ) -> pd.DataFrame:
        """Build a single-row feature frame for a (future) match.

        Args:
            home: Home team (already normalized to current name).
            away: Away team (already normalized to current name).
            date: Match date.
            neutral: Whether the match is on neutral ground.
            tournament_category: One of :data:`TOURNAMENT_CATEGORIES`.
            importance: Optional override; defaults to the category weight.

        Returns:
            A one-row DataFrame with exactly ``feature_columns_`` in order.
        """
        if not self.feature_columns_:
            raise RuntimeError("FeatureBuilder must be fitted before transform_one().")
        date = pd.Timestamp(date)
        weight = (
            importance
            if importance is not None
            else tournament_importance_weight(tournament_category)
        )
        feats = self._compute_features(
            home, away, date, neutral, tournament_category, float(weight),
            int(date.month), int(date.dayofweek),
        )
        return pd.DataFrame([feats]).reindex(columns=self.feature_columns_, fill_value=0.0)

    # ------------------------------------------------------------------ #
    # Convenience accessors (for the UI / CLI)
    # ------------------------------------------------------------------ #
    def team_elo(self, team: str) -> float:
        """Current Elo rating of a team."""
        return self.elo.rating(team)

    def recent_form(self, team: str, window: int = 5) -> dict[str, float]:
        """Latest rolling-form summary for a team (for display)."""
        prefix = "team"
        stats = self._form_stats(self._team(team).log, prefix)
        return {k.replace(f"{prefix}_", ""): v for k, v in stats.items() if f"_last_{window}" in k}


def build_features(
    df: pd.DataFrame,
    *,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, FeatureBuilder]:
    """Convenience wrapper returning ``(feature_matrix, fitted_builder)``."""
    builder = FeatureBuilder()
    features = builder.fit_transform(df, use_cache=use_cache)
    return features, builder


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from .preprocess import preprocess

    frame, _ = preprocess()
    feats, builder = build_features(frame, use_cache=False)
    logger.info("Feature matrix shape: %s", feats.shape)
    logger.info("Number of model features: %d", len(builder.feature_columns_))
    print(feats[builder.feature_columns_].describe().T.head(20))
