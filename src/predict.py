"""Inference: load the trained stack and predict a single (future) match.

Provides both a reusable :class:`Predictor` (used by the Streamlit app) and an
interactive command-line interface::

    python -m src.predict
    python predict.py --home Belgium --away Netherlands --country Belgium \\
        --tournament Friendly --date 2026-09-10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .config import get_logger
from .dixon_coles import DixonColesModel
from .features import FeatureBuilder
from .poisson import most_likely_score, top_scorelines
from .preprocess import (
    TOURNAMENT_CATEGORIES,
    NameMap,
    categorize_tournament,
    tournament_importance_weight,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class PredictionResult:
    """Structured prediction output."""

    home_team: str
    away_team: str
    country: str
    tournament: str
    tournament_category: str
    date: str
    neutral: bool

    p_home_win: float
    p_draw: float
    p_away_win: float

    expected_home_goals: float
    expected_away_goals: float

    most_likely_score: tuple[int, int]
    most_likely_prob: float
    top_scorelines: list[tuple[tuple[int, int], float]]

    home_elo: float
    away_elo: float

    score_matrix: np.ndarray = field(repr=False)
    model_probs: dict[str, tuple[float, float, float]] = field(default_factory=dict, repr=False)

    def format_text(self) -> str:
        """Render a human-readable summary (matches the project spec layout)."""
        lines = [
            f"{self.home_team} vs {self.away_team}",
            f"({self.tournament} · {self.date} · "
            f"{'neutral venue' if self.neutral else self.country})",
            "",
            f"Home Win: {self.p_home_win * 100:5.1f}%",
            f"Draw:     {self.p_draw * 100:5.1f}%",
            f"Away Win: {self.p_away_win * 100:5.1f}%",
            "",
            "Expected Goals:",
            f"  {self.home_team}: {self.expected_home_goals:.2f}",
            f"  {self.away_team}: {self.expected_away_goals:.2f}",
            "",
            f"Most likely score:",
            f"  {self.most_likely_score[0]}-{self.most_likely_score[1]} "
            f"({self.most_likely_prob * 100:.1f}%)",
            "",
            "Top 5 outcomes:",
        ]
        for (h, a), prob in self.top_scorelines:
            lines.append(f"  {h}-{a}  ({prob * 100:4.1f}%)")
        lines += [
            "",
            f"Elo: {self.home_team} {self.home_elo:.0f}  |  {self.away_team} {self.away_elo:.0f}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
class Predictor:
    """Loads trained artifacts and scores hypothetical matches."""

    def __init__(self, models_dir: Path | str = config.MODELS_DIR) -> None:
        models_dir = Path(models_dir)
        self._require(config.FEATURE_BUILDER_PKL)
        self.builder: FeatureBuilder = joblib.load(config.FEATURE_BUILDER_PKL)
        self.feature_columns: list[str] = joblib.load(config.FEATURE_LIST_PKL)
        self.name_map: NameMap = joblib.load(config.NAME_MAP_PKL)
        self.xgb_model = joblib.load(config.XGB_MODEL_PKL)
        self.lgb_model = joblib.load(config.LGB_MODEL_PKL)
        self.dc: DixonColesModel = joblib.load(config.DIXON_COLES_PKL)
        self.stacker = joblib.load(config.STACKER_PKL)
        self.stack_order: list[str] = joblib.load(config.ENSEMBLE_WEIGHTS_PKL)
        logger.info("Predictor loaded (%d teams, last data %s)",
                    len(self.builder.teams_), self.builder.last_date_)

    @staticmethod
    def _require(path: Path) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Missing model artifact: {path}. Run `python -m src.train` first."
            )

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    @property
    def teams(self) -> list[str]:
        """Sorted list of known teams (current names)."""
        return self.builder.teams_

    @property
    def tournaments(self) -> list[str]:
        """Selectable tournament categories."""
        return list(TOURNAMENT_CATEGORIES)

    def normalize_team(self, name: str) -> str:
        """Normalize a (possibly historical) team name to its current form."""
        return self.name_map.normalize(name)

    def resolve_category(self, tournament: str) -> str:
        """Resolve a raw tournament string or category label to a category."""
        if tournament in TOURNAMENT_CATEGORIES:
            return tournament
        return categorize_tournament(tournament)

    @staticmethod
    def detect_neutral(home_team: str, country: str) -> bool:
        """Auto-detect the neutral flag: neutral unless played in the home country."""
        return home_team.strip().lower() != country.strip().lower()

    # ------------------------------------------------------------------ #
    # Core prediction
    # ------------------------------------------------------------------ #
    def _raw_scores(
        self,
        home: str,
        away: str,
        date: pd.Timestamp,
        neutral: bool,
        category: str,
        importance: float,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], float, float]:
        """Score one ordering with every model + the stacked ensemble.

        Returns ``(stacked_probs, model_probs, lam_home, lam_away)`` where each
        probability vector is ordered ``[away, draw, home]``.
        """
        X = self.builder.transform_one(home, away, date, neutral, category, importance)
        X = X[self.feature_columns]
        model_probs = {
            "xgboost": np.asarray(self.xgb_model.predict_proba(X)[0], dtype=float),
            "lightgbm": np.asarray(self.lgb_model.predict_proba(X)[0], dtype=float),
            "dixon_coles": self.dc.outcome_probabilities(home, away, neutral),
        }
        stacked = self._stack(model_probs)
        lam_home, lam_away = self.dc.predict_expected(home, away, neutral)
        return stacked, model_probs, float(lam_home), float(lam_away)

    def _stack(self, model_probs: dict[str, np.ndarray]) -> np.ndarray:
        """Combine per-model probabilities with the logistic stacker."""
        feats = np.hstack([
            np.log(np.clip(model_probs[name], 1e-6, 1.0)) for name in self.stack_order
        ]).reshape(1, -1)
        return np.asarray(self.stacker.predict_proba(feats)[0], dtype=float)

    @staticmethod
    def _symmetrize(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Average a fixture's probabilities with its swapped-ordering version.

        ``p1`` is ``[away, draw, home]`` for ``(home, away)`` and ``p2`` for the
        reversed ``(away, home)``; the result is again ``[away, draw, home]``.
        """
        p_away = (p1[0] + p2[2]) / 2.0
        p_draw = (p1[1] + p2[1]) / 2.0
        p_home = (p1[2] + p2[0]) / 2.0
        out = np.array([p_away, p_draw, p_home], dtype=float)
        total = out.sum()
        return out / total if total > 0 else out

    def predict(
        self,
        home_team: str,
        away_team: str,
        country: str,
        tournament: str,
        date: str,
    ) -> PredictionResult:
        """Predict the outcome of a single match.

        For neutral matches the prediction is symmetrized over both team
        orderings so the result never depends on which team the user happened to
        type first (the source data lists the winner as "home" for neutral
        games, so a single ordering would be biased).

        Args:
            home_team: Home team name (historical names are normalized).
            away_team: Away team name.
            country: Country where the match is played (drives neutral flag).
            tournament: Tournament category or raw tournament label.
            date: Match date as ``YYYY-MM-DD``.

        Returns:
            A populated :class:`PredictionResult`.
        """
        home = self.normalize_team(home_team)
        away = self.normalize_team(away_team)
        host = self.normalize_team(country)
        category = self.resolve_category(tournament)
        neutral = self.detect_neutral(home, host)
        match_date = pd.Timestamp(date)
        importance = tournament_importance_weight(category)

        if home not in self.builder.teams_:
            logger.warning("Home team '%s' unseen in training data; using neutral priors.", home)
        if away not in self.builder.teams_:
            logger.warning("Away team '%s' unseen in training data; using neutral priors.", away)

        stacked1, mprobs1, lam_home, lam_away = self._raw_scores(
            home, away, match_date, neutral, category, importance)
        if neutral:
            # Average both orderings -> exactly order-invariant neutral prediction.
            stacked2, mprobs2, lam_h2, lam_a2 = self._raw_scores(
                away, home, match_date, neutral, category, importance)
            stacked = self._symmetrize(stacked1, stacked2)
            model_probs = {n: self._symmetrize(mprobs1[n], mprobs2[n]) for n in mprobs1}
            lam_home = (lam_home + lam_a2) / 2.0
            lam_away = (lam_away + lam_h2) / 2.0
        else:
            stacked = stacked1
            model_probs = mprobs1

        p_away, p_draw, p_home = float(stacked[0]), float(stacked[1]), float(stacked[2])

        # --- scorelines (Dixon-Coles corrected) ---------------------------- #
        matrix = self.dc.matrix_from_lambdas(lam_home, lam_away)
        (ml_h, ml_a), ml_prob = most_likely_score(matrix)
        tops = top_scorelines(matrix, top_n=5)

        return PredictionResult(
            home_team=home,
            away_team=away,
            country=host,
            tournament=tournament,
            tournament_category=category,
            date=str(match_date.date()),
            neutral=neutral,
            p_home_win=p_home,
            p_draw=p_draw,
            p_away_win=p_away,
            expected_home_goals=lam_home,
            expected_away_goals=lam_away,
            most_likely_score=(ml_h, ml_a),
            most_likely_prob=ml_prob,
            top_scorelines=tops,
            home_elo=self.builder.team_elo(home),
            away_elo=self.builder.team_elo(away),
            score_matrix=matrix,
            model_probs={
                name: (float(p[2]), float(p[1]), float(p[0])) for name, p in model_probs.items()
            },
        )


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #
def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or (default or "")


def _run_once(predictor: Predictor, args: argparse.Namespace) -> None:
    result = predictor.predict(args.home, args.away, args.country, args.tournament, args.date)
    print("\n" + result.format_text() + "\n")


def main() -> None:
    """Interactive / one-shot CLI entry point."""
    parser = argparse.ArgumentParser(description="Predict an international football match.")
    parser.add_argument("--home", help="Home team")
    parser.add_argument("--away", help="Away team")
    parser.add_argument("--country", help="Country where the match is played")
    parser.add_argument("--tournament", default="Friendly", help="Tournament (category or raw name)")
    parser.add_argument("--date", help="Match date YYYY-MM-DD")
    args = parser.parse_args()

    try:
        predictor = Predictor()
    except FileNotFoundError as exc:
        print(f"\n[error] {exc}\n")
        raise SystemExit(1)

    # One-shot mode if the essential flags are provided.
    if args.home and args.away and args.country and args.date:
        _run_once(predictor, args)
        return

    print("\n=== Football Match Predictor ===")
    print("(press Ctrl+C to quit)\n")
    while True:
        try:
            args.home = _prompt("Home Team", "Belgium")
            args.away = _prompt("Away Team", "Netherlands")
            args.country = _prompt("Country", args.home)
            args.tournament = _prompt("Tournament", "Friendly")
            args.date = _prompt("Date (YYYY-MM-DD)", "2026-09-10")
            _run_once(predictor, args)
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        except Exception as exc:  # keep the REPL alive on bad input
            print(f"[error] {exc}\n")


if __name__ == "__main__":
    main()
