"""Dixon-Coles style attack/defense goal model with time decay.

This is the football-modelling gold standard for goals and scorelines and a
clear upgrade over a generic-feature Poisson regression. Each team gets an
*attack* and *defense* strength; a home-advantage term applies on non-neutral
ground. Strengths are estimated by a convex, weighted Poisson regression on a
one-hot team design matrix (so it is fast and always converges), with an
exponential **time decay** that down-weights old matches. The classic
Dixon-Coles low-score correction ``rho`` is then fitted to better capture
0-0/1-0/0-1/1-1 results (and therefore draws).

Expected goals come straight from the fitted rates; the corrected scoreline
matrix yields the most-likely score, top-N scorelines and an independent
home/draw/away estimate used in the ensemble.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from . import config
from .config import get_logger

logger = get_logger(__name__)

# Default daily decay rate -> ~2.5 year half-life (suited to the sparse
# international calendar). half_life_days = ln(2) / xi.
DEFAULT_XI: float = 0.00076


def _tau(home_goals: np.ndarray, away_goals: np.ndarray, lam: np.ndarray,
         mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles low-score correction factor (vectorized)."""
    tau = np.ones_like(lam, dtype=float)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


class DixonColesModel:
    """Time-weighted attack/defense Poisson goal model."""

    def __init__(self, xi: float = DEFAULT_XI, alpha: float = 1e-3) -> None:
        self.xi = float(xi)
        self.alpha = float(alpha)
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.intercept: float = 0.0
        self.home_adv: float = 0.0
        self.rho: float = 0.0
        self.teams_: list[str] = []

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> "DixonColesModel":
        """Fit attack/defense strengths, home advantage and rho.

        Args:
            df: Matches with ``date, home_team, away_team, home_score,
                away_score, neutral`` (chronological order not required).
            reference_date: Date that decay weights are measured back from
                (defaults to the latest match date).
        """
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)
        self.teams_ = teams

        ref = pd.Timestamp(reference_date) if reference_date is not None else df["date"].max()
        age_days = (ref - df["date"]).dt.days.to_numpy().clip(min=0)
        w = np.exp(-self.xi * age_days)

        home = df["home_team"].map(idx).to_numpy()
        away = df["away_team"].map(idx).to_numpy()
        hs = df["home_score"].to_numpy(dtype=float)
        as_ = df["away_score"].to_numpy(dtype=float)
        not_neutral = (~df["neutral"].to_numpy(dtype=bool)).astype(float)
        n = len(df)

        # Two observations per match: (home scoring) and (away scoring).
        # Columns: [attack_0..attack_{T-1}, defense_0..defense_{T-1}, home_field]
        n_cols = 2 * n_teams + 1
        rows = np.arange(2 * n)
        # home-scoring obs (rows 0..n-1): attack[home], defense[away], home_field
        # away-scoring obs (rows n..2n-1): attack[away], defense[home], 0
        data, col, row = [], [], []

        def add(r, c):
            row.append(r); col.append(c); data.append(1.0)

        # attack terms
        col_attack_home = home
        col_attack_away = away
        col_def_away = n_teams + away
        col_def_home = n_teams + home
        # Build sparse triples
        r_home = np.arange(n)
        r_away = np.arange(n, 2 * n)
        # attack
        row_idx = np.concatenate([r_home, r_away])
        col_idx = np.concatenate([col_attack_home, col_attack_away])
        # defense
        row_idx = np.concatenate([row_idx, r_home, r_away])
        col_idx = np.concatenate([col_idx, col_def_away, col_def_home])
        vals = np.ones(len(row_idx))
        # home field indicator (only for home-scoring obs, value = not_neutral)
        row_idx = np.concatenate([row_idx, r_home])
        col_idx = np.concatenate([col_idx, np.full(n, 2 * n_teams)])
        vals = np.concatenate([vals, not_neutral])

        X = sparse.csr_matrix((vals, (row_idx, col_idx)), shape=(2 * n, n_cols))
        y = np.concatenate([hs, as_])
        sw = np.concatenate([w, w])

        model = PoissonRegressor(alpha=self.alpha, fit_intercept=True, max_iter=600, tol=1e-7)
        model.fit(X, y, sample_weight=sw)

        coef = model.coef_
        self.attack = {t: float(coef[idx[t]]) for t in teams}
        self.defense = {t: float(coef[n_teams + idx[t]]) for t in teams}
        self.home_adv = float(coef[2 * n_teams])
        self.intercept = float(model.intercept_)

        self._fit_rho(home, away, hs, as_, not_neutral, w)
        logger.info("Dixon-Coles fitted: %d teams, home_adv=%.3f, rho=%.3f",
                    n_teams, self.home_adv, self.rho)
        return self

    def _fit_rho(self, home, away, hs, as_, not_neutral, w) -> None:
        """Estimate the low-score correction rho by weighted MLE."""
        attack = np.array([self.attack[t] for t in self.teams_])
        defense = np.array([self.defense[t] for t in self.teams_])
        log_lh = self.intercept + attack[home] + defense[away] + self.home_adv * not_neutral
        log_la = self.intercept + attack[away] + defense[home]
        lam = np.clip(np.exp(log_lh), config.MIN_EXPECTED_GOALS, config.MAX_EXPECTED_GOALS)
        mu = np.clip(np.exp(log_la), config.MIN_EXPECTED_GOALS, config.MAX_EXPECTED_GOALS)

        def neg_ll(rho: float) -> float:
            tau = _tau(hs, as_, lam, mu, rho)
            tau = np.clip(tau, 1e-6, None)
            return -np.sum(w * np.log(tau))

        res = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
        self.rho = float(res.x) if res.success else 0.0

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_expected(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        """Expected ``(home_goals, away_goals)`` for a fixture."""
        a_home = self.attack.get(home, 0.0)
        a_away = self.attack.get(away, 0.0)
        d_home = self.defense.get(home, 0.0)
        d_away = self.defense.get(away, 0.0)
        log_lh = self.intercept + a_home + d_away + (0.0 if neutral else self.home_adv)
        log_la = self.intercept + a_away + d_home
        lam = float(np.clip(np.exp(log_lh), config.MIN_EXPECTED_GOALS, config.MAX_EXPECTED_GOALS))
        mu = float(np.clip(np.exp(log_la), config.MIN_EXPECTED_GOALS, config.MAX_EXPECTED_GOALS))
        return lam, mu

    def matrix_from_lambdas(self, lam: float, mu: float,
                            max_goals: int = config.SCORE_MATRIX_MAX_GOALS) -> np.ndarray:
        """Dixon-Coles-corrected scoreline matrix from explicit goal rates."""
        goals = np.arange(max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        # Apply the low-score correction to the 2x2 corner.
        matrix[0, 0] *= 1.0 - lam * mu * self.rho
        matrix[0, 1] *= 1.0 + lam * self.rho
        matrix[1, 0] *= 1.0 + mu * self.rho
        matrix[1, 1] *= 1.0 - self.rho
        matrix = np.clip(matrix, 0.0, None)
        total = matrix.sum()
        return matrix / total if total > 0 else matrix

    def scoreline_matrix(self, home: str, away: str, neutral: bool,
                         max_goals: int = config.SCORE_MATRIX_MAX_GOALS) -> np.ndarray:
        """Dixon-Coles-corrected scoreline probability matrix for a fixture."""
        lam, mu = self.predict_expected(home, away, neutral)
        return self.matrix_from_lambdas(lam, mu, max_goals)

    def outcome_probabilities(self, home: str, away: str, neutral: bool) -> np.ndarray:
        """Return ``[p_away, p_draw, p_home]`` for a fixture."""
        matrix = self.scoreline_matrix(home, away, neutral)
        p_home = np.tril(matrix, k=-1).sum()
        p_draw = np.trace(matrix)
        p_away = np.triu(matrix, k=1).sum()
        probs = np.array([p_away, p_draw, p_home], dtype=float)
        total = probs.sum()
        return probs / total if total > 0 else probs

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized-ish outcome probabilities for a frame of fixtures.

        Args:
            df: Must contain ``home_team``, ``away_team`` and ``neutral``.

        Returns:
            Array of shape ``(len(df), 3)`` ordered ``[away, draw, home]``.
        """
        homes = df["home_team"].to_numpy()
        aways = df["away_team"].to_numpy()
        neutrals = df["neutral"].to_numpy(dtype=bool)
        out = np.empty((len(df), 3), dtype=float)
        for i in range(len(df)):
            out[i] = self.outcome_probabilities(homes[i], aways[i], bool(neutrals[i]))
        return out
