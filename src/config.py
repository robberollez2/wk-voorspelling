"""Central configuration: paths, constants and logging setup.

Every other module imports its paths and tunable constants from here so the
project has a single source of truth.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SRC_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SRC_DIR.parent

DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"

RAW_MATCHES_CSV: Path = DATA_DIR / "all_matches.csv"
COUNTRIES_CSV: Path = DATA_DIR / "countries_names.csv"

# Cached / generated artifacts -------------------------------------------------
PREPROCESSED_PARQUET: Path = MODELS_DIR / "preprocessed.parquet"
FEATURES_PARQUET: Path = MODELS_DIR / "feature_matrix.parquet"

FEATURE_BUILDER_PKL: Path = MODELS_DIR / "feature_builder.joblib"
FEATURE_LIST_PKL: Path = MODELS_DIR / "feature_list.joblib"
NAME_MAP_PKL: Path = MODELS_DIR / "name_map.joblib"

XGB_MODEL_PKL: Path = MODELS_DIR / "xgb_model.joblib"
LGB_MODEL_PKL: Path = MODELS_DIR / "lgb_model.joblib"
POISSON_HOME_PKL: Path = MODELS_DIR / "poisson_home.joblib"
POISSON_AWAY_PKL: Path = MODELS_DIR / "poisson_away.joblib"
ENSEMBLE_WEIGHTS_PKL: Path = MODELS_DIR / "ensemble_weights.joblib"

METRICS_JSON: Path = MODELS_DIR / "metrics.json"
SHAP_SUMMARY_PNG: Path = MODELS_DIR / "shap_summary.png"
SHAP_VALUES_PKL: Path = MODELS_DIR / "shap_summary.joblib"

# --------------------------------------------------------------------------- #
# Modelling constants
# --------------------------------------------------------------------------- #
# Outcome label encoding (kept stable everywhere in the project).
LABEL_AWAY_WIN: int = 0
LABEL_DRAW: int = 1
LABEL_HOME_WIN: int = 2
CLASS_LABELS: tuple[int, int, int] = (LABEL_AWAY_WIN, LABEL_DRAW, LABEL_HOME_WIN)
CLASS_NAMES: tuple[str, str, str] = ("Away Win", "Draw", "Home Win")

# Rolling-form windows (number of previous matches per team).
FORM_WINDOWS: tuple[int, ...] = (5, 10, 20)

# Recency weighting (exponential decay over the last N matches).
RECENCY_WINDOW: int = 20
RECENCY_HALFLIFE: float = 5.0  # in matches; weight halves every 5 games back

# Head-to-head look-back horizon.
H2H_YEARS: int = 10

# Elo system.
ELO_START: float = 1500.0
ELO_HOME_ADVANTAGE: float = 65.0  # added to home rating when match is not neutral
ELO_BASE_K: float = 32.0

# Scoreline modelling.
SCORE_MATRIX_MAX_GOALS: int = 10   # internal grid size for probability mass
SCORE_REPORT_MAX_GOALS: int = 6    # 0-0 .. 6-6 reported to the user
MIN_EXPECTED_GOALS: float = 0.05
MAX_EXPECTED_GOALS: float = 6.0

# Time-based data splits (inclusive year ranges).
TRAIN_END_YEAR: int = 2020
VAL_START_YEAR: int = 2021
VAL_END_YEAR: int = 2023
TEST_START_YEAR: int = 2024

# Optuna.
DEFAULT_OPTUNA_TRIALS: int = 100
RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger configured with a single stdout handler.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level for the logger.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def ensure_dirs() -> None:
    """Create the data and models directories if they do not yet exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
