"""Data loading, name normalization and preprocessing.

Responsibilities
----------------
* Load ``countries_names.csv`` and build a historical -> current name mapping
  (plus team colours used by the web app).
* Robustly load ``all_matches.csv`` (skipping malformed rows).
* Normalize every team name and the host-country name to its current name.
* Derive date features and the 3-class match-outcome target.
* Categorize tournaments into importance buckets (shared by the Elo and
  feature modules).

The cleaned frame is cached to parquet so the (cheap) preprocessing only runs
once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .config import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Tournament categorization (shared by elo.py and features.py)
# --------------------------------------------------------------------------- #
TOURNAMENT_CATEGORIES: tuple[str, ...] = (
    "Friendly",
    "Nations League",
    "World Cup Qualification",
    "Continental Qualification",
    "Continental Championship",
    "Confederations Cup",
    "World Cup",
    "Other",
)

# Relative importance weight per category. Used to scale the Elo K-factor and
# exposed as an ordinal model feature.
TOURNAMENT_IMPORTANCE: dict[str, float] = {
    "Friendly": 0.60,
    "Other": 0.70,
    "Nations League": 0.90,
    "Continental Qualification": 0.90,
    "World Cup Qualification": 1.00,
    "Confederations Cup": 1.00,
    "Continental Championship": 1.20,
    "World Cup": 1.40,
}

# Keywords identifying the *finals* of a confederation championship.
_CONTINENTAL_KEYWORDS: tuple[str, ...] = (
    "european championship",
    "african nations cup",
    "africa cup",
    "african cup of nations",
    "copa america",
    "south american championship",
    "south american champ",
    "asian cup",
    "afc asian cup",
    "concacaf championship",
    "concacaf champ",
    "concacaf ch",
    "gold cup",
    "oceania nations cup",
    "ofc nations",
)

_QUAL_RE = re.compile(r"\bq\b")


def _is_qualifier(text: str) -> bool:
    """Return True when the tournament string denotes a qualification stage."""
    return "qual" in text or _QUAL_RE.search(text) is not None


def categorize_tournament(tournament: str) -> str:
    """Map a raw tournament string onto one of :data:`TOURNAMENT_CATEGORIES`.

    The mapping is heuristic but deterministic; anything not recognised as a
    major competition falls back to ``"Other"`` (regional cups, games, etc.).

    Args:
        tournament: Raw tournament label from the dataset.

    Returns:
        One of the canonical category names.
    """
    if not isinstance(tournament, str) or not tournament.strip():
        return "Other"

    text = tournament.lower().strip()
    qual = _is_qualifier(text)

    if "confederations cup" in text:
        return "Confederations Cup"
    if "nations league" in text:
        return "Nations League"
    if "world cup" in text:
        return "World Cup Qualification" if qual else "World Cup"
    if any(keyword in text for keyword in _CONTINENTAL_KEYWORDS):
        return "Continental Qualification" if qual else "Continental Championship"
    if "friendly" in text:
        return "Friendly"
    return "Other"


def tournament_importance_weight(category: str) -> float:
    """Return the importance weight for a tournament category."""
    return TOURNAMENT_IMPORTANCE.get(category, TOURNAMENT_IMPORTANCE["Other"])


# --------------------------------------------------------------------------- #
# Name mapping
# --------------------------------------------------------------------------- #
@dataclass
class NameMap:
    """Historical -> current team-name mapping and per-team colours."""

    to_current: dict[str, str] = field(default_factory=dict)
    colors: dict[str, tuple[str, str]] = field(default_factory=dict)

    def normalize(self, name: str) -> str:
        """Normalize a single name to its current form (identity if unknown)."""
        if not isinstance(name, str):
            return name
        clean = name.strip()
        return self.to_current.get(clean, clean)

    def color(self, current_name: str, fallback: str = "#3366CC") -> str:
        """Return the primary colour for a (current) team name."""
        pair = self.colors.get(current_name)
        return pair[0] if pair else fallback

    def secondary_color(self, current_name: str, fallback: str = "#888888") -> str:
        """Return the secondary colour for a (current) team name."""
        pair = self.colors.get(current_name)
        return pair[1] if pair else fallback


def load_name_map(path: Path | str = config.COUNTRIES_CSV) -> NameMap:
    """Load the country-name mapping file into a :class:`NameMap`.

    The file has columns ``original_name, current_name, color_code,
    secondary_color_code``; colour columns may be missing for some rows.

    Args:
        path: Path to ``countries_names.csv``.

    Returns:
        A populated :class:`NameMap`.
    """
    path = Path(path)
    logger.info("Loading name map from %s", path)
    df = pd.read_csv(path, dtype=str).fillna("")

    to_current: dict[str, str] = {}
    colors: dict[str, tuple[str, str]] = {}

    for row in df.itertuples(index=False):
        original = str(row.original_name).strip()
        current = str(row.current_name).strip() or original
        if not original:
            continue
        to_current[original] = current

        primary = str(getattr(row, "color_code", "") or "").strip()
        secondary = str(getattr(row, "secondary_color_code", "") or "").strip()
        if current and primary and current not in colors:
            colors[current] = (primary, secondary or "#888888")

    # A current name should always map to itself as well.
    for current in set(to_current.values()):
        to_current.setdefault(current, current)

    logger.info("Name map: %d aliases, %d teams with colours", len(to_current), len(colors))
    return NameMap(to_current=to_current, colors=colors)


# --------------------------------------------------------------------------- #
# Raw match loading
# --------------------------------------------------------------------------- #
def load_raw_matches(path: Path | str = config.RAW_MATCHES_CSV) -> pd.DataFrame:
    """Load the raw matches CSV, skipping malformed rows.

    Args:
        path: Path to ``all_matches.csv``.

    Returns:
        A DataFrame with the raw (un-normalized) columns.
    """
    path = Path(path)
    logger.info("Loading raw matches from %s", path)
    df = pd.read_csv(
        path,
        dtype={
            "home_team": "string",
            "away_team": "string",
            "tournament": "string",
            "country": "string",
        },
        on_bad_lines="skip",
    )
    logger.info("Loaded %d raw rows", len(df))
    return df


# --------------------------------------------------------------------------- #
# Preprocessing pipeline
# --------------------------------------------------------------------------- #
def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce a (possibly string) neutral column into a clean boolean."""
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "t"})
    )


def preprocess(
    matches_path: Path | str = config.RAW_MATCHES_CSV,
    countries_path: Path | str = config.COUNTRIES_CSV,
    *,
    use_cache: bool = True,
    cache_path: Path | str = config.PREPROCESSED_PARQUET,
) -> tuple[pd.DataFrame, NameMap]:
    """Run the full preprocessing pipeline.

    Steps: load -> normalize names -> clean -> date features -> target ->
    tournament category. Result is cached to parquet.

    Args:
        matches_path: Path to the raw matches CSV.
        countries_path: Path to the country-name mapping CSV.
        use_cache: When True, reuse a cached parquet if present.
        cache_path: Where to read/write the cached preprocessed frame.

    Returns:
        ``(clean_df, name_map)`` where ``clean_df`` is sorted by date and
        contains the engineered base columns and target.
    """
    config.ensure_dirs()
    cache_path = Path(cache_path)
    name_map = load_name_map(countries_path)

    if use_cache and cache_path.exists():
        logger.info("Loading cached preprocessed data from %s", cache_path)
        df = pd.read_parquet(cache_path)
        return df, name_map

    df = load_raw_matches(matches_path)

    # --- normalize names (vectorized) -------------------------------------- #
    mapping = name_map.to_current
    for col in ("home_team", "away_team", "country"):
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].map(lambda x: mapping.get(x, x) if pd.notna(x) else x)

    # --- clean ------------------------------------------------------------- #
    before = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df = df[df["home_team"] != df["away_team"]]
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df[(df["home_score"] >= 0) & (df["away_score"] >= 0)]
    df["neutral"] = _to_bool(df["neutral"])
    df["tournament"] = df["tournament"].fillna("Friendly").astype(str)
    df = df.drop_duplicates(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    )
    logger.info("Cleaning removed %d rows (%d -> %d)", before - len(df), before, len(df))

    # --- sort chronologically (stable) ------------------------------------- #
    df = df.sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)

    # --- date features (vectorized) ---------------------------------------- #
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["day"] = df["date"].dt.day.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek.astype(int)

    # --- target (0 away win, 1 draw, 2 home win) --------------------------- #
    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] == df["away_score"],
    ]
    df["result"] = np.select(
        conditions, [config.LABEL_HOME_WIN, config.LABEL_DRAW], default=config.LABEL_AWAY_WIN
    ).astype(int)

    # --- tournament category & importance ---------------------------------- #
    df["tournament_category"] = df["tournament"].map(categorize_tournament).astype("category")
    df["tournament_importance"] = df["tournament_category"].map(tournament_importance_weight).astype(float)

    df.to_parquet(cache_path, index=False)
    logger.info("Saved preprocessed data (%d rows) to %s", len(df), cache_path)
    return df, name_map


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    frame, _ = preprocess(use_cache=False)
    logger.info("Preprocessed %d matches spanning %s..%s", len(frame), frame["date"].min().date(), frame["date"].max().date())
    print(frame[["date", "home_team", "away_team", "home_score", "away_score", "result", "tournament_category"]].head())
    print(frame["tournament_category"].value_counts())
