"""StatsBomb open-data player layer: squad-strength features.

FBref blocks automated access, so the only feasible free player-level source in
this environment is StatsBomb's open-data repository, which covers a selection
of major men's national-team tournaments (World Cups, Euro 2020/2024, Copa
America 2024, AFCON 2023).

For each covered match we read the starting line-ups and maintain a per-player
result rating (a player-level Elo, updated by the match outcome against the
opponent's squad strength). A team's **squad strength** for a match is the mean
rating of its starting XI *before* that match (leak-free). These strengths are
exposed as features keyed by ``(date, {teamA, teamB})`` and merged into the main
feature matrix; matches without StatsBomb coverage (≈97% of the data, and all
future fixtures unless a line-up is supplied) fall back to a neutral default
plus a ``squad_data_available = 0`` flag.

The whole thing is cached to parquet so the network fetch only happens once.
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from . import config
from .config import get_logger
from .preprocess import NameMap, load_name_map

logger = get_logger(__name__)

_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
SQUAD_CACHE: Path = config.MODELS_DIR / "squad_strength.parquet"

# Covered men's national-team competitions: (competition_id, season_id, label).
COVERED_COMPETITIONS: tuple[tuple[int, int, str], ...] = (
    (43, 3, "World Cup 2018"),
    (43, 106, "World Cup 2022"),
    (55, 43, "Euro 2020"),
    (55, 282, "Euro 2024"),
    (223, 282, "Copa America 2024"),
    (1267, 107, "AFCON 2023"),
)

_PLAYER_ELO_START = 1500.0
_PLAYER_ELO_K = 24.0


def _fetch_json(url: str, timeout: int = 30) -> object | None:
    """Fetch and parse a JSON document, returning None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "football-ai/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as exc:  # network errors are non-fatal
        logger.debug("fetch failed for %s: %s", url, exc)
        return None


def _starters(team_block: dict) -> list[int]:
    """Return the player ids that started the match for one team."""
    starters: list[int] = []
    for player in team_block.get("lineup", []):
        for pos in player.get("positions", []):
            if str(pos.get("from", "")).startswith("00:00"):
                starters.append(player["player_id"])
                break
    return starters


def _collect_matches(name_map: NameMap) -> list[dict]:
    """Fetch the match lists for every covered competition."""
    matches: list[dict] = []
    for comp_id, season_id, label in COVERED_COMPETITIONS:
        data = _fetch_json(f"{_BASE}/matches/{comp_id}/{season_id}.json")
        if not data:
            logger.warning("No match list for %s (%d/%d)", label, comp_id, season_id)
            continue
        for m in data:
            matches.append({
                "match_id": m["match_id"],
                "date": pd.Timestamp(m["match_date"]),
                "home_team": name_map.normalize(m["home_team"]["home_team_name"]),
                "away_team": name_map.normalize(m["away_team"]["away_team_name"]),
                "home_score": int(m["home_score"]),
                "away_score": int(m["away_score"]),
            })
        logger.info("StatsBomb %s: %d matches", label, len(data))
    matches.sort(key=lambda r: r["date"])
    return matches


def _fetch_lineups(match_ids: list[int]) -> dict[int, list]:
    """Fetch all line-up files in parallel, keyed by match id."""
    results: dict[int, list] = {}

    def one(mid: int) -> tuple[int, object | None]:
        return mid, _fetch_json(f"{_BASE}/lineups/{mid}.json")

    with ThreadPoolExecutor(max_workers=8) as pool:
        for mid, data in pool.map(one, match_ids):
            if data:
                results[mid] = data
    return results


def build_squad_strength(
    name_map: NameMap | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build (and cache) leak-free per-match squad strengths from StatsBomb.

    Returns:
        DataFrame with columns ``date, team1, team2, strength1, strength2``
        where strengths are centred on 0 (mean starter player-rating minus the
        1500 baseline) and measured *before* the match. Empty if StatsBomb is
        unreachable.
    """
    if use_cache and SQUAD_CACHE.exists():
        logger.info("Loading cached squad strengths from %s", SQUAD_CACHE)
        return pd.read_parquet(SQUAD_CACHE)

    name_map = name_map or load_name_map()
    logger.info("Fetching StatsBomb open-data (covered tournaments only)...")
    matches = _collect_matches(name_map)
    if not matches:
        logger.warning("StatsBomb unreachable; squad layer will be empty.")
        empty = pd.DataFrame(columns=["date", "team1", "team2", "strength1", "strength2"])
        return empty

    lineups = _fetch_lineups([m["match_id"] for m in matches])
    logger.info("Fetched %d / %d line-ups", len(lineups), len(matches))

    ratings: dict[int, float] = {}
    rows: list[dict] = []

    for m in matches:
        lineup = lineups.get(m["match_id"])
        if not lineup or len(lineup) < 2:
            continue
        # Map StatsBomb team blocks to our normalized names.
        blocks = {name_map.normalize(b["team_name"]): b for b in lineup}
        home, away = m["home_team"], m["away_team"]
        if home not in blocks or away not in blocks:
            continue
        home_xi = _starters(blocks[home])
        away_xi = _starters(blocks[away])
        if not home_xi or not away_xi:
            continue

        home_strength = sum(ratings.get(p, _PLAYER_ELO_START) for p in home_xi) / len(home_xi)
        away_strength = sum(ratings.get(p, _PLAYER_ELO_START) for p in away_xi) / len(away_xi)

        team1, team2 = sorted((home, away))
        rows.append({
            "date": m["date"].normalize(),
            "team1": team1,
            "team2": team2,
            "strength1": (home_strength if team1 == home else away_strength) - _PLAYER_ELO_START,
            "strength2": (home_strength if team2 == home else away_strength) - _PLAYER_ELO_START,
        })

        # Update player ratings by the (shared) match result.
        expected_home = 1.0 / (1.0 + 10.0 ** ((away_strength - home_strength) / 400.0))
        if m["home_score"] > m["away_score"]:
            actual_home = 1.0
        elif m["home_score"] == m["away_score"]:
            actual_home = 0.5
        else:
            actual_home = 0.0
        delta = _PLAYER_ELO_K * (actual_home - expected_home)
        for p in home_xi:
            ratings[p] = ratings.get(p, _PLAYER_ELO_START) + delta
        for p in away_xi:
            ratings[p] = ratings.get(p, _PLAYER_ELO_START) - delta

    out = pd.DataFrame(rows)
    if use_cache and not out.empty:
        out.to_parquet(SQUAD_CACHE, index=False)
        logger.info("Cached %d squad-strength rows to %s", len(out), SQUAD_CACHE)
    return out


def squad_lookup(df: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, float]]:
    """Turn the squad-strength frame into a fast ``(date, t1, t2) -> {team: s}`` map."""
    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    for r in df.itertuples(index=False):
        key = (pd.Timestamp(r.date).strftime("%Y-%m-%d"), r.team1, r.team2)
        lookup[key] = {r.team1: float(r.strength1), r.team2: float(r.strength2)}
    return lookup


if __name__ == "__main__":  # pragma: no cover - manual fetch
    frame = build_squad_strength(use_cache=False)
    logger.info("Built squad strengths: %d matches", len(frame))
    if not frame.empty:
        print(frame.sort_values("strength1", ascending=False).head(10).to_string(index=False))
