"""Streamlit web interface for the football match predictor.

Run with::

    streamlit run app.py

Inputs: Home Team, Away Team, Country, Tournament, Date.
Outputs: win/draw/loss probabilities, expected goals, most likely score, the
top-5 scorelines, plus win-probability, Elo and recent-form charts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402
from src.predict import Predictor, PredictionResult  # noqa: E402

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")


@st.cache_resource(show_spinner="Loading trained models...")
def load_predictor() -> Predictor:
    """Load the predictor once and cache it for the session."""
    return Predictor()


def _team_color(predictor: Predictor, team: str, fallback: str) -> str:
    return predictor.name_map.color(team, fallback)


def win_probability_chart(predictor: Predictor, result: PredictionResult) -> plt.Figure:
    """Horizontal bar chart of the three outcome probabilities."""
    home_c = _team_color(predictor, result.home_team, "#1565C0")
    away_c = _team_color(predictor, result.away_team, "#C62828")
    labels = [f"{result.home_team} win", "Draw", f"{result.away_team} win"]
    values = [result.p_home_win, result.p_draw, result.p_away_win]
    colors = [home_c, "#9E9E9E", away_c]

    fig, ax = plt.subplots(figsize=(7, 2.6))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    for bar, value in zip(bars, values):
        ax.text(min(value + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{value * 100:.1f}%", va="center", fontweight="bold")
    fig.tight_layout()
    return fig


def elo_chart(predictor: Predictor, result: PredictionResult) -> plt.Figure:
    """Bar chart comparing the two teams' current Elo ratings."""
    home_c = _team_color(predictor, result.home_team, "#1565C0")
    away_c = _team_color(predictor, result.away_team, "#C62828")
    fig, ax = plt.subplots(figsize=(7, 2.6))
    teams = [result.home_team, result.away_team]
    elos = [result.home_elo, result.away_elo]
    bars = ax.bar(teams, elos, color=[home_c, away_c])
    ax.set_ylabel("Elo rating")
    ax.set_ylim(min(elos) - 80, max(elos) + 80)
    for bar, value in zip(bars, elos):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 5, f"{value:.0f}",
                ha="center", fontweight="bold")
    fig.tight_layout()
    return fig


def form_chart(predictor: Predictor, result: PredictionResult) -> plt.Figure:
    """Grouped bars of points won over the last 5/10/20 matches."""
    home = result.home_team
    away = result.away_team
    windows = list(config.FORM_WINDOWS)
    home_form = predictor.builder._team(home).log  # noqa: SLF001 (internal accessor)
    away_form = predictor.builder._team(away).log  # noqa: SLF001

    def points(log, window: int) -> int:
        recent = list(log)[-window:]
        return sum(3 if o == 2 else 1 if o == 1 else 0 for *_, o in recent)

    home_pts = [points(home_form, w) for w in windows]
    away_pts = [points(away_form, w) for w in windows]
    max_pts = [3 * w for w in windows]
    home_rate = [p / m for p, m in zip(home_pts, max_pts)]
    away_rate = [p / m for p, m in zip(away_pts, max_pts)]

    x = np.arange(len(windows))
    width = 0.38
    home_c = _team_color(predictor, home, "#1565C0")
    away_c = _team_color(predictor, away, "#C62828")

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(x - width / 2, home_rate, width, label=home, color=home_c)
    ax.bar(x + width / 2, away_rate, width, label=away, color=away_c)
    ax.set_xticks(x, [f"Last {w}" for w in windows])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Points won (rate)")
    ax.legend()
    fig.tight_layout()
    return fig


def score_heatmap(result: PredictionResult, max_goals: int = config.SCORE_REPORT_MAX_GOALS) -> plt.Figure:
    """Heatmap of scoreline probabilities (0-0 .. max-max)."""
    sub = result.score_matrix[: max_goals + 1, : max_goals + 1]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sub, cmap="Greens", origin="upper")
    ax.set_xticks(range(max_goals + 1))
    ax.set_yticks(range(max_goals + 1))
    ax.set_xlabel(f"{result.away_team} goals")
    ax.set_ylabel(f"{result.home_team} goals")
    best = result.most_likely_score
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            ax.text(j, i, f"{sub[i, j] * 100:.0f}", ha="center", va="center",
                    fontsize=7, color="black" if sub[i, j] < sub.max() * 0.6 else "white")
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, edgecolor="#C62828", lw=2))
    fig.colorbar(im, ax=ax, label="Probability (%)")
    fig.tight_layout()
    return fig


def main() -> None:
    """Render the Streamlit page."""
    st.title("⚽ International Football Match Predictor")
    st.caption(
        "Ensemble of XGBoost + LightGBM + Poisson with Elo, rolling form and "
        "head-to-head features. The neutral-venue flag is detected automatically."
    )

    try:
        predictor = load_predictor()
    except FileNotFoundError as exc:
        st.error(f"{exc}")
        st.info("Train the models first:  `python -m src.train`")
        st.stop()

    teams = predictor.teams
    default_home = teams.index("Belgium") if "Belgium" in teams else 0
    default_away = teams.index("Netherlands") if "Netherlands" in teams else 1

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            home = st.selectbox("Home Team", teams, index=default_home)
            country = st.selectbox(
                "Country (where it is played)", teams,
                index=teams.index(home) if home in teams else 0,
            )
            date = st.date_input("Date", value=pd.Timestamp("2026-09-10"))
        with c2:
            away = st.selectbox("Away Team", teams, index=default_away)
            tournament = st.selectbox("Tournament", predictor.tournaments)
        submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

    if not submitted:
        st.stop()

    if home == away:
        st.warning("Home and away team must differ.")
        st.stop()

    result = predictor.predict(home, away, country, tournament, str(date))

    venue = "Neutral venue" if result.neutral else f"Home advantage ({result.country})"
    st.subheader(f"{result.home_team} vs {result.away_team}")
    st.caption(f"{result.tournament} · {result.date} · {venue}")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{result.home_team} win", f"{result.p_home_win * 100:.1f}%")
    m2.metric("Draw", f"{result.p_draw * 100:.1f}%")
    m3.metric(f"{result.away_team} win", f"{result.p_away_win * 100:.1f}%")

    g1, g2, g3 = st.columns(3)
    g1.metric(f"Expected goals · {result.home_team}", f"{result.expected_home_goals:.2f}")
    g2.metric(f"Expected goals · {result.away_team}", f"{result.expected_away_goals:.2f}")
    g3.metric("Most likely score",
              f"{result.most_likely_score[0]}-{result.most_likely_score[1]}",
              f"{result.most_likely_prob * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("**Win probability**")
        st.pyplot(win_probability_chart(predictor, result))
        st.markdown("**Recent form (points rate)**")
        st.pyplot(form_chart(predictor, result))
    with right:
        st.markdown("**Elo comparison**")
        st.pyplot(elo_chart(predictor, result))
        st.markdown("**Top 5 scorelines**")
        top_df = pd.DataFrame(
            [{"Score": f"{h}-{a}", "Probability": f"{p * 100:.1f}%"}
             for (h, a), p in result.top_scorelines]
        )
        st.table(top_df)

    st.markdown("**Scoreline probability matrix** (home rows × away columns, %)")
    st.pyplot(score_heatmap(result))

    with st.expander("Model transparency"):
        st.write("Ensemble weights:", {k: round(v, 3) for k, v in predictor.weights.items()})
        rows = []
        for name, (ph, pd_, pa) in result.model_probs.items():
            rows.append({"model": name, "home": f"{ph*100:.1f}%", "draw": f"{pd_*100:.1f}%", "away": f"{pa*100:.1f}%"})
        rows.append({"model": "ensemble",
                     "home": f"{result.p_home_win*100:.1f}%",
                     "draw": f"{result.p_draw*100:.1f}%",
                     "away": f"{result.p_away_win*100:.1f}%"})
        st.table(pd.DataFrame(rows))
        if config.SHAP_SUMMARY_PNG.exists():
            st.markdown("**Global feature importance (SHAP)**")
            st.image(str(config.SHAP_SUMMARY_PNG))


if __name__ == "__main__":
    main()
