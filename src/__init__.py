"""Football match prediction AI for international national teams.

A production-ready pipeline that normalizes historical team names, engineers
leak-free temporal features (rolling form, Elo, head-to-head, tournament
importance), and trains an ensemble of XGBoost, LightGBM and Poisson models to
predict match outcomes, expected goals and the most likely scorelines.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
