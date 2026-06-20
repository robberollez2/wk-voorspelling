"""Training pipeline.

Trains and serializes the full prediction stack:

1. XGBoost classifier (Optuna-tuned on a time-based validation split).
2. LightGBM classifier (Optuna-tuned likewise).
3. Poisson goal model (expected home/away goals -> scoreline matrix).
4. A weighted-average ensemble of the three outcome-probability sources.

Validation strictly respects time order (no random splits): train on
1872-2020, validate on 2021-2023, test on 2024+.  A TimeSeriesSplit
cross-validation report is also produced.  Metrics (accuracy, log loss, Brier,
ROC AUC), SHAP feature importances and all model artifacts are written to
``models/``.

Run ``python -m src.train --help`` for options.
"""

from __future__ import annotations

import argparse
import json
import warnings
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb
import xgboost as xgb

from . import config
from .config import get_logger
from .features import build_features
from .poisson import PoissonGoalModel, outcome_probabilities, scoreline_matrix
from .preprocess import preprocess

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

_LABELS = list(config.CLASS_LABELS)


# --------------------------------------------------------------------------- #
# Splitting & metrics
# --------------------------------------------------------------------------- #
def make_splits(features: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Split the feature matrix into time-ordered train/val/test partitions."""
    year = features["year"].to_numpy()
    train_mask = year <= config.TRAIN_END_YEAR
    val_mask = (year >= config.VAL_START_YEAR) & (year <= config.VAL_END_YEAR)
    test_mask = year >= config.TEST_START_YEAR

    def subset(mask: np.ndarray) -> dict[str, Any]:
        part = features.loc[mask]
        return {
            "X": part[feature_columns],
            "y": part["result"].to_numpy(),
            "home_goals": part["home_score"].to_numpy(),
            "away_goals": part["away_score"].to_numpy(),
        }

    splits = {"train": subset(train_mask), "val": subset(val_mask), "test": subset(test_mask)}
    for name, part in splits.items():
        logger.info("Split %-5s: %d matches", name, len(part["y"]))
    return splits


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean multiclass Brier score (lower is better)."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Compute accuracy, log loss, Brier score and macro ROC AUC."""
    preds = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "log_loss": float(log_loss(y_true, proba, labels=_LABELS)),
        "brier_score": multiclass_brier(y_true, proba),
        "roc_auc_ovr_macro": float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=_LABELS)
        ),
    }


# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #
def _fit_xgb(params: dict[str, Any], splits: dict[str, Any]) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        early_stopping_rounds=50,
        **params,
    )
    model.fit(
        splits["train"]["X"], splits["train"]["y"],
        eval_set=[(splits["val"]["X"], splits["val"]["y"])],
        verbose=False,
    )
    return model


def tune_xgb(splits: dict[str, Any], n_trials: int) -> tuple[dict[str, Any], xgb.XGBClassifier]:
    """Optuna-tune XGBoost minimising validation log loss."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        }
        model = _fit_xgb(params, splits)
        proba = model.predict_proba(splits["val"]["X"])
        return log_loss(splits["val"]["y"], proba, labels=_LABELS)

    logger.info("Tuning XGBoost (%d trials)...", n_trials)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {"n_estimators": 3000, **study.best_params}
    logger.info("Best XGBoost val log loss: %.4f", study.best_value)
    model = _fit_xgb(best, splits)
    best["best_iteration"] = int(getattr(model, "best_iteration", best["n_estimators"]) or best["n_estimators"])
    return best, model


# --------------------------------------------------------------------------- #
# LightGBM
# --------------------------------------------------------------------------- #
def _fit_lgb(params: dict[str, Any], splits: dict[str, Any]) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        verbose=-1,
        **params,
    )
    model.fit(
        splits["train"]["X"], splits["train"]["y"],
        eval_set=[(splits["val"]["X"], splits["val"]["y"])],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def tune_lgb(splits: dict[str, Any], n_trials: int) -> tuple[dict[str, Any], lgb.LGBMClassifier]:
    """Optuna-tune LightGBM minimising validation log loss."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = _fit_lgb(params, splits)
        proba = model.predict_proba(splits["val"]["X"])
        return log_loss(splits["val"]["y"], proba, labels=_LABELS)

    logger.info("Tuning LightGBM (%d trials)...", n_trials)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {"n_estimators": 3000, "subsample_freq": 1, **study.best_params}
    logger.info("Best LightGBM val log loss: %.4f", study.best_value)
    model = _fit_lgb(best, splits)
    best["best_iteration"] = int(getattr(model, "best_iteration_", best["n_estimators"]) or best["n_estimators"])
    return best, model


# --------------------------------------------------------------------------- #
# Poisson outcome probabilities
# --------------------------------------------------------------------------- #
def poisson_outcome_probs(model: PoissonGoalModel, X: pd.DataFrame) -> np.ndarray:
    """Outcome probabilities ``[away, draw, home]`` per row from the Poisson model."""
    lam_home, lam_away = model.predict_expected(X)
    probs = np.empty((len(X), 3), dtype=float)
    for i in range(len(X)):
        matrix = scoreline_matrix(lam_home[i], lam_away[i])
        probs[i] = outcome_probabilities(matrix)
    return probs


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #
def optimize_ensemble_weights(
    probas: dict[str, np.ndarray], y_val: np.ndarray, step: float = 0.05,
    min_weight: float = 0.10,
) -> dict[str, float]:
    """Grid-search simplex weights minimising validation log loss.

    A per-model floor (``min_weight``) keeps every model contributing to the
    blend, which both honours the three-model ensemble design and regularises
    the weights against over-fitting the validation split.
    """
    names = list(probas)
    grid = np.arange(min_weight, 1.0 - min_weight + 1e-9, step)
    best_loss = np.inf
    best_w = np.array([1.0 / len(names)] * len(names))
    for w0 in grid:
        for w1 in grid:
            w2 = 1.0 - w0 - w1
            if w2 < min_weight - 1e-9 or w2 > 1.0 + 1e-9:
                continue
            weights = np.array([w0, w1, max(w2, 0.0)])
            blended = (
                weights[0] * probas[names[0]]
                + weights[1] * probas[names[1]]
                + weights[2] * probas[names[2]]
            )
            blended /= blended.sum(axis=1, keepdims=True)
            loss = log_loss(y_val, blended, labels=_LABELS)
            if loss < best_loss:
                best_loss = loss
                best_w = weights
    logger.info("Best ensemble weights %s -> val log loss %.4f", dict(zip(names, best_w.round(3))), best_loss)
    return {name: float(w) for name, w in zip(names, best_w)}


def blend(probas: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of probability matrices, renormalised per row."""
    blended = sum(weights[name] * probas[name] for name in weights)
    return blended / blended.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# TimeSeriesSplit report & SHAP
# --------------------------------------------------------------------------- #
def time_series_cv_report(
    params: dict[str, Any], X: pd.DataFrame, y: np.ndarray, n_splits: int = 5
) -> dict[str, float]:
    """Cross-validated XGBoost performance using TimeSeriesSplit."""
    logger.info("Running TimeSeriesSplit report (%d splits)...", n_splits)
    n_estimators = int(params.get("best_iteration", 300)) or 300
    fixed = {k: v for k, v in params.items() if k not in {"n_estimators", "best_iteration"}}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses, accuracies = [], []
    for train_idx, test_idx in tscv.split(X):
        model = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            tree_method="hist", n_jobs=-1, random_state=config.RANDOM_STATE,
            n_estimators=n_estimators, **fixed,
        )
        model.fit(X.iloc[train_idx], y[train_idx], verbose=False)
        proba = model.predict_proba(X.iloc[test_idx])
        losses.append(log_loss(y[test_idx], proba, labels=_LABELS))
        accuracies.append(accuracy_score(y[test_idx], proba.argmax(axis=1)))
    return {
        "cv_log_loss_mean": float(np.mean(losses)),
        "cv_log_loss_std": float(np.std(losses)),
        "cv_accuracy_mean": float(np.mean(accuracies)),
        "cv_accuracy_std": float(np.std(accuracies)),
    }


def compute_shap(model: xgb.XGBClassifier, X: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    """Compute mean |SHAP| importance per feature and save a summary plot."""
    try:
        import shap  # local import keeps base pipeline importable without shap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample = X.sample(min(1000, len(X)), random_state=config.RANDOM_STATE)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
        # `values` may be a list (one array per class) or a 3-D array; in either
        # case collapse every axis except the feature axis into mean |SHAP|.
        arr = np.abs(np.asarray(values))
        n_features = len(feature_columns)
        feat_axis = next(ax for ax, size in enumerate(arr.shape) if size == n_features)
        mean_abs = arr.mean(axis=tuple(a for a in range(arr.ndim) if a != feat_axis))
        mean_abs = np.asarray(mean_abs).ravel()[:n_features]
        importance = dict(sorted(zip(feature_columns, mean_abs.tolist()), key=lambda kv: kv[1], reverse=True))

        top = list(importance.items())[:20][::-1]
        plt.figure(figsize=(9, 8))
        plt.barh([k for k, _ in top], [v for _, v in top], color="#2E7D32")
        plt.xlabel("Mean |SHAP value|")
        plt.title("Top 20 feature importances (SHAP)")
        plt.tight_layout()
        plt.savefig(config.SHAP_SUMMARY_PNG, dpi=120)
        plt.close()
        joblib.dump(importance, config.SHAP_VALUES_PKL)
        logger.info("Saved SHAP summary to %s", config.SHAP_SUMMARY_PNG)
        return importance
    except Exception as exc:  # pragma: no cover - SHAP is best-effort
        logger.warning("SHAP computation failed: %s", exc)
        return {}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train(n_trials: int = config.DEFAULT_OPTUNA_TRIALS, use_cache: bool = True) -> dict[str, Any]:
    """Run the end-to-end training pipeline and persist all artifacts."""
    config.ensure_dirs()
    frame, name_map = preprocess(use_cache=use_cache)
    features, builder = build_features(frame, use_cache=use_cache)
    feature_columns = builder.feature_columns_
    logger.info("Using %d features", len(feature_columns))

    splits = make_splits(features, feature_columns)

    # --- classifiers ------------------------------------------------------- #
    xgb_params, xgb_model = tune_xgb(splits, n_trials)
    lgb_params, lgb_model = tune_lgb(splits, n_trials)

    # --- poisson ----------------------------------------------------------- #
    poisson_model = PoissonGoalModel().fit(
        splits["train"]["X"], splits["train"]["home_goals"], splits["train"]["away_goals"]
    )

    # --- ensemble weights on validation ------------------------------------ #
    val_probas = {
        "xgboost": xgb_model.predict_proba(splits["val"]["X"]),
        "lightgbm": lgb_model.predict_proba(splits["val"]["X"]),
        "poisson": poisson_outcome_probs(poisson_model, splits["val"]["X"]),
    }
    weights = optimize_ensemble_weights(val_probas, splits["val"]["y"])

    # --- test evaluation --------------------------------------------------- #
    test_probas = {
        "xgboost": xgb_model.predict_proba(splits["test"]["X"]),
        "lightgbm": lgb_model.predict_proba(splits["test"]["X"]),
        "poisson": poisson_outcome_probs(poisson_model, splits["test"]["X"]),
    }
    ensemble_test = blend(test_probas, weights)
    y_test = splits["test"]["y"]

    metrics: dict[str, Any] = {
        "n_features": len(feature_columns),
        "n_train": int(len(splits["train"]["y"])),
        "n_val": int(len(splits["val"]["y"])),
        "n_test": int(len(splits["test"]["y"])),
        "test": {
            "xgboost": evaluate(y_test, test_probas["xgboost"]),
            "lightgbm": evaluate(y_test, test_probas["lightgbm"]),
            "poisson": evaluate(y_test, test_probas["poisson"]),
            "ensemble": evaluate(y_test, ensemble_test),
        },
        "ensemble_weights": weights,
    }

    # Poisson expected-goals error on the test set.
    lam_home, lam_away = poisson_model.predict_expected(splits["test"]["X"])
    metrics["test"]["poisson_goal_mae"] = {
        "home": float(np.mean(np.abs(lam_home - splits["test"]["home_goals"]))),
        "away": float(np.mean(np.abs(lam_away - splits["test"]["away_goals"]))),
    }

    # TimeSeriesSplit CV report.
    metrics["time_series_cv"] = time_series_cv_report(
        xgb_params, features[feature_columns], features["result"].to_numpy()
    )

    # SHAP explainability.
    metrics["shap_top_features"] = dict(
        list(compute_shap(xgb_model, splits["test"]["X"], feature_columns).items())[:15]
    )

    # --- persist artifacts ------------------------------------------------- #
    joblib.dump(builder, config.FEATURE_BUILDER_PKL)
    joblib.dump(feature_columns, config.FEATURE_LIST_PKL)
    joblib.dump(name_map, config.NAME_MAP_PKL)
    joblib.dump(xgb_model, config.XGB_MODEL_PKL)
    joblib.dump(lgb_model, config.LGB_MODEL_PKL)
    joblib.dump(poisson_model.home_model, config.POISSON_HOME_PKL)
    joblib.dump(poisson_model.away_model, config.POISSON_AWAY_PKL)
    joblib.dump(weights, config.ENSEMBLE_WEIGHTS_PKL)
    with open(config.METRICS_JSON, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Saved all artifacts to %s", config.MODELS_DIR)
    _log_summary(metrics)
    return metrics


def _log_summary(metrics: dict[str, Any]) -> None:
    """Pretty-print the headline test metrics."""
    logger.info("=" * 64)
    logger.info("TEST-SET METRICS (2024+)")
    logger.info("%-10s %8s %9s %8s %8s", "model", "acc", "logloss", "brier", "roc_auc")
    for name in ("xgboost", "lightgbm", "poisson", "ensemble"):
        m = metrics["test"][name]
        logger.info(
            "%-10s %8.4f %9.4f %8.4f %8.4f",
            name, m["accuracy"], m["log_loss"], m["brier_score"], m["roc_auc_ovr_macro"],
        )
    logger.info("=" * 64)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the football prediction models.")
    parser.add_argument("--trials", type=int, default=config.DEFAULT_OPTUNA_TRIALS,
                        help="Number of Optuna trials per classifier (default: 100).")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached preprocessing/features.")
    parser.add_argument("--quick", action="store_true", help="Quick run with 10 trials (for smoke testing).")
    args = parser.parse_args()

    n_trials = 10 if args.quick else args.trials
    train(n_trials=n_trials, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
