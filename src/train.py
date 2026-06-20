"""Training pipeline.

Trains and serializes the full prediction stack:

1. XGBoost classifier (Optuna-tuned on a time-based validation split).
2. LightGBM classifier (Optuna-tuned likewise).
3. Dixon-Coles attack/defense goal model (expected goals + scorelines).
4. A logistic-regression **stacking** meta-learner that combines and calibrates
   the three outcome-probability sources.

Validation strictly respects time order (no random splits): train on
1872-2020, validate on 2021-2023, test on 2024+.  Each Dixon-Coles fit only
uses matches strictly before the split it scores.  A TimeSeriesSplit
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb
import xgboost as xgb

from . import config
from .config import get_logger
from .dixon_coles import DixonColesModel
from .features import build_features
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
            "meta": part[["home_team", "away_team", "neutral"]].reset_index(drop=True),
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
# Dixon-Coles fitting (with a small time-decay search)
# --------------------------------------------------------------------------- #
def fit_dixon_coles(frame: pd.DataFrame, cutoff_year: int, xi: float) -> DixonColesModel:
    """Fit a Dixon-Coles model on all matches up to (and including) a year."""
    train = frame[frame["year"] <= cutoff_year]
    return DixonColesModel(xi=xi).fit(train)


def search_dixon_coles_xi(frame: pd.DataFrame, val_meta: pd.DataFrame, y_val: np.ndarray) -> float:
    """Pick the time-decay rate minimising validation log loss."""
    best_xi, best_loss = 0.0, np.inf
    for xi in (0.0, 0.0004, 0.00076, 0.0012, 0.002):
        model = fit_dixon_coles(frame, config.TRAIN_END_YEAR, xi)
        proba = model.predict_proba(val_meta)
        loss = log_loss(y_val, proba, labels=_LABELS)
        if loss < best_loss:
            best_loss, best_xi = loss, xi
    logger.info("Dixon-Coles best xi=%.5f (val log loss %.4f)", best_xi, best_loss)
    return best_xi


# --------------------------------------------------------------------------- #
# Ensemble: stacking meta-learner + simple-average baseline
# --------------------------------------------------------------------------- #
_STACK_ORDER = ("xgboost", "lightgbm", "dixon_coles")


def _stack_features(probas: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate per-model log-probabilities into a stacker design matrix."""
    parts = [np.log(np.clip(probas[name], 1e-6, 1.0)) for name in _STACK_ORDER]
    return np.hstack(parts)


def fit_stacker(probas: dict[str, np.ndarray], y_val: np.ndarray) -> LogisticRegression:
    """Fit a multinomial logistic-regression stacker on validation predictions."""
    meta_X = _stack_features(probas)
    # scikit-learn >=1.7 uses multinomial by default for multiclass with lbfgs.
    stacker = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    stacker.fit(meta_X, y_val)
    loss = log_loss(y_val, stacker.predict_proba(meta_X), labels=_LABELS)
    logger.info("Stacker fitted (val log loss %.4f)", loss)
    return stacker


def stack_predict(stacker: LogisticRegression, probas: dict[str, np.ndarray]) -> np.ndarray:
    """Apply the stacker to per-model probabilities -> ensemble probabilities."""
    return stacker.predict_proba(_stack_features(probas))


def mean_blend(probas: dict[str, np.ndarray]) -> np.ndarray:
    """Simple equal-weight average baseline (for comparison)."""
    blended = sum(probas[name] for name in _STACK_ORDER) / len(_STACK_ORDER)
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
def train(
    n_trials: int = config.DEFAULT_OPTUNA_TRIALS,
    use_cache: bool = True,
    use_player_data: bool = True,
) -> dict[str, Any]:
    """Run the end-to-end training pipeline and persist all artifacts."""
    config.ensure_dirs()
    frame, name_map = preprocess(use_cache=use_cache)

    squad = None
    if use_player_data:
        try:
            from .player_data import build_squad_strength, squad_lookup
            squad_df = build_squad_strength(name_map, use_cache=use_cache)
            squad = squad_lookup(squad_df) if not squad_df.empty else None
            logger.info("Player layer: %d StatsBomb squad-strength matches", 0 if squad is None else len(squad))
        except Exception as exc:  # player layer is optional/best-effort
            logger.warning("Player layer unavailable (%s); continuing without it.", exc)

    features, builder = build_features(frame, use_cache=use_cache, squad_lookup=squad)
    feature_columns = builder.feature_columns_
    logger.info("Using %d features", len(feature_columns))

    splits = make_splits(features, feature_columns)

    # --- classifiers ------------------------------------------------------- #
    xgb_params, xgb_model = tune_xgb(splits, n_trials)
    lgb_params, lgb_model = tune_lgb(splits, n_trials)

    # --- Dixon-Coles goal model (leak-free: each fit uses only earlier data) #
    xi = search_dixon_coles_xi(frame, splits["val"]["meta"], splits["val"]["y"])
    dc_val = fit_dixon_coles(frame, config.TRAIN_END_YEAR, xi)    # scores val
    dc_test = fit_dixon_coles(frame, config.VAL_END_YEAR, xi)     # scores test
    dc_prod = DixonColesModel(xi=xi).fit(frame)                   # for inference

    # --- stacking meta-learner fitted on validation predictions ------------ #
    val_probas = {
        "xgboost": xgb_model.predict_proba(splits["val"]["X"]),
        "lightgbm": lgb_model.predict_proba(splits["val"]["X"]),
        "dixon_coles": dc_val.predict_proba(splits["val"]["meta"]),
    }
    stacker = fit_stacker(val_probas, splits["val"]["y"])

    # --- test evaluation --------------------------------------------------- #
    test_probas = {
        "xgboost": xgb_model.predict_proba(splits["test"]["X"]),
        "lightgbm": lgb_model.predict_proba(splits["test"]["X"]),
        "dixon_coles": dc_test.predict_proba(splits["test"]["meta"]),
    }
    ensemble_test = stack_predict(stacker, test_probas)
    mean_test = mean_blend(test_probas)
    y_test = splits["test"]["y"]

    metrics: dict[str, Any] = {
        "n_features": len(feature_columns),
        "n_train": int(len(splits["train"]["y"])),
        "n_val": int(len(splits["val"]["y"])),
        "n_test": int(len(splits["test"]["y"])),
        "dixon_coles_xi": xi,
        "test": {
            "xgboost": evaluate(y_test, test_probas["xgboost"]),
            "lightgbm": evaluate(y_test, test_probas["lightgbm"]),
            "dixon_coles": evaluate(y_test, test_probas["dixon_coles"]),
            "ensemble_mean": evaluate(y_test, mean_test),
            "ensemble": evaluate(y_test, ensemble_test),
        },
    }

    # Dixon-Coles expected-goals error on the test set.
    dc_lam = np.array([dc_test.predict_expected(r.home_team, r.away_team, r.neutral)
                       for r in splits["test"]["meta"].itertuples(index=False)])
    metrics["test"]["goal_mae"] = {
        "home": float(np.mean(np.abs(dc_lam[:, 0] - splits["test"]["home_goals"]))),
        "away": float(np.mean(np.abs(dc_lam[:, 1] - splits["test"]["away_goals"]))),
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
    joblib.dump(dc_prod, config.DIXON_COLES_PKL)
    joblib.dump(stacker, config.STACKER_PKL)
    joblib.dump(list(_STACK_ORDER), config.ENSEMBLE_WEIGHTS_PKL)  # stack input order
    with open(config.METRICS_JSON, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Saved all artifacts to %s", config.MODELS_DIR)
    _log_summary(metrics)
    return metrics


def _log_summary(metrics: dict[str, Any]) -> None:
    """Pretty-print the headline test metrics."""
    logger.info("=" * 64)
    logger.info("TEST-SET METRICS (2024+)")
    logger.info("%-13s %8s %9s %8s %8s", "model", "acc", "logloss", "brier", "roc_auc")
    for name in ("xgboost", "lightgbm", "dixon_coles", "ensemble_mean", "ensemble"):
        m = metrics["test"][name]
        logger.info(
            "%-13s %8.4f %9.4f %8.4f %8.4f",
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
    parser.add_argument("--no-player-data", action="store_true",
                        help="Skip the StatsBomb squad-strength player layer.")
    args = parser.parse_args()

    n_trials = 10 if args.quick else args.trials
    train(n_trials=n_trials, use_cache=not args.no_cache, use_player_data=not args.no_player_data)


if __name__ == "__main__":
    main()
