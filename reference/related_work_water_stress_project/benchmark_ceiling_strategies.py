#!/usr/bin/env python3
"""Benchmark advanced strategies to push BHSE Ksat performance above baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import boxcox
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from train_ksat_models import (
    RANDOM_STATE_DEFAULT,
    create_stratified_holdout_split,
    load_dataset,
    prepare_dataset,
    quartile_strata,
    regression_metrics,
)

try:
    import shap

    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

BASELINE_TARGET_R2_LOG = 0.7502
EPS = 1e-12
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class Candidate:
    name: str
    feature_cols: List[str]
    method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark feature engineering, RF/LGBM tuning, stacking, Box-Cox and residual "
            "hybrid models for BHSE Ksat."
        )
    )
    parser.add_argument("--data-path", type=Path, default=Path("Pontos_Ksat_Sanga_Ellert.xlsx"))
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_ceiling_strategies"))
    parser.add_argument(
        "--baseline-params-json",
        type=Path,
        default=Path("outputs_mlp_tuning_optuna/mlp_best_params.json"),
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE_DEFAULT)
    parser.add_argument("--max-iter", type=int, default=2500)
    parser.add_argument("--rf-trials", type=int, default=80)
    parser.add_argument("--lgbm-trials", type=int, default=100)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--mccv-iter", type=int, default=120)
    parser.add_argument("--mccv-test-size", type=float, default=0.2)
    return parser.parse_args()


def load_mlp_params(path: Path) -> Dict[str, object]:
    defaults = {
        "hidden_layer_sizes": (128, 64),
        "activation": "tanh",
        "alpha": 0.0011380907649808582,
        "learning_rate_init": 0.0024792013326598596,
    }
    if not path.exists():
        return defaults

    data = json.loads(path.read_text(encoding="utf-8"))
    params = data.get("best_params", {})
    hidden = params.get("hidden_layer_sizes", defaults["hidden_layer_sizes"])
    if isinstance(hidden, list):
        hidden = tuple(int(v) for v in hidden)
    return {
        "hidden_layer_sizes": tuple(hidden),
        "activation": str(params.get("activation", defaults["activation"])),
        "alpha": float(params.get("alpha", defaults["alpha"])),
        "learning_rate_init": float(
            params.get("learning_rate_init", defaults["learning_rate_init"])
        ),
    }


def build_mlp_pipeline(params: Dict[str, object], seed: int, max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
                    activation=str(params["activation"]),
                    solver="adam",
                    alpha=float(params["alpha"]),
                    learning_rate_init=float(params["learning_rate_init"]),
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=40,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_rf(params: Dict[str, object], seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=float(params["max_features"]),
        max_samples=float(params["max_samples"]),
        random_state=seed,
        n_jobs=-1,
    )


def build_lgbm(params: Dict[str, object], seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params["min_child_samples"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sand_clay_ratio"] = out["sand"] / (out["clay"] + 1.0)
    out["macro_ds_ratio"] = out["macroporosity"] / (out["bulk_density"] + EPS)
    out["clay_ds_product"] = out["clay"] * out["bulk_density"]
    out["macro_squared"] = out["macroporosity"] ** 2
    out["sand_macro"] = out["sand"] * out["macroporosity"]
    out["log_ds"] = np.log(np.clip(out["bulk_density"], EPS, None))
    out["inv_clay"] = 1.0 / (out["clay"] + 1.0)

    x_std = float(out["x"].std(ddof=0)) + EPS
    y_std = float(out["y"].std(ddof=0)) + EPS
    out["x_norm"] = (out["x"] - float(out["x"].mean())) / x_std
    out["y_norm"] = (out["y"] - float(out["y"].mean())) / y_std
    out["dist_center"] = np.sqrt(out["x_norm"] ** 2 + out["y_norm"] ** 2)
    out["spatial_interaction"] = out["x_norm"] * out["y_norm"]
    return out


def tune_rf_optuna(
    x: np.ndarray, y: np.ndarray, seed: int, n_trials: int, cv_folds: int
) -> Tuple[Dict[str, object], float, pd.DataFrame]:
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
        }
        model = build_rf(params=params, seed=seed)
        scores = cross_val_score(model, x, y, cv=splitter, scoring="r2", n_jobs=1)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    rows = []
    for trial in study.trials:
        if trial.value is None:
            continue
        row = {"trial": int(trial.number), "value": float(trial.value)}
        row.update(trial.params)
        rows.append(row)
    tuning_df = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
    return study.best_params, float(study.best_value), tuning_df


def tune_lgbm_optuna(
    x: np.ndarray, y: np.ndarray, seed: int, n_trials: int, cv_folds: int
) -> Tuple[Dict[str, object], float, pd.DataFrame]:
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 96),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        }
        model = build_lgbm(params=params, seed=seed)
        scores = cross_val_score(model, x, y, cv=splitter, scoring="r2", n_jobs=1)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed + 17)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    rows = []
    for trial in study.trials:
        if trial.value is None:
            continue
        row = {"trial": int(trial.number), "value": float(trial.value)}
        row.update(trial.params)
        rows.append(row)
    tuning_df = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
    return study.best_params, float(study.best_value), tuning_df


def shap_feature_selection(
    x_df: pd.DataFrame, y: np.ndarray, rf_params: Dict[str, object], seed: int
) -> Tuple[List[str], pd.DataFrame]:
    model = build_rf(params=rf_params, seed=seed)
    model.fit(x_df.to_numpy(dtype=float), y)
    features = list(x_df.columns)

    if HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_df.to_numpy(dtype=float))
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        mean_abs = np.mean(np.abs(np.asarray(shap_values, dtype=float)), axis=0)
        source = "shap"
    else:
        mean_abs = np.abs(model.feature_importances_)
        source = "rf_feature_importances"

    threshold = float(np.mean(mean_abs) * 0.10)
    selected_mask = mean_abs >= threshold
    if int(np.sum(selected_mask)) < 5:
        top_idx = np.argsort(mean_abs)[::-1][:5]
        selected_mask = np.zeros_like(selected_mask, dtype=bool)
        selected_mask[top_idx] = True

    importance_df = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_importance": mean_abs,
            "selected": selected_mask,
            "selector_source": source,
            "threshold": threshold,
        }
    ).sort_values("mean_abs_importance", ascending=False)

    selected = importance_df.loc[importance_df["selected"], "feature"].tolist()
    return selected, importance_df.reset_index(drop=True)


def fit_predict_candidate(
    candidate: Candidate,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    max_iter: int,
    mlp_params: Dict[str, object],
    rf_params: Dict[str, object],
    lgbm_params: Dict[str, object],
) -> np.ndarray:
    x_train = train_df[candidate.feature_cols].to_numpy(dtype=float)
    x_test = test_df[candidate.feature_cols].to_numpy(dtype=float)
    y_train_log = train_df["log_ksat"].to_numpy(dtype=float)
    y_train_orig = train_df["ksat"].to_numpy(dtype=float)

    if candidate.method == "mlp_log":
        model = build_mlp_pipeline(params=mlp_params, seed=seed, max_iter=max_iter)
        model.fit(x_train, y_train_log)
        pred = model.predict(x_test)
        if not np.all(np.isfinite(pred)):
            raise ValueError("Non-finite predictions from mlp_log.")
        return pred

    if candidate.method == "rf_log":
        model = build_rf(params=rf_params, seed=seed)
        model.fit(x_train, y_train_log)
        pred = model.predict(x_test)
        if not np.all(np.isfinite(pred)):
            raise ValueError("Non-finite predictions from rf_log.")
        return pred

    if candidate.method == "lgbm_log":
        model = build_lgbm(params=lgbm_params, seed=seed)
        model.fit(x_train, y_train_log)
        pred = model.predict(x_test)
        if not np.all(np.isfinite(pred)):
            raise ValueError("Non-finite predictions from lgbm_log.")
        return pred

    if candidate.method == "stack_log":
        estimators = [
            ("mlp", build_mlp_pipeline(params=mlp_params, seed=seed, max_iter=max_iter)),
            ("rf", build_rf(params=rf_params, seed=seed)),
            ("lgbm", build_lgbm(params=lgbm_params, seed=seed)),
        ]
        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            n_jobs=-1,
            passthrough=False,
        )
        stack.fit(x_train, y_train_log)
        pred = stack.predict(x_test)
        if not np.all(np.isfinite(pred)):
            raise ValueError("Non-finite predictions from stack_log.")
        return pred

    if candidate.method == "boxcox_mlp":
        base = build_mlp_pipeline(params=mlp_params, seed=seed, max_iter=max_iter)
        reg = TransformedTargetRegressor(
            regressor=base,
            transformer=PowerTransformer(method="box-cox", standardize=False),
        )
        reg.fit(x_train, y_train_orig)
        pred_orig = np.clip(reg.predict(x_test), EPS, None)
        pred = np.log10(pred_orig)
        if not np.all(np.isfinite(pred)):
            raise ValueError("Non-finite predictions from boxcox_mlp.")
        return pred

    if candidate.method == "residual_hybrid":
        trend = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
        trend.fit(x_train, y_train_log)
        residual_train = y_train_log - trend.predict(x_train)
        residual_model = build_mlp_pipeline(params=mlp_params, seed=seed, max_iter=max_iter)
        residual_model.fit(x_train, residual_train)
        pred_log = trend.predict(x_test) + residual_model.predict(x_test)
        if not np.all(np.isfinite(pred_log)):
            raise ValueError("Non-finite predictions from residual_hybrid.")
        return pred_log

    raise ValueError(f"Unknown candidate method: {candidate.method}")


def evaluate_holdout(
    df: pd.DataFrame,
    candidates: Sequence[Candidate],
    seed: int,
    max_iter: int,
    mlp_params: Dict[str, object],
    rf_params: Dict[str, object],
    lgbm_params: Dict[str, object],
) -> pd.DataFrame:
    base_cols = ["x", "y", "sand", "clay", "bulk_density", "macroporosity"]
    split = create_stratified_holdout_split(
        x=df[base_cols].to_numpy(dtype=float),
        y=df["log_ksat"].to_numpy(dtype=float),
        random_state=seed,
        train_fraction=0.70,
        val_fraction=0.15,
        test_fraction=0.15,
    )

    idx_train_val = np.concatenate([split.idx_train, split.idx_val])
    train_df = df.iloc[idx_train_val].reset_index(drop=True)
    test_df = df.iloc[split.idx_test].reset_index(drop=True)
    y_true_log = test_df["log_ksat"].to_numpy(dtype=float)

    rows = []
    for i, candidate in enumerate(candidates):
        try:
            pred_log = fit_predict_candidate(
                candidate=candidate,
                train_df=train_df,
                test_df=test_df,
                seed=seed + i,
                max_iter=max_iter,
                mlp_params=mlp_params,
                rf_params=rf_params,
                lgbm_params=lgbm_params,
            )
            metrics = regression_metrics(y_true_log, pred_log)
            rows.append(
                {
                    "evaluation": "holdout_70_15_15",
                    "model": candidate.name,
                    "n_trainval": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "status": "ok",
                    "r2_log": float(metrics["r2_log"]),
                    "rmse_log": float(metrics["rmse_log"]),
                    "mae_log": float(metrics["mae_log"]),
                    "r2_orig": float(metrics["r2_orig"]),
                    "rmse_orig": float(metrics["rmse_orig"]),
                    "mae_orig": float(metrics["mae_orig"]),
                    "beat_baseline_07502": bool(metrics["r2_log"] > BASELINE_TARGET_R2_LOG),
                    "details": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "evaluation": "holdout_70_15_15",
                    "model": candidate.name,
                    "n_trainval": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "status": "failed",
                    "r2_log": float("nan"),
                    "rmse_log": float("nan"),
                    "mae_log": float("nan"),
                    "r2_orig": float("nan"),
                    "rmse_orig": float("nan"),
                    "mae_orig": float("nan"),
                    "beat_baseline_07502": False,
                    "details": str(exc),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["status", "r2_log"], ascending=[True, False]).reset_index(drop=True)


def evaluate_mccv(
    df: pd.DataFrame,
    candidates: Sequence[Candidate],
    seed: int,
    max_iter: int,
    n_iter: int,
    test_size: float,
    mlp_params: Dict[str, object],
    rf_params: Dict[str, object],
    lgbm_params: Dict[str, object],
) -> pd.DataFrame:
    strata = quartile_strata(df["log_ksat"].to_numpy(dtype=float))
    idx_all = np.arange(len(df))
    rows = []

    for it in range(n_iter):
        train_idx, test_idx = train_test_split(
            idx_all,
            test_size=test_size,
            random_state=seed + it,
            stratify=strata,
        )
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        y_true_log = test_df["log_ksat"].to_numpy(dtype=float)

        for m, candidate in enumerate(candidates):
            try:
                pred_log = fit_predict_candidate(
                    candidate=candidate,
                    train_df=train_df,
                    test_df=test_df,
                    seed=seed + it * 100 + m,
                    max_iter=max_iter,
                    mlp_params=mlp_params,
                    rf_params=rf_params,
                    lgbm_params=lgbm_params,
                )
                metrics = regression_metrics(y_true_log, pred_log)
                rows.append(
                    {
                        "evaluation": "mccv",
                        "iteration": int(it + 1),
                        "model": candidate.name,
                        "n_train": int(len(train_df)),
                        "n_test": int(len(test_df)),
                        "status": "ok",
                        "r2_log": float(metrics["r2_log"]),
                        "rmse_log": float(metrics["rmse_log"]),
                        "mae_log": float(metrics["mae_log"]),
                        "details": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "evaluation": "mccv",
                        "iteration": int(it + 1),
                        "model": candidate.name,
                        "n_train": int(len(train_df)),
                        "n_test": int(len(test_df)),
                        "status": "failed",
                        "r2_log": float("nan"),
                        "rmse_log": float("nan"),
                        "mae_log": float("nan"),
                        "details": str(exc),
                    }
                )

        if (it + 1) % 20 == 0 or it == n_iter - 1:
            print(f"MCCV progress: {it + 1}/{n_iter}")

    return pd.DataFrame(rows)


def summarize_mccv(scores_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in scores_df.groupby("model"):
        ok = group[group["status"] == "ok"]
        vals = ok["r2_log"].to_numpy(dtype=float)
        if vals.size == 0:
            rows.append(
                {
                    "model": model,
                    "n": 0,
                    "n_failed": int(len(group)),
                    "r2_log_mean": float("nan"),
                    "r2_log_std": float("nan"),
                    "r2_log_ci95_low": float("nan"),
                    "r2_log_ci95_high": float("nan"),
                    "r2_log_median": float("nan"),
                    "max_r2_log": float("nan"),
                    "beat_baseline_rate_07502": float("nan"),
                }
            )
            continue

        rows.append(
            {
                "model": model,
                "n": int(vals.size),
                "n_failed": int(len(group) - vals.size),
                "r2_log_mean": float(np.mean(vals)),
                "r2_log_std": float(np.std(vals, ddof=0)),
                "r2_log_ci95_low": float(np.percentile(vals, 2.5)),
                "r2_log_ci95_high": float(np.percentile(vals, 97.5)),
                "r2_log_median": float(np.median(vals)),
                "max_r2_log": float(np.max(vals)),
                "beat_baseline_rate_07502": float(np.mean(vals > BASELINE_TARGET_R2_LOG)),
            }
        )
    return pd.DataFrame(rows).sort_values("r2_log_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mlp_params = load_mlp_params(args.baseline_params_json)
    df_raw = load_dataset(args.data_path, sheet_name=args.sheet_name)
    dataset = prepare_dataset(
        df_raw=df_raw,
        include_silt=False,
        include_ratio=False,
        include_sinusoidal=False,
    )

    cols = ["x", "y", "sand", "clay", "bulk_density", "macroporosity", "ksat", "log_ksat"]
    df = dataset.df_model[cols].copy().reset_index(drop=True)
    df = add_engineered_features(df)

    base_features = ["x", "y", "sand", "clay", "bulk_density", "macroporosity"]
    engineered_features = base_features + [
        "sand_clay_ratio",
        "macro_ds_ratio",
        "clay_ds_product",
        "macro_squared",
        "sand_macro",
        "log_ds",
        "inv_clay",
        "x_norm",
        "y_norm",
        "dist_center",
        "spatial_interaction",
    ]

    x_eng = df[engineered_features].to_numpy(dtype=float)
    y_log = df["log_ksat"].to_numpy(dtype=float)
    y_orig = df["ksat"].to_numpy(dtype=float)

    boxcox_all, lambda_boxcox = boxcox(y_orig)
    _ = boxcox_all
    print(f"Global Box-Cox lambda (for reference): {lambda_boxcox:.4f}")

    print("Tuning RandomForest with Optuna...")
    rf_best_params, rf_best_cv_r2, rf_trials_df = tune_rf_optuna(
        x=x_eng,
        y=y_log,
        seed=args.seed,
        n_trials=args.rf_trials,
        cv_folds=args.cv_folds,
    )
    rf_trials_df.to_csv(args.output_dir / "rf_optuna_trials.csv", index=False)

    print("Tuning LightGBM with Optuna...")
    lgbm_best_params, lgbm_best_cv_r2, lgbm_trials_df = tune_lgbm_optuna(
        x=x_eng,
        y=y_log,
        seed=args.seed,
        n_trials=args.lgbm_trials,
        cv_folds=args.cv_folds,
    )
    lgbm_trials_df.to_csv(args.output_dir / "lgbm_optuna_trials.csv", index=False)

    selected_features, shap_df = shap_feature_selection(
        x_df=df[engineered_features],
        y=y_log,
        rf_params=rf_best_params,
        seed=args.seed,
    )
    shap_df.to_csv(args.output_dir / "feature_selection_importance.csv", index=False)

    candidates = [
        Candidate(name="mlp_baseline_6f", feature_cols=base_features, method="mlp_log"),
        Candidate(name="mlp_engineered", feature_cols=engineered_features, method="mlp_log"),
        Candidate(
            name="mlp_engineered_shap_selected",
            feature_cols=selected_features,
            method="mlp_log",
        ),
        Candidate(name="rf_optuna_engineered", feature_cols=engineered_features, method="rf_log"),
        Candidate(name="lgbm_optuna_engineered", feature_cols=engineered_features, method="lgbm_log"),
        Candidate(
            name="stack_mlp_rf_lgbm",
            feature_cols=engineered_features,
            method="stack_log",
        ),
        Candidate(
            name="mlp_engineered_boxcox_target",
            feature_cols=engineered_features,
            method="boxcox_mlp",
        ),
        Candidate(
            name="hybrid_linear_plus_mlp_residual",
            feature_cols=engineered_features,
            method="residual_hybrid",
        ),
    ]

    print("Evaluating holdout split 70/15/15...")
    holdout_df = evaluate_holdout(
        df=df,
        candidates=candidates,
        seed=args.seed,
        max_iter=args.max_iter,
        mlp_params=mlp_params,
        rf_params=rf_best_params,
        lgbm_params=lgbm_best_params,
    )
    holdout_df.to_csv(args.output_dir / "ceiling_holdout_results.csv", index=False)

    print("Evaluating MCCV...")
    mccv_scores_df = evaluate_mccv(
        df=df,
        candidates=candidates,
        seed=args.seed,
        max_iter=args.max_iter,
        n_iter=args.mccv_iter,
        test_size=args.mccv_test_size,
        mlp_params=mlp_params,
        rf_params=rf_best_params,
        lgbm_params=lgbm_best_params,
    )
    mccv_scores_df.to_csv(args.output_dir / "ceiling_mccv_scores.csv", index=False)
    mccv_summary_df = summarize_mccv(mccv_scores_df)
    mccv_summary_df.to_csv(args.output_dir / "ceiling_mccv_summary.csv", index=False)

    best_holdout_row = holdout_df.iloc[0].to_dict()
    best_mccv_row = mccv_summary_df.iloc[0].to_dict()

    payload = {
        "baseline_target_r2_log": BASELINE_TARGET_R2_LOG,
        "config": {
            "seed": args.seed,
            "max_iter": args.max_iter,
            "rf_trials": args.rf_trials,
            "lgbm_trials": args.lgbm_trials,
            "cv_folds": args.cv_folds,
            "mccv_iter": args.mccv_iter,
            "mccv_test_size": args.mccv_test_size,
            "n_samples_total": int(len(df)),
            "selected_features_count": int(len(selected_features)),
            "selected_features": selected_features,
            "shap_available": HAS_SHAP,
        },
        "boxcox_lambda_global": float(lambda_boxcox),
        "mlp_params": {
            "hidden_layer_sizes": list(mlp_params["hidden_layer_sizes"]),
            "activation": mlp_params["activation"],
            "alpha": mlp_params["alpha"],
            "learning_rate_init": mlp_params["learning_rate_init"],
        },
        "rf_tuning": {
            "best_cv_r2_log": float(rf_best_cv_r2),
            "best_params": rf_best_params,
        },
        "lgbm_tuning": {
            "best_cv_r2_log": float(lgbm_best_cv_r2),
            "best_params": lgbm_best_params,
        },
        "best_holdout": best_holdout_row,
        "best_mccv": best_mccv_row,
    }
    with open(args.output_dir / "ceiling_summary.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print("\n=== Ceiling benchmark completed ===")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(
        "Best holdout R2_log: "
        f"{best_holdout_row['model']} -> {best_holdout_row['r2_log']:.4f}"
    )
    print(
        "Best MCCV mean R2_log: "
        f"{best_mccv_row['model']} -> {best_mccv_row['r2_log_mean']:.4f} ± {best_mccv_row['r2_log_std']:.4f}"
    )
    print(
        "Baseline target comparison (0.7502) holdout: "
        f"{'YES' if float(best_holdout_row['r2_log']) > BASELINE_TARGET_R2_LOG else 'NO'}"
    )
    print(
        "Baseline target comparison (0.7502) MCCV mean: "
        f"{'YES' if float(best_mccv_row['r2_log_mean']) > BASELINE_TARGET_R2_LOG else 'NO'}"
    )


if __name__ == "__main__":
    main()
