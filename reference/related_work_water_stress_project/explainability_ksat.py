#!/usr/bin/env python3
"""Comprehensive explainability workflow for BHSE Ksat prediction."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_ksat_models import (
    RANDOM_STATE_DEFAULT,
    create_stratified_holdout_split,
    load_dataset,
    prepare_dataset,
    regression_metrics,
)

try:
    import lime.lime_tabular

    HAS_LIME = True
except Exception:
    HAS_LIME = False

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run global/local/spatial explainability for Ksat prediction in BHSE "
            "(SHAP, permutation importance, PDP/ICE, LIME, counterfactuals, residual maps, variogram)."
        )
    )
    parser.add_argument("--data-path", type=Path, default=Path("Pontos_Ksat_Sanga_Ellert.xlsx"))
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_explainability"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE_DEFAULT)
    parser.add_argument("--model", choices=["rf", "lgbm", "mlp"], default="rf")
    parser.add_argument("--feature-set", choices=["base", "engineered"], default="engineered")
    parser.add_argument("--max-iter-mlp", type=int, default=2500)
    parser.add_argument("--perm-repeats", type=int, default=30)
    parser.add_argument("--sloo-buffer-m", type=float, default=50.0)
    parser.add_argument("--sloo-min-train", type=int, default=20)
    parser.add_argument("--sloo-max-points", type=int, default=169)
    parser.add_argument(
        "--ceiling-summary-json",
        type=Path,
        default=Path("outputs_ceiling_strategies/ceiling_summary.json"),
    )
    parser.add_argument(
        "--mlp-params-json",
        type=Path,
        default=Path("outputs_mlp_tuning_optuna/mlp_best_params.json"),
    )
    return parser.parse_args()


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


def load_reference_params(
    model_name: str,
    ceiling_summary_json: Path,
    mlp_params_json: Path,
) -> Dict[str, object]:
    defaults: Dict[str, object] = {}
    if model_name == "rf":
        defaults = {
            "n_estimators": 582,
            "max_depth": 21,
            "min_samples_split": 3,
            "min_samples_leaf": 5,
            "max_features": 0.3898326250967136,
            "max_samples": 0.8788375533311741,
        }
    elif model_name == "lgbm":
        defaults = {
            "n_estimators": 166,
            "learning_rate": 0.03314802915422099,
            "num_leaves": 68,
            "min_child_samples": 33,
            "subsample": 0.944434530559701,
            "colsample_bytree": 0.9352227350018216,
            "reg_alpha": 0.01343049404303324,
            "reg_lambda": 0.03889673458223428,
        }
    else:
        defaults = {
            "hidden_layer_sizes": (128, 64),
            "activation": "tanh",
            "alpha": 0.0011380907649808582,
            "learning_rate_init": 0.0024792013326598596,
        }

    if model_name in {"rf", "lgbm"} and ceiling_summary_json.exists():
        data = json.loads(ceiling_summary_json.read_text(encoding="utf-8"))
        key = "rf_tuning" if model_name == "rf" else "lgbm_tuning"
        best = data.get(key, {}).get("best_params", {})
        if best:
            return best

    if model_name == "mlp" and mlp_params_json.exists():
        data = json.loads(mlp_params_json.read_text(encoding="utf-8"))
        best = data.get("best_params", {})
        if best:
            hidden = best.get("hidden_layer_sizes", defaults["hidden_layer_sizes"])
            if isinstance(hidden, list):
                hidden = tuple(int(v) for v in hidden)
            return {
                "hidden_layer_sizes": tuple(hidden),
                "activation": str(best.get("activation", defaults["activation"])),
                "alpha": float(best.get("alpha", defaults["alpha"])),
                "learning_rate_init": float(
                    best.get("learning_rate_init", defaults["learning_rate_init"])
                ),
            }

    return defaults


def build_model_factory(
    model_name: str,
    params: Dict[str, object],
    seed: int,
    max_iter_mlp: int,
) -> Callable[[int], RegressorMixin]:
    def make_rf(local_seed: int) -> RegressorMixin:
        return RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=float(params["max_features"]),
            max_samples=float(params["max_samples"]),
            random_state=local_seed,
            n_jobs=-1,
        )

    def make_lgbm(local_seed: int) -> RegressorMixin:
        return LGBMRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            random_state=local_seed,
            n_jobs=-1,
            verbosity=-1,
        )

    def make_mlp(local_seed: int) -> RegressorMixin:
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
                        max_iter=max_iter_mlp,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=40,
                        random_state=local_seed,
                    ),
                ),
            ]
        )

    if model_name == "rf":
        return make_rf
    if model_name == "lgbm":
        return make_lgbm
    return make_mlp


def ensure_dirs(base: Path) -> Dict[str, Path]:
    dirs = {
        "global": base / "global",
        "local": base / "local",
        "spatial": base / "spatial",
        "diagnostics": base / "diagnostics",
    }
    base.mkdir(parents=True, exist_ok=True)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(path: Path, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def shap_global(
    model: RegressorMixin,
    model_name: str,
    x_train_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
    out_dir: Path,
    seed: int,
) -> Tuple[np.ndarray, object]:
    if model_name in {"rf", "lgbm"}:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test_df)
    else:
        background = shap.sample(x_train_df, min(50, len(x_train_df)), random_state=seed)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(x_test_df, nsamples=300)

    shap_values_arr = np.asarray(shap_values, dtype=float)
    if shap_values_arr.ndim > 2:
        shap_values_arr = shap_values_arr[0]

    mean_abs = np.mean(np.abs(shap_values_arr), axis=0)
    pd.DataFrame(
        {
            "feature": x_test_df.columns,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False).to_csv(
        out_dir / "shap_mean_abs.csv", index=False
    )

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values_arr, x_test_df, plot_type="bar", show=False)
    plt.title("SHAP Summary (bar)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values_arr, x_test_df, show=False)
    plt.title("SHAP Summary (beeswarm)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_beeswarm.png", dpi=180)
    plt.close()

    return shap_values_arr, explainer


def run_permutation_importance(
    model: RegressorMixin,
    x_test_df: pd.DataFrame,
    y_test: np.ndarray,
    repeats: int,
    seed: int,
    out_dir: Path,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x_test_df,
        y_test,
        n_repeats=repeats,
        random_state=seed,
        scoring="r2",
        n_jobs=-1,
    )
    imp_df = pd.DataFrame(
        {
            "feature": x_test_df.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(out_dir / "permutation_importance.csv", index=False)

    plot_df = imp_df.sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        plot_df["feature"],
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
        capsize=3,
        color="steelblue",
        alpha=0.85,
    )
    ax.set_xlabel("Queda em R2 ao permutar a feature")
    ax.set_title("Permutation Importance (com desvio-padrao)")
    fig.tight_layout()
    fig.savefig(out_dir / "permutation_importance.png", dpi=180)
    plt.close(fig)
    return imp_df


def run_pdp_ice(
    model: RegressorMixin,
    x_train_df: pd.DataFrame,
    out_dir: Path,
) -> List[str]:
    candidates = ["macroporosity", "bulk_density", "sand", "clay"]
    features = [c for c in candidates if c in x_train_df.columns]
    if not features:
        return []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.ravel()
    PartialDependenceDisplay.from_estimator(
        model,
        x_train_df,
        features=features,
        kind="both",
        subsample=min(100, len(x_train_df)),
        ax=axes_flat[: len(features)],
    )

    for j in range(len(features), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("PDP + ICE para log(Ksat)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "pdp_ice.png", dpi=180)
    plt.close(fig)
    return features


def run_shap_interactions(
    model: RegressorMixin,
    model_name: str,
    explainer: object,
    shap_values: np.ndarray,
    x_test_df: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, object]:
    payload: Dict[str, object] = {"available": False}
    if model_name not in {"rf", "lgbm"}:
        return payload

    try:
        interaction = explainer.shap_interaction_values(x_test_df)
        interaction_arr = np.asarray(interaction, dtype=float)
        if interaction_arr.ndim > 3:
            interaction_arr = interaction_arr[0]
        mean_interaction = np.mean(np.abs(interaction_arr), axis=0)

        inter_df = pd.DataFrame(
            mean_interaction,
            index=x_test_df.columns,
            columns=x_test_df.columns,
        )
        inter_df.to_csv(out_dir / "shap_interaction_matrix.csv")

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(mean_interaction, cmap="Blues")
        ax.set_xticks(range(len(x_test_df.columns)))
        ax.set_yticks(range(len(x_test_df.columns)))
        ax.set_xticklabels(x_test_df.columns, rotation=45, ha="right")
        ax.set_yticklabels(x_test_df.columns)
        ax.set_title("Media |SHAP interaction|")
        fig.colorbar(im, ax=ax, label="media absoluta")
        fig.tight_layout()
        fig.savefig(out_dir / "shap_interaction_heatmap.png", dpi=180)
        plt.close(fig)

        if "macroporosity" in x_test_df.columns and "bulk_density" in x_test_df.columns:
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(
                "macroporosity",
                shap_values,
                x_test_df,
                interaction_index="bulk_density",
                show=False,
            )
            plt.tight_layout()
            plt.savefig(out_dir / "shap_dependence_macro_vs_ds.png", dpi=180)
            plt.close()

        payload["available"] = True
    except Exception as exc:
        payload["available"] = False
        payload["error"] = str(exc)
    return payload


def select_local_points(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    abs_err = np.abs(y_true - y_pred)
    picks = {
        "best_prediction": int(np.argmin(abs_err)),
        "worst_prediction": int(np.argmax(abs_err)),
        "highest_ksat": int(np.argmax(y_true)),
        "lowest_ksat": int(np.argmin(y_true)),
    }
    # remove duplicates while preserving first occurrence labels
    seen: Dict[int, str] = {}
    for label, idx in picks.items():
        if idx not in seen:
            seen[idx] = label
    return {label: idx for idx, label in seen.items()}


def run_local_shap_waterfalls(
    shap_values: np.ndarray,
    explainer: object,
    x_test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
) -> pd.DataFrame:
    picks = select_local_points(y_true=y_true, y_pred=y_pred)

    base_value = explainer.expected_value
    base = float(np.atleast_1d(base_value)[0])
    rows: List[Dict[str, object]] = []

    for label, idx in picks.items():
        exp = shap.Explanation(
            values=shap_values[idx],
            base_values=base,
            data=x_test_df.iloc[idx].to_numpy(dtype=float),
            feature_names=list(x_test_df.columns),
        )
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, max_display=min(15, x_test_df.shape[1]), show=False)
        plt.title(f"SHAP Waterfall - {label}")
        plt.tight_layout()
        fig.savefig(out_dir / f"waterfall_{label}.png", dpi=180)
        plt.close(fig)

        rows.append(
            {
                "label": label,
                "test_index": int(idx),
                "ksat_true_cm_day": float(10.0**y_true[idx]),
                "ksat_pred_cm_day": float(10.0**y_pred[idx]),
                "error_log": float(y_true[idx] - y_pred[idx]),
                "abs_error_log": float(abs(y_true[idx] - y_pred[idx])),
            }
        )

    local_df = pd.DataFrame(rows).sort_values("abs_error_log", ascending=False)
    local_df.to_csv(out_dir / "local_points_summary.csv", index=False)
    return local_df


def run_lime_local(
    model: RegressorMixin,
    x_train_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
    local_df: pd.DataFrame,
    out_dir: Path,
    seed: int,
) -> pd.DataFrame:
    if not HAS_LIME:
        return pd.DataFrame(
            [{"status": "skipped", "reason": "lime_not_installed"}]
        )

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=x_train_df.to_numpy(dtype=float),
        feature_names=list(x_train_df.columns),
        mode="regression",
        discretize_continuous=False,
        random_state=seed,
    )

    rows: List[Dict[str, object]] = []
    for _, row in local_df.iterrows():
        idx = int(row["test_index"])
        label = str(row["label"])
        explanation = explainer.explain_instance(
            x_test_df.iloc[idx].to_numpy(dtype=float),
            model.predict,
            num_features=min(len(x_train_df.columns), 12),
        )
        fig = explanation.as_pyplot_figure()
        fig.suptitle(f"LIME - {label}", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"lime_{label}.png", dpi=180)
        plt.close(fig)

        for feat_rule, weight in explanation.as_list():
            rows.append(
                {
                    "label": label,
                    "test_index": idx,
                    "feature_rule": feat_rule,
                    "weight": float(weight),
                }
            )

    lime_df = pd.DataFrame(rows)
    lime_df.to_csv(out_dir / "lime_local_weights.csv", index=False)
    return lime_df


def run_counterfactual_simple(
    model: RegressorMixin,
    x_train_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
    y_true: np.ndarray,
    worst_idx: int,
    out_dir: Path,
) -> pd.DataFrame:
    x0 = x_test_df.iloc[worst_idx].to_numpy(dtype=float)
    target = float(y_true[worst_idx])

    rows: List[Dict[str, object]] = []
    for j, col in enumerate(x_train_df.columns):
        low = float(np.min(x_train_df[col].to_numpy(dtype=float)))
        high = float(np.max(x_train_df[col].to_numpy(dtype=float)))
        best_val = float(x0[j])
        best_err = float("inf")
        best_pred = float("nan")

        for value in np.linspace(low, high, 120):
            x_cf = x0.copy()
            x_cf[j] = value
            pred = float(model.predict([x_cf])[0])
            err = abs(pred - target)
            if err < best_err:
                best_err = err
                best_val = float(value)
                best_pred = pred

        rows.append(
            {
                "feature": col,
                "original_value": float(x0[j]),
                "counterfactual_value": best_val,
                "delta_feature": float(best_val - x0[j]),
                "target_log_ksat": target,
                "pred_log_ksat_after_change": best_pred,
                "abs_error_after_change": best_err,
            }
        )

    cf_df = pd.DataFrame(rows).sort_values("abs_error_after_change", ascending=True)
    cf_df.to_csv(out_dir / "counterfactual_worst_point.csv", index=False)
    return cf_df


def run_neighbor_influence(
    x_train_df: pd.DataFrame,
    y_train: np.ndarray,
    x_test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    local_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    train = x_train_df.to_numpy(dtype=float)
    test = x_test_df.to_numpy(dtype=float)
    mu = train.mean(axis=0)
    sigma = train.std(axis=0) + EPS
    train_z = (train - mu) / sigma
    test_z = (test - mu) / sigma

    rows: List[Dict[str, object]] = []
    for _, local_row in local_df.iterrows():
        idx = int(local_row["test_index"])
        label = str(local_row["label"])
        dists = np.linalg.norm(train_z - test_z[idx], axis=1)
        nn_idx = np.argsort(dists)[:5]
        for rank, tidx in enumerate(nn_idx, start=1):
            rows.append(
                {
                    "label": label,
                    "test_index": idx,
                    "neighbor_rank": rank,
                    "neighbor_train_index": int(tidx),
                    "distance_z": float(dists[tidx]),
                    "neighbor_log_ksat": float(y_train[tidx]),
                    "test_log_ksat_true": float(y_test[idx]),
                    "test_log_ksat_pred": float(y_pred[idx]),
                }
            )
    nn_df = pd.DataFrame(rows)
    nn_df.to_csv(out_dir / "local_nearest_neighbors.csv", index=False)
    return nn_df


def spatial_loo_with_buffer(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_factory: Callable[[int], RegressorMixin],
    seed: int,
    buffer_radius_m: float,
    min_train: int,
    max_points: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    use_df = df.copy().reset_index(drop=True)
    if max_points < len(use_df):
        use_df = use_df.iloc[:max_points].reset_index(drop=True)

    coords = use_df[["x", "y"]].to_numpy(dtype=float)
    x_all = use_df[list(feature_cols)].to_numpy(dtype=float)
    y_all = use_df["log_ksat"].to_numpy(dtype=float)
    dmat = cdist(coords, coords)

    preds = np.zeros(len(use_df), dtype=float)
    train_sizes = np.zeros(len(use_df), dtype=int)
    fallback_count = 0

    for i in range(len(use_df)):
        mask = dmat[i] > buffer_radius_m
        mask[i] = False
        if int(np.sum(mask)) < min_train:
            mask = dmat[i] > 0.0
            fallback_count += 1
        train_idx = np.where(mask)[0]
        train_sizes[i] = int(len(train_idx))

        model = model_factory(seed + i)
        model.fit(x_all[train_idx], y_all[train_idx])
        preds[i] = float(model.predict(x_all[[i]])[0])

        if (i + 1) % 30 == 0 or i == len(use_df) - 1:
            print(f"SLOO progress: {i + 1}/{len(use_df)}")

    residual = y_all - preds
    out_df = use_df[["x", "y", "log_ksat"]].copy()
    out_df["log_ksat_pred"] = preds
    out_df["residual_log"] = residual
    out_df["abs_error_log"] = np.abs(residual)
    out_df["train_size"] = train_sizes

    metrics = regression_metrics(y_all, preds)
    summary = {
        "n_points": int(len(use_df)),
        "buffer_radius_m": float(buffer_radius_m),
        "min_train": int(min_train),
        "fallback_count": int(fallback_count),
        "avg_train_size": float(np.mean(train_sizes)),
        "r2_log": float(metrics["r2_log"]),
        "rmse_log": float(metrics["rmse_log"]),
        "mae_log": float(metrics["mae_log"]),
        "r2_orig": float(metrics["r2_orig"]),
        "rmse_orig": float(metrics["rmse_orig"]),
        "mae_orig": float(metrics["mae_orig"]),
    }
    return out_df, summary


def plot_residual_maps(df_res: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    residuals = df_res["residual_log"].to_numpy(dtype=float)
    abs_err = np.abs(residuals)
    coords = df_res[["x", "y"]].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    norm = TwoSlopeNorm(
        vmin=float(np.min(residuals)),
        vcenter=0.0,
        vmax=float(np.max(residuals)),
    )
    sc1 = axes[0].scatter(
        coords[:, 0],
        coords[:, 1],
        c=residuals,
        cmap="RdBu_r",
        norm=norm,
        s=70,
        edgecolors="gray",
        linewidths=0.3,
    )
    axes[0].set_title("Mapa de Residuo (log)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    fig.colorbar(sc1, ax=axes[0], label="real - previsto")

    sc2 = axes[1].scatter(
        coords[:, 0],
        coords[:, 1],
        c=abs_err,
        cmap="YlOrRd",
        s=70,
        edgecolors="gray",
        linewidths=0.3,
    )
    axes[1].set_title("Mapa de Erro Absoluto |log|")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    fig.colorbar(sc2, ax=axes[1], label="|erro|")

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_residual_maps.png", dpi=180)
    plt.close(fig)


def experimental_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 12,
) -> pd.DataFrame:
    distances = pdist(coords, metric="euclidean")
    diffs = pdist(values.reshape(-1, 1), metric="euclidean")
    semivariance = 0.5 * (diffs**2)

    max_dist = float(np.max(distances))
    bins = np.linspace(0.0, max_dist, n_lags + 1)
    bin_ids = np.digitize(distances, bins, right=True)

    rows: List[Dict[str, float]] = []
    for b in range(1, len(bins)):
        mask = bin_ids == b
        if np.sum(mask) < 5:
            continue
        rows.append(
            {
                "lag_id": float(b),
                "distance_mean": float(np.mean(distances[mask])),
                "gamma_mean": float(np.mean(semivariance[mask])),
                "pairs": float(np.sum(mask)),
            }
        )
    return pd.DataFrame(rows)


def spherical_model(h: np.ndarray, nugget: float, sill: float, rng: float) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    x = h / (rng + EPS)
    core = np.where(h <= rng, 1.5 * x - 0.5 * (x**3), 1.0)
    return nugget + sill * core


def fit_variogram_model(vario_df: pd.DataFrame) -> Dict[str, float]:
    x = vario_df["distance_mean"].to_numpy(dtype=float)
    y = vario_df["gamma_mean"].to_numpy(dtype=float)

    nugget0 = float(np.percentile(y, 10))
    sill0 = float(max(np.percentile(y, 90) - nugget0, EPS))
    range0 = float(np.percentile(x, 70))

    try:
        popt, _ = curve_fit(
            spherical_model,
            x,
            y,
            p0=[nugget0, sill0, range0],
            bounds=([0.0, 0.0, EPS], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
        nugget, sill, rng = [float(v) for v in popt]
    except Exception:
        nugget, sill, rng = nugget0, sill0, range0

    gde = float((nugget / (nugget + sill + EPS)) * 100.0)
    return {"nugget": nugget, "sill": sill, "range": rng, "gde_percent": gde}


def save_variogram_plot(
    vario_df: pd.DataFrame,
    params: Dict[str, float],
    out_path: Path,
) -> None:
    x = vario_df["distance_mean"].to_numpy(dtype=float)
    y = vario_df["gamma_mean"].to_numpy(dtype=float)
    x_line = np.linspace(0.0, max(float(np.max(x)), EPS), 200)
    y_line = spherical_model(
        x_line,
        params["nugget"],
        params["sill"],
        params["range"],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="steelblue", label="Semivariograma experimental")
    ax.plot(x_line, y_line, color="darkred", linewidth=2, label="Ajuste esferico")
    ax.set_xlabel("Distancia")
    ax.set_ylabel("Semivariancia")
    ax.set_title("Variograma dos residuos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def residual_correlation(
    df_res: pd.DataFrame,
    feature_cols: Sequence[str],
    x_source: pd.DataFrame,
) -> pd.DataFrame:
    abs_error = np.abs(df_res["residual_log"].to_numpy(dtype=float))
    rows = []
    for col in feature_cols:
        r, p = spearmanr(x_source[col].to_numpy(dtype=float), abs_error)
        rows.append({"feature": col, "spearman_r": float(r), "p_value": float(p)})
    out = pd.DataFrame(rows).sort_values("p_value", ascending=True).reset_index(drop=True)
    return out


def quadrant_diagnostics(df_res: pd.DataFrame) -> pd.DataFrame:
    x_mid = float(df_res["x"].median())
    y_mid = float(df_res["y"].median())

    quad = (
        (df_res["x"].to_numpy(dtype=float) > x_mid).astype(int) * 2
        + (df_res["y"].to_numpy(dtype=float) > y_mid).astype(int)
    )
    names = {0: "SW", 1: "NW", 2: "SE", 3: "NE"}

    rows = []
    for q in [0, 1, 2, 3]:
        mask = quad == q
        if int(np.sum(mask)) < 3:
            continue
        y_true = df_res.loc[mask, "log_ksat"].to_numpy(dtype=float)
        y_pred = df_res.loc[mask, "log_ksat_pred"].to_numpy(dtype=float)
        res = y_true - y_pred
        rows.append(
            {
                "quadrant": names[q],
                "n": int(np.sum(mask)),
                "r2_log": float(r2_score(y_true, y_pred)),
                "mae_log": float(np.mean(np.abs(res))),
                "mean_residual_log": float(np.mean(res)),
            }
        )
    return pd.DataFrame(rows)


def physical_consistency_checks(
    model: RegressorMixin,
    x_ref_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    checks = {
        "macroporosity": "positive",
        "bulk_density": "negative",
        "sand": "positive",
        "clay": "negative",
    }
    available = [k for k in checks if k in x_ref_df.columns]
    x_mean = x_ref_df.mean(axis=0).to_numpy(dtype=float)

    rows: List[Dict[str, object]] = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.ravel()
    for i, feat in enumerate(available):
        idx = list(x_ref_df.columns).index(feat)
        values = np.linspace(
            float(np.min(x_ref_df[feat])),
            float(np.max(x_ref_df[feat])),
            80,
        )
        preds = []
        for v in values:
            x = x_mean.copy()
            x[idx] = v
            preds.append(float(model.predict([x])[0]))
        r, p = spearmanr(values, preds)
        expected = checks[feat]
        consistent = bool((r > 0.5 and expected == "positive") or (r < -0.5 and expected == "negative"))
        rows.append(
            {
                "feature": feat,
                "expected_direction": expected,
                "observed_spearman_r": float(r),
                "p_value": float(p),
                "physically_consistent": consistent,
            }
        )
        ax = axes_flat[i]
        ax.plot(values, preds, color="teal")
        ax.set_title(f"{feat} (r={r:+.2f})")
        ax.set_xlabel(feat)
        ax.set_ylabel("Predicao log(Ksat)")
        ax.grid(alpha=0.2)

    for j in range(len(available), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Checagem de consistencia fisica")
    fig.tight_layout()
    fig.savefig(out_dir / "physical_consistency_curves.png", dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    dirs = ensure_dirs(args.output_dir)

    # 1) Load and prepare dataset
    df_raw = load_dataset(args.data_path, sheet_name=args.sheet_name)
    dataset = prepare_dataset(
        df_raw=df_raw,
        include_silt=False,
        include_ratio=False,
        include_sinusoidal=False,
    )
    cols = ["x", "y", "sand", "clay", "bulk_density", "macroporosity", "ksat", "log_ksat"]
    df = dataset.df_model[cols].copy().reset_index(drop=True)
    if args.feature_set == "engineered":
        df = add_engineered_features(df)

    base_features = ["x", "y", "sand", "clay", "bulk_density", "macroporosity"]
    engineered_plus = base_features + [
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
    feature_cols = base_features if args.feature_set == "base" else engineered_plus

    # 2) Split holdout
    split = create_stratified_holdout_split(
        x=df[base_features].to_numpy(dtype=float),
        y=df["log_ksat"].to_numpy(dtype=float),
        random_state=args.seed,
        train_fraction=0.70,
        val_fraction=0.15,
        test_fraction=0.15,
    )
    idx_trainval = np.concatenate([split.idx_train, split.idx_val])
    train_df = df.iloc[idx_trainval].reset_index(drop=True)
    test_df = df.iloc[split.idx_test].reset_index(drop=True)

    x_train_df = train_df[feature_cols].copy()
    x_test_df = test_df[feature_cols].copy()
    y_train = train_df["log_ksat"].to_numpy(dtype=float)
    y_test = test_df["log_ksat"].to_numpy(dtype=float)

    # 3) Train chosen model
    params = load_reference_params(
        model_name=args.model,
        ceiling_summary_json=args.ceiling_summary_json,
        mlp_params_json=args.mlp_params_json,
    )
    model_factory = build_model_factory(
        model_name=args.model,
        params=params,
        seed=args.seed,
        max_iter_mlp=args.max_iter_mlp,
    )
    model = model_factory(args.seed)
    model.fit(x_train_df, y_train)
    y_pred = np.asarray(model.predict(x_test_df), dtype=float)

    if not np.all(np.isfinite(y_pred)):
        raise RuntimeError("Non-finite predictions in holdout set.")

    metrics_holdout = regression_metrics(y_test, y_pred)
    holdout_payload = {
        "model": args.model,
        "feature_set": args.feature_set,
        "n_trainval": int(len(train_df)),
        "n_test": int(len(test_df)),
        **{k: float(v) for k, v in metrics_holdout.items()},
    }
    save_json(args.output_dir / "holdout_metrics.json", holdout_payload)
    pd.DataFrame(
        {
            "x": test_df["x"],
            "y": test_df["y"],
            "log_ksat_true": y_test,
            "log_ksat_pred": y_pred,
            "residual_log": y_test - y_pred,
            "ksat_true": np.power(10.0, y_test),
            "ksat_pred": np.power(10.0, y_pred),
        }
    ).to_csv(args.output_dir / "holdout_predictions.csv", index=False)

    print(
        f"Holdout R2_log={metrics_holdout['r2_log']:.4f} RMSE_log={metrics_holdout['rmse_log']:.4f}"
    )

    # 4) Global explainability
    shap_values, explainer = shap_global(
        model=model,
        model_name=args.model,
        x_train_df=x_train_df,
        x_test_df=x_test_df,
        out_dir=dirs["global"],
        seed=args.seed,
    )
    run_permutation_importance(
        model=model,
        x_test_df=x_test_df,
        y_test=y_test,
        repeats=args.perm_repeats,
        seed=args.seed,
        out_dir=dirs["global"],
    )
    used_pdp_features = run_pdp_ice(model=model, x_train_df=x_train_df, out_dir=dirs["global"])
    interaction_payload = run_shap_interactions(
        model=model,
        model_name=args.model,
        explainer=explainer,
        shap_values=shap_values,
        x_test_df=x_test_df,
        out_dir=dirs["global"],
    )

    # 5) Local explainability
    local_df = run_local_shap_waterfalls(
        shap_values=shap_values,
        explainer=explainer,
        x_test_df=x_test_df,
        y_true=y_test,
        y_pred=y_pred,
        out_dir=dirs["local"],
    )
    run_lime_local(
        model=model,
        x_train_df=x_train_df,
        x_test_df=x_test_df,
        local_df=local_df,
        out_dir=dirs["local"],
        seed=args.seed,
    )
    worst_idx = int(local_df.sort_values("abs_error_log", ascending=False).iloc[0]["test_index"])
    run_counterfactual_simple(
        model=model,
        x_train_df=x_train_df,
        x_test_df=x_test_df,
        y_true=y_test,
        worst_idx=worst_idx,
        out_dir=dirs["local"],
    )
    run_neighbor_influence(
        x_train_df=x_train_df,
        y_train=y_train,
        x_test_df=x_test_df,
        y_test=y_test,
        y_pred=y_pred,
        local_df=local_df,
        out_dir=dirs["local"],
    )

    # 6) Spatial diagnostics on holdout
    holdout_res_df = pd.DataFrame(
        {
            "x": test_df["x"],
            "y": test_df["y"],
            "log_ksat": y_test,
            "log_ksat_pred": y_pred,
            "residual_log": y_test - y_pred,
            "abs_error_log": np.abs(y_test - y_pred),
        }
    )
    plot_residual_maps(holdout_res_df, dirs["spatial"], prefix="holdout")
    holdout_res_df.to_csv(dirs["spatial"] / "holdout_residuals.csv", index=False)

    # 7) Spatial LOO + buffer for all points
    spatial_cols = list(dict.fromkeys(["x", "y", "log_ksat"] + list(feature_cols)))
    sloo_df, sloo_summary = spatial_loo_with_buffer(
        df=df[spatial_cols].copy(),
        feature_cols=feature_cols,
        model_factory=model_factory,
        seed=args.seed,
        buffer_radius_m=args.sloo_buffer_m,
        min_train=args.sloo_min_train,
        max_points=args.sloo_max_points,
    )
    sloo_df.to_csv(dirs["spatial"] / "sloo_predictions.csv", index=False)
    save_json(dirs["spatial"] / "sloo_summary.json", sloo_summary)
    plot_residual_maps(sloo_df, dirs["spatial"], prefix="sloo")

    # 8) Residual variogram
    vario_df = experimental_variogram(
        coords=sloo_df[["x", "y"]].to_numpy(dtype=float),
        values=sloo_df["residual_log"].to_numpy(dtype=float),
        n_lags=12,
    )
    vario_df.to_csv(dirs["spatial"] / "residual_variogram_experimental.csv", index=False)
    if len(vario_df) >= 4:
        vario_params = fit_variogram_model(vario_df)
        save_json(dirs["spatial"] / "residual_variogram_fit.json", vario_params)
        save_variogram_plot(
            vario_df=vario_df,
            params=vario_params,
            out_path=dirs["spatial"] / "residual_variogram_plot.png",
        )
    else:
        vario_params = {"status": "insufficient_points"}
        save_json(dirs["spatial"] / "residual_variogram_fit.json", vario_params)

    # 9) Error-covariate correlation and quadrants (using SLOO residuals)
    x_source = df.iloc[: len(sloo_df)][feature_cols].reset_index(drop=True)
    corr_df = residual_correlation(
        df_res=sloo_df,
        feature_cols=feature_cols,
        x_source=x_source,
    )
    corr_df.to_csv(dirs["diagnostics"] / "error_covariate_spearman.csv", index=False)

    quad_df = quadrant_diagnostics(sloo_df)
    quad_df.to_csv(dirs["diagnostics"] / "error_by_quadrant.csv", index=False)

    physical_df = physical_consistency_checks(
        model=model,
        x_ref_df=x_train_df,
        out_dir=dirs["diagnostics"],
    )
    physical_df.to_csv(dirs["diagnostics"] / "physical_consistency_checks.csv", index=False)

    # 10) Global summary
    summary = {
        "config": {
            "seed": args.seed,
            "model": args.model,
            "feature_set": args.feature_set,
            "perm_repeats": args.perm_repeats,
            "sloo_buffer_m": args.sloo_buffer_m,
            "sloo_min_train": args.sloo_min_train,
            "sloo_max_points": args.sloo_max_points,
            "has_lime": HAS_LIME,
            "n_samples_total": int(len(df)),
            "n_trainval": int(len(train_df)),
            "n_test": int(len(test_df)),
            "feature_cols": feature_cols,
            "model_params": params,
        },
        "holdout_metrics": holdout_payload,
        "sloo_summary": sloo_summary,
        "used_pdp_features": used_pdp_features,
        "shap_interactions": interaction_payload,
    }
    save_json(args.output_dir / "explainability_summary.json", summary)

    print("\n=== Explainability pipeline completed ===")
    print(f"Output dir: {args.output_dir.resolve()}")
    print(f"Holdout R2_log: {holdout_payload['r2_log']:.4f}")
    print(f"SLOO R2_log: {sloo_summary['r2_log']:.4f}")
    if isinstance(vario_params, dict) and "gde_percent" in vario_params:
        print(
            "Residual variogram GDE: "
            f"{vario_params['gde_percent']:.1f}% | range={vario_params['range']:.1f} m"
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
