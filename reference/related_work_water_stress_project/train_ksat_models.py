#!/usr/bin/env python3
"""Train neural and benchmark regressors for Ksat prediction in BHSE."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None
    HAS_XGBOOST = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False


RANDOM_STATE_DEFAULT = 42
EPS = 1e-12


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    normalized = {col: normalize_text(col) for col in df.columns}
    mapping: Dict[str, str] = {}

    for col, norm in normalized.items():
        if norm in {"x", "coord x", "coordenada x", "utm x", "x utm"}:
            mapping.setdefault("x", col)
        elif norm in {"y", "coord y", "coordenada y", "utm y", "y utm"}:
            mapping.setdefault("y", col)
        elif "areia" in norm:
            mapping.setdefault("sand", col)
        elif "silte" in norm:
            mapping.setdefault("silt", col)
        elif "argila" in norm:
            mapping.setdefault("clay", col)
        elif norm.startswith("ds ") or norm == "ds" or "densidade" in norm:
            mapping.setdefault("bulk_density", col)
        elif "macroporosidade" in norm or "macro porosidade" in norm:
            mapping.setdefault("macroporosity", col)
        elif "ksat" in norm:
            mapping.setdefault("ksat", col)

    missing = [
        key
        for key in ["x", "y", "sand", "silt", "clay", "bulk_density", "macroporosity", "ksat"]
        if key not in mapping
    ]
    if missing:
        raise ValueError(
            f"Could not resolve expected columns: {missing}. Available columns: {list(df.columns)}"
        )
    return mapping


def quartile_strata(y: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(y, [0.0, 0.25, 0.5, 0.75, 1.0])
    quantiles = np.unique(quantiles)
    if quantiles.size <= 2:
        return np.zeros_like(y, dtype=int)
    return np.digitize(y, quantiles[1:-1], right=True).astype(int)


def compute_spatial_blocks(
    x: np.ndarray,
    y: np.ndarray,
    blocks_x: int,
    blocks_y: int,
) -> np.ndarray:
    if blocks_x < 1 or blocks_y < 1:
        raise ValueError("blocks_x and blocks_y must be >= 1.")
    x_edges = np.linspace(np.min(x), np.max(x), blocks_x + 1)
    y_edges = np.linspace(np.min(y), np.max(y), blocks_y + 1)

    x_bin = np.digitize(x, x_edges[1:-1], right=False)
    y_bin = np.digitize(y, y_edges[1:-1], right=False)
    return (x_bin * blocks_y + y_bin).astype(int)


def pbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(y_true))
    if math.isclose(denom, 0.0, abs_tol=EPS):
        return float("nan")
    return 100.0 * float(np.sum(y_pred - y_true)) / denom


def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    y_true_orig = np.power(10.0, y_true_log)
    y_pred_orig = np.power(10.0, y_pred_log)

    metrics = {
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "pbias_log": float(pbias(y_true_log, y_pred_log)),
        "r2_orig": float(r2_score(y_true_orig, y_pred_orig)),
        "rmse_orig": float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))),
        "mae_orig": float(mean_absolute_error(y_true_orig, y_pred_orig)),
        "pbias_orig": float(pbias(y_true_orig, y_pred_orig)),
    }
    return metrics


class AdaptiveRBFRegressor(BaseEstimator, RegressorMixin):
    """RBF features + ridge regressor with optional adaptive gamma."""

    def __init__(
        self,
        n_components: int = 600,
        gamma: Optional[float] = None,
        alpha: float = 1.0,
        random_state: int = RANDOM_STATE_DEFAULT,
        max_points_gamma: int = 128,
    ) -> None:
        self.n_components = n_components
        self.gamma = gamma
        self.alpha = alpha
        self.random_state = random_state
        self.max_points_gamma = max_points_gamma

    def _estimate_gamma(self, x: np.ndarray) -> float:
        rng = np.random.default_rng(self.random_state)
        n_points = x.shape[0]
        subset_size = min(n_points, self.max_points_gamma)
        subset_idx = rng.choice(n_points, size=subset_size, replace=False)
        subset = x[subset_idx]

        gram = np.dot(subset, subset.T)
        sq_norm = np.diag(gram)
        sq_dist = sq_norm[:, None] + sq_norm[None, :] - 2.0 * gram
        sq_dist = sq_dist[np.triu_indices_from(sq_dist, k=1)]
        sq_dist = sq_dist[sq_dist > 0.0]

        if sq_dist.size == 0:
            return 1.0 / max(1.0, x.shape[1])
        median_sq_dist = float(np.median(sq_dist))
        return 1.0 / (2.0 * median_sq_dist)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "AdaptiveRBFRegressor":
        gamma = self.gamma if self.gamma is not None else self._estimate_gamma(x)
        self.gamma_ = float(gamma)

        self.rbf_ = RBFSampler(
            gamma=self.gamma_,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        z = self.rbf_.fit_transform(x)

        self.regressor_ = Ridge(alpha=self.alpha)
        self.regressor_.fit(z, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.rbf_.transform(x)
        return self.regressor_.predict(z)


@dataclass
class DatasetBundle:
    df_raw: pd.DataFrame
    df_model: pd.DataFrame
    feature_names: List[str]
    target_name: str
    column_mapping: Dict[str, str]


def load_dataset(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_excel(path, sheet_name=sheet_name)


def prepare_dataset(
    df_raw: pd.DataFrame,
    include_silt: bool,
    include_ratio: bool,
    include_sinusoidal: bool,
) -> DatasetBundle:
    column_mapping = resolve_columns(df_raw)
    df_model = pd.DataFrame(
        {
            "x": pd.to_numeric(df_raw[column_mapping["x"]], errors="coerce"),
            "y": pd.to_numeric(df_raw[column_mapping["y"]], errors="coerce"),
            "sand": pd.to_numeric(df_raw[column_mapping["sand"]], errors="coerce"),
            "silt": pd.to_numeric(df_raw[column_mapping["silt"]], errors="coerce"),
            "clay": pd.to_numeric(df_raw[column_mapping["clay"]], errors="coerce"),
            "bulk_density": pd.to_numeric(df_raw[column_mapping["bulk_density"]], errors="coerce"),
            "macroporosity": pd.to_numeric(df_raw[column_mapping["macroporosity"]], errors="coerce"),
            "ksat": pd.to_numeric(df_raw[column_mapping["ksat"]], errors="coerce"),
        }
    )

    before_drop = len(df_model)
    df_model = df_model.dropna().copy()
    dropped = before_drop - len(df_model)
    if dropped > 0:
        warnings.warn(f"Dropped {dropped} rows with missing values.")

    if (df_model["ksat"] <= 0).any():
        n_non_positive = int((df_model["ksat"] <= 0).sum())
        raise ValueError(f"Ksat contains {n_non_positive} non-positive values; log10 undefined.")

    df_model["log_ksat"] = np.log10(df_model["ksat"])

    feature_names = ["x", "y", "sand", "clay", "bulk_density", "macroporosity"]
    if include_silt:
        feature_names.append("silt")

    if include_ratio:
        df_model["sand_clay_ratio"] = df_model["sand"] / (df_model["clay"] + EPS)
        feature_names.append("sand_clay_ratio")

    if include_sinusoidal:
        x_min, x_max = float(df_model["x"].min()), float(df_model["x"].max())
        y_min, y_max = float(df_model["y"].min()), float(df_model["y"].max())
        x_norm = (df_model["x"] - x_min) / (x_max - x_min + EPS)
        y_norm = (df_model["y"] - y_min) / (y_max - y_min + EPS)
        df_model["sin_x"] = np.sin(2.0 * np.pi * x_norm)
        df_model["cos_x"] = np.cos(2.0 * np.pi * x_norm)
        df_model["sin_y"] = np.sin(2.0 * np.pi * y_norm)
        df_model["cos_y"] = np.cos(2.0 * np.pi * y_norm)
        feature_names.extend(["sin_x", "cos_x", "sin_y", "cos_y"])

    return DatasetBundle(
        df_raw=df_raw,
        df_model=df_model,
        feature_names=feature_names,
        target_name="log_ksat",
        column_mapping=column_mapping,
    )


@dataclass
class HoldoutSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray


def create_stratified_holdout_split(
    x: np.ndarray,
    y: np.ndarray,
    random_state: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> HoldoutSplit:
    total = train_fraction + val_fraction + test_fraction
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train_fraction + val_fraction + test_fraction must be 1.0.")

    idx = np.arange(len(y))
    strata = quartile_strata(y)
    stratify_main = strata if np.unique(strata).size > 1 else None

    x_train_val, x_test, y_train_val, y_test, idx_train_val, idx_test, strata_train_val, _ = train_test_split(
        x,
        y,
        idx,
        strata,
        test_size=test_fraction,
        random_state=random_state,
        stratify=stratify_main,
    )

    val_within_train_val = val_fraction / (train_fraction + val_fraction)
    stratify_secondary = strata_train_val if np.unique(strata_train_val).size > 1 else None

    x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(
        x_train_val,
        y_train_val,
        idx_train_val,
        test_size=val_within_train_val,
        random_state=random_state,
        stratify=stratify_secondary,
    )

    return HoldoutSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )


def build_model_factories(
    model_names: Sequence[str],
    seed: int,
    n_features: int,
    mlp_hidden: Sequence[int],
    mlp_lr: float,
    mlp_alpha: float,
    mlp_max_iter: int,
    rbf_components: int,
    rbf_gamma: Optional[float],
    rbf_alpha: float,
    rf_estimators: int,
    use_gpu: bool,
) -> Dict[str, Callable[[], object]]:
    factories: Dict[str, Callable[[], object]] = {}

    if "mlp" in model_names:
        factories["MLP"] = lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=tuple(mlp_hidden),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=mlp_lr,
                        alpha=mlp_alpha,
                        max_iter=mlp_max_iter,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if "rbf" in model_names:
        factories["RBFNN"] = lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    AdaptiveRBFRegressor(
                        n_components=rbf_components,
                        gamma=rbf_gamma,
                        alpha=rbf_alpha,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if "rf" in model_names:
        factories["RandomForest"] = lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=rf_estimators,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if "xgb" in model_names:
        if not HAS_XGBOOST:
            warnings.warn("xgboost not installed; skipping XGBoost model.")
        else:
            xgb_factory = _xgboost_factory(seed=seed, use_gpu=use_gpu)
            factories["XGBoost"] = lambda: Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", xgb_factory()),
                ]
            )

    if not factories:
        raise ValueError("No valid model selected.")
    return factories


def _xgboost_factory(seed: int, use_gpu: bool) -> Callable[[], object]:
    base_params = {
        "objective": "reg:squarederror",
        "n_estimators": 800,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "random_state": seed,
        "tree_method": "hist",
    }

    if use_gpu:
        base_params["device"] = "cuda"

    def factory() -> object:
        try:
            return XGBRegressor(**base_params)
        except TypeError:
            fallback = dict(base_params)
            fallback.pop("device", None)
            if use_gpu:
                fallback["tree_method"] = "gpu_hist"
            return XGBRegressor(**fallback)

    return factory


def evaluate_holdout(
    model_factories: Dict[str, Callable[[], object]],
    split: HoldoutSplit,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, Dict[str, np.ndarray]]]:
    rows = []
    fitted_models: Dict[str, object] = {}
    predictions: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name, factory in model_factories.items():
        model = factory()
        model.fit(split.x_train, split.y_train)
        fitted_models[model_name] = model
        predictions[model_name] = {}

        for split_name, x_data, y_data in [
            ("train", split.x_train, split.y_train),
            ("val", split.x_val, split.y_val),
            ("test", split.x_test, split.y_test),
        ]:
            y_pred = model.predict(x_data)
            predictions[model_name][split_name] = y_pred
            metric_row = regression_metrics(y_data, y_pred)
            metric_row.update(
                {
                    "evaluation": "holdout",
                    "model": model_name,
                    "split": split_name,
                    "n_samples": int(len(y_data)),
                }
            )
            rows.append(metric_row)

    return pd.DataFrame(rows), fitted_models, predictions


def evaluate_spatial_cv(
    model_factories: Dict[str, Callable[[], object]],
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, unique_groups.size)
    if n_splits < 2:
        raise ValueError(
            "Not enough unique spatial groups for CV. Increase number of blocks or reduce n_splits."
        )

    splitter = GroupKFold(n_splits=n_splits)
    fold_rows = []

    for model_name, factory in model_factories.items():
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups=groups), start=1):
            model = factory()
            model.fit(x[train_idx], y[train_idx])
            y_pred = model.predict(x[test_idx])
            metric_row = regression_metrics(y[test_idx], y_pred)
            metric_row.update(
                {
                    "evaluation": "spatial_cv",
                    "model": model_name,
                    "fold": int(fold),
                    "n_samples": int(len(test_idx)),
                    "n_train_groups": int(np.unique(groups[train_idx]).size),
                    "n_test_groups": int(np.unique(groups[test_idx]).size),
                }
            )
            fold_rows.append(metric_row)

    fold_df = pd.DataFrame(fold_rows)
    summary = (
        fold_df.groupby("model")
        .agg(
            r2_log_mean=("r2_log", "mean"),
            r2_log_std=("r2_log", "std"),
            rmse_log_mean=("rmse_log", "mean"),
            rmse_log_std=("rmse_log", "std"),
            mae_log_mean=("mae_log", "mean"),
            mae_log_std=("mae_log", "std"),
            r2_orig_mean=("r2_orig", "mean"),
            r2_orig_std=("r2_orig", "std"),
            rmse_orig_mean=("rmse_orig", "mean"),
            rmse_orig_std=("rmse_orig", "std"),
            mae_orig_mean=("mae_orig", "mean"),
            mae_orig_std=("mae_orig", "std"),
            pbias_orig_mean=("pbias_orig", "mean"),
            pbias_orig_std=("pbias_orig", "std"),
        )
        .reset_index()
    )
    return fold_df, summary


def build_test_predictions_table(
    df_model: pd.DataFrame,
    split: HoldoutSplit,
    predictions: Dict[str, Dict[str, np.ndarray]],
) -> pd.DataFrame:
    test_df = df_model.iloc[split.idx_test][["x", "y", "ksat", "log_ksat"]].copy()
    test_df = test_df.rename(columns={"ksat": "ksat_true", "log_ksat": "log_ksat_true"})
    test_df["sample_idx"] = split.idx_test
    test_df = test_df.reset_index(drop=True)

    for model_name, split_preds in predictions.items():
        pred_log = split_preds["test"]
        test_df[f"log_ksat_pred_{model_name}"] = pred_log
        test_df[f"ksat_pred_{model_name}"] = np.power(10.0, pred_log)
        test_df[f"residual_log_{model_name}"] = pred_log - test_df["log_ksat_true"].values
        test_df[f"residual_orig_{model_name}"] = (
            test_df[f"ksat_pred_{model_name}"] - test_df["ksat_true"]
        )

    return test_df


def save_permutation_importance(
    model_name: str,
    model: object,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    output_dir: Path,
    seed: int,
) -> Optional[pd.DataFrame]:
    try:
        result = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=40,
            random_state=seed,
            scoring="r2",
            n_jobs=-1,
        )
    except Exception as exc:
        warnings.warn(f"Permutation importance failed for {model_name}: {exc}")
        return None

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(output_dir / f"permutation_importance_{model_name}.csv", index=False)
    return importance_df


def maybe_plot_outputs(
    holdout_metrics: pd.DataFrame,
    test_predictions: pd.DataFrame,
    blocks: np.ndarray,
    output_dir: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not installed; skipping plots.")
        return

    for model_name in sorted(
        {
            col.replace("ksat_pred_", "")
            for col in test_predictions.columns
            if col.startswith("ksat_pred_")
        }
    ):
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        x = test_predictions["ksat_true"].values
        y = test_predictions[f"ksat_pred_{model_name}"].values
        ax.scatter(x, y, alpha=0.8, edgecolor="black", linewidth=0.4)
        xy_min = min(np.min(x), np.min(y))
        xy_max = max(np.max(x), np.max(y))
        ax.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--")
        ax.set_xlabel("Ksat observado (cm/dia)")
        ax.set_ylabel("Ksat previsto (cm/dia)")
        ax.set_title(f"Teste Holdout - {model_name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"scatter_observado_vs_previsto_{model_name}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    subset = holdout_metrics[holdout_metrics["split"] == "test"].copy()
    subset = subset.sort_values("r2_log", ascending=False)
    ax.bar(subset["model"], subset["r2_log"])
    ax.set_ylabel("R2 no espaco log10(Ksat)")
    ax.set_title("Comparacao Holdout - R2 log (teste)")
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_r2_log_comparison.png", dpi=180)
    plt.close(fig)

    spatial_df = test_predictions[["x", "y"]].copy()
    spatial_df["block_id"] = blocks[test_predictions["sample_idx"].values]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    scatter = ax.scatter(spatial_df["x"], spatial_df["y"], c=spatial_df["block_id"], s=40)
    ax.set_xlabel("X (UTM)")
    ax.set_ylabel("Y (UTM)")
    ax.set_title("Blocos espaciais (amostras de teste)")
    fig.colorbar(scatter, ax=ax, label="Bloco")
    fig.tight_layout()
    fig.savefig(output_dir / "spatial_blocks_test.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ksat prediction in BHSE using MLP baseline + spatially aware evaluation."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("Pontos_Ksat_Sanga_Ellert.xlsx"),
        help="Path to source Excel file.",
    )
    parser.add_argument(
        "--sheet-name",
        default=0,
        help="Sheet name or index in the Excel file (default: first sheet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for metrics, predictions and figures.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE_DEFAULT)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--spatial-folds", type=int, default=5)
    parser.add_argument("--blocks-x", type=int, default=4)
    parser.add_argument("--blocks-y", type=int, default=4)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "rbf", "rf", "xgb"],
        choices=["mlp", "rbf", "rf", "xgb"],
        help="Models to train and compare.",
    )
    parser.add_argument("--include-silt", action="store_true")
    parser.add_argument("--include-ratio", action="store_true")
    parser.add_argument("--include-sinusoidal", action="store_true")
    parser.add_argument("--mlp-hidden", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    parser.add_argument("--mlp-max-iter", type=int, default=3000)
    parser.add_argument("--rbf-components", type=int, default=900)
    parser.add_argument("--rbf-gamma", type=float, default=None)
    parser.add_argument("--rbf-alpha", type=float, default=1.0)
    parser.add_argument("--rf-estimators", type=int, default=700)
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU for XGBoost when available (MLP/RBF/RF remain on CPU).",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_gpu and "xgb" not in args.models:
        warnings.warn("--use-gpu was set, but xgb is not selected.")

    df_raw = load_dataset(args.data_path, sheet_name=args.sheet_name)
    dataset = prepare_dataset(
        df_raw=df_raw,
        include_silt=args.include_silt,
        include_ratio=args.include_ratio,
        include_sinusoidal=args.include_sinusoidal,
    )

    x_all = dataset.df_model[dataset.feature_names].to_numpy(dtype=float)
    y_all = dataset.df_model[dataset.target_name].to_numpy(dtype=float)

    split = create_stratified_holdout_split(
        x=x_all,
        y=y_all,
        random_state=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    model_factories = build_model_factories(
        model_names=args.models,
        seed=args.seed,
        n_features=len(dataset.feature_names),
        mlp_hidden=args.mlp_hidden,
        mlp_lr=args.mlp_lr,
        mlp_alpha=args.mlp_alpha,
        mlp_max_iter=args.mlp_max_iter,
        rbf_components=args.rbf_components,
        rbf_gamma=args.rbf_gamma,
        rbf_alpha=args.rbf_alpha,
        rf_estimators=args.rf_estimators,
        use_gpu=args.use_gpu,
    )

    holdout_metrics, fitted_models, holdout_predictions = evaluate_holdout(
        model_factories=model_factories,
        split=split,
    )
    holdout_metrics = holdout_metrics.sort_values(
        by=["split", "r2_log"], ascending=[True, False]
    ).reset_index(drop=True)

    spatial_groups = compute_spatial_blocks(
        x=dataset.df_model["x"].to_numpy(),
        y=dataset.df_model["y"].to_numpy(),
        blocks_x=args.blocks_x,
        blocks_y=args.blocks_y,
    )
    spatial_folds, spatial_summary = evaluate_spatial_cv(
        model_factories=model_factories,
        x=x_all,
        y=y_all,
        groups=spatial_groups,
        n_splits=args.spatial_folds,
    )

    test_predictions = build_test_predictions_table(
        df_model=dataset.df_model,
        split=split,
        predictions=holdout_predictions,
    )

    holdout_metrics.to_csv(args.output_dir / "metrics_holdout.csv", index=False)
    spatial_folds.to_csv(args.output_dir / "metrics_spatial_cv_folds.csv", index=False)
    spatial_summary.to_csv(args.output_dir / "metrics_spatial_cv_summary.csv", index=False)
    test_predictions.to_csv(args.output_dir / "predictions_test.csv", index=False)

    block_table = dataset.df_model[["x", "y"]].copy()
    block_table["block_id"] = spatial_groups
    block_table.to_csv(args.output_dir / "spatial_blocks_all_points.csv", index=False)

    if "MLP" in fitted_models:
        save_permutation_importance(
            model_name="MLP",
            model=fitted_models["MLP"],
            x_test=split.x_test,
            y_test=split.y_test,
            feature_names=dataset.feature_names,
            output_dir=args.output_dir,
            seed=args.seed,
        )

    if not args.no_plots:
        maybe_plot_outputs(
            holdout_metrics=holdout_metrics,
            test_predictions=test_predictions,
            blocks=spatial_groups,
            output_dir=args.output_dir,
        )

    test_subset = holdout_metrics[holdout_metrics["split"] == "test"].copy()
    test_subset = test_subset.sort_values("r2_log", ascending=False).reset_index(drop=True)
    best_model = test_subset.iloc[0]["model"] if not test_subset.empty else "n/a"

    print("\n=== Ksat modeling run completed ===")
    print(f"Rows used: {len(dataset.df_model)}")
    print(f"Features used ({len(dataset.feature_names)}): {dataset.feature_names}")
    print(f"Models evaluated: {list(model_factories.keys())}")
    print(f"Best holdout model (R2 log on test): {best_model}")
    print(f"Output directory: {args.output_dir.resolve()}")

    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        config_dict = {
            "data_path": str(args.data_path),
            "sheet_name": args.sheet_name,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "spatial_folds": args.spatial_folds,
            "blocks_x": args.blocks_x,
            "blocks_y": args.blocks_y,
            "feature_names": dataset.feature_names,
            "column_mapping": dataset.column_mapping,
            "models": list(model_factories.keys()),
        }
        json.dump(config_dict, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
