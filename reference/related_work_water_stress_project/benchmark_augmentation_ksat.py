#!/usr/bin/env python3
"""Benchmark augmentation strategies for Ksat prediction in BHSE."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
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
    from pykrige.ok import OrdinaryKriging

    HAS_PYKRIGE = True
except Exception:
    OrdinaryKriging = None
    HAS_PYKRIGE = False

try:
    import smogn

    HAS_SMOGN = True
except Exception:
    smogn = None
    HAS_SMOGN = False

try:
    from copulas.multivariate import GaussianMultivariate

    HAS_COPULAS = True
except Exception:
    GaussianMultivariate = None
    HAS_COPULAS = False


BASELINE_TARGET_R2_LOG = 0.7502
EPS = 1e-12


@dataclass
class MethodResult:
    method: str
    family: str
    status: str
    n_train_samples: int
    r2_log: float
    rmse_log: float
    mae_log: float
    pbias_log: float
    r2_orig: float
    rmse_orig: float
    mae_orig: float
    pbias_orig: float
    beat_baseline_07502: bool
    details: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test multiple augmentation strategies for BHSE Ksat prediction."
    )
    parser.add_argument("--data-path", type=Path, default=Path("Pontos_Ksat_Sanga_Ellert.xlsx"))
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_augmentation_benchmark"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE_DEFAULT)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument(
        "--baseline-params-json",
        type=Path,
        default=Path("outputs_mlp_tuning_optuna/mlp_best_params.json"),
    )
    return parser.parse_args()


def clip_physical_limits(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sand" in out.columns:
        out["sand"] = out["sand"].clip(0.0, 100.0)
    if "clay" in out.columns:
        out["clay"] = out["clay"].clip(0.0, 100.0)
    if "bulk_density" in out.columns:
        out["bulk_density"] = out["bulk_density"].clip(0.8, 1.9)
    if "macroporosity" in out.columns:
        out["macroporosity"] = out["macroporosity"].clip(0.0, 1.0)
    return out


def load_baseline_mlp_params(path: Path) -> Dict[str, object]:
    default_params = {
        "hidden_layer_sizes": (128, 64),
        "activation": "tanh",
        "alpha": 0.0011380907649808582,
        "learning_rate_init": 0.0024792013326598596,
    }
    if not path.exists():
        return default_params

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        params = data.get("best_params", {})
        hidden = params.get("hidden_layer_sizes", default_params["hidden_layer_sizes"])
        if isinstance(hidden, list):
            hidden = tuple(int(v) for v in hidden)
        return {
            "hidden_layer_sizes": hidden,
            "activation": str(params.get("activation", default_params["activation"])),
            "alpha": float(params.get("alpha", default_params["alpha"])),
            "learning_rate_init": float(
                params.get("learning_rate_init", default_params["learning_rate_init"])
            ),
        }
    except Exception:
        warnings.warn("Failed to read baseline params JSON; using defaults.")
        return default_params


def build_mlp_pipeline(
    params: Dict[str, object],
    seed: int,
    max_iter: int,
) -> Pipeline:
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


def train_eval_single(
    method: str,
    family: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
    mlp_params: Dict[str, object],
    seed: int,
    max_iter: int,
    details: str,
) -> MethodResult:
    x_train = train_df[feature_names].to_numpy(dtype=float)
    y_train = train_df["log_ksat"].to_numpy(dtype=float)
    x_test = test_df[feature_names].to_numpy(dtype=float)
    y_test = test_df["log_ksat"].to_numpy(dtype=float)

    model = build_mlp_pipeline(params=mlp_params, seed=seed, max_iter=max_iter)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    metrics = regression_metrics(y_test, y_pred)
    return MethodResult(
        method=method,
        family=family,
        status="ok",
        n_train_samples=int(len(train_df)),
        r2_log=float(metrics["r2_log"]),
        rmse_log=float(metrics["rmse_log"]),
        mae_log=float(metrics["mae_log"]),
        pbias_log=float(metrics["pbias_log"]),
        r2_orig=float(metrics["r2_orig"]),
        rmse_orig=float(metrics["rmse_orig"]),
        mae_orig=float(metrics["mae_orig"]),
        pbias_orig=float(metrics["pbias_orig"]),
        beat_baseline_07502=bool(metrics["r2_log"] > BASELINE_TARGET_R2_LOG),
        details=details,
    )


def method_baseline(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    context: Dict[str, object],
) -> Tuple[pd.DataFrame, str]:
    _ = rng
    _ = context
    return train_df.copy(), "No augmentation; train on real train+val only."


def method_jitter_gaussian(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    context: Dict[str, object],
) -> Tuple[pd.DataFrame, str]:
    _ = context
    copies = [train_df.copy()]
    noise_fracs = {
        "sand": 0.02,
        "clay": 0.02,
        "bulk_density": 0.015,
        "macroporosity": 0.03,
    }
    for _ in range(3):
        df_aug = train_df.copy()
        for col, frac in noise_fracs.items():
            sigma = float(train_df[col].std(ddof=0)) * frac
            df_aug[col] += rng.normal(0.0, sigma, size=len(df_aug))
        copies.append(clip_physical_limits(df_aug))
    return pd.concat(copies, ignore_index=True), "3x Gaussian jitter on physical covariates."


def method_coordinate_jitter(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    context: Dict[str, object],
) -> Tuple[pd.DataFrame, str]:
    min_x = float(context["min_x"])
    max_x = float(context["max_x"])
    min_y = float(context["min_y"])
    max_y = float(context["max_y"])

    copies = [train_df.copy()]
    radius = 10.0
    for _ in range(2):
        df_aug = train_df.copy()
        angles = rng.uniform(0.0, 2.0 * np.pi, size=len(df_aug))
        distances = rng.uniform(0.0, radius, size=len(df_aug))
        df_aug["x"] += distances * np.cos(angles)
        df_aug["y"] += distances * np.sin(angles)
        df_aug["x"] = df_aug["x"].clip(min_x, max_x)
        df_aug["y"] = df_aug["y"].clip(min_y, max_y)
        copies.append(df_aug)
    return pd.concat(copies, ignore_index=True), "2x coordinate jitter (radius <= 10m)."


def method_ksat_multiplicative_noise(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    context: Dict[str, object],
) -> Tuple[pd.DataFrame, str]:
    _ = context
    sigma_log10 = 0.065
    copies = [train_df.copy()]
    for _ in range(2):
        df_aug = train_df.copy()
        df_aug["log_ksat"] += rng.normal(0.0, sigma_log10, size=len(df_aug))
        copies.append(df_aug)
    return pd.concat(copies, ignore_index=True), "2x multiplicative target noise (~15%)."


def method_combined_noise(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    context: Dict[str, object],
) -> Tuple[pd.DataFrame, str]:
    min_x = float(context["min_x"])
    max_x = float(context["max_x"])
    min_y = float(context["min_y"])
    max_y = float(context["max_y"])

    copies = [train_df.copy()]
    for _ in range(3):
        df_aug = train_df.copy()
        df_aug["sand"] += rng.normal(0.0, 0.02 * train_df["sand"].std(ddof=0), size=len(df_aug))
        df_aug["clay"] += rng.normal(0.0, 0.02 * train_df["clay"].std(ddof=0), size=len(df_aug))
        df_aug["bulk_density"] += rng.normal(
            0.0, 0.015 * train_df["bulk_density"].std(ddof=0), size=len(df_aug)
        )
        df_aug["macroporosity"] += rng.normal(
            0.0, 0.03 * train_df["macroporosity"].std(ddof=0), size=len(df_aug)
        )

        angles = rng.uniform(0.0, 2.0 * np.pi, size=len(df_aug))
        distances = rng.uniform(0.0, 8.0, size=len(df_aug))
        df_aug["x"] += distances * np.cos(angles)
        df_aug["y"] += distances * np.sin(angles)
        df_aug["x"] = df_aug["x"].clip(min_x, max_x)
        df_aug["y"] = df_aug["y"].clip(min_y, max_y)

        df_aug["log_ksat"] += rng.normal(0.0, 0.05, size=len(df_aug))
        copies.append(clip_physical_limits(df_aug))

    return pd.concat(copies, ignore_index=True), "Combined covariate + coordinate + target noise."


def method_monte_carlo_measurement(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    context: Dict[str, object],
) -> Tuple[pd.DataFrame, str]:
    _ = context
    n_samples = len(train_df) * 5
    idx = rng.integers(0, len(train_df), size=n_samples)
    df_aug = train_df.iloc[idx].reset_index(drop=True)

    df_aug["sand"] += rng.normal(0.0, 1.5, size=n_samples)
    df_aug["clay"] += rng.normal(0.0, 1.5, size=n_samples)
    df_aug["bulk_density"] += rng.normal(0.0, 0.03, size=n_samples)
    df_aug["macroporosity"] += rng.normal(0.0, 0.015, size=n_samples)
    df_aug["log_ksat"] += rng.normal(0.0, 0.065, size=n_samples)
    df_aug = clip_physical_limits(df_aug)

    return (
        pd.concat([train_df, df_aug], ignore_index=True),
        "Monte Carlo perturbation from measurement uncertainty.",
    )


def method_idw_rbf_linear(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    context: Dict[str, object],
) -> Tuple[pd.DataFrame, str]:
    min_x = float(context["min_x"])
    max_x = float(context["max_x"])
    min_y = float(context["min_y"])
    max_y = float(context["max_y"])

    coords = train_df[["x", "y"]].to_numpy(dtype=float)
    n_new = 700
    x_new = rng.uniform(min_x, max_x, size=n_new)
    y_new = rng.uniform(min_y, max_y, size=n_new)
    new_coords = np.column_stack([x_new, y_new])

    synth = {"x": x_new, "y": y_new}
    for col in ["sand", "clay", "bulk_density", "macroporosity", "log_ksat"]:
        interp = RBFInterpolator(coords, train_df[col].to_numpy(dtype=float), kernel="linear", epsilon=50.0)
        synth[col] = interp(new_coords)

    df_syn = clip_physical_limits(pd.DataFrame(synth))
    return (
        pd.concat([train_df, df_syn], ignore_index=True),
        "IDW-like interpolation via RBF linear kernel for features and log_ksat.",
    )


def method_thin_plate_spline(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    context: Dict[str, object],
) -> Tuple[pd.DataFrame, str]:
    min_x = float(context["min_x"])
    max_x = float(context["max_x"])
    min_y = float(context["min_y"])
    max_y = float(context["max_y"])

    coords = train_df[["x", "y"]].to_numpy(dtype=float)
    n_new = 700
    x_new = rng.uniform(min_x, max_x, size=n_new)
    y_new = rng.uniform(min_y, max_y, size=n_new)
    new_coords = np.column_stack([x_new, y_new])

    synth = {"x": x_new, "y": y_new}
    for col in ["sand", "clay", "bulk_density", "macroporosity", "log_ksat"]:
        interp = RBFInterpolator(coords, train_df[col].to_numpy(dtype=float), kernel="thin_plate_spline")
        synth[col] = interp(new_coords)

    df_syn = clip_physical_limits(pd.DataFrame(synth))
    return (
        pd.concat([train_df, df_syn], ignore_index=True),
        "Thin plate spline interpolation for features and log_ksat.",
    )


def method_kriging_covariate_composition(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    context: Dict[str, object],
) -> Tuple[pd.DataFrame, str]:
    if not HAS_PYKRIGE:
        raise RuntimeError("pykrige is not installed.")

    min_x = float(context["min_x"])
    max_x = float(context["max_x"])
    min_y = float(context["min_y"])
    max_y = float(context["max_y"])

    x_new = rng.uniform(min_x, max_x, size=550)
    y_new = rng.uniform(min_y, max_y, size=550)
    synth = {"x": x_new, "y": y_new}

    for col in ["sand", "clay", "bulk_density", "macroporosity", "log_ksat"]:
        ok = OrdinaryKriging(
            train_df["x"].to_numpy(dtype=float),
            train_df["y"].to_numpy(dtype=float),
            train_df[col].to_numpy(dtype=float),
            variogram_model="exponential",
            verbose=False,
            enable_plotting=False,
        )
        pred, _ = ok.execute("points", x_new, y_new)
        synth[col] = np.asarray(pred).astype(float)

    df_syn = clip_physical_limits(pd.DataFrame(synth))
    return (
        pd.concat([train_df, df_syn], ignore_index=True),
        "Ordinary kriging per covariate + log_ksat pseudo-target.",
    )


def method_smogn(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    context: Dict[str, object],
) -> Tuple[pd.DataFrame, str]:
    _ = context
    feature_cols = ["x", "y", "sand", "clay", "bulk_density", "macroporosity"]

    if HAS_SMOGN:
        try:
            df_aug = smogn.smoter(
                data=train_df.copy(),
                y="log_ksat",
                samp_method="balance",
                rel_thres=0.8,
                rel_method="auto",
            )
            df_aug = clip_physical_limits(df_aug)
            return df_aug, "SMOGN oversampling for regression target tails."
        except Exception:
            warnings.warn("smogn failed with current pandas; using fallback SMOGN-like augmentation.")

    y = train_df["log_ksat"].to_numpy(dtype=float)
    q_low = float(np.quantile(y, 0.10))
    q_high = float(np.quantile(y, 0.90))
    rare_mask = (y <= q_low) | (y >= q_high)
    rare_df = train_df.loc[rare_mask].reset_index(drop=True)
    common_df = train_df.loc[~rare_mask].reset_index(drop=True)

    if len(rare_df) < 6:
        return train_df.copy(), "SMOGN-like fallback skipped (few rare samples)."

    n_new = max(len(common_df) - len(rare_df), len(rare_df))
    k = min(7, len(rare_df))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(rare_df[feature_cols].to_numpy(dtype=float))
    neigh_idx = nn.kneighbors(return_distance=False)

    synthetic_rows = []
    for _ in range(n_new):
        i = int(rng.integers(0, len(rare_df)))
        neighbors = neigh_idx[i]
        if neighbors.size <= 1:
            j = i
        else:
            j = int(rng.choice(neighbors[1:]))
        a = rare_df.iloc[i]
        b = rare_df.iloc[j]
        lam = float(rng.uniform(0.0, 1.0))

        row = {}
        for col in feature_cols + ["log_ksat"]:
            row[col] = float(a[col] + lam * (b[col] - a[col]))
        row["log_ksat"] += float(rng.normal(0.0, 0.03))
        synthetic_rows.append(row)

    df_syn = pd.DataFrame(synthetic_rows)
    df_syn = clip_physical_limits(df_syn)
    df_aug = pd.concat([train_df, df_syn], ignore_index=True)
    return df_aug, "SMOGN-like fallback (extreme-target interpolation among rare neighbors)."


def method_copula(
    train_df: pd.DataFrame,
    rng: np.random.Generator,
    context: Dict[str, object],
) -> Tuple[pd.DataFrame, str]:
    _ = rng
    _ = context
    if not HAS_COPULAS:
        raise RuntimeError("copulas is not installed.")

    model = GaussianMultivariate()
    model.fit(train_df)
    df_syn = model.sample(num_rows=len(train_df) * 4)
    df_syn = clip_physical_limits(df_syn)
    return (
        pd.concat([train_df, df_syn], ignore_index=True),
        "Gaussian copula synthetic sampling over features+target.",
    )


def method_fpt_pretrain_finetune_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
    mlp_params: Dict[str, object],
    seed: int,
    max_iter: int,
) -> MethodResult:
    _ = mlp_params
    rng = np.random.default_rng(seed)
    n_syn = 3000

    sand_min, sand_max = float(train_df["sand"].min()), float(train_df["sand"].max())
    clay_min, clay_max = float(train_df["clay"].min()), float(train_df["clay"].max())
    bd_min, bd_max = float(train_df["bulk_density"].min()), float(train_df["bulk_density"].max())
    x_min, x_max = float(train_df["x"].min()), float(train_df["x"].max())
    y_min, y_max = float(train_df["y"].min()), float(train_df["y"].max())

    reg_macro = LinearRegression()
    reg_macro.fit(train_df[["sand", "clay", "bulk_density"]], train_df["macroporosity"])

    sand = rng.uniform(sand_min, sand_max, size=n_syn * 2)
    clay = rng.uniform(clay_min, clay_max, size=n_syn * 2)
    mask = (sand + clay) <= 98.0
    sand = sand[mask][:n_syn]
    clay = clay[mask][:n_syn]
    if sand.size < n_syn:
        extra = n_syn - sand.size
        sand = np.concatenate([sand, rng.uniform(sand_min, sand_max, size=extra)])
        clay = np.concatenate([clay, rng.uniform(clay_min, clay_max, size=extra)])

    bulk = rng.uniform(bd_min, bd_max, size=n_syn)
    macro = reg_macro.predict(np.column_stack([sand, clay, bulk])) + rng.normal(0.0, 0.01, size=n_syn)
    macro = np.clip(macro, 0.0, 1.0)
    x_syn = rng.uniform(x_min, x_max, size=n_syn)
    y_syn = rng.uniform(y_min, y_max, size=n_syn)

    log_ksat_syn = (
        -0.60
        + 1.15 * np.log10(np.clip(sand, 0.0, None) + 1.0)
        - 0.50 * np.log10(np.clip(clay, 0.0, None) + 1.0)
        - 2.80 * np.log10(np.clip(bulk, 0.8, None))
    )
    log_ksat_syn += rng.normal(0.0, 0.12, size=n_syn)

    df_syn = pd.DataFrame(
        {
            "x": x_syn,
            "y": y_syn,
            "sand": sand,
            "clay": clay,
            "bulk_density": bulk,
            "macroporosity": macro,
            "log_ksat": log_ksat_syn,
        }
    )
    df_syn = clip_physical_limits(df_syn)

    scaler = StandardScaler()
    x_syn_scaled = scaler.fit_transform(df_syn[feature_names].to_numpy(dtype=float))
    x_real_scaled = scaler.transform(train_df[feature_names].to_numpy(dtype=float))
    x_test_scaled = scaler.transform(test_df[feature_names].to_numpy(dtype=float))

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="tanh",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=8e-4,
        max_iter=1200,
        warm_start=True,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(x_syn_scaled, df_syn["log_ksat"].to_numpy(dtype=float))
    model.max_iter = max_iter
    model.fit(x_real_scaled, train_df["log_ksat"].to_numpy(dtype=float))

    y_pred = model.predict(x_test_scaled)
    metrics = regression_metrics(test_df["log_ksat"].to_numpy(dtype=float), y_pred)
    return MethodResult(
        method="ptf_pretrain_finetune",
        family="physics",
        status="ok",
        n_train_samples=int(len(train_df) + len(df_syn)),
        r2_log=float(metrics["r2_log"]),
        rmse_log=float(metrics["rmse_log"]),
        mae_log=float(metrics["mae_log"]),
        pbias_log=float(metrics["pbias_log"]),
        r2_orig=float(metrics["r2_orig"]),
        rmse_orig=float(metrics["rmse_orig"]),
        mae_orig=float(metrics["mae_orig"]),
        pbias_orig=float(metrics["pbias_orig"]),
        beat_baseline_07502=bool(metrics["r2_log"] > BASELINE_TARGET_R2_LOG),
        details="Pretrain on 3k FPT synthetic samples then finetune on real train+val.",
    )


def method_spatial_bootstrap_ensemble_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
    mlp_params: Dict[str, object],
    seed: int,
    max_iter: int,
) -> MethodResult:
    rng = np.random.default_rng(seed)
    block_size = 150.0
    x = train_df["x"].to_numpy(dtype=float)
    y = train_df["y"].to_numpy(dtype=float)
    x0, y0 = float(np.min(x)), float(np.min(y))

    block_x = np.floor((x - x0) / block_size).astype(int)
    block_y = np.floor((y - y0) / block_size).astype(int)
    block_id = block_x * 1000 + block_y
    unique_blocks = np.unique(block_id)

    n_bootstrap = 12
    x_test = test_df[feature_names].to_numpy(dtype=float)
    test_preds = []
    for i in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        sampled_idx = []
        for blk in sampled_blocks:
            blk_idx = np.where(block_id == blk)[0]
            if blk_idx.size == 0:
                continue
            sampled_idx.append(rng.choice(blk_idx, size=blk_idx.size, replace=True))
        if not sampled_idx:
            continue
        idx = np.concatenate(sampled_idx)
        df_boot = train_df.iloc[idx].reset_index(drop=True)
        model = build_mlp_pipeline(params=mlp_params, seed=seed + i, max_iter=max_iter)
        model.fit(
            df_boot[feature_names].to_numpy(dtype=float),
            df_boot["log_ksat"].to_numpy(dtype=float),
        )
        test_preds.append(model.predict(x_test))

    if not test_preds:
        raise RuntimeError("No bootstrap models were trained.")
    y_pred = np.mean(np.vstack(test_preds), axis=0)
    metrics = regression_metrics(test_df["log_ksat"].to_numpy(dtype=float), y_pred)
    return MethodResult(
        method="spatial_block_bootstrap_ensemble",
        family="bootstrap",
        status="ok",
        n_train_samples=int(len(train_df)),
        r2_log=float(metrics["r2_log"]),
        rmse_log=float(metrics["rmse_log"]),
        mae_log=float(metrics["mae_log"]),
        pbias_log=float(metrics["pbias_log"]),
        r2_orig=float(metrics["r2_orig"]),
        rmse_orig=float(metrics["rmse_orig"]),
        mae_orig=float(metrics["mae_orig"]),
        pbias_orig=float(metrics["pbias_orig"]),
        beat_baseline_07502=bool(metrics["r2_log"] > BASELINE_TARGET_R2_LOG),
        details=f"Mean prediction of {len(test_preds)} spatial bootstrap MLPs.",
    )


def run_augmentation_methods(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
    mlp_params: Dict[str, object],
    seed: int,
    max_iter: int,
    context: Dict[str, object],
) -> List[MethodResult]:
    rng_master = np.random.default_rng(seed)

    methods: List[Tuple[str, str, Callable[..., Tuple[pd.DataFrame, str]]]] = [
        ("baseline_no_aug", "baseline", method_baseline),
        ("gaussian_jitter_covariates", "noise", method_jitter_gaussian),
        ("coordinate_jitter", "noise", method_coordinate_jitter),
        ("ksat_multiplicative_noise", "noise", method_ksat_multiplicative_noise),
        ("combined_noise", "noise", method_combined_noise),
        ("monte_carlo_measurement", "physics", method_monte_carlo_measurement),
        ("idw_rbf_linear", "interpolation", method_idw_rbf_linear),
        ("thin_plate_spline", "interpolation", method_thin_plate_spline),
        ("kriging_covariate_composition", "interpolation", method_kriging_covariate_composition),
        ("smogn_regression", "synthetic", method_smogn),
        ("copula_gaussian", "synthetic", method_copula),
    ]

    results: List[MethodResult] = []
    for method_name, family, fn in methods:
        method_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(method_seed)
        try:
            augmented_df, details = fn(train_df=train_df, rng=rng, context=context)
            augmented_df = augmented_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
            result = train_eval_single(
                method=method_name,
                family=family,
                train_df=augmented_df,
                test_df=test_df,
                feature_names=feature_names,
                mlp_params=mlp_params,
                seed=seed,
                max_iter=max_iter,
                details=details,
            )
        except Exception as exc:
            result = MethodResult(
                method=method_name,
                family=family,
                status="failed",
                n_train_samples=0,
                r2_log=float("nan"),
                rmse_log=float("nan"),
                mae_log=float("nan"),
                pbias_log=float("nan"),
                r2_orig=float("nan"),
                rmse_orig=float("nan"),
                mae_orig=float("nan"),
                pbias_orig=float("nan"),
                beat_baseline_07502=False,
                details=f"FAILED: {exc}",
            )
        results.append(result)

    try:
        results.append(
            method_spatial_bootstrap_ensemble_eval(
                train_df=train_df,
                test_df=test_df,
                feature_names=feature_names,
                mlp_params=mlp_params,
                seed=seed,
                max_iter=max_iter,
            )
        )
    except Exception as exc:
        results.append(
            MethodResult(
                method="spatial_block_bootstrap_ensemble",
                family="bootstrap",
                status="failed",
                n_train_samples=0,
                r2_log=float("nan"),
                rmse_log=float("nan"),
                mae_log=float("nan"),
                pbias_log=float("nan"),
                r2_orig=float("nan"),
                rmse_orig=float("nan"),
                mae_orig=float("nan"),
                pbias_orig=float("nan"),
                beat_baseline_07502=False,
                details=f"FAILED: {exc}",
            )
        )

    try:
        results.append(
            method_fpt_pretrain_finetune_eval(
                train_df=train_df,
                test_df=test_df,
                feature_names=feature_names,
                mlp_params=mlp_params,
                seed=seed,
                max_iter=max_iter,
            )
        )
    except Exception as exc:
        results.append(
            MethodResult(
                method="ptf_pretrain_finetune",
                family="physics",
                status="failed",
                n_train_samples=0,
                r2_log=float("nan"),
                rmse_log=float("nan"),
                mae_log=float("nan"),
                pbias_log=float("nan"),
                r2_orig=float("nan"),
                rmse_orig=float("nan"),
                mae_orig=float("nan"),
                pbias_orig=float("nan"),
                beat_baseline_07502=False,
                details=f"FAILED: {exc}",
            )
        )

    return results


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mlp_params = load_baseline_mlp_params(args.baseline_params_json)
    df_raw = load_dataset(args.data_path, sheet_name=args.sheet_name)
    dataset = prepare_dataset(
        df_raw=df_raw,
        include_silt=False,
        include_ratio=False,
        include_sinusoidal=False,
    )

    feature_names = ["x", "y", "sand", "clay", "bulk_density", "macroporosity"]
    df_model = dataset.df_model[feature_names + ["log_ksat"]].copy().reset_index(drop=True)

    split = create_stratified_holdout_split(
        x=df_model[feature_names].to_numpy(dtype=float),
        y=df_model["log_ksat"].to_numpy(dtype=float),
        random_state=args.seed,
        train_fraction=0.70,
        val_fraction=0.15,
        test_fraction=0.15,
    )

    idx_trainval = np.concatenate([split.idx_train, split.idx_val])
    train_df = df_model.iloc[idx_trainval].reset_index(drop=True)
    test_df = df_model.iloc[split.idx_test].reset_index(drop=True)

    context = {
        "min_x": float(train_df["x"].min()),
        "max_x": float(train_df["x"].max()),
        "min_y": float(train_df["y"].min()),
        "max_y": float(train_df["y"].max()),
    }

    results = run_augmentation_methods(
        train_df=train_df,
        test_df=test_df,
        feature_names=feature_names,
        mlp_params=mlp_params,
        seed=args.seed,
        max_iter=args.max_iter,
        context=context,
    )

    results_df = pd.DataFrame([r.__dict__ for r in results])
    results_df = results_df.sort_values(
        by=["status", "r2_log"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    results_df.to_csv(args.output_dir / "augmentation_benchmark_results.csv", index=False)
    with open(args.output_dir / "augmentation_benchmark_config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "baseline_target_r2_log": BASELINE_TARGET_R2_LOG,
                "seed": args.seed,
                "max_iter": args.max_iter,
                "mlp_params": {
                    "hidden_layer_sizes": list(mlp_params["hidden_layer_sizes"]),
                    "activation": mlp_params["activation"],
                    "alpha": mlp_params["alpha"],
                    "learning_rate_init": mlp_params["learning_rate_init"],
                },
                "feature_names": feature_names,
                "n_trainval": int(len(train_df)),
                "n_test": int(len(test_df)),
                "has_pykrige": HAS_PYKRIGE,
                "has_smogn": HAS_SMOGN,
                "has_copulas": HAS_COPULAS,
            },
            file,
            indent=2,
        )

    best_ok = results_df[results_df["status"] == "ok"].copy()
    best_ok = best_ok.sort_values("r2_log", ascending=False).reset_index(drop=True)
    best_method = best_ok.iloc[0]["method"] if not best_ok.empty else "n/a"
    best_r2 = float(best_ok.iloc[0]["r2_log"]) if not best_ok.empty else float("nan")

    print("\n=== Augmentation benchmark completed ===")
    print(f"Train+Val samples (real): {len(train_df)}")
    print(f"Test samples (real only): {len(test_df)}")
    print(f"Baseline target to beat: {BASELINE_TARGET_R2_LOG:.4f}")
    print(f"Best method by test R2_log: {best_method}")
    print(f"Best test R2_log: {best_r2:.4f}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print("\nTop results:")
    print(
        best_ok[
            ["method", "family", "r2_log", "rmse_orig", "mae_orig", "beat_baseline_07502", "details"]
        ]
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
