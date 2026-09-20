#!/usr/bin/env python3
"""Robust virtual test evaluation for BHSE Ksat: baseline MLP vs copula augmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_ksat_models import RANDOM_STATE_DEFAULT, load_dataset, prepare_dataset, quartile_strata

try:
    from copulas.multivariate import GaussianMultivariate

    HAS_COPULAS = True
except Exception:
    GaussianMultivariate = None
    HAS_COPULAS = False


BASELINE_SINGLE_SPLIT_R2_LOG = 0.7502
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs copula-augmented MLP using repeated and spatial CV strategies."
    )
    parser.add_argument("--data-path", type=Path, default=Path("Pontos_Ksat_Sanga_Ellert.xlsx"))
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_virtual_test_eval"))
    parser.add_argument(
        "--baseline-params-json",
        type=Path,
        default=Path("outputs_mlp_tuning_optuna/mlp_best_params.json"),
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE_DEFAULT)
    parser.add_argument("--max-iter", type=int, default=2500)
    parser.add_argument("--rkf-splits", type=int, default=5)
    parser.add_argument("--rkf-repeats", type=int, default=10)
    parser.add_argument("--mccv-iter", type=int, default=200)
    parser.add_argument("--mccv-test-size", type=float, default=0.2)
    parser.add_argument("--sloo-buffer-m", type=float, default=50.0)
    parser.add_argument("--sloo-min-train", type=int, default=20)
    parser.add_argument("--copula-multiplier", type=float, default=4.0)
    return parser.parse_args()


def load_baseline_params(path: Path) -> Dict[str, object]:
    default_params = {
        "hidden_layer_sizes": (128, 64),
        "activation": "tanh",
        "alpha": 0.0011380907649808582,
        "learning_rate_init": 0.0024792013326598596,
    }
    if not path.exists():
        return default_params

    data = json.loads(path.read_text(encoding="utf-8"))
    params = data.get("best_params", {})
    hidden = params.get("hidden_layer_sizes", default_params["hidden_layer_sizes"])
    if isinstance(hidden, list):
        hidden = tuple(int(v) for v in hidden)
    return {
        "hidden_layer_sizes": tuple(hidden),
        "activation": str(params.get("activation", default_params["activation"])),
        "alpha": float(params.get("alpha", default_params["alpha"])),
        "learning_rate_init": float(
            params.get("learning_rate_init", default_params["learning_rate_init"])
        ),
    }


def build_model(params: Dict[str, object], seed: int, max_iter: int) -> Pipeline:
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


def clip_physical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sand"] = out["sand"].clip(0.0, 100.0)
    out["clay"] = out["clay"].clip(0.0, 100.0)
    out["bulk_density"] = out["bulk_density"].clip(0.8, 1.9)
    out["macroporosity"] = out["macroporosity"].clip(0.0, 1.0)
    return out


def make_copula_augmented_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    rng: np.random.Generator,
    multiplier: float,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    if not HAS_COPULAS:
        return x_train, y_train, True

    train_df = pd.DataFrame(x_train, columns=feature_names)
    train_df["log_ksat"] = y_train
    train_df = clip_physical(train_df)

    n_syn = max(1, int(len(train_df) * multiplier))
    try:
        model = GaussianMultivariate()
        model.fit(train_df)
        synth = model.sample(num_rows=n_syn)
        synth = clip_physical(synth).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        if synth.empty:
            return x_train, y_train, True
        x_aug = np.vstack([x_train, synth[list(feature_names)].to_numpy(dtype=float)])
        y_aug = np.concatenate([y_train, synth["log_ksat"].to_numpy(dtype=float)])
        return x_aug, y_aug, False
    except Exception:
        # fallback: no augmentation for this split if copula fails
        return x_train, y_train, True


def split_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2_log": float(r2_score(y_true, y_pred)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_log": float(mean_absolute_error(y_true, y_pred)),
    }


def summarize_distribution(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "median": float(np.median(arr)),
        "n": int(arr.size),
    }


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    try:
        stat, pvalue = wilcoxon(arr_a, arr_b, zero_method="wilcox", alternative="two-sided")
        return {"statistic": float(stat), "pvalue": float(pvalue)}
    except Exception:
        return {"statistic": float("nan"), "pvalue": float("nan")}


def evaluate_repeated_stratified_kfold(
    x: np.ndarray,
    y: np.ndarray,
    model_params: Dict[str, object],
    seed: int,
    max_iter: int,
    n_splits: int,
    n_repeats: int,
    copula_multiplier: float,
    feature_names: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    strata = quartile_strata(y)
    rows: List[Dict[str, object]] = []
    copula_fallback_count = 0
    split_id = 0

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for fold, (train_idx, test_idx) in enumerate(skf.split(x, strata), start=1):
            split_seed = seed * 10_000 + rep * 100 + fold
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            baseline_model = build_model(model_params, seed=split_seed, max_iter=max_iter)
            baseline_model.fit(x_train, y_train)
            pred_base = baseline_model.predict(x_test)
            base_metrics = split_metrics(y_test, pred_base)

            x_aug, y_aug, used_fallback = make_copula_augmented_train(
                x_train=x_train,
                y_train=y_train,
                feature_names=feature_names,
                rng=np.random.default_rng(split_seed),
                multiplier=copula_multiplier,
            )
            if used_fallback:
                copula_fallback_count += 1
            copula_model = build_model(model_params, seed=split_seed, max_iter=max_iter)
            copula_model.fit(x_aug, y_aug)
            pred_cop = copula_model.predict(x_test)
            cop_metrics = split_metrics(y_test, pred_cop)

            split_id += 1
            rows.append(
                {
                    "strategy": "repeated_stratified_kfold",
                    "repeat": rep + 1,
                    "fold": fold,
                    "split_id": split_id,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "model": "baseline",
                    **base_metrics,
                }
            )
            rows.append(
                {
                    "strategy": "repeated_stratified_kfold",
                    "repeat": rep + 1,
                    "fold": fold,
                    "split_id": split_id,
                    "n_train": int(x_aug.shape[0]),
                    "n_test": int(len(test_idx)),
                    "model": "copula",
                    **cop_metrics,
                }
            )

    return pd.DataFrame(rows), {"copula_fallback_count": int(copula_fallback_count)}


def evaluate_mccv(
    x: np.ndarray,
    y: np.ndarray,
    model_params: Dict[str, object],
    seed: int,
    max_iter: int,
    n_iter: int,
    test_size: float,
    copula_multiplier: float,
    feature_names: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    strata = quartile_strata(y)
    rows: List[Dict[str, object]] = []
    copula_fallback_count = 0

    for i in range(n_iter):
        split_seed = seed + i
        x_train, x_test, y_train, y_test, strata_train, _ = train_test_split(
            x,
            y,
            strata,
            test_size=test_size,
            random_state=split_seed,
            stratify=strata,
        )
        _ = strata_train

        baseline_model = build_model(model_params, seed=split_seed, max_iter=max_iter)
        baseline_model.fit(x_train, y_train)
        pred_base = baseline_model.predict(x_test)
        base_metrics = split_metrics(y_test, pred_base)

        x_aug, y_aug, used_fallback = make_copula_augmented_train(
            x_train=x_train,
            y_train=y_train,
            feature_names=feature_names,
            rng=np.random.default_rng(split_seed),
            multiplier=copula_multiplier,
        )
        if used_fallback:
            copula_fallback_count += 1
        copula_model = build_model(model_params, seed=split_seed, max_iter=max_iter)
        copula_model.fit(x_aug, y_aug)
        pred_cop = copula_model.predict(x_test)
        cop_metrics = split_metrics(y_test, pred_cop)

        rows.append(
            {
                "strategy": "mccv",
                "iteration": i + 1,
                "split_id": i + 1,
                "n_train": int(x_train.shape[0]),
                "n_test": int(x_test.shape[0]),
                "model": "baseline",
                **base_metrics,
            }
        )
        rows.append(
            {
                "strategy": "mccv",
                "iteration": i + 1,
                "split_id": i + 1,
                "n_train": int(x_aug.shape[0]),
                "n_test": int(x_test.shape[0]),
                "model": "copula",
                **cop_metrics,
            }
        )

    return pd.DataFrame(rows), {"copula_fallback_count": int(copula_fallback_count)}


def evaluate_spatial_loo_buffer(
    coords: np.ndarray,
    x_features: np.ndarray,
    y: np.ndarray,
    model_params: Dict[str, object],
    seed: int,
    max_iter: int,
    buffer_radius_m: float,
    min_train: int,
    copula_multiplier: float,
    feature_names: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    n = len(y)
    dmat = cdist(coords, coords)
    pred_base = np.zeros(n, dtype=float)
    pred_cop = np.zeros(n, dtype=float)
    train_sizes = np.zeros(n, dtype=int)
    train_sizes_cop = np.zeros(n, dtype=int)
    used_fallback = np.zeros(n, dtype=bool)
    used_min_train_fallback = np.zeros(n, dtype=bool)

    for i in range(n):
        dist = dmat[i]
        mask_train = dist > buffer_radius_m
        mask_train[i] = False
        if int(mask_train.sum()) < min_train:
            mask_train = dist > 0.0
            used_min_train_fallback[i] = True

        train_idx = np.where(mask_train)[0]
        x_train, y_train = x_features[train_idx], y[train_idx]
        x_test = x_features[[i]]
        train_sizes[i] = int(len(train_idx))

        split_seed = seed + i
        model_base = build_model(model_params, seed=split_seed, max_iter=max_iter)
        model_base.fit(x_train, y_train)
        pred_base[i] = float(model_base.predict(x_test)[0])

        x_aug, y_aug, fallback = make_copula_augmented_train(
            x_train=x_train,
            y_train=y_train,
            feature_names=feature_names,
            rng=np.random.default_rng(split_seed),
            multiplier=copula_multiplier,
        )
        used_fallback[i] = fallback
        train_sizes_cop[i] = int(x_aug.shape[0])

        model_cop = build_model(model_params, seed=split_seed, max_iter=max_iter)
        model_cop.fit(x_aug, y_aug)
        pred_cop[i] = float(model_cop.predict(x_test)[0])

    r2_base = float(r2_score(y, pred_base))
    r2_cop = float(r2_score(y, pred_cop))
    rmse_base = float(np.sqrt(mean_squared_error(y, pred_base)))
    rmse_cop = float(np.sqrt(mean_squared_error(y, pred_cop)))
    mae_base = float(mean_absolute_error(y, pred_base))
    mae_cop = float(mean_absolute_error(y, pred_cop))
    abs_err_base = np.abs(y - pred_base)
    abs_err_cop = np.abs(y - pred_cop)
    wil_abs = safe_wilcoxon(abs_err_base, abs_err_cop)

    pred_df = pd.DataFrame(
        {
            "idx": np.arange(n),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "y_true_log": y,
            "y_pred_baseline_log": pred_base,
            "y_pred_copula_log": pred_cop,
            "abs_err_baseline": abs_err_base,
            "abs_err_copula": abs_err_cop,
            "train_size_baseline": train_sizes,
            "train_size_copula": train_sizes_cop,
            "used_min_train_fallback": used_min_train_fallback,
            "used_copula_fallback": used_fallback,
        }
    )

    summary = {
        "r2_log_baseline": r2_base,
        "r2_log_copula": r2_cop,
        "rmse_log_baseline": rmse_base,
        "rmse_log_copula": rmse_cop,
        "mae_log_baseline": mae_base,
        "mae_log_copula": mae_cop,
        "mae_log_delta_copula_minus_baseline": float(mae_cop - mae_base),
        "r2_log_delta_copula_minus_baseline": float(r2_cop - r2_base),
        "wilcoxon_abs_error_statistic": wil_abs["statistic"],
        "wilcoxon_abs_error_pvalue": wil_abs["pvalue"],
        "avg_train_size_baseline": float(np.mean(train_sizes)),
        "avg_train_size_copula": float(np.mean(train_sizes_cop)),
        "copula_fallback_count": int(np.sum(used_fallback)),
        "min_train_fallback_count": int(np.sum(used_min_train_fallback)),
        "n_points": int(n),
    }
    return pred_df, summary


def paired_summary_from_scores(df: pd.DataFrame, strategy: str) -> Dict[str, object]:
    sub = df[df["strategy"] == strategy].copy()
    if sub.empty:
        return {}
    piv = sub.pivot(index="split_id", columns="model", values="r2_log").dropna()
    base_scores = piv["baseline"].to_numpy(dtype=float)
    cop_scores = piv["copula"].to_numpy(dtype=float)
    deltas = cop_scores - base_scores
    wil = safe_wilcoxon(base_scores, cop_scores)
    return {
        "baseline_r2_log": summarize_distribution(base_scores),
        "copula_r2_log": summarize_distribution(cop_scores),
        "delta_r2_log_copula_minus_baseline": summarize_distribution(deltas),
        "wilcoxon_r2_log_statistic": wil["statistic"],
        "wilcoxon_r2_log_pvalue": wil["pvalue"],
        "copula_beats_baseline_rate": float(np.mean(deltas > 0)),
        "n_pairs": int(len(deltas)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    params = load_baseline_params(args.baseline_params_json)
    df_raw = load_dataset(args.data_path, sheet_name=args.sheet_name)
    dataset = prepare_dataset(
        df_raw=df_raw,
        include_silt=False,
        include_ratio=False,
        include_sinusoidal=False,
    )

    feature_names = ["x", "y", "sand", "clay", "bulk_density", "macroporosity"]
    df = dataset.df_model[feature_names + ["log_ksat"]].copy().reset_index(drop=True)
    x = df[feature_names].to_numpy(dtype=float)
    y = df["log_ksat"].to_numpy(dtype=float)
    coords = df[["x", "y"]].to_numpy(dtype=float)

    rkf_df, rkf_info = evaluate_repeated_stratified_kfold(
        x=x,
        y=y,
        model_params=params,
        seed=args.seed,
        max_iter=args.max_iter,
        n_splits=args.rkf_splits,
        n_repeats=args.rkf_repeats,
        copula_multiplier=args.copula_multiplier,
        feature_names=feature_names,
    )

    mccv_df, mccv_info = evaluate_mccv(
        x=x,
        y=y,
        model_params=params,
        seed=args.seed,
        max_iter=args.max_iter,
        n_iter=args.mccv_iter,
        test_size=args.mccv_test_size,
        copula_multiplier=args.copula_multiplier,
        feature_names=feature_names,
    )

    sloo_pred_df, sloo_summary = evaluate_spatial_loo_buffer(
        coords=coords,
        x_features=x,
        y=y,
        model_params=params,
        seed=args.seed,
        max_iter=args.max_iter,
        buffer_radius_m=args.sloo_buffer_m,
        min_train=args.sloo_min_train,
        copula_multiplier=args.copula_multiplier,
        feature_names=feature_names,
    )

    all_scores_df = pd.concat([rkf_df, mccv_df], ignore_index=True)
    all_scores_df.to_csv(args.output_dir / "virtual_eval_split_scores.csv", index=False)
    sloo_pred_df.to_csv(args.output_dir / "virtual_eval_spatial_loo_predictions.csv", index=False)

    rkf_summary = paired_summary_from_scores(all_scores_df, "repeated_stratified_kfold")
    mccv_summary = paired_summary_from_scores(all_scores_df, "mccv")

    summary = {
        "baseline_single_split_reference_r2_log": BASELINE_SINGLE_SPLIT_R2_LOG,
        "config": {
            "seed": args.seed,
            "max_iter": args.max_iter,
            "rkf_splits": args.rkf_splits,
            "rkf_repeats": args.rkf_repeats,
            "mccv_iter": args.mccv_iter,
            "mccv_test_size": args.mccv_test_size,
            "sloo_buffer_m": args.sloo_buffer_m,
            "sloo_min_train": args.sloo_min_train,
            "copula_multiplier": args.copula_multiplier,
            "has_copulas": HAS_COPULAS,
            "n_samples_total": int(len(df)),
        },
        "model_params": {
            "hidden_layer_sizes": list(params["hidden_layer_sizes"]),
            "activation": params["activation"],
            "alpha": params["alpha"],
            "learning_rate_init": params["learning_rate_init"],
        },
        "repeated_stratified_kfold": {**rkf_summary, **rkf_info},
        "mccv": {**mccv_summary, **mccv_info},
        "spatial_loo_with_buffer": sloo_summary,
    }

    with open(args.output_dir / "virtual_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Virtual test evaluation completed ===")
    print(f"Output directory: {args.output_dir.resolve()}")
    if rkf_summary:
        print(
            "Repeated Stratified KFold (baseline vs copula) - mean R2_log: "
            f"{rkf_summary['baseline_r2_log']['mean']:.4f} vs {rkf_summary['copula_r2_log']['mean']:.4f}"
        )
        print(
            "Repeated Stratified KFold Wilcoxon p-value: "
            f"{rkf_summary['wilcoxon_r2_log_pvalue']:.6f}"
        )
    if mccv_summary:
        print(
            "MCCV (baseline vs copula) - mean R2_log: "
            f"{mccv_summary['baseline_r2_log']['mean']:.4f} vs {mccv_summary['copula_r2_log']['mean']:.4f}"
        )
        print(f"MCCV Wilcoxon p-value: {mccv_summary['wilcoxon_r2_log_pvalue']:.6f}")
    print(
        "Spatial LOO+buffer R2_log (baseline vs copula): "
        f"{sloo_summary['r2_log_baseline']:.4f} vs {sloo_summary['r2_log_copula']:.4f}"
    )


if __name__ == "__main__":
    main()
