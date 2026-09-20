#!/usr/bin/env python3
"""Automatic hyperparameter tuning for MLP in BHSE Ksat prediction."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_ksat_models import (
    RANDOM_STATE_DEFAULT,
    compute_spatial_blocks,
    create_stratified_holdout_split,
    load_dataset,
    prepare_dataset,
    regression_metrics,
)

try:
    import optuna

    HAS_OPTUNA = True
except Exception:
    optuna = None
    HAS_OPTUNA = False


def parse_hidden_layers(values: Sequence[str]) -> List[Tuple[int, ...]]:
    parsed: List[Tuple[int, ...]] = []
    for value in values:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError(f"Invalid hidden layer definition: '{value}'")
        parsed.append(tuple(int(part) for part in parts))
    return parsed


def build_mlp_pipeline(
    hidden_layer_sizes: Tuple[int, ...],
    activation: str,
    alpha: float,
    learning_rate_init: float,
    max_iter: int,
    seed: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    solver="adam",
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=40,
                    random_state=seed,
                ),
            ),
        ]
    )


def run_grid_search(
    x_trainval: np.ndarray,
    y_trainval: np.ndarray,
    groups_trainval: np.ndarray,
    hidden_candidates: Sequence[Tuple[int, ...]],
    activations: Sequence[str],
    alphas: Sequence[float],
    learning_rates: Sequence[float],
    max_iter: int,
    seed: int,
    n_splits: int,
    verbose: int,
) -> Tuple[Dict[str, object], float, pd.DataFrame]:
    unique_groups = np.unique(groups_trainval)
    folds = min(n_splits, unique_groups.size)
    if folds < 2:
        raise ValueError("Not enough spatial groups for GridSearchCV.")

    base_pipeline = build_mlp_pipeline(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        seed=seed,
    )

    param_grid = {
        "model__hidden_layer_sizes": list(hidden_candidates),
        "model__activation": list(activations),
        "model__alpha": list(alphas),
        "model__learning_rate_init": list(learning_rates),
    }

    search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=GroupKFold(n_splits=folds),
        refit=True,
        n_jobs=-1,
        verbose=verbose,
        return_train_score=True,
    )
    search.fit(x_trainval, y_trainval, groups=groups_trainval)

    results_df = (
        pd.DataFrame(search.cv_results_)
        .sort_values(["rank_test_score", "mean_test_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    best_params = {
        "hidden_layer_sizes": tuple(search.best_params_["model__hidden_layer_sizes"]),
        "activation": str(search.best_params_["model__activation"]),
        "alpha": float(search.best_params_["model__alpha"]),
        "learning_rate_init": float(search.best_params_["model__learning_rate_init"]),
    }
    return best_params, float(search.best_score_), results_df


def run_optuna_search(
    x_trainval: np.ndarray,
    y_trainval: np.ndarray,
    groups_trainval: np.ndarray,
    hidden_candidates: Sequence[Tuple[int, ...]],
    activations: Sequence[str],
    max_iter: int,
    seed: int,
    n_splits: int,
    n_trials: int,
    timeout: int | None,
) -> Tuple[Dict[str, object], float, pd.DataFrame]:
    if not HAS_OPTUNA:
        raise RuntimeError(
            "Optuna is not installed. Install with `.venv/bin/pip install optuna` "
            "or use --method grid."
        )

    unique_groups = np.unique(groups_trainval)
    folds = min(n_splits, unique_groups.size)
    if folds < 2:
        raise ValueError("Not enough spatial groups for Optuna search.")

    fold_indices = list(GroupKFold(n_splits=folds).split(x_trainval, y_trainval, groups=groups_trainval))
    hidden_as_text = [",".join(str(unit) for unit in hidden) for hidden in hidden_candidates]

    def objective(trial: optuna.Trial) -> float:
        hidden_text = trial.suggest_categorical("hidden_layer_sizes", hidden_as_text)
        hidden = tuple(int(value) for value in hidden_text.split(","))
        activation = trial.suggest_categorical("activation", list(activations))
        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)

        fold_scores: List[float] = []
        for train_idx, valid_idx in fold_indices:
            model = build_mlp_pipeline(
                hidden_layer_sizes=hidden,
                activation=activation,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                seed=seed,
            )
            try:
                model.fit(x_trainval[train_idx], y_trainval[train_idx])
                pred = model.predict(x_trainval[valid_idx])
                score = float(r2_score(y_trainval[valid_idx], pred))
            except Exception:
                score = -1e9
            fold_scores.append(score)
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_hidden = tuple(int(value) for value in study.best_params["hidden_layer_sizes"].split(","))
    best_params = {
        "hidden_layer_sizes": best_hidden,
        "activation": str(study.best_params["activation"]),
        "alpha": float(study.best_params["alpha"]),
        "learning_rate_init": float(study.best_params["learning_rate_init"]),
    }
    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    trials_df = trials_df.sort_values("value", ascending=False).reset_index(drop=True)
    return best_params, float(study.best_value), trials_df


def evaluate_model_splits(
    model: Pipeline,
    split_name: str,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> Dict[str, float]:
    y_pred = model.predict(x_data)
    row = regression_metrics(y_data, y_pred)
    row.update({"model": split_name, "n_samples": int(len(y_data))})
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune MLP hyperparameters for BHSE Ksat prediction using GridSearchCV or Optuna."
    )
    parser.add_argument("--data-path", type=Path, default=Path("Pontos_Ksat_Sanga_Ellert.xlsx"))
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_mlp_tuning"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE_DEFAULT)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--spatial-folds", type=int, default=5)
    parser.add_argument("--blocks-x", type=int, default=4)
    parser.add_argument("--blocks-y", type=int, default=4)
    parser.add_argument("--include-silt", action="store_true")
    parser.add_argument("--include-ratio", action="store_true")
    parser.add_argument("--include-sinusoidal", action="store_true")
    parser.add_argument("--method", choices=["grid", "optuna"], default="grid")
    parser.add_argument(
        "--grid-hidden",
        nargs="+",
        default=["32,16", "64,32", "128,64", "128,64,32"],
        help="Hidden layer candidate list. Example: --grid-hidden 64,32 128,64,32",
    )
    parser.add_argument("--grid-activation", nargs="+", default=["relu", "tanh"])
    parser.add_argument("--grid-alpha", nargs="+", type=float, default=[1e-5, 1e-4, 1e-3])
    parser.add_argument("--grid-lr", nargs="+", type=float, default=[1e-4, 5e-4, 1e-3, 5e-3])
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--optuna-trials", type=int, default=80)
    parser.add_argument("--optuna-timeout", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hidden_candidates = parse_hidden_layers(args.grid_hidden)
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

    trainval_idx = np.concatenate([split.idx_train, split.idx_val])
    x_trainval = x_all[trainval_idx]
    y_trainval = y_all[trainval_idx]

    spatial_groups_all = compute_spatial_blocks(
        x=dataset.df_model["x"].to_numpy(),
        y=dataset.df_model["y"].to_numpy(),
        blocks_x=args.blocks_x,
        blocks_y=args.blocks_y,
    )
    groups_trainval = spatial_groups_all[trainval_idx]

    if args.method == "grid":
        best_params, best_cv_score, tuning_df = run_grid_search(
            x_trainval=x_trainval,
            y_trainval=y_trainval,
            groups_trainval=groups_trainval,
            hidden_candidates=hidden_candidates,
            activations=args.grid_activation,
            alphas=args.grid_alpha,
            learning_rates=args.grid_lr,
            max_iter=args.max_iter,
            seed=args.seed,
            n_splits=args.spatial_folds,
            verbose=args.verbose,
        )
    else:
        best_params, best_cv_score, tuning_df = run_optuna_search(
            x_trainval=x_trainval,
            y_trainval=y_trainval,
            groups_trainval=groups_trainval,
            hidden_candidates=hidden_candidates,
            activations=args.grid_activation,
            max_iter=args.max_iter,
            seed=args.seed,
            n_splits=args.spatial_folds,
            n_trials=args.optuna_trials,
            timeout=args.optuna_timeout,
        )

    baseline_model = build_mlp_pipeline(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    tuned_model = build_mlp_pipeline(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        activation=best_params["activation"],
        alpha=best_params["alpha"],
        learning_rate_init=best_params["learning_rate_init"],
        max_iter=args.max_iter,
        seed=args.seed,
    )

    baseline_model.fit(x_trainval, y_trainval)
    tuned_model.fit(x_trainval, y_trainval)

    comparison_rows = []
    for tag, model in [("MLP_Baseline", baseline_model), ("MLP_Tuned", tuned_model)]:
        row_trainval = evaluate_model_splits(model, tag, x_trainval, y_trainval)
        row_trainval["split"] = "trainval"
        comparison_rows.append(row_trainval)

        row_test = evaluate_model_splits(model, tag, split.x_test, split.y_test)
        row_test["split"] = "test"
        comparison_rows.append(row_test)

    comparison_df = pd.DataFrame(comparison_rows)

    tuned_pred_test = tuned_model.predict(split.x_test)
    test_df = dataset.df_model.iloc[split.idx_test][["x", "y", "ksat", "log_ksat"]].copy()
    test_df = test_df.rename(columns={"ksat": "ksat_true", "log_ksat": "log_ksat_true"})
    test_df["log_ksat_pred_tuned_mlp"] = tuned_pred_test
    test_df["ksat_pred_tuned_mlp"] = np.power(10.0, tuned_pred_test)
    test_df["residual_log_tuned_mlp"] = test_df["log_ksat_pred_tuned_mlp"] - test_df["log_ksat_true"]
    test_df["residual_orig_tuned_mlp"] = test_df["ksat_pred_tuned_mlp"] - test_df["ksat_true"]
    test_df = test_df.reset_index(drop=True)

    tuning_df.to_csv(args.output_dir / "mlp_tuning_trials.csv", index=False)
    comparison_df.to_csv(args.output_dir / "mlp_tuned_vs_baseline_metrics.csv", index=False)
    test_df.to_csv(args.output_dir / "mlp_tuned_predictions_test.csv", index=False)

    best_params_serializable = dict(best_params)
    best_params_serializable["hidden_layer_sizes"] = list(best_params["hidden_layer_sizes"])
    with open(args.output_dir / "mlp_best_params.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "method": args.method,
                "best_cv_r2_log": best_cv_score,
                "best_params": best_params_serializable,
            },
            file,
            indent=2,
        )

    with open(args.output_dir / "mlp_tuning_config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "data_path": str(args.data_path),
                "sheet_name": args.sheet_name,
                "seed": args.seed,
                "features": dataset.feature_names,
                "method": args.method,
                "spatial_folds": args.spatial_folds,
                "blocks_x": args.blocks_x,
                "blocks_y": args.blocks_y,
                "grid_hidden": args.grid_hidden,
                "grid_activation": args.grid_activation,
                "grid_alpha": args.grid_alpha,
                "grid_lr": args.grid_lr,
                "max_iter": args.max_iter,
                "optuna_trials": args.optuna_trials,
                "optuna_timeout": args.optuna_timeout,
            },
            file,
            indent=2,
        )

    best_test = comparison_df[(comparison_df["model"] == "MLP_Tuned") & (comparison_df["split"] == "test")]
    if best_test.empty:
        warnings.warn("No tuned test row available.")
    else:
        best_test = best_test.iloc[0]
        print("\n=== MLP tuning completed ===")
        print(f"Method: {args.method}")
        print(f"Best CV R2 (log): {best_cv_score:.4f}")
        print(f"Best params: {best_params}")
        print(f"Tuned test R2 (log): {best_test['r2_log']:.4f}")
        print(f"Tuned test RMSE (orig): {best_test['rmse_orig']:.2f} cm/dia")
        print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
