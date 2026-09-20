#!/usr/bin/env python3
"""Round 3: generalization-focused MLP tuning with repeated spatial CV and ensembling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_ksat_models import (
    EPS,
    RANDOM_STATE_DEFAULT,
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
    for item in values:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"Invalid hidden layer definition: '{item}'")
        parsed.append(tuple(int(p) for p in parts))
    return parsed


def parse_seed_list(values: Sequence[int]) -> List[int]:
    seeds = [int(v) for v in values]
    if not seeds:
        raise ValueError("Seed list cannot be empty.")
    return seeds


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
                    n_iter_no_change=50,
                    random_state=seed,
                ),
            ),
        ]
    )


def periodic_spatial_blocks(
    x: np.ndarray,
    y: np.ndarray,
    blocks_x: int,
    blocks_y: int,
    offset_x: float,
    offset_y: float,
) -> np.ndarray:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    x_scaled = ((x - x_min) / (x_max - x_min + EPS) * blocks_x + offset_x) % blocks_x
    y_scaled = ((y - y_min) / (y_max - y_min + EPS) * blocks_y + offset_y) % blocks_y

    x_bin = np.floor(x_scaled).astype(int)
    y_bin = np.floor(y_scaled).astype(int)
    return (x_bin * blocks_y + y_bin).astype(int)


def make_offsets(n_repeats: int) -> List[Tuple[float, float]]:
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1.")
    offsets = [(0.0, 0.0)]
    for i in range(1, n_repeats):
        offsets.append((((i * 0.61803398875) % 1.0), ((i * 0.41421356237) % 1.0)))
    return offsets


def build_repeated_spatial_splits(
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    blocks_x: int,
    blocks_y: int,
    n_splits: int,
    n_repeats: int,
) -> List[Dict[str, object]]:
    offsets = make_offsets(n_repeats)
    split_specs: List[Dict[str, object]] = []

    for repeat_id, (off_x, off_y) in enumerate(offsets, start=1):
        groups = periodic_spatial_blocks(
            x=x_coord,
            y=y_coord,
            blocks_x=blocks_x,
            blocks_y=blocks_y,
            offset_x=off_x,
            offset_y=off_y,
        )
        unique_groups = np.unique(groups)
        folds = min(n_splits, unique_groups.size)
        if folds < 2:
            continue

        # Manual group-fold split to keep explicit control and metadata.
        group_ids = np.array(sorted(unique_groups.tolist()))
        fold_bins = np.array_split(group_ids, folds)
        for fold_id, test_groups in enumerate(fold_bins, start=1):
            test_mask = np.isin(groups, test_groups)
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            if len(test_idx) < 2 or len(train_idx) == 0:
                continue
            split_specs.append(
                {
                    "repeat": repeat_id,
                    "fold": fold_id,
                    "offset_x": float(off_x),
                    "offset_y": float(off_y),
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )

    if not split_specs:
        raise ValueError("No valid repeated spatial splits were generated.")
    return split_specs


def evaluate_params_repeated_cv(
    params: Dict[str, object],
    x_trainval: np.ndarray,
    y_trainval: np.ndarray,
    split_specs: Sequence[Dict[str, object]],
    seed: int,
    max_iter: int,
) -> Tuple[float, float, List[float]]:
    scores: List[float] = []
    for spec in split_specs:
        train_idx = spec["train_idx"]
        test_idx = spec["test_idx"]
        model = build_mlp_pipeline(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation=params["activation"],
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            max_iter=max_iter,
            seed=seed,
        )
        model.fit(x_trainval[train_idx], y_trainval[train_idx])
        pred = model.predict(x_trainval[test_idx])
        scores.append(float(r2_score(y_trainval[test_idx], pred)))

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    return mean_score, std_score, scores


def trial_params_to_model_params(trial: "optuna.trial.FrozenTrial") -> Dict[str, object]:
    hidden = tuple(int(p) for p in str(trial.params["hidden_layer_sizes"]).split(","))
    return {
        "hidden_layer_sizes": hidden,
        "activation": str(trial.params["activation"]),
        "alpha": float(trial.params["alpha"]),
        "learning_rate_init": float(trial.params["learning_rate_init"]),
    }


def build_prediction_table(
    df_model: pd.DataFrame,
    idx_test: np.ndarray,
    preds: Dict[str, np.ndarray],
) -> pd.DataFrame:
    table = df_model.iloc[idx_test][["x", "y", "ksat", "log_ksat"]].copy()
    table = table.rename(columns={"ksat": "ksat_true", "log_ksat": "log_ksat_true"})
    table = table.reset_index(drop=True)
    for label, pred_log in preds.items():
        table[f"log_ksat_pred_{label}"] = pred_log
        table[f"ksat_pred_{label}"] = np.power(10.0, pred_log)
        table[f"residual_log_{label}"] = pred_log - table["log_ksat_true"]
        table[f"residual_orig_{label}"] = table[f"ksat_pred_{label}"] - table["ksat_true"]
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round 3 MLP tuning: repeated spatial CV + ensemble for better generalization."
    )
    parser.add_argument("--data-path", type=Path, default=Path("Pontos_Ksat_Sanga_Ellert.xlsx"))
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_mlp_round3"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE_DEFAULT)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--spatial-folds", type=int, default=5)
    parser.add_argument("--blocks-x", type=int, default=4)
    parser.add_argument("--blocks-y", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--stability-weight", type=float, default=0.20)
    parser.add_argument("--optuna-trials", type=int, default=250)
    parser.add_argument("--optuna-timeout", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=7000)
    parser.add_argument(
        "--hidden-candidates",
        nargs="+",
        default=[
            "32,16",
            "64,32",
            "128,64",
            "256,128",
            "64,64,32",
            "128,64,32",
            "256,128,64",
        ],
    )
    parser.add_argument("--activation-candidates", nargs="+", default=["tanh", "relu"])
    parser.add_argument("--top-k-configs", type=int, default=5)
    parser.add_argument("--members-per-config", type=int, default=3)
    parser.add_argument("--ensemble-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46, 47])
    parser.add_argument("--include-silt", action="store_true")
    parser.add_argument(
        "--disable-ratio",
        action="store_true",
        help="Disable sand/clay ratio feature (enabled by default in round 3).",
    )
    parser.add_argument(
        "--disable-sinusoidal",
        action="store_true",
        help="Disable sinusoidal coordinate features (enabled by default in round 3).",
    )
    return parser.parse_args()


def main() -> None:
    if not HAS_OPTUNA:
        raise RuntimeError(
            "Optuna is required for round 3. Install with `.venv/bin/pip install optuna`."
        )

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hidden_candidates = parse_hidden_layers(args.hidden_candidates)
    ensemble_seeds = parse_seed_list(args.ensemble_seeds)

    include_ratio = not args.disable_ratio
    include_sinusoidal = not args.disable_sinusoidal

    df_raw = load_dataset(args.data_path, sheet_name=args.sheet_name)
    dataset = prepare_dataset(
        df_raw=df_raw,
        include_silt=args.include_silt,
        include_ratio=include_ratio,
        include_sinusoidal=include_sinusoidal,
    )

    x_all = dataset.df_model[dataset.feature_names].to_numpy(dtype=float)
    y_all = dataset.df_model[dataset.target_name].to_numpy(dtype=float)
    x_coord_all = dataset.df_model["x"].to_numpy(dtype=float)
    y_coord_all = dataset.df_model["y"].to_numpy(dtype=float)

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
    x_coord_trainval = x_coord_all[trainval_idx]
    y_coord_trainval = y_coord_all[trainval_idx]

    split_specs = build_repeated_spatial_splits(
        x_coord=x_coord_trainval,
        y_coord=y_coord_trainval,
        blocks_x=args.blocks_x,
        blocks_y=args.blocks_y,
        n_splits=args.spatial_folds,
        n_repeats=args.repeats,
    )

    def objective(trial: "optuna.Trial") -> float:
        hidden_text = trial.suggest_categorical(
            "hidden_layer_sizes",
            [",".join(str(v) for v in hidden) for hidden in hidden_candidates],
        )
        activation = trial.suggest_categorical("activation", list(args.activation_candidates))
        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True)

        params = {
            "hidden_layer_sizes": tuple(int(v) for v in hidden_text.split(",")),
            "activation": activation,
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
        }
        mean_score, std_score, _ = evaluate_params_repeated_cv(
            params=params,
            x_trainval=x_trainval,
            y_trainval=y_trainval,
            split_specs=split_specs,
            seed=args.seed,
            max_iter=args.max_iter,
        )
        objective_value = mean_score - args.stability_weight * std_score
        trial.set_user_attr("mean_r2_log", mean_score)
        trial.set_user_attr("std_r2_log", std_score)
        return objective_value

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=args.optuna_trials,
        timeout=args.optuna_timeout,
        show_progress_bar=False,
    )

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
    if not completed_trials:
        raise RuntimeError("No successful Optuna trial completed.")

    top_k = min(args.top_k_configs, len(completed_trials))
    top_trials = completed_trials[:top_k]
    best_trial = top_trials[0]
    best_params = trial_params_to_model_params(best_trial)

    baseline_params = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
    }

    baseline_model = build_mlp_pipeline(
        hidden_layer_sizes=baseline_params["hidden_layer_sizes"],
        activation=baseline_params["activation"],
        alpha=baseline_params["alpha"],
        learning_rate_init=baseline_params["learning_rate_init"],
        max_iter=args.max_iter,
        seed=args.seed,
    )
    baseline_model.fit(x_trainval, y_trainval)

    best_single_model = build_mlp_pipeline(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        activation=best_params["activation"],
        alpha=best_params["alpha"],
        learning_rate_init=best_params["learning_rate_init"],
        max_iter=args.max_iter,
        seed=args.seed,
    )
    best_single_model.fit(x_trainval, y_trainval)

    ensemble_members: List[Dict[str, object]] = []
    ensemble_preds_trainval: List[np.ndarray] = []
    ensemble_preds_test: List[np.ndarray] = []
    for config_rank, trial in enumerate(top_trials, start=1):
        config = trial_params_to_model_params(trial)
        for member_i in range(args.members_per_config):
            seed = ensemble_seeds[(config_rank + member_i - 1) % len(ensemble_seeds)]
            model = build_mlp_pipeline(
                hidden_layer_sizes=config["hidden_layer_sizes"],
                activation=config["activation"],
                alpha=config["alpha"],
                learning_rate_init=config["learning_rate_init"],
                max_iter=args.max_iter,
                seed=seed,
            )
            model.fit(x_trainval, y_trainval)
            ensemble_preds_trainval.append(model.predict(x_trainval))
            ensemble_preds_test.append(model.predict(split.x_test))
            ensemble_members.append(
                {
                    "config_rank": config_rank,
                    "trial_number": int(trial.number),
                    "seed": int(seed),
                    "hidden_layer_sizes": list(config["hidden_layer_sizes"]),
                    "activation": config["activation"],
                    "alpha": float(config["alpha"]),
                    "learning_rate_init": float(config["learning_rate_init"]),
                    "trial_objective": float(trial.value),
                }
            )

    pred_baseline_trainval = baseline_model.predict(x_trainval)
    pred_baseline_test = baseline_model.predict(split.x_test)
    pred_best_trainval = best_single_model.predict(x_trainval)
    pred_best_test = best_single_model.predict(split.x_test)
    pred_ens_trainval = np.mean(np.vstack(ensemble_preds_trainval), axis=0)
    pred_ens_test = np.mean(np.vstack(ensemble_preds_test), axis=0)

    metrics_rows = []
    for model_name, y_tr_pred, y_te_pred in [
        ("MLP_Baseline", pred_baseline_trainval, pred_baseline_test),
        ("MLP_BestSingle", pred_best_trainval, pred_best_test),
        ("MLP_Ensemble", pred_ens_trainval, pred_ens_test),
    ]:
        row_trainval = regression_metrics(y_trainval, y_tr_pred)
        row_trainval.update({"model": model_name, "split": "trainval", "n_samples": int(len(y_trainval))})
        metrics_rows.append(row_trainval)

        row_test = regression_metrics(split.y_test, y_te_pred)
        row_test.update({"model": model_name, "split": "test", "n_samples": int(len(split.y_test))})
        metrics_rows.append(row_test)

    metrics_df = pd.DataFrame(metrics_rows)

    pred_table = build_prediction_table(
        df_model=dataset.df_model,
        idx_test=split.idx_test,
        preds={
            "baseline": pred_baseline_test,
            "best_single": pred_best_test,
            "ensemble": pred_ens_test,
        },
    )

    trial_rows = []
    for trial in completed_trials:
        params = trial_params_to_model_params(trial)
        trial_rows.append(
            {
                "number": int(trial.number),
                "objective_value": float(trial.value),
                "mean_r2_log": float(trial.user_attrs.get("mean_r2_log", np.nan)),
                "std_r2_log": float(trial.user_attrs.get("std_r2_log", np.nan)),
                "hidden_layer_sizes": ",".join(str(v) for v in params["hidden_layer_sizes"]),
                "activation": params["activation"],
                "alpha": float(params["alpha"]),
                "learning_rate_init": float(params["learning_rate_init"]),
            }
        )
    trials_df = pd.DataFrame(trial_rows).sort_values("objective_value", ascending=False).reset_index(drop=True)

    metrics_df.to_csv(args.output_dir / "round3_metrics.csv", index=False)
    pred_table.to_csv(args.output_dir / "round3_predictions_test.csv", index=False)
    trials_df.to_csv(args.output_dir / "round3_trials.csv", index=False)

    with open(args.output_dir / "round3_best_params.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "objective_name": "mean_r2_log_minus_stability_weight_times_std",
                "stability_weight": args.stability_weight,
                "best_objective_value": float(best_trial.value),
                "best_trial_number": int(best_trial.number),
                "best_params": {
                    "hidden_layer_sizes": list(best_params["hidden_layer_sizes"]),
                    "activation": best_params["activation"],
                    "alpha": best_params["alpha"],
                    "learning_rate_init": best_params["learning_rate_init"],
                },
            },
            file,
            indent=2,
        )

    with open(args.output_dir / "round3_ensemble_members.json", "w", encoding="utf-8") as file:
        json.dump(ensemble_members, file, indent=2)

    with open(args.output_dir / "round3_config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "data_path": str(args.data_path),
                "sheet_name": args.sheet_name,
                "seed": args.seed,
                "feature_names": dataset.feature_names,
                "include_ratio": include_ratio,
                "include_sinusoidal": include_sinusoidal,
                "include_silt": args.include_silt,
                "spatial_folds": args.spatial_folds,
                "blocks_x": args.blocks_x,
                "blocks_y": args.blocks_y,
                "repeats": args.repeats,
                "n_repeated_splits": len(split_specs),
                "optuna_trials": args.optuna_trials,
                "optuna_timeout": args.optuna_timeout,
                "max_iter": args.max_iter,
                "hidden_candidates": args.hidden_candidates,
                "activation_candidates": args.activation_candidates,
                "top_k_configs": top_k,
                "members_per_config": args.members_per_config,
                "ensemble_seeds": ensemble_seeds,
            },
            file,
            indent=2,
        )

    best_test_row = metrics_df[(metrics_df["model"] == "MLP_Ensemble") & (metrics_df["split"] == "test")].iloc[0]
    print("\n=== Round 3 completed ===")
    print(f"Rows used: {len(dataset.df_model)}")
    print(f"Features used ({len(dataset.feature_names)}): {dataset.feature_names}")
    print(f"Repeated spatial splits: {len(split_specs)}")
    print(f"Best trial objective: {float(best_trial.value):.4f}")
    print(f"Ensemble test R2 (log): {float(best_test_row['r2_log']):.4f}")
    print(f"Ensemble test RMSE (orig): {float(best_test_row['rmse_orig']):.2f} cm/dia")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
