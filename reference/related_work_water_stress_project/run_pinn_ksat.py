#!/usr/bin/env python3
"""Physics-Informed Neural Network for Ksat prediction with multiple constraint strategies."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_ksat_models import (
    EPS,
    RANDOM_STATE_DEFAULT,
    create_stratified_holdout_split,
    load_dataset,
    prepare_dataset,
    regression_metrics,
)

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

STRATEGIES = ["baseline", "mono", "ptf", "mono+ptf", "full"]


# ---------------------------------------------------------------------------
# PINN Model
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        activation: str = "tanh",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        act_fn = nn.Tanh if activation == "tanh" else nn.ReLU
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PINNPipeline:
    """Wraps a StandardScaler (numpy) + PINN (torch) into a single object."""

    def __init__(self, model: PINN, scaler_mean: np.ndarray, scaler_std: np.ndarray) -> None:
        self.model = model
        self.scaler_mean = torch.tensor(scaler_mean, dtype=torch.float32)
        self.scaler_std = torch.tensor(scaler_std, dtype=torch.float32)

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.scaler_mean.to(x.device)) / self.scaler_std.to(x.device)

    def unscale(self, x_scaled: torch.Tensor) -> torch.Tensor:
        return x_scaled * self.scaler_std.to(x_scaled.device) + self.scaler_mean.to(x_scaled.device)


# ---------------------------------------------------------------------------
# Physics loss functions (operate on SCALED inputs via pipeline)
# ---------------------------------------------------------------------------

def _monotonicity_violations(
    pipeline: PINNPipeline,
    x_scaled: torch.Tensor,
    feature_idx: Dict[str, int],
) -> torch.Tensor:
    x_scaled = x_scaled.clone().detach().requires_grad_(True)
    y = pipeline.model(pipeline.scale(x_scaled))
    violations = torch.tensor(0.0, device=x_scaled.device)

    grad_map = {
        "macroporosity": 1,
        "bulk_density": -1,
        "sand": 1,
        "clay": -1,
    }
    for feat, sign in grad_map.items():
        idx = feature_idx[feat]
        grad = torch.autograd.grad(
            y, x_scaled, grad_outputs=torch.ones_like(y), create_graph=True
        )[0]
        violation = torch.relu(-sign * grad[:, idx])
        violations = violations + violation.mean()

    return violations


def _ptf_residual(
    pipeline: PINNPipeline,
    x_scaled: torch.Tensor,
    y_pred: torch.Tensor,
    feature_idx: Dict[str, int],
    ptf_coeffs: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    x_orig = pipeline.unscale(x_scaled)
    sand = x_orig[:, feature_idx["sand"]]
    clay = x_orig[:, feature_idx["clay"]]
    bulk = x_orig[:, feature_idx["bulk_density"]]

    if ptf_coeffs is not None:
        intercept = ptf_coeffs.get("intercept", 4.542)
        c_sand = ptf_coeffs.get("sand", 0.213)
        c_clay = ptf_coeffs.get("clay", -1.273)
        c_bulk = ptf_coeffs.get("bulk_density", -4.126)
    else:
        intercept, c_sand, c_clay, c_bulk = -0.60, 1.15, -0.50, -2.80

    log_ksat_ptf = (
        intercept
        + c_sand * torch.log10(torch.clamp(sand, min=0.0) + 1.0)
        + c_clay * torch.log10(torch.clamp(clay, min=0.0) + 1.0)
        + c_bulk * torch.log10(torch.clamp(bulk, min=0.8))
    )
    return torch.mean((y_pred - log_ksat_ptf) ** 2)


def _boundary_loss(y_pred: torch.Tensor, y_min: float = -1.0, y_max: float = 5.0) -> torch.Tensor:
    return torch.relu(-y_pred + y_min).mean() + torch.relu(y_pred - y_max).mean()


def compute_physics_loss(
    pipeline: PINNPipeline,
    x_scaled: torch.Tensor,
    y_pred: torch.Tensor,
    strategy: str,
    feature_idx: Dict[str, int],
    lambdas: Dict[str, float],
    ptf_coeffs: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses: Dict[str, float] = {}
    total = torch.tensor(0.0, device=x_scaled.device)

    if strategy in ("mono", "mono+ptf", "full"):
        l_mono = _monotonicity_violations(pipeline, x_scaled, feature_idx)
        total = total + lambdas.get("lambda_mono", 1.0) * l_mono
        losses["loss_mono"] = float(l_mono)

    if strategy in ("ptf", "mono+ptf", "full"):
        l_ptf = _ptf_residual(pipeline, x_scaled, y_pred, feature_idx, ptf_coeffs)
        total = total + lambdas.get("lambda_ptf", 0.5) * l_ptf
        losses["loss_ptf"] = float(l_ptf)

    if strategy == "full":
        l_bound = _boundary_loss(y_pred)
        total = total + lambdas.get("lambda_boundary", 0.1) * l_bound
        losses["loss_boundary"] = float(l_bound)

    return total, losses


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    physics_loss: List[float] = field(default_factory=list)
    epoch: List[int] = field(default_factory=list)


class PINNTrainer:
    def __init__(
        self,
        pipeline: PINNPipeline,
        strategy: str,
        lambdas: Dict[str, float],
        lr: float,
        patience: int,
        feature_idx: Dict[str, int],
        ptf_coeffs: Optional[Dict[str, float]] = None,
    ) -> None:
        self.pipeline = pipeline
        self.strategy = strategy
        self.lambdas = lambdas
        self.lr = lr
        self.patience = patience
        self.feature_idx = feature_idx
        self.ptf_coeffs = ptf_coeffs
        self.history = TrainingHistory()

    def _compute_loss(
        self, x_scaled: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.pipeline.model.train()
        y_pred = self.pipeline.model(x_scaled)
        data_loss = nn.functional.mse_loss(y_pred, y)

        if self.strategy == "baseline":
            return data_loss, {"loss_data": float(data_loss.detach())}

        phys_loss, phys_details = compute_physics_loss(
            self.pipeline, x_scaled, y_pred, self.strategy, self.feature_idx, self.lambdas, self.ptf_coeffs
        )
        total = data_loss + phys_loss
        details = {"loss_data": float(data_loss.detach())}
        details.update(phys_details)
        return total, details

    @torch.no_grad()
    def _val_loss(self, x_scaled: torch.Tensor, y: torch.Tensor) -> float:
        self.pipeline.model.eval()
        pred = self.pipeline.model(x_scaled)
        return float(nn.functional.mse_loss(pred, y))

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> TrainingHistory:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train)
        x_val_sc = scaler.transform(x_val)

        self.pipeline.scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
        self.pipeline.scaler_std = torch.tensor(scaler.scale_, dtype=torch.float32)

        x_t = torch.tensor(x_train_sc, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        x_v = torch.tensor(x_val_sc, dtype=torch.float32)
        y_v = torch.tensor(y_val, dtype=torch.float32)

        ds = TensorDataset(x_t, y_t)
        loader = DataLoader(ds, batch_size=min(batch_size, len(x_t)), shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(self.pipeline.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.patience // 3
        )

        best_val = float("inf")
        best_state = None
        wait = 0

        for ep in range(1, epochs + 1):
            self.pipeline.model.train()
            epoch_loss = 0.0
            epoch_phys = 0.0
            n_batches = 0

            for xb, yb in loader:
                optimizer.zero_grad()
                total_loss, details = self._compute_loss(xb, yb)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pipeline.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += float(details.get("loss_data", total_loss))
                epoch_phys += float(details.get("loss_mono", 0)) + float(
                    details.get("loss_ptf", 0)
                )
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            avg_phys = epoch_phys / max(n_batches, 1)
            vl = self._val_loss(x_v, y_v)

            self.history.train_loss.append(avg_train)
            self.history.val_loss.append(vl)
            self.history.physics_loss.append(avg_phys)
            self.history.epoch.append(ep)

            scheduler.step(vl)

            if vl < best_val - 1e-6:
                best_val = vl
                best_state = {k: v.clone() for k, v in self.pipeline.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.pipeline.model.load_state_dict(best_state)
        return self.history

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        self.pipeline.model.eval()
        x_sc = (x - self.pipeline.scaler_mean.numpy()) / self.pipeline.scaler_std.numpy()
        xt = torch.tensor(x_sc, dtype=torch.float32)
        return self.pipeline.model(xt).numpy()


# ---------------------------------------------------------------------------
# Feature index helper
# ---------------------------------------------------------------------------

def build_feature_idx(feature_names: Sequence[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(feature_names)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_strategy(
    strategy: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    hidden_layers: Sequence[int],
    activation: str,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    lambdas: Dict[str, float],
    seed: int,
    ptf_coeffs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PINN(
        input_dim=len(feature_names),
        hidden_layers=hidden_layers,
        activation=activation,
    )
    feature_idx = build_feature_idx(feature_names)

    pipeline = PINNPipeline(model, scaler_mean=np.zeros(len(feature_names)), scaler_std=np.ones(len(feature_names)))

    trainer = PINNTrainer(
        pipeline=pipeline,
        strategy=strategy,
        lambdas=lambdas,
        lr=lr,
        patience=patience,
        feature_idx=feature_idx,
        ptf_coeffs=ptf_coeffs,
    )

    history = trainer.fit(x_train, y_train, x_val, y_val, epochs, batch_size)

    pred_train = trainer.predict(x_train)
    pred_val = trainer.predict(x_val)
    pred_test = trainer.predict(x_test)

    metrics = {}
    for split_name, yt, yp in [
        ("train", y_train, pred_train),
        ("val", y_val, pred_val),
        ("test", y_test, pred_test),
    ]:
        row = regression_metrics(yt, yp)
        row.update({"strategy": strategy, "split": split_name, "n_samples": int(len(yt))})
        metrics[split_name] = row

    return {
        "strategy": strategy,
        "model": model,
        "trainer": trainer,
        "history": history,
        "metrics": metrics,
        "pred_test": pred_test,
    }


def run_spatial_cv(
    strategy: str,
    x_all: np.ndarray,
    y_all: np.ndarray,
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    feature_names: Sequence[str],
    blocks_x: int,
    blocks_y: int,
    n_splits: int,
    hidden_layers: Sequence[int],
    activation: str,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    lambdas: Dict[str, float],
    seed: int,
) -> pd.DataFrame:
    from train_ksat_models import compute_spatial_blocks

    groups = compute_spatial_blocks(x_coord, y_coord, blocks_x, blocks_y)
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, unique_groups.size)
    if n_splits < 2:
        warnings.warn("Not enough spatial groups for CV.")
        return pd.DataFrame()

    from sklearn.model_selection import GroupKFold

    splitter = GroupKFold(n_splits=n_splits)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x_all, y_all, groups=groups), start=1):
        torch.manual_seed(seed)
        np.random.seed(seed)

        x_tr, y_tr = x_all[train_idx], y_all[train_idx]
        x_te, y_te = x_all[test_idx], y_all[test_idx]

        n_val = max(1, int(len(x_tr) * 0.15))
        rng = np.random.default_rng(seed)
        val_idx = rng.choice(len(x_tr), size=n_val, replace=False)
        mask = np.ones(len(x_tr), dtype=bool)
        mask[val_idx] = False
        xv_tr, yv_tr = x_tr[mask], y_tr[mask]
        xv_val, yv_val = x_tr[val_idx], y_tr[val_idx]

        feature_idx = build_feature_idx(feature_names)
        model = PINN(
            input_dim=len(feature_names),
            hidden_layers=hidden_layers,
            activation=activation,
        )
        pipeline = PINNPipeline(model, scaler_mean=np.zeros(len(feature_names)), scaler_std=np.ones(len(feature_names)))
        trainer = PINNTrainer(
            pipeline=pipeline,
            strategy=strategy,
            lambdas=lambdas,
            lr=lr,
            patience=patience,
            feature_idx=feature_idx,
        )
        trainer.fit(xv_tr, yv_tr, xv_val, yv_val, epochs, batch_size)
        pred = trainer.predict(x_te)

        row = regression_metrics(y_te, pred)
        row.update({
            "strategy": strategy,
            "fold": int(fold),
            "n_samples": int(len(test_idx)),
        })
        fold_rows.append(row)

    return pd.DataFrame(fold_rows)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def build_test_predictions_table(
    y_test: np.ndarray,
    pred_test: Dict[str, np.ndarray],
    idx_test: np.ndarray,
    df_model: pd.DataFrame,
) -> pd.DataFrame:
    table = df_model.iloc[idx_test][["x", "y", "ksat", "log_ksat"]].copy()
    table = table.rename(columns={"ksat": "ksat_true", "log_ksat": "log_ksat_true"})
    table["sample_idx"] = idx_test
    table = table.reset_index(drop=True)

    for strat, pred_log in pred_test.items():
        table[f"log_ksat_pred_{strat}"] = pred_log
        table[f"ksat_pred_{strat}"] = np.power(10.0, pred_log)
        table[f"residual_log_{strat}"] = pred_log - table["log_ksat_true"].values

    return table


def maybe_plot_outputs(
    all_results: List[Dict[str, object]],
    test_predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not installed; skipping plots.")
        return

    for res in all_results:
        strat = res["strategy"]
        hist = res["history"]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist.epoch, hist.train_loss, label="Train (data)", alpha=0.8)
        ax.plot(hist.epoch, hist.val_loss, label="Val", alpha=0.8)
        if any(v > 0 for v in hist.physics_loss):
            ax.plot(hist.epoch, hist.physics_loss, label="Physics", alpha=0.6, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Curva de Treino - {strat}")
        ax.legend()
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(output_dir / f"loss_curve_{strat}.png", dpi=150)
        plt.close(fig)

    for strat in [r["strategy"] for r in all_results]:
        col_pred = f"log_ksat_pred_{strat}"
        if col_pred not in test_predictions.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        x = test_predictions["ksat_true"].values
        y = test_predictions[f"ksat_pred_{strat}"].values
        ax.scatter(x, y, alpha=0.7, edgecolor="black", linewidth=0.3, s=30)
        xy_min = min(np.min(x), np.min(y))
        xy_max = max(np.max(x), np.max(y))
        ax.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", color="red")
        ax.set_xlabel("Ksat observado (cm/dia)")
        ax.set_ylabel("Ksat previsto (cm/dia)")
        ax.set_title(f"PINN Teste Holdout - {strat}")
        fig.tight_layout()
        fig.savefig(output_dir / f"scatter_pinn_{strat}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    strats = [r["strategy"] for r in all_results]
    r2_vals = [r["metrics"]["test"]["r2_log"] for r in all_results]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]
    ax.bar(strats, r2_vals, color=colors[: len(strats)])
    ax.set_ylabel("R2 log (teste)")
    ax.set_title("Comparacao PINN - Todas as Estrategias")
    fig.tight_layout()
    fig.savefig(output_dir / "pinn_strategy_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PINN for Ksat prediction: physics-informed neural network with multiple constraint strategies."
    )
    parser.add_argument("--data-path", type=Path, default=Path("data.xlsx"))
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_pinn"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE_DEFAULT)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)

    parser.add_argument(
        "--strategy",
        nargs="+",
        default=STRATEGIES,
        choices=STRATEGIES,
        help="Physics constraint strategies to evaluate.",
    )
    parser.add_argument("--hidden-layers", nargs="+", type=int, default=[128, 64, 32])
    parser.add_argument("--activation", default="tanh", choices=["tanh", "relu"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--lambda-mono", type=float, default=1.0)
    parser.add_argument("--lambda-ptf", type=float, default=0.5)
    parser.add_argument("--lambda-boundary", type=float, default=0.1)

    parser.add_argument("--ptf-intercept", type=float, default=4.542)
    parser.add_argument("--ptf-sand", type=float, default=0.213)
    parser.add_argument("--ptf-clay", type=float, default=-1.273)
    parser.add_argument("--ptf-bulk", type=float, default=-4.126)

    parser.add_argument("--spatial-cv", action="store_true", help="Run spatial CV on best strategy.")
    parser.add_argument("--spatial-folds", type=int, default=5)
    parser.add_argument("--blocks-x", type=int, default=4)
    parser.add_argument("--blocks-y", type=int, default=4)

    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--include-silt", action="store_true")
    parser.add_argument("--include-ratio", action="store_true")
    parser.add_argument("--include-sinusoidal", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lambdas = {
        "lambda_mono": args.lambda_mono,
        "lambda_ptf": args.lambda_ptf,
        "lambda_boundary": args.lambda_boundary,
    }

    ptf_coeffs = {
        "intercept": args.ptf_intercept,
        "sand": args.ptf_sand,
        "clay": args.ptf_clay,
        "bulk_density": args.ptf_bulk,
    }

    df_raw = load_dataset(args.data_path, sheet_name=args.sheet_name)
    dataset = prepare_dataset(
        df_raw=df_raw,
        include_silt=args.include_silt,
        include_ratio=args.include_ratio,
        include_sinusoidal=args.include_sinusoidal,
    )

    x_all = dataset.df_model[dataset.feature_names].to_numpy(dtype=float)
    y_all = dataset.df_model["log_ksat"].to_numpy(dtype=float)

    split = create_stratified_holdout_split(
        x=x_all,
        y=y_all,
        random_state=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    print(f"\nDataset: {len(dataset.df_model)} samples, {len(dataset.feature_names)} features")
    print(f"Features: {dataset.feature_names}")
    print(f"Strategies: {args.strategy}")
    print()

    all_results: List[Dict[str, object]] = []
    pred_test_dict: Dict[str, np.ndarray] = {}

    for strat in args.strategy:
        print(f"--- Strategy: {strat} ---")
        result = evaluate_strategy(
            strategy=strat,
            x_train=split.x_train,
            y_train=split.y_train,
            x_val=split.x_val,
            y_val=split.y_val,
            x_test=split.x_test,
            y_test=split.y_test,
            feature_names=dataset.feature_names,
            hidden_layers=args.hidden_layers,
            activation=args.activation,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            lambdas=lambdas,
            seed=args.seed,
            ptf_coeffs=ptf_coeffs,
        )
        all_results.append(result)
        pred_test_dict[strat] = result["pred_test"]

        m = result["metrics"]["test"]
        print(f"  Test R2 (log): {m['r2_log']:.4f}  |  RMSE (orig): {m['rmse_orig']:.2f} cm/dia")
        print(f"  Train R2 (log): {result['metrics']['train']['r2_log']:.4f}")
        print()

    holdout_rows = []
    for res in all_results:
        for split_name in ("train", "val", "test"):
            holdout_rows.append(res["metrics"][split_name])
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df = holdout_df.sort_values(["split", "r2_log"], ascending=[True, False]).reset_index(drop=True)
    holdout_df.to_csv(args.output_dir / "pinn_metrics_holdout.csv", index=False)

    test_preds = build_test_predictions_table(
        y_test=split.y_test,
        pred_test=pred_test_dict,
        idx_test=split.idx_test,
        df_model=dataset.df_model,
    )
    test_preds.to_csv(args.output_dir / "pinn_predictions_test.csv", index=False)

    best_result = max(all_results, key=lambda r: r["metrics"]["test"]["r2_log"])
    best_strat = best_result["strategy"]

    history_rows = []
    for res in all_results:
        hist = res["history"]
        for i in range(len(hist.epoch)):
            history_rows.append({
                "strategy": res["strategy"],
                "epoch": hist.epoch[i],
                "train_loss": hist.train_loss[i],
                "val_loss": hist.val_loss[i],
                "physics_loss": hist.physics_loss[i],
            })
    pd.DataFrame(history_rows).to_csv(args.output_dir / "pinn_training_curves.csv", index=False)

    if args.spatial_cv:
        print(f"--- Spatial CV ({best_strat}) ---")
        x_coord = dataset.df_model["x"].to_numpy(dtype=float)
        y_coord = dataset.df_model["y"].to_numpy(dtype=float)

        spatial_df = run_spatial_cv(
            strategy=best_strat,
            x_all=x_all,
            y_all=y_all,
            x_coord=x_coord,
            y_coord=y_coord,
            feature_names=dataset.feature_names,
            blocks_x=args.blocks_x,
            blocks_y=args.blocks_y,
            n_splits=args.spatial_folds,
            hidden_layers=args.hidden_layers,
            activation=args.activation,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            lambdas=lambdas,
            seed=args.seed,
        )
        if not spatial_df.empty:
            summary = (
                spatial_df.groupby("strategy")
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
                )
                .reset_index()
            )
            spatial_df.to_csv(args.output_dir / "pinn_metrics_spatial_cv_folds.csv", index=False)
            summary.to_csv(args.output_dir / "pinn_metrics_spatial_cv_summary.csv", index=False)
            print(f"  Spatial CV R2 (log): {summary['r2_log_mean'].iloc[0]:.4f} +/- {summary['r2_log_std'].iloc[0]:.4f}")
        print()

    if not args.no_plots:
        maybe_plot_outputs(all_results, test_preds, args.output_dir)

    baseline_path = Path("outputs_data") / "metrics_holdout.csv"
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        baseline_test = baseline_df[baseline_df["split"] == "test"].copy()
        comp_rows = []
        for _, row in baseline_test.iterrows():
            comp_rows.append({
                "model": row["model"],
                "r2_log": row["r2_log"],
                "rmse_orig": row["rmse_orig"],
                "source": "baseline",
            })
        for res in all_results:
            m = res["metrics"]["test"]
            comp_rows.append({
                "model": f"PINN_{res['strategy']}",
                "r2_log": m["r2_log"],
                "rmse_orig": m["rmse_orig"],
                "source": "pinn",
            })
        pd.DataFrame(comp_rows).to_csv(args.output_dir / "pinn_comparison_baseline.csv", index=False)

    with open(args.output_dir / "pinn_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": str(args.data_path),
                "sheet_name": args.sheet_name,
                "seed": args.seed,
                "feature_names": dataset.feature_names,
                "strategies": args.strategy,
                "hidden_layers": args.hidden_layers,
                "activation": args.activation,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "lambdas": lambdas,
                "best_strategy": best_strat,
                "best_test_r2_log": float(best_result["metrics"]["test"]["r2_log"]),
                "best_test_rmse_orig": float(best_result["metrics"]["test"]["rmse_orig"]),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=== PINN Results Summary ===")
    print(f"Rows used: {len(dataset.df_model)}")
    print(f"Features: {dataset.feature_names}")
    print()
    for res in all_results:
        m = res["metrics"]["test"]
        tag = " <-- BEST" if res["strategy"] == best_strat else ""
        print(
            f"  {res['strategy']:12s}  R2_log={m['r2_log']:.4f}  RMSE_orig={m['rmse_orig']:.2f} cm/dia{tag}"
        )
    print(f"\nBest strategy: {best_strat}")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
