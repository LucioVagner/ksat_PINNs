"""
Aplica as 5 estrategias de PINN do colega (baseline, mono, ptf, mono+ptf, full)
no dataset COMPLETO (5 fontes), usando leave-one-source-out (mesma validacao
rigorosa que ja usamos) + varias seeds.
"""
import numpy as np
import jax
import jax.numpy as jnp
import json

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import load_data
from pinn_model import init_params, forward_batch, make_loss_fn_v2, train

N_SEEDS = 4
N_EPOCHS = 4000
PATIENCE = 12
VERBOSE_EVERY = 200
LR = 2e-3
STRATEGIES = ['baseline', 'mono', 'ptf', 'mono+ptf', 'full']
LAMBDAS = {'lambda_mono': 0.05, 'lambda_ptf': 0.3, 'lambda_boundary': 0.05}

d = load_data(os.path.join(os.path.dirname(__file__), '..', 'data', 'ksat_dataset.xlsx'))
df = d['df']
X_cont, X_cat = d['X_cont'], d['X_cat']
cont_features, cat_features = d['cont_features'], d['cat_features']
y_log = d['y_log']
feat_pos = {f: i for i, f in enumerate(cont_features)}
n_cont = X_cont.shape[1]
layer_sizes = [X_cont.shape[1] + X_cat.shape[1], 32, 32, 1]
fontes = df['Fonte dos dados'].unique().tolist()


def metrics(obs, pred):
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return rmse, r2


def run_fold(test_fonte, strategy, seed):
    test_mask = (df['Fonte dos dados'] == test_fonte).values
    trainval_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    rng_np = np.random.RandomState(1000 * seed + (hash(test_fonte) % 997))
    perm = rng_np.permutation(len(trainval_idx))
    n_val = max(int(0.15 * len(trainval_idx)), 5)
    val_idx = trainval_idx[perm[:n_val]]
    train_idx = trainval_idx[perm[n_val:]]

    mu = X_cont[train_idx].mean(axis=0)
    sigma = X_cont[train_idx].std(axis=0)
    sigma[sigma == 0] = 1.0
    X_cont_std = (X_cont - mu) / sigma
    X_all = np.concatenate([X_cont_std, X_cat], axis=1)

    y_mu, y_sigma = y_log[train_idx].mean(), y_log[train_idx].std()
    y_std = (y_log - y_mu) / y_sigma

    X_train = jnp.array(X_all[train_idx]); y_train = jnp.array(y_std[train_idx])
    X_val = jnp.array(X_all[val_idx]); y_val = jnp.array(y_std[val_idx])
    X_test = jnp.array(X_all[test_idx])

    key = jax.random.PRNGKey(seed)
    params0 = init_params(key, layer_sizes)
    loss_fn = make_loss_fn_v2(feat_pos, mu, sigma, y_mu, y_sigma, strategy, LAMBDAS, lambda_l2=1e-3)
    params, hist = train(params0, X_train, y_train, X_val, y_val, loss_fn,
                          n_epochs=N_EPOCHS, lr=LR, verbose_every=VERBOSE_EVERY, patience=PATIENCE)

    pred_test_std = np.array(forward_batch(params, X_test))
    pred_test_log = pred_test_std * y_sigma + y_mu
    obs_test_log = y_log[test_idx]
    rmse, r2 = metrics(obs_test_log, pred_test_log)
    return dict(fonte=test_fonte, strategy=strategy, seed=seed, n_test=len(test_idx), rmse=rmse, r2=r2)


def ci95(values):
    values = np.array(values)
    m = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return m, 1.96 * se


all_rows = []
for strategy in STRATEGIES:
    print(f"\n========== Estrategia: {strategy} ==========")
    for fonte in fontes:
        for s in range(N_SEEDS):
            res = run_fold(fonte, strategy, seed=s)
            all_rows.append(res)
    r2_strat = [r['r2'] for r in all_rows if r['strategy'] == strategy]
    m, ci = ci95(r2_strat)
    print(f"  >> {strategy}: R2 medio (todas as fontes/seeds) = {m:.3f} +/- {ci:.3f}")

print("\n\n================ RESUMO FINAL (leave-one-source-out, %d seeds) ================" % N_SEEDS)
summary = {}
for strategy in STRATEGIES:
    rows = [r for r in all_rows if r['strategy'] == strategy]
    r2_all = [r['r2'] for r in rows]
    rmse_all = [r['rmse'] for r in rows]
    m_r2, ci_r2 = ci95(r2_all)
    m_rmse, ci_rmse = ci95(rmse_all)
    per_fonte = {}
    for fonte in fontes:
        r2_f = [r['r2'] for r in rows if r['fonte'] == fonte]
        m_f, ci_f = ci95(r2_f)
        per_fonte[fonte] = {'r2_mean': m_f, 'r2_ci95': ci_f}
    summary[strategy] = {'r2_mean': m_r2, 'r2_ci95': ci_r2, 'rmse_mean': m_rmse,
                          'rmse_ci95': ci_rmse, 'per_fonte': per_fonte, 'n_runs': len(rows)}
    print(f"\n--- {strategy} ---")
    print(f"R2 geral: {m_r2:.3f} +/- {ci_r2:.3f}  |  RMSE geral (log10): {m_rmse:.3f} +/- {ci_rmse:.3f}")
    for fonte, v in per_fonte.items():
        print(f"    {fonte[:35]:35s} R2={v['r2_mean']:.3f}+/-{v['r2_ci95']:.3f}")

with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics', 'physics_constraint_strategies_metrics.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics', 'physics_constraint_strategies_raw_results.json'), 'w') as f:
    json.dump(all_rows, f, indent=2, ensure_ascii=False)
print("\nSalvo em physics_constraint_strategies_metrics.json")
