"""
Validacao por grupo (leave-one-source-out, k=5) para o modelo PINN de Ksat.

Cada uma das 5 fontes de dados (estudos/regioes) e usada como conjunto de TESTE
uma vez, treinando nas outras 4. Isso testa se o modelo generaliza para um
estudo/regiao inteiramente nao vista (mais rigoroso que k-fold aleatorio).

Para cada fold, repetimos o treino com varias seeds (inicializacao da rede +
divisao treino/validacao interna) para quantificar a variabilidade do
treinamento em si, alem da variabilidade entre folds.

Resultado: media +/- IC 95% do R2 e RMSE, por fold e agregado.
"""
import numpy as np
import jax
import jax.numpy as jnp
import json

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import load_data
from pinn_model import init_params, forward_batch, make_loss_fn, train

N_SEEDS = 5
N_EPOCHS = 4000
PATIENCE = 12
VERBOSE_EVERY = 100
LR = 2e-3

d = load_data(os.path.join(os.path.dirname(__file__), '..', 'data', 'ksat_dataset.xlsx'))
df = d['df']
X_cont, X_cat = d['X_cont'], d['X_cat']
cont_features, cat_features = d['cont_features'], d['cat_features']
y_log = d['y_log']
feat_pos = {f: i for i, f in enumerate(cont_features)}
phys_idx = [feat_pos['Porosidade'], feat_pos['Macroporosidade'],
            feat_pos['Densidade'], feat_pos['Argila']]
phys_sign = [+1, +1, -1, -1]
n_cont = X_cont.shape[1]
layer_sizes = [X_cont.shape[1] + X_cat.shape[1], 32, 32, 1]

fontes = df['Fonte dos dados'].unique().tolist()
print("Fontes (grupos) para leave-one-group-out:", fontes)


def metrics(obs, pred):
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    mae = float(np.mean(np.abs(obs - pred)))
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return rmse, mae, r2


def run_fold(test_fonte, lambda_phys, seed, rng_np):
    test_mask = (df['Fonte dos dados'] == test_fonte).values
    trainval_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    # separa 15% do trainval para early stopping (validacao interna)
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
    X_test = jnp.array(X_all[test_idx]); y_test_std = jnp.array(y_std[test_idx])

    key = jax.random.PRNGKey(seed)
    params0 = init_params(key, layer_sizes)
    loss_fn = make_loss_fn(n_cont, phys_idx, phys_sign, lambda_phys=lambda_phys, lambda_l2=1e-3)
    params, hist = train(params0, X_train, y_train, X_val, y_val, loss_fn,
                          n_epochs=N_EPOCHS, lr=LR, verbose_every=VERBOSE_EVERY,
                          patience=PATIENCE)

    pred_test_std = np.array(forward_batch(params, X_test))
    pred_test_log = pred_test_std * y_sigma + y_mu
    obs_test_log = y_log[test_idx]
    rmse, mae, r2 = metrics(obs_test_log, pred_test_log)
    return dict(fonte=test_fonte, seed=seed, n_test=len(test_idx),
                rmse=rmse, mae=mae, r2=r2)


def ci95(values):
    values = np.array(values)
    m = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return m, 1.96 * se


results = {'baseline': [], 'pinn': []}
for model_name, lam in [('baseline', 0.0), ('pinn', 0.05)]:
    print(f"\n========== Modelo: {model_name} (lambda_fisica={lam}) ==========")
    for fonte in fontes:
        for s in range(N_SEEDS):
            rng_np = np.random.RandomState(1000 * s + hash(fonte) % 997)
            res = run_fold(fonte, lam, seed=s, rng_np=rng_np)
            results[model_name].append(res)
            print(f"  [{model_name}] fonte-teste='{fonte[:25]:25s}' seed={s} "
                  f"n_test={res['n_test']:3d}  R2={res['r2']:.3f}  RMSE={res['rmse']:.3f}")

# ---------------------------------------------------------------------------
# Agregacao
# ---------------------------------------------------------------------------
summary = {}
for model_name in ['baseline', 'pinn']:
    rows = results[model_name]
    r2_all = [r['r2'] for r in rows]
    rmse_all = [r['rmse'] for r in rows]
    m_r2, ci_r2 = ci95(r2_all)
    m_rmse, ci_rmse = ci95(rmse_all)

    per_fonte = {}
    for fonte in fontes:
        r2_f = [r['r2'] for r in rows if r['fonte'] == fonte]
        rmse_f = [r['rmse'] for r in rows if r['fonte'] == fonte]
        m_r2f, ci_r2f = ci95(r2_f)
        m_rmsef, ci_rmsef = ci95(rmse_f)
        per_fonte[fonte] = {'r2_mean': m_r2f, 'r2_ci95': ci_r2f,
                             'rmse_mean': m_rmsef, 'rmse_ci95': ci_rmsef,
                             'n_test': rows[0]['n_test'] if False else
                             [r['n_test'] for r in rows if r['fonte'] == fonte][0]}

    summary[model_name] = {
        'r2_mean': m_r2, 'r2_ci95': ci_r2,
        'rmse_mean': m_rmse, 'rmse_ci95': ci_rmse,
        'per_fonte': per_fonte,
        'n_runs': len(rows),
    }

print("\n\n================ RESUMO FINAL (leave-one-source-out, %d seeds) ================" % N_SEEDS)
for model_name in ['baseline', 'pinn']:
    s = summary[model_name]
    print(f"\n--- {model_name} ---")
    print(f"R2 geral:   {s['r2_mean']:.3f} +/- {s['r2_ci95']:.3f} (IC 95%%, n={s['n_runs']} execucoes)")
    print(f"RMSE geral: {s['rmse_mean']:.3f} +/- {s['rmse_ci95']:.3f} (log10 Ksat)")
    print("Por fonte (grupo mantido de fora):")
    for fonte, v in s['per_fonte'].items():
        print(f"  {fonte[:35]:35s} n={v['n_test']:3d}  R2={v['r2_mean']:.3f}+/-{v['r2_ci95']:.3f}  "
              f"RMSE={v['rmse_mean']:.3f}+/-{v['rmse_ci95']:.3f}")

with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics', 'leave_one_group_out_metrics.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics', 'leave_one_group_out_raw_results.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nSalvo em leave_one_group_out_metrics.json e leave_one_group_out_raw_results.json")
