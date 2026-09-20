import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import load_data
from pinn_model import init_params, forward_batch, make_loss_fn, train

np.random.seed(42)
RNG = np.random.RandomState(42)

# ---------------------------------------------------------------------------
# 1. Dados
# ---------------------------------------------------------------------------
d = load_data(os.path.join(os.path.dirname(__file__), '..', 'data', 'ksat_dataset.xlsx'))
X_cont, X_cat = d['X_cont'], d['X_cat']
cont_features, cat_features = d['cont_features'], d['cat_features']
y_log = d['y_log']
n = X_cont.shape[0]

idx = RNG.permutation(n)
n_test = int(0.15 * n)
n_val = int(0.15 * n)
test_idx = idx[:n_test]
val_idx = idx[n_test:n_test + n_val]
train_idx = idx[n_test + n_val:]

# padronizacao (so com stats de treino)
mu = X_cont[train_idx].mean(axis=0)
sigma = X_cont[train_idx].std(axis=0)
X_cont_std = (X_cont - mu) / sigma

X_all = np.concatenate([X_cont_std, X_cat], axis=1)
n_cont = X_cont.shape[1]

y_mu, y_sigma = y_log[train_idx].mean(), y_log[train_idx].std()
y_std = (y_log - y_mu) / y_sigma

X_train, X_val, X_test = jnp.array(X_all[train_idx]), jnp.array(X_all[val_idx]), jnp.array(X_all[test_idx])
y_train, y_val, y_test = jnp.array(y_std[train_idx]), jnp.array(y_std[val_idx]), jnp.array(y_std[test_idx])

print(f"Treino: {len(train_idx)}  Validação: {len(val_idx)}  Teste: {len(test_idx)}  "
      f"Features: {X_all.shape[1]} ({n_cont} continuas + {len(cat_features)} uso do solo)")

# indices das variaveis com restricao fisica (posicao dentro das continuas padronizadas)
feat_pos = {f: i for i, f in enumerate(cont_features)}
phys_idx = [feat_pos['Porosidade'], feat_pos['Macroporosidade'],
            feat_pos['Densidade'], feat_pos['Argila']]
phys_sign = [+1, +1, -1, -1]

layer_sizes = [X_all.shape[1], 32, 32, 1]

# ---------------------------------------------------------------------------
# 2. Modelo A: NN puramente orientada a dados (lambda_phys = 0)  -- baseline
# ---------------------------------------------------------------------------
key = jax.random.PRNGKey(0)
params0 = init_params(key, layer_sizes)
loss_plain = make_loss_fn(n_cont, phys_idx, phys_sign, lambda_phys=0.0, lambda_l2=1e-3)
print("\n=== Treinando modelo baseline (sem restricao fisica) ===")
params_plain, hist_plain = train(params0, X_train, y_train, X_val, y_val,
                                  loss_plain, n_epochs=5000, lr=2e-3, verbose_every=100, patience=15)

# ---------------------------------------------------------------------------
# 3. Modelo B: PINN (com restricao fisica de Darcy/Kozeny-Carman)
# ---------------------------------------------------------------------------
key2 = jax.random.PRNGKey(0)  # mesma inicializacao para comparacao justa
params0b = init_params(key2, layer_sizes)
loss_pinn = make_loss_fn(n_cont, phys_idx, phys_sign, lambda_phys=0.05, lambda_l2=1e-3)
print("\n=== Treinando PINN (com restricao fisica Darcy/Kozeny-Carman) ===")
params_pinn, hist_pinn = train(params0b, X_train, y_train, X_val, y_val,
                                loss_pinn, n_epochs=5000, lr=2e-3, verbose_every=100, patience=15)

# ---------------------------------------------------------------------------
# 4. Avaliacao
# ---------------------------------------------------------------------------
def evaluate(params, name):
    pred_train_std = np.array(forward_batch(params, X_train))
    pred_test_std = np.array(forward_batch(params, X_test))

    pred_train_log = pred_train_std * y_sigma + y_mu
    pred_test_log = pred_test_std * y_sigma + y_mu
    obs_train_log = y_log[train_idx]
    obs_test_log = y_log[test_idx]

    def metrics(obs, pred):
        rmse = np.sqrt(np.mean((obs - pred) ** 2))
        mae = np.mean(np.abs(obs - pred))
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - obs.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        return rmse, mae, r2

    rmse_tr, mae_tr, r2_tr = metrics(obs_train_log, pred_train_log)
    rmse_te, mae_te, r2_te = metrics(obs_test_log, pred_test_log)

    # tambem em escala original (cm/dia)
    obs_test_orig = 10 ** obs_test_log
    pred_test_orig = 10 ** pred_test_log
    rmse_orig = np.sqrt(np.mean((obs_test_orig - pred_test_orig) ** 2))

    print(f"\n--- {name} ---")
    print(f"Treino  (log10 Ksat): RMSE={rmse_tr:.3f}  MAE={mae_tr:.3f}  R2={r2_tr:.3f}")
    print(f"Teste   (log10 Ksat): RMSE={rmse_te:.3f}  MAE={mae_te:.3f}  R2={r2_te:.3f}")
    print(f"Teste   (cm/dia, escala original): RMSE={rmse_orig:.1f}")

    return dict(pred_train_log=pred_train_log, pred_test_log=pred_test_log,
                obs_train_log=obs_train_log, obs_test_log=obs_test_log,
                rmse_te=rmse_te, mae_te=mae_te, r2_te=r2_te)

res_plain = evaluate(params_plain, "Baseline (NN sem fisica)")
res_pinn = evaluate(params_pinn, "PINN (com restricao de Darcy/Kozeny-Carman)")

# ---------------------------------------------------------------------------
# 5. Checagem da consistencia fisica (sinais das derivadas no conjunto de teste)
# ---------------------------------------------------------------------------
import jax as _jax
def grad_check(params, name):
    grad_fn = _jax.grad(lambda p, x: forward_batch(p, x[None, :])[0], argnums=1)
    grads = np.array([grad_fn(params, X_test[i]) for i in range(X_test.shape[0])])
    print(f"\n--- Sinais das derivadas (fracao fisicamente consistente) -- {name} ---")
    labels = {'Porosidade': '+', 'Macroporosidade': '+', 'Densidade': '-', 'Argila': '-'}
    for feat, sign in labels.items():
        j = feat_pos[feat]
        g = grads[:, j]
        frac_ok = np.mean(g >= 0) if sign == '+' else np.mean(g <= 0)
        print(f"  d(Ksat)/d({feat}) deveria ser {sign} -> consistente em {frac_ok*100:.1f}% dos pontos de teste")

grad_check(params_plain, "Baseline")
grad_check(params_pinn, "PINN")

# ---------------------------------------------------------------------------
# 6. Graficos
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, res, title in zip(axes, [res_plain, res_pinn],
                           ["Baseline (sem física)", "PINN (Darcy/Kozeny-Carman)"]):
    ax.scatter(res['obs_test_log'], res['pred_test_log'], alpha=0.6, s=25, edgecolor='k', linewidth=0.3)
    lims = [min(res['obs_test_log'].min(), res['pred_test_log'].min()) - 0.2,
            max(res['obs_test_log'].max(), res['pred_test_log'].max()) + 0.2]
    ax.plot(lims, lims, 'r--', lw=1.5, label='1:1')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('log10(Ksat) observado')
    ax.set_ylabel('log10(Ksat) previsto')
    ax.set_title(f"{title}\nR²={res['r2_te']:.3f}  RMSE={res['rmse_te']:.3f}")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'parity_plot.png'), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
verbose_every = 100
ep_plain = np.arange(len(hist_plain['val_loss'])) * verbose_every
ep_pinn = np.arange(len(hist_pinn['val_loss'])) * verbose_every
ax.plot(ep_plain, hist_plain['val_loss'], label='Baseline - val loss', color='tab:orange')
ax.plot(ep_pinn, hist_pinn['val_loss'], label='PINN - val loss', color='tab:blue')
ax.plot(ep_pinn, hist_pinn['phys_loss'], label='PINN - resíduo físico', color='tab:green', linestyle='--')
ax.set_xlabel('época'); ax.set_ylabel('loss'); ax.set_yscale('log')
ax.set_title('Curvas de treinamento')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'training_curves.png'), dpi=150)
plt.close()

# Partial dependence: variar uma feature, manter as demais na mediana
def partial_dependence(params, feat_name, n_points=60):
    j = feat_pos[feat_name]
    x_grid_orig = np.linspace(X_cont[:, j].min(), X_cont[:, j].max(), n_points)
    x_grid_std = (x_grid_orig - mu[j]) / sigma[j]

    base_cont = np.median(X_cont_std, axis=0)
    base_cat = np.median(X_cat, axis=0)  # moda aproximada (uso mais comum via mediana one-hot)
    base = np.concatenate([base_cont, base_cat])
    X_pd = np.tile(base, (n_points, 1))
    X_pd[:, j] = x_grid_std
    preds_std = np.array(forward_batch(params, jnp.array(X_pd)))
    preds_log = preds_std * y_sigma + y_mu
    return x_grid_orig, 10 ** preds_log

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
pd_feats = ['Porosidade', 'Macroporosidade', 'Densidade', 'Argila']
for ax, feat in zip(axes.flat, pd_feats):
    xg, yb = partial_dependence(params_plain, feat)
    _, yp = partial_dependence(params_pinn, feat)
    ax.plot(xg, yb, label='Baseline', color='tab:orange')
    ax.plot(xg, yp, label='PINN', color='tab:blue')
    ax.set_xlabel(feat)
    ax.set_ylabel('Ksat previsto (cm/dia)')
    ax.set_yscale('log')
    ax.set_title(f'Dependência parcial: {feat}')
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'partial_dependence.png'), dpi=150)
plt.close()

print("\nGráficos salvos: parity_plot.png, training_curves.png, partial_dependence.png")

# salva metricas resumo
import json
summary = {
    'n_treino': int(len(train_idx)), 'n_validacao': int(len(val_idx)), 'n_teste': int(len(test_idx)),
    'baseline': {'r2_teste': res_plain['r2_te'], 'rmse_teste_log10': res_plain['rmse_te'],
                 'mae_teste_log10': res_plain['mae_te']},
    'pinn': {'r2_teste': res_pinn['r2_te'], 'rmse_teste_log10': res_pinn['rmse_te'],
             'mae_teste_log10': res_pinn['mae_te']},
}
with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics', 'baseline_vs_pinn_metrics.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
