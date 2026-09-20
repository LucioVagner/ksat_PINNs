"""
PINN (Physics-Informed Neural Network) para prever a condutividade hidraulica
saturada do solo (Ksat), usando a Lei de Darcy / Kozeny-Carman como restricao
fisica suave (soft constraint) sobre as derivadas da rede em relacao as
propriedades fisicas do solo.

Lei de Darcy (fluxo saturado, regime permanente):
    q = -Ksat * dh/dz   =>   sob gradiente unitario padrao de ensaio, q = Ksat

Equacao de Kozeny-Carman (derivada da Lei de Darcy para meios porosos):
    Ksat ~ (rho*g/mu) * phi^3 / ((1-phi)^2 * S0^2)

Dessa relacao extraimos os SINAIS fisicos esperados (nao os valores exatos,
pois Kozeny-Carman e idealizada para meios granulares nao agregados; solos
reais tem estrutura/agregacao). Esses sinais sao impostos via autograd
(dK/dx) da propria rede, exatamente como um PINN impoe o residuo de uma PDE:

    dKsat/d(Porosidade)      >= 0   (mais poros -> mais conducao, Darcy)
    dKsat/d(Macroporosidade) >= 0   (poros maiores -> mais fluxo, Darcy/Poiseuille)
    dKsat/d(Densidade solo)  <= 0   (mais compactacao -> menos poros -> Kozeny-Carman)
    dKsat/d(Argila)          <= 0   (particulas finas -> maior tortuosidade/S0 -> Kozeny-Carman)
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial

# ---------------------------------------------------------------------------
# Arquitetura da rede (MLP totalmente conectada)
# ---------------------------------------------------------------------------
def init_params(key, layer_sizes):
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for k, (n_in, n_out) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        wk, bk = jax.random.split(k)
        scale = jnp.sqrt(2.0 / n_in)
        W = jax.random.normal(wk, (n_in, n_out)) * scale
        b = jnp.zeros((n_out,))
        params.append((W, b))
    return params

def forward_single(params, x):
    """x: vetor 1D com todas as features (continuas padronizadas + one-hot uso)."""
    a = x
    for (W, b) in params[:-1]:
        a = jnp.tanh(a @ W + b)
    W, b = params[-1]
    out = a @ W + b
    return out[0]  # predicao escalar: log10(Ksat)

forward_batch = jax.vmap(forward_single, in_axes=(None, 0))

# ---------------------------------------------------------------------------
# Loss: dados + fisica (Darcy / Kozeny-Carman via autograd)
# ---------------------------------------------------------------------------
def make_loss_fn_v2(feat_pos, mu, sigma, y_mu, y_sigma, strategy, lambdas, lambda_l2,
                     ptf_coeffs=None):
    """
    Loss generalizada com 3 termos fisicos opcionais, inspirados no PINN do colega
    (run_pinn_ksat.py):

    - 'mono'     : monotonicidade das derivadas (igual ao make_loss_fn original)
    - 'ptf'      : residuo contra uma equacao de pedotransferencia (PTF) da literatura
                   (tipo Cosby et al.) relacionando log10(Ksat) a areia/argila/densidade
    - 'boundary' : penaliza previsoes fora de uma faixa fisicamente plausivel de log10(Ksat)

    strategy: 'baseline' | 'mono' | 'ptf' | 'mono+ptf' | 'full' (mono+ptf+boundary)
    """
    mono_idx = jnp.array([feat_pos['Porosidade'], feat_pos['Macroporosidade'],
                          feat_pos['Densidade'], feat_pos['Argila']])
    mono_sign = jnp.array([+1, +1, -1, -1], dtype=jnp.float32)

    if ptf_coeffs is None:
        ptf_coeffs = {'intercept': -0.60, 'sand': 1.15, 'clay': -0.50, 'bulk_density': -2.80}

    mu_j, sigma_j = jnp.array(mu), jnp.array(sigma)
    idx_sand, idx_clay, idx_bulk = feat_pos['Areia'], feat_pos['Argila'], feat_pos['Densidade']

    grad_fn = jax.grad(forward_single, argnums=1)
    grad_batch = jax.vmap(grad_fn, in_axes=(None, 0))

    def mono_loss(params, X):
        dK_dx = grad_batch(params, X)
        dK_phys = dK_dx[:, mono_idx]
        violation = jnp.maximum(0.0, -mono_sign * dK_phys)
        return jnp.mean(violation ** 2)

    def ptf_loss(params, X, y_pred_std):
        # desfaz a padronizacao para obter valores reais de areia/argila/densidade
        x_cont_std = X[:, :len(mu)]
        x_orig = x_cont_std * sigma_j + mu_j
        sand = x_orig[:, idx_sand]
        clay = x_orig[:, idx_clay]
        bulk = x_orig[:, idx_bulk]
        log_ksat_ptf = (
            ptf_coeffs['intercept']
            + ptf_coeffs['sand'] * jnp.log10(jnp.clip(sand, 0.0, None) + 1.0)
            + ptf_coeffs['clay'] * jnp.log10(jnp.clip(clay, 0.0, None) + 1.0)
            + ptf_coeffs['bulk_density'] * jnp.log10(jnp.clip(bulk, 0.8, None))
        )
        y_pred_log = y_pred_std * y_sigma + y_mu
        return jnp.mean((y_pred_log - log_ksat_ptf) ** 2)

    def boundary_loss(y_pred_std, y_min=-1.0, y_max=5.0):
        y_pred_log = y_pred_std * y_sigma + y_mu
        return jnp.mean(jnp.maximum(0.0, y_min - y_pred_log) ** 2 +
                         jnp.maximum(0.0, y_pred_log - y_max) ** 2)

    def loss_fn(params, X, y):
        preds = forward_batch(params, X)
        data_loss = jnp.mean((preds - y) ** 2)
        total = data_loss
        aux = {'data_loss': data_loss, 'mono_loss': 0.0, 'ptf_loss': 0.0, 'boundary_loss': 0.0}

        if strategy in ('mono', 'mono+ptf', 'full'):
            l_mono = mono_loss(params, X)
            total = total + lambdas.get('lambda_mono', 1.0) * l_mono
            aux['mono_loss'] = l_mono
        if strategy in ('ptf', 'mono+ptf', 'full'):
            l_ptf = ptf_loss(params, X, preds)
            total = total + lambdas.get('lambda_ptf', 0.5) * l_ptf
            aux['ptf_loss'] = l_ptf
        if strategy == 'full':
            l_bound = boundary_loss(preds)
            total = total + lambdas.get('lambda_boundary', 0.1) * l_bound
            aux['boundary_loss'] = l_bound

        l2 = sum(jnp.sum(W ** 2) for (W, _) in params)
        total = total + lambda_l2 * l2
        phys_total = total - data_loss - lambda_l2 * l2
        return total, (data_loss, phys_total)

    return loss_fn


def make_loss_fn(n_cont, phys_idx, phys_sign, lambda_phys, lambda_l2):
    """
    n_cont: numero de features continuas (as primeiras n_cont colunas de X)
    phys_idx: indices (na secao continua, ja padronizada) das variaveis com restricao fisica
    phys_sign: +1 (derivada deve ser >=0) ou -1 (derivada deve ser <=0), mesmo tamanho de phys_idx
    """
    phys_idx = jnp.array(phys_idx)
    phys_sign = jnp.array(phys_sign, dtype=jnp.float32)

    grad_fn = jax.grad(forward_single, argnums=1)  # dK_hat/dx completo
    grad_batch = jax.vmap(grad_fn, in_axes=(None, 0))

    def loss_fn(params, X, y):
        preds = forward_batch(params, X)
        data_loss = jnp.mean((preds - y) ** 2)

        # residuo fisico: penaliza derivadas com sinal fisicamente errado
        dK_dx = grad_batch(params, X)                      # (N, n_features)
        dK_phys = dK_dx[:, phys_idx]                        # (N, n_phys)
        violation = jnp.maximum(0.0, -phys_sign * dK_phys)  # hinge
        physics_loss = jnp.mean(violation ** 2)

        l2 = sum(jnp.sum(W ** 2) for (W, _) in params)

        total = data_loss + lambda_phys * physics_loss + lambda_l2 * l2
        return total, (data_loss, physics_loss)

    return loss_fn

# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------
def train(params, X_train, y_train, X_val, y_val, loss_fn,
          n_epochs=3000, lr=1e-3, seed=0, verbose_every=250, patience=15):
    """Treina com early stopping baseado em X_val/y_val (nao deve ser o teste final)."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, X, y):
        (loss, (dloss, ploss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, X, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, dloss, ploss

    history = {'train_loss': [], 'data_loss': [], 'phys_loss': [], 'val_loss': []}
    best_val = float('inf')
    best_params = params
    bad_checks = 0
    for epoch in range(n_epochs):
        params, opt_state, loss, dloss, ploss = step(params, opt_state, X_train, y_train)
        if epoch % verbose_every == 0 or epoch == n_epochs - 1:
            val_preds = forward_batch(params, X_val)
            val_loss = float(jnp.mean((val_preds - y_val) ** 2))
            history['train_loss'].append(float(loss))
            history['data_loss'].append(float(dloss))
            history['phys_loss'].append(float(ploss))
            history['val_loss'].append(val_loss)
            print(f"epoch {epoch:5d} | loss {float(loss):.5f} | data {float(dloss):.5f} "
                  f"| phys {float(ploss):.6f} | val {val_loss:.5f}")
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_params = params
                bad_checks = 0
            else:
                bad_checks += 1
                if bad_checks >= patience:
                    print(f"  early stopping na epoca {epoch} (melhor val={best_val:.5f})")
                    break
    return best_params, history
