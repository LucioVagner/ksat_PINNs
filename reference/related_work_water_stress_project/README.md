# Previsao de Ksat com Redes Neurais (BHSE)

Pipeline para modelagem de `Ksat` na Bacia Hidrografica da Sanga Ellert (Cangucu/RS), com foco em:

- alvo transformado: `log10(Ksat)`;
- split estratificado 70/15/15 por quartis do alvo;
- validacao cruzada espacial por blocos geograficos;
- comparacao entre `MLP` (baseline), `RBFNN`, `RandomForest` e `XGBoost` (opcional).

## 1. Instalacao

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## 2. Execucao rapida (recomendada)

```bash
.venv/bin/python train_ksat_models.py
```

Isso usa por padrao:

- arquivo `Pontos_Ksat_Sanga_Ellert.xlsx`;
- modelos `mlp rbf rf xgb` (se `xgboost` nao estiver instalado, ele e ignorado);
- split `70/15/15`;
- `Spatial K-Fold` com `k=5` e blocos `4x4`.

## 3. Execucao focada no baseline MLP

```bash
.venv/bin/python train_ksat_models.py --models mlp --mlp-hidden 128 64 32
```

## 4. Execucao com features espaciais derivadas

```bash
.venv/bin/python train_ksat_models.py \
  --include-ratio \
  --include-sinusoidal \
  --models mlp rbf rf
```

## 5. GPU (apenas XGBoost)

```bash
.venv/bin/python train_ksat_models.py --models mlp rbf rf xgb --use-gpu
```

Observacao: no script atual, `--use-gpu` acelera apenas `XGBoost`.  
`MLPRegressor` do scikit-learn roda em CPU (adequado para `n=169`).

## 6. Principais saidas

Arquivos gerados em `outputs/`:

- `metrics_holdout.csv`: metricas por modelo em treino/val/teste;
- `metrics_spatial_cv_folds.csv`: metricas por fold espacial;
- `metrics_spatial_cv_summary.csv`: media e desvio padrao por modelo;
- `predictions_test.csv`: predicoes no conjunto de teste (log e original);
- `spatial_blocks_all_points.csv`: bloco geografico de cada ponto;
- `permutation_importance_MLP.csv`: importancia por permutacao (se MLP executado);
- `run_config.json`: parametros e mapeamento de colunas.

Se `matplotlib` estiver disponivel, tambem gera graficos de dispersao e comparacao.

## 7. Metricas implementadas

Todas no espaco `log10` e no espaco original (`cm/dia`):

- `R2`
- `RMSE`
- `MAE`
- `PBIAS (%)`

## 8. Colunas esperadas na planilha

O script detecta automaticamente colunas equivalentes, com base em:

- `X`, `Y`
- `Areia (%)`, `Silte (%)`, `Argila (%)`
- `DS (g/cm3)`
- `Macroporosidade (%)`
- `Ksat (cm/dia)`

## 9. Tuning automatico do MLP (Grid/Optuna)

Script dedicado:

```bash
.venv/bin/python tune_mlp.py --method grid
```

Arquivos gerados (padrao em `outputs_mlp_tuning/`):

- `mlp_tuning_trials.csv`
- `mlp_best_params.json`
- `mlp_tuned_vs_baseline_metrics.csv`
- `mlp_tuned_predictions_test.csv`
- `mlp_tuning_config.json`

Exemplo com grade menor (mais rapido):

```bash
.venv/bin/python tune_mlp.py \
  --method grid \
  --grid-hidden 64,32 128,64 \
  --grid-alpha 1e-4 1e-3 \
  --grid-lr 5e-4 1e-3
```

Exemplo com Optuna:

```bash
.venv/bin/python tune_mlp.py --method optuna --optuna-trials 80
```

## 10. Rodada 3 (generalizacao): Repeated Spatial CV + Ensemble MLP

Script dedicado:

```bash
.venv/bin/python run_mlp_round3.py
```

Padrao da rodada 3:

- tuning com Optuna em CV espacial repetida (blocos com offsets);
- objetivo de tuning = media do `R2_log` - penalidade por instabilidade entre folds;
- features derivadas espaciais ligadas por padrao (`sand_clay_ratio` + `sin/cos` de X/Y);
- ensemble com multiplas configuracoes top + multiplas seeds.

Saidas em `outputs_mlp_round3/`:

- `round3_metrics.csv`
- `round3_predictions_test.csv`
- `round3_trials.csv`
- `round3_best_params.json`
- `round3_ensemble_members.json`
- `round3_config.json`
