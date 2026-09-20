# Processos e Metodos - Previsao de Ksat (BHSE)

## 1. Objetivo do baseline

Este documento consolida o que foi feito para construir e avaliar a baseline de previsao de `Ksat` na BHSE, incluindo:

- arquitetura do modelo;
- pipeline de dados e processos de treino/validacao;
- abordagem de IA/ML usada;
- resultados obtidos;
- comparacao com tecnicas de augmentation.

Baseline de referencia considerada neste projeto:

- `MLP tuned` com `R2_log = 0.7502` no conjunto de teste real.


## 2. Dados usados

Fonte:

- `Pontos_Ksat_Sanga_Ellert.xlsx`

Amostras:

- total: `169` pontos;
- camada: `0-20 cm`;
- teste final: apenas pontos reais medidos.

Colunas mapeadas no pipeline:

- `X` -> `x`
- `Y` -> `y`
- `Areia (%)` -> `sand`
- `Argila (%)` -> `clay`
- `DS (g/cm3)` -> `bulk_density`
- `Macroporosidade (%)` -> `macroporosity`
- `Ksat (cm/dia)` -> alvo bruto

Transformacao do alvo:

- alvo de treino: `log_ksat = log10(Ksat)`.


## 3. Arquitetura da baseline (MLP)

Implementacao:

- `scikit-learn` com `Pipeline(StandardScaler -> MLPRegressor)`.

Arquitetura final da baseline (tuning Optuna):

- `hidden_layer_sizes = (128, 64)`
- `activation = tanh`
- `solver = adam`
- `alpha = 0.0011380907649808582`
- `learning_rate_init = 0.0024792013326598596`
- `early_stopping = True`
- `validation_fraction = 0.15`
- `n_iter_no_change = 40`
- `random_state = 42`

Features de entrada da baseline:

- `x`, `y`, `sand`, `clay`, `bulk_density`, `macroporosity`.


## 4. Processo de treino e validacao

### 4.1 Split e preprocessamento

- split estratificado por quartis de `log_ksat`;
- proporcao `70/15/15` (`train/val/test`);
- `seed = 42`;
- padronizacao com `StandardScaler` dentro do pipeline do modelo.

### 4.2 Avaliacao

Metricas calculadas em dois espacos:

- espaco log: `R2_log`, `RMSE_log`, `MAE_log`, `PBIAS_log`;
- espaco original: `R2_orig`, `RMSE_orig`, `MAE_orig`, `PBIAS_orig`.

### 4.3 Validacao espacial

Tambem foi feita avaliacao por blocos geograficos (`GroupKFold` espacial), para reduzir otimismo por dependencia espacial local.


## 5. Processo de IA executado no projeto

### Etapa A - Baseline inicial e comparativos

Script:

- `train_ksat_models.py`

Modelos comparados:

- `MLP`, `RBFNN`, `RandomForest`, `XGBoost`.

Resultado holdout (teste) nessa etapa:

- `RandomForest`: `R2_log = 0.7658`
- `RBFNN`: `R2_log = 0.7406`
- `XGBoost`: `R2_log = 0.7322`
- `MLP inicial`: `R2_log = -0.1465`

Conclusao:

- MLP inicial precisava tuning agressivo.

### Etapa B - Tuning do MLP

Script:

- `tune_mlp.py`

Rodadas:

- GridSearch;
- Optuna.

Melhor resultado do MLP (definido como baseline oficial do MLP):

- `R2_log teste = 0.7501716919` (`~0.7502`).

### Etapa C - Rodadas adicionais de tuning

Rodada 2 com busca maior (`grid` e `optuna`) foi executada e nao superou a baseline `0.7502` no teste.

### Etapa D - Benchmark de augmentation

Script:

- `benchmark_augmentation_ksat.py`

Tecnicas testadas:

- interpolacao: `kriging`, `idw/rbf linear`, `thin plate spline`;
- ruido: `gaussian jitter`, `coordinate jitter`, `ksat multiplicative`, `combined noise`;
- sintetico: `copula gaussian`, `smogn-like` (fallback);
- bootstrap: `spatial block bootstrap ensemble`;
- fisica: `monte carlo measurement`, `ptf pretrain finetune`.


## 6. Resultados principais

## 6.1 Baseline oficial do MLP (sem augmentation)

Fonte:

- `outputs_mlp_tuning_optuna/mlp_tuned_vs_baseline_metrics.csv`

`MLP_Tuned` (teste):

- `R2_log = 0.7501716919`
- `RMSE_log = 0.2744442907`
- `MAE_log = 0.2191963685`
- `R2_orig = 0.6121214180`
- `RMSE_orig = 1323.8716709`
- `MAE_orig = 935.7350256`

## 6.2 Benchmark de augmentation (comparado com baseline 0.7502)

Fonte:

- `outputs_augmentation_benchmark/augmentation_benchmark_results.csv`

Melhores resultados no teste:

1. `copula_gaussian`: `R2_log = 0.7644664071` -> **superou baseline 0.7502**
2. `baseline_no_aug`: `R2_log = 0.7501716919`
3. `kriging_covariate_composition`: `R2_log = 0.6589143584`
4. `spatial_block_bootstrap_ensemble`: `R2_log = 0.6568195946`

Demais tecnicas nao superaram a baseline.


## 7. Interpretacao tecnica

- O melhor MLP sem augmentation foi obtido com tuning de hiperparametros e alvo em log.
- Algumas tecnicas de augmentation degradaram forte a generalizacao (especialmente interpoladores sinteticos diretos em `log_ksat` e pretrain PTF desta configuracao).
- A tecnica que trouxe ganho real sobre a baseline foi:
  - `copula_gaussian` (`+0.0143` em `R2_log` sobre `0.7502`).
- Esse ganho deve ser interpretado com cautela porque:
  - o teste tem `26` amostras;
  - em datasets pequenos, variacoes de split podem alterar ranking entre metodos.


## 8. Artefatos e reprodutibilidade

Scripts principais:

- `train_ksat_models.py`
- `tune_mlp.py`
- `benchmark_augmentation_ksat.py`

Arquivos de resultado usados neste documento:

- `outputs/metrics_holdout.csv`
- `outputs/metrics_spatial_cv_summary.csv`
- `outputs/run_config.json`
- `outputs_mlp_tuning_optuna/mlp_best_params.json`
- `outputs_mlp_tuning_optuna/mlp_tuned_vs_baseline_metrics.csv`
- `outputs_augmentation_benchmark/augmentation_benchmark_results.csv`
- `outputs_augmentation_benchmark/augmentation_benchmark_config.json`

Comandos principais:

```bash
.venv/bin/python train_ksat_models.py --no-plots
.venv/bin/python tune_mlp.py --method optuna --optuna-trials 60 --output-dir outputs_mlp_tuning_optuna
.venv/bin/python benchmark_augmentation_ksat.py --output-dir outputs_augmentation_benchmark --max-iter 4000
```

