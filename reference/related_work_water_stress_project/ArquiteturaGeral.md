# Arquitetura Geral - Projeto de Previsao de Ksat (BHSE)

## 1. Objetivo deste documento

Este documento descreve, em detalhe, a arquitetura tecnica e o processo completo dos experimentos executados para previsao de `Ksat` na BHSE, cobrindo:

- objetivo de cada experimento;
- desenho metodologico;
- configuracoes e hiperparametros usados;
- resultados obtidos;
- interpretacao tecnica de cada etapa;
- consolidacao final do que funcionou e do que nao funcionou.

Escopo temporal dos experimentos: todas as rodadas realizadas neste repositorio ate a geracao de `outputs_ceiling_strategies/`.

## 1.1 Infraestrutura computacional

Ambiente de execucao informado para o projeto:

- GPU: `NVIDIA RTX 5090`
- CPU: `Intel Core Ultra 9 285K`
- Ambiente Python: `.venv` local do repositorio

Embora varios modelos usados aqui sejam CPU-friendly (MLP sklearn, RF, LGBM), a infraestrutura permite escalar para buscas maiores.


## 2. Contexto do problema e dados

### 2.1 Dataset e alvo

- Arquivo fonte: `Pontos_Ksat_Sanga_Ellert.xlsx`
- Amostras: `169` pontos
- Camada: `0-20 cm`
- Variavel alvo fisica: `Ksat (cm/dia)`
- Variavel alvo de modelagem: `log_ksat = log10(Ksat)`

O alvo em log foi usado em todos os pipelines principais por estabilidade numerica e reducao do efeito de cauda longa.

### 2.2 Mapeamento de colunas para variaveis internas

- `X` -> `x`
- `Y` -> `y`
- `Areia (%)` -> `sand`
- `Silte (%)` -> `silt`
- `Argila (%)` -> `clay`
- `DS (g/cm3)` -> `bulk_density`
- `Macroporosidade (%)` -> `macroporosity`
- `Ksat (cm/dia)` -> `ksat`

### 2.3 Features base

Conjunto base usado na maior parte dos experimentos:

- `x`, `y`, `sand`, `clay`, `bulk_density`, `macroporosity`


## 3. Regras gerais de avaliacao

### 3.1 Split principal

Quando o experimento era de holdout:

- `train/val/test = 70% / 15% / 15%`
- `seed = 42`
- estratificacao por quartis de `log_ksat`
- conjunto de teste sempre com pontos reais medidos

### 3.2 Validacao espacial

Nos experimentos com CV espacial:

- divisao por blocos geograficos (`4x4`) usando `GroupKFold`
- `5` folds espaciais

Objetivo: reduzir vazamento espacial e estimar melhor generalizacao fora da vizinhanca imediata.

### 3.3 Metricas

Todas as rotinas calcularam metricas em dois espacos:

- espaco log: `R2_log`, `RMSE_log`, `MAE_log`, `PBIAS_log`
- espaco original: `R2_orig`, `RMSE_orig`, `MAE_orig`, `PBIAS_orig`

### 3.4 Controle de vazamento e consistencia experimental

Regras adotadas para evitar inflacao artificial de resultado:

- `log10(Ksat)` calculado antes da modelagem, mas sem uso de informacao do teste para ajuste de parametros de modelo;
- padronizacao (`StandardScaler`) dentro de `Pipeline`, evitando fit do scaler no conjunto de teste;
- nos experimentos de augmentation, amostras sinteticas foram geradas apenas no conjunto de treino;
- o conjunto de teste holdout permaneceu composto por dados reais medidos;
- em validacoes robustas (RKF, MCCV, SLOO), as metricas foram agregadas por distribuicao e nao por um unico split.


## 4. Experimento A - Baseline multimodelos (primeira fotografia)

Script: `train_ksat_models.py`  
Saidas: `outputs/`

### 4.1 Objetivo

Criar uma linha de base inicial com modelos diversos para:

- medir desempenho de referencia;
- verificar se MLP sem tuning ja era competitivo;
- comparar modelos com vies diferente (rede neural, kernel, arvores boosting/bagging).

### 4.2 Modelos avaliados

- `MLP`
- `RBFNN` (RBF features + Ridge)
- `RandomForest`
- `XGBoost`

### 4.3 Resultado holdout (teste, 26 pontos)

`outputs/metrics_holdout.csv`:

- `RandomForest`: `R2_log = 0.7658`
- `RBFNN`: `R2_log = 0.7406`
- `XGBoost`: `R2_log = 0.7322`
- `MLP`: `R2_log = -0.1465`

### 4.4 Resultado CV espacial (media dos folds)

`outputs/metrics_spatial_cv_summary.csv`:

- `RandomForest`: `R2_log_mean = 0.4142`
- `RBFNN`: `R2_log_mean = 0.4007`
- `XGBoost`: `R2_log_mean = 0.3393`
- `MLP`: `R2_log_mean = 0.1522`

### 4.5 Conclusao da etapa

- MLP inicial estava subotimo e precisava tuning.
- RandomForest abriu com melhor resultado em holdout e tambem boa robustez espacial relativa.
- Queda entre holdout e CV espacial mostrou que o problema tem dependencia espacial relevante e maior dificuldade fora da vizinhanca local.


## 5. Experimento B - Tuning do MLP (rodada 1)

Script: `tune_mlp.py`  
Saidas: `outputs_mlp_tuning/` e `outputs_mlp_tuning_optuna/`

### 5.1 Objetivo

Recuperar desempenho do MLP e testar se era possivel aproximar/superar modelos de arvore sem mudar features base.

### 5.2 Configuracao rodada 1

- Features: `x, y, sand, clay, bulk_density, macroporosity`
- CV espacial: `5` folds em blocos `4x4`
- Metodos de busca:
  - Grid (`outputs_mlp_tuning`)
  - Optuna (`outputs_mlp_tuning_optuna`, `60` trials)

### 5.3 Melhores hiperparametros (Optuna rodada 1)

`outputs_mlp_tuning_optuna/mlp_best_params.json`:

- `hidden_layer_sizes = (128, 64)`
- `activation = tanh`
- `alpha = 0.0011380907649808582`
- `learning_rate_init = 0.0024792013326598596`

### 5.4 Resultados

`outputs_mlp_tuning_optuna/mlp_tuned_vs_baseline_metrics.csv`:

- `MLP_Tuned` no teste: `R2_log = 0.7501716919` (referencia `0.7502`)
- `MLP_Baseline` no teste: `R2_log = 0.3031283011`

### 5.5 Conclusao da etapa

- Tuning foi efetivo para o MLP no split fixo.
- O valor `0.7502` virou baseline oficial da linha MLP neste projeto.
- Em paralelo, o resultado de RF da etapa A (`0.7658`) permaneceu como referencia forte de metodo alternativo.


## 6. Experimento C - Tuning do MLP (rodada 2, busca maior)

Script: `tune_mlp.py`  
Saidas: `outputs_mlp_tuning_round2_grid/` e `outputs_mlp_tuning_round2_optuna/`

### 6.1 Objetivo

Testar espaco de busca mais amplo (mais camadas/regularizacao/taxa de aprendizado) para tentar empurrar MLP acima de `0.80`.

### 6.2 Configuracao rodada 2

- Grid expandido com arquiteturas ate `256,128,64`
- `max_iter = 6000`
- Optuna com `300` trials
- mesmas 6 features base

### 6.3 Resultados no teste

- Melhor Grid R2 teste: `0.6625`
- Melhor Optuna R2 teste: `0.7055`

Nenhum superou a baseline MLP `0.7502`.

### 6.4 Conclusao da etapa

- A expansao do espaco de tuning nao trouxe ganho no split de teste.
- Sinal de que o teto do MLP com estas features e esta amostragem pode estar abaixo do esperado em varios cenarios.


## 7. Experimento D - Round 3 parcial (MLP com features espaciais derivadas + ensemble)

Script: `run_mlp_round3.py`  
Saidas: `outputs_mlp_round3_partial/`

### 7.1 Objetivo

Fazer uma rodada mais avancada para MLP com:

- feature engineering espacial (`sand_clay_ratio`, codificacao senoidal);
- repeticao de splits;
- montagem de ensemble de candidatos.

### 7.2 Configuracao executada

`outputs_mlp_round3_partial/round3_partial_config.json`:

- features:
  - base 6
  - `sand_clay_ratio`
  - `sin_x`, `cos_x`, `sin_y`, `cos_y`
- `repeats = 6`
- `spatial_folds = 5`
- `candidate_count = 6`
- `ensemble_member_count = 18`
- `max_iter = 8000`

Obs.: rodada interrompida antes do fim planejado.

### 7.3 Resultados (teste)

`outputs_mlp_round3_partial/round3_partial_metrics.csv`:

- `MLP_Baseline`: `R2_log = 0.5330`
- `MLP_BestSingle_Partial`: `R2_log = 0.5133`
- `MLP_Ensemble_Partial`: `R2_log = 0.6524`

### 7.4 Conclusao da etapa

- Nesta rodada parcial, nao houve superacao da baseline `0.7502`.
- Ensemble parcial melhorou sobre candidatos isolados, mas ficou abaixo da meta.
- Como a rodada foi interrompida antes do plano completo, os resultados devem ser tratados como exploratorios.


## 8. Experimento E - Benchmark de Augmentation

Script: `benchmark_augmentation_ksat.py`  
Saidas: `outputs_augmentation_benchmark/`

### 8.1 Objetivo

Testar se augmentations fisicos/espaciais/sinteticos melhoram o MLP tuned no teste real.

### 8.2 Metodologia

- Treino em `train+val` real (`143` pontos)
- Teste em `26` pontos reais
- MLP base fixo (parametros Optuna rodada 1)
- Cada metodo gera dados adicionais apenas no treino

### 8.3 Familias e metodos testados

- Baseline sem augment
- Ruido: jitter gaussiano, jitter coordenada, ruido multiplicativo em Ksat, combinado
- Interpolacao: IDW/RBF linear, thin plate spline, kriging composicional
- Sintetico: copula gaussiana, SMOGN-like
- Bootstrap espacial em ensemble
- Fisica: Monte Carlo e pretrain/fine-tune com PTF simplificada

### 8.4 Ranking principal (R2_log no teste)

`outputs_augmentation_benchmark/augmentation_benchmark_results.csv`:

1. `copula_gaussian`: `0.7645` (superou `0.7502`)
2. `baseline_no_aug`: `0.7502`
3. `kriging_covariate_composition`: `0.6589`
4. `spatial_block_bootstrap_ensemble`: `0.6568`

Metodos com piora acentuada:

- `idw_rbf_linear`: `-22.1360`
- `thin_plate_spline`: `-38.6921`
- `ptf_pretrain_finetune`: `-21.5161`

### 8.5 Conclusao da etapa

- Em split unico, copula gaussiana deu ganho positivo (`+0.0143` sobre `0.7502`).
- Varios augmentations causaram degradacao severa, indicando risco alto de gerar distribuicoes artificiais nao alinhadas ao teste real.


## 9. Experimento F - Validacao robusta "virtual test" (estabilidade estatistica)

Script: `evaluate_virtual_test_strategies.py`  
Saidas: `outputs_virtual_test_eval/`

### 9.1 Objetivo

Verificar se ganhos em split unico eram reais ou ruido de amostragem.

Comparacao principal:

- `baseline` (MLP tuned sem augment)
- `copula` (treino aumentado com copula)

### 9.2 Estrategias avaliadas

- Repeated Stratified K-Fold: `5 folds x 10 repeticoes` (`50` pares)
- MCCV: `200` iteracoes (`200` pares)
- Spatial LOO com buffer: `169` avaliacoes (1 por ponto), `buffer=50m`

### 9.3 Resultados

`outputs_virtual_test_eval/virtual_eval_summary.json`:

Repeated Stratified K-Fold:

- Baseline: `0.4590 +/- 0.1637`
- Copula: `0.5471 +/- 0.1090`
- Delta medio: `+0.0880`
- Wilcoxon: `p = 1.98e-05`

MCCV:

- Baseline: `0.4807 +/- 0.1591`
- Copula: `0.5593 +/- 0.1135`
- Delta medio: `+0.0786`
- Wilcoxon: `p = 1.22e-18`

Spatial LOO + buffer:

- Baseline: `R2_log = 0.5102`
- Copula: `R2_log = 0.5545`
- Wilcoxon erro absoluto: `p = 0.3806` (nao significativo)

### 9.4 Conclusao da etapa

- O ganho da copula apareceu de forma consistente em RKF e MCCV.
- Entretanto, o nivel absoluto medio ficou muito abaixo de `0.7502`.
- Isso indica que `0.7502` e um valor possivelmente otimista de split unico, nao um desempenho robusto medio.
- Em termos de publicacao tecnica, a avaliacao robusta e mais confiavel para inferencia de desempenho esperado fora da amostra.


## 10. Experimento G - Estrategias para quebrar teto 0.77-0.78

Script: `benchmark_ceiling_strategies.py`  
Saidas: `outputs_ceiling_strategies/`

### 10.1 Objetivo

Testar pacote completo de melhorias:

- feature engineering fisica e espacial;
- tuning de RandomForest com Optuna;
- tuning de LightGBM com Optuna;
- stacking (`MLP + RF + LGBM`);
- transformacao Box-Cox do alvo;
- selecao de features por SHAP;
- modelagem hibrida de residuos (Linear + MLP).

### 10.2 Features derivadas implementadas

- `sand_clay_ratio`
- `macro_ds_ratio`
- `clay_ds_product`
- `macro_squared`
- `sand_macro`
- `log_ds`
- `inv_clay`
- `x_norm`, `y_norm`
- `dist_center`
- `spatial_interaction`

### 10.3 Tuning executado

RF (`70` trials):

- melhor CV: `0.5419`
- params: `n_estimators=582`, `max_depth=21`, `min_samples_split=3`, `min_samples_leaf=5`, `max_features=0.3898`, `max_samples=0.8788`

LightGBM (`90` trials):

- melhor CV: `0.5207`
- params: `n_estimators=166`, `learning_rate=0.0331`, `num_leaves=68`, `min_child_samples=33`, `subsample=0.9444`, `colsample_bytree=0.9352`, `reg_alpha=0.0134`, `reg_lambda=0.0389`

### 10.4 Box-Cox

- `lambda global = 0.2484` (diferente de zero, portanto log puro nao e transformacao identica ao Box-Cox otimo global)

### 10.5 Selecao por SHAP

- `16` features selecionadas (de 17 candidatas)
- mais importantes: `sand_macro`, `macroporosity`, `macro_squared`, `macro_ds_ratio`

### 10.6 Resultado holdout (70/15/15)

`outputs_ceiling_strategies/ceiling_holdout_results.csv`:

- `rf_optuna_engineered`: `R2_log = 0.7593` (superou `0.7502`)
- `mlp_baseline_6f`: `0.7502` (quase empate, abaixo por margem minima)
- `stack_mlp_rf_lgbm`: `0.7471`
- `hybrid_linear_plus_mlp_residual`: `0.7094`
- `lgbm_optuna_engineered`: `0.7049`
- `mlp_engineered`: `0.7005`
- `mlp_engineered_shap_selected`: `0.6936`
- `mlp_engineered_boxcox_target`: `0.6097`

### 10.7 Resultado MCCV (80 iteracoes)

`outputs_ceiling_strategies/ceiling_mccv_summary.csv`:

- `rf_optuna_engineered`: `0.6125 +/- 0.0986`
- `stack_mlp_rf_lgbm`: `0.6053 +/- 0.0999`
- `lgbm_optuna_engineered`: `0.5718 +/- 0.1046`
- `mlp_baseline_6f`: `0.4920 +/- 0.1749`
- `hybrid_linear_plus_mlp_residual`: `0.4893 +/- 0.1754`
- `mlp_engineered`: `0.4438 +/- 0.2174`
- `mlp_engineered_shap_selected`: `0.4058 +/- 0.2638`
- `mlp_engineered_boxcox_target`: `0.2259 +/- 0.5904` (`3` falhas por predito nao finito)

### 10.8 Conclusao da etapa

- Melhor modelo robusto deste pacote foi RF tunado com features engenheiradas.
- Houve superacao de `0.7502` no holdout, mas nao em media robusta MCCV.
- Teto de `0.77-0.78` nao foi quebrado de forma consistente.
- O componente que mais agregou sinal foi a engenharia de interacoes com macroporosidade, especialmente em modelos de arvore.


## 11. Consolidacao global dos experimentos

### 11.0 Linha do tempo sintetica (objetivo -> resultado)

1. Baseline multimodelos:
   objetivo: estabelecer referencia inicial.
   resultado: RF (`0.7658`) melhor no holdout; MLP inicial muito fraco.
2. Tuning MLP rodada 1:
   objetivo: recuperar MLP.
   resultado: MLP tuned chegou a `0.7502` no teste.
3. Tuning MLP rodada 2:
   objetivo: romper teto do MLP com busca maior.
   resultado: sem ganho sobre `0.7502`.
4. Round3 parcial:
   objetivo: adicionar features espaciais derivadas e ensemble.
   resultado: melhoria parcial, sem bater baseline.
5. Augmentation benchmark:
   objetivo: testar ganho de dados sinteticos.
   resultado: copula (`0.7645`) venceu no split unico.
6. Virtual test robusto:
   objetivo: medir estabilidade estatistica do ganho.
   resultado: copula melhora baseline MLP, mas medias ficam na faixa `~0.55`.
7. Ceiling strategies:
   objetivo: pacote completo para superar `0.77-0.78`.
   resultado: RF tuned atinge `0.7593` holdout, mas MCCV medio `0.6125`.

### 11.1 Melhores numeros por perspectiva

Melhor split holdout observado:

- `RandomForest` da etapa inicial: `R2_log = 0.7658`
- `copula + MLP` em benchmark de augmentation: `R2_log = 0.7645`
- `RF tuned + engineered` no benchmark de teto: `R2_log = 0.7593`

Melhor robustez (media MCCV/estrategias repetidas):

- Benchmark de teto (MCCV): `RF tuned = 0.6125`
- Virtual test (MCCV copula MLP): `0.5593`
- Virtual test (MCCV baseline MLP): `0.4807`

### 11.2 Leitura tecnica final

- Split unico de 26 pontos pode superestimar desempenho.
- Avaliacoes repetidas/robustas mostraram queda relevante no R2 medio.
- RF (principalmente tunado) foi o metodo mais estavel ate aqui.
- MLP teve desempenho sensivel a split e a configuracao.
- Augmentation por copula foi o unico aumento sintetico com ganho consistente frente ao MLP baseline, mas insuficiente para consolidar `R2_log > 0.75` em media robusta.

### 11.3 O que estes resultados significam na pratica

- Se a meta for "melhor numero de holdout", ha cenarios > `0.75`.
- Se a meta for "desempenho esperado em uso real", a evidencia atual aponta para faixa media menor.
- Para decisoes tecnicas de producao/publicacao, as metricas robustas (MCCV/RKF/SLOO) devem ter prioridade sobre um unico split.


## 12. Artefatos de codigo e resultados

### 12.1 Scripts

- `train_ksat_models.py`
- `tune_mlp.py`
- `run_mlp_round3.py`
- `benchmark_augmentation_ksat.py`
- `evaluate_virtual_test_strategies.py`
- `benchmark_ceiling_strategies.py`

### 12.2 Pastas de saida

- `outputs/`
- `outputs_mlp_tuning/`
- `outputs_mlp_tuning_optuna/`
- `outputs_mlp_tuning_round2_grid/`
- `outputs_mlp_tuning_round2_optuna/`
- `outputs_mlp_round3_partial/`
- `outputs_augmentation_benchmark/`
- `outputs_virtual_test_eval/`
- `outputs_ceiling_strategies/`

### 12.3 Comandos executados (reproducao)

```bash
.venv/bin/python train_ksat_models.py --no-plots
.venv/bin/python tune_mlp.py --method grid --output-dir outputs_mlp_tuning
.venv/bin/python tune_mlp.py --method optuna --optuna-trials 60 --output-dir outputs_mlp_tuning_optuna
.venv/bin/python tune_mlp.py --method grid --max-iter 6000 --output-dir outputs_mlp_tuning_round2_grid
.venv/bin/python tune_mlp.py --method optuna --max-iter 6000 --optuna-trials 300 --output-dir outputs_mlp_tuning_round2_optuna
.venv/bin/python run_mlp_round3.py --output-dir outputs_mlp_round3_partial
.venv/bin/python benchmark_augmentation_ksat.py --output-dir outputs_augmentation_benchmark --max-iter 4000
.venv/bin/python evaluate_virtual_test_strategies.py --output-dir outputs_virtual_test_eval
.venv/bin/python benchmark_ceiling_strategies.py --rf-trials 70 --lgbm-trials 90 --mccv-iter 80 --output-dir outputs_ceiling_strategies
```

### 12.4 Limitacoes metodologicas registradas

- Tamanho amostral pequeno (`n=169`) e teste holdout pequeno (`n=26`) tornam o ranking de split unico instavel.
- Parte dos resultados depende de seed e composicao de fold, mesmo com estratificacao.
- A rodada `round3` ficou parcial (interrompida), logo nao representa busca exaustiva.
- Alguns pipelines mais agressivos (ex.: Box-Cox + MLP) produziram instabilidades numericas em parte das iteracoes.


## 13. Conclusao executiva

1. O projeto conseguiu superar `0.7502` em cenarios de holdout especificos (ex.: RF tuned `0.7593`, RF inicial `0.7658`, copula+MLP `0.7645`).
2. Quando a avaliacao fica robusta (MCCV, RKF, Spatial-LOO), os R2 medios caem para faixa `~0.48-0.61`.
3. O melhor compromisso atual entre desempenho e estabilidade foi `RandomForest` com tuning Optuna e features engenheiradas.
4. O teto `0.77-0.78` ainda nao foi quebrado de forma estatisticamente robusta com os dados atuais.
