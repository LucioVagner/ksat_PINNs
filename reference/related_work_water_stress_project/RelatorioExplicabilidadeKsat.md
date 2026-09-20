# Relatório Técnico de Explicabilidade
## Previsão de Ksat (BHSE) - Modelo RF Tunado com Features Engenheiradas

## 1. Escopo e objetivo

Este relatório consolida a análise de explicabilidade do modelo de previsão de `log10(Ksat)` para a BHSE, com foco em três dimensões:

1. Explicabilidade global: o que o modelo aprendeu como padrão geral.
2. Explicabilidade local: por que o modelo acertou ou errou em pontos específicos.
3. Diagnóstico espacial de resíduos: onde o modelo erra e se os erros têm estrutura geográfica.

Modelo analisado:

- Tipo: `RandomForestRegressor`
- Feature set: `engineered` (17 variáveis)
- Base de treino/validação/teste: `70/15/15`, seed `42`
- Referência de robustez espacial adicional: `Spatial-LOO` com buffer de `50 m`

Arquivos de entrada principais:

- `outputs_explainability/explainability_summary.json`
- `outputs_explainability/holdout_metrics.json`
- `outputs_explainability/spatial/sloo_summary.json`
- tabelas e figuras em `outputs_explainability/global`, `local`, `spatial`, `diagnostics`


## 2. Configuração experimental

### 2.1 Dados e alvo

- Fonte: `Pontos_Ksat_Sanga_Ellert.xlsx`
- Amostras totais: `169`
- Alvo modelado: `log_ksat = log10(Ksat)`

### 2.2 Features usadas no modelo final

Base física e espacial:

- `x`, `y`, `sand`, `clay`, `bulk_density`, `macroporosity`

Features derivadas:

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

### 2.3 Hiperparâmetros do modelo analisado

- `n_estimators = 582`
- `max_depth = 21`
- `min_samples_split = 3`
- `min_samples_leaf = 5`
- `max_features = 0.3898`
- `max_samples = 0.8788`


## 3. Desempenho de referência para explicabilidade

### 3.1 Holdout (26 pontos de teste)

- `R2_log = 0.7602`
- `RMSE_log = 0.2689`
- `MAE_log = 0.2093`
- `PBIAS_log = -2.35%`
- `R2_orig = 0.5246`
- `RMSE_orig = 1465.65 cm/dia`
- `MAE_orig = 881.50 cm/dia`

Interpretação:

- O modelo atinge bom desempenho no split holdout específico, consistente com os melhores resultados vistos anteriormente.

### 3.2 Spatial-LOO com buffer 50 m (169 pontos)

- `R2_log = 0.5990`
- `RMSE_log = 0.3521`
- `MAE_log = 0.2701`
- `R2_orig = 0.4050`
- `RMSE_orig = 1486.82 cm/dia`
- `MAE_orig = 968.18 cm/dia`
- Tamanho médio de treino por fold: `164.26` pontos
- Fallback por falta de treino mínimo: `0`

Interpretação:

- A queda de performance no SLOO confirma que o problema tem componente espacial desafiador e que o holdout pode ser otimista.


## 4. Explicabilidade global

### 4.1 SHAP summary (global)

Figura:

- `outputs_explainability/global/shap_summary_bar.png`
- `outputs_explainability/global/shap_summary_beeswarm.png`

Top features por `mean(|SHAP|)`:

1. `sand_macro` (0.1400)
2. `macroporosity` (0.1004)
3. `macro_squared` (0.0880)
4. `macro_ds_ratio` (0.0716)
5. `sand` (0.0178)
6. `log_ds` (0.0156)
7. `bulk_density` (0.0133)
8. `x` (0.0131)

Leitura:

- O modelo está centrado em variáveis de estrutura de poros/interações com macro, não só em textura bruta.
- A presença de termos de interação (`sand_macro`, `macro_squared`) como dominantes sugere forte não-linearidade física aprendida.

### 4.2 Permutation importance com incerteza

Figura:

- `outputs_explainability/global/permutation_importance.png`

Tabela:

- `outputs_explainability/global/permutation_importance.csv`

Top features por queda média em R2:

1. `sand_macro` (0.1812 +/- 0.0700)
2. `macroporosity` (0.0710 +/- 0.0434)
3. `macro_squared` (0.0545 +/- 0.0378)
4. `macro_ds_ratio` (0.0416 +/- 0.0349)

Leitura:

- SHAP e permutation convergem para o mesmo núcleo de importância, reforçando robustez da interpretação.
- Algumas variáveis (`log_ds`, `bulk_density`, `y`) aparecem com importância média ~0 ou negativa em permutation, sinalizando possível redundância no conjunto final.

### 4.3 PDP + ICE

Figura:

- `outputs_explainability/global/pdp_ice.png`

Variáveis analisadas:

- `macroporosity`, `bulk_density`, `sand`, `clay`

Leitura:

- As curvas confirmam dependências não lineares e heterogeneidade local (ICE), principalmente para variáveis estruturais.
- O efeito médio não é homogêneo entre amostras, o que justifica interações explícitas no feature engineering.

### 4.4 SHAP interactions

Figuras/tabelas:

- `outputs_explainability/global/shap_interaction_heatmap.png`
- `outputs_explainability/global/shap_dependence_macro_vs_ds.png`
- `outputs_explainability/global/shap_interaction_matrix.csv`

Leitura:

- Interações envolvendo `macroporosity` são centrais, especialmente com `bulk_density` e termos derivados.
- O modelo usa mecanismos combinados de textura + estrutura, não relações univariadas simples.


## 5. Explicabilidade local

### 5.1 Pontos de referência analisados

Tabela:

- `outputs_explainability/local/local_points_summary.csv`

Pontos selecionados:

- `best_prediction`: erro absoluto log `0.0079`
- `worst_prediction`: erro absoluto log `0.6518`
- `highest_ksat`: erro absoluto log `0.4218`
- `lowest_ksat`: erro absoluto log `0.3082`

### 5.2 SHAP waterfall (caso a caso)

Figuras:

- `outputs_explainability/local/waterfall_best_prediction.png`
- `outputs_explainability/local/waterfall_worst_prediction.png`
- `outputs_explainability/local/waterfall_highest_ksat.png`
- `outputs_explainability/local/waterfall_lowest_ksat.png`

Leitura:

- O pior erro ocorre em combinação de variáveis onde efeitos de macro/interação empurram fortemente a previsão para cima.
- Em extremos de Ksat, o modelo tende a reduzir amplitude (subestimando picos altos e superestimando alguns baixos), comportamento comum em regressão com amostra pequena.

### 5.3 LIME local

Figuras:

- `outputs_explainability/local/lime_best_prediction.png`
- `outputs_explainability/local/lime_worst_prediction.png`
- `outputs_explainability/local/lime_highest_ksat.png`
- `outputs_explainability/local/lime_lowest_ksat.png`

Tabela:

- `outputs_explainability/local/lime_local_weights.csv`

Leitura:

- LIME confirma localmente o protagonismo de variáveis ligadas a macro e interações para a maioria dos pontos explicados.

### 5.4 Vizinhos influentes (proximidade no espaço de features)

Tabela:

- `outputs_explainability/local/local_nearest_neighbors.csv`

Leitura:

- Nos casos de maior erro, os vizinhos no espaço de features nem sempre têm alvo próximo, sugerindo overlap de regimes físicos distintos.

### 5.5 Contrafactual simples (pior ponto)

Tabela:

- `outputs_explainability/local/counterfactual_worst_point.csv`

Top ajustes que mais reduzem erro no pior ponto:

1. reduzir `macroporosity`
2. reduzir `sand_macro`
3. reduzir `macro_squared`
4. reduzir `macro_ds_ratio`

Leitura:

- O erro extremo está associado principalmente ao bloco de variáveis de macroestrutura, coerente com SHAP global/local.


## 6. Diagnóstico espacial dos erros

### 6.1 Mapas de resíduos

Figuras:

- `outputs_explainability/spatial/holdout_residual_maps.png`
- `outputs_explainability/spatial/sloo_residual_maps.png`

Leitura:

- Há padrão espacial de super/subestimação em regiões específicas, mais claro no SLOO.
- Isso sugere que parte da estrutura espacial do processo ainda não foi totalmente capturada.

### 6.2 Variograma dos resíduos

Arquivos:

- `outputs_explainability/spatial/residual_variogram_experimental.csv`
- `outputs_explainability/spatial/residual_variogram_fit.json`
- `outputs_explainability/spatial/residual_variogram_plot.png`

Parâmetros ajustados (esférico):

- `nugget = 0.0997`
- `sill = 0.0634`
- `range = 1134.4 m`
- `GDE = 61.1%`

Interpretação:

- `GDE` intermediário (não dominado por pepita pura) indica que os resíduos **não** são totalmente aleatórios no espaço.
- Há sinal de dependência espacial remanescente, implicando oportunidade para modelagem residual geoespacial adicional.

### 6.3 Correlação |erro| x covariáveis

Tabela:

- `outputs_explainability/diagnostics/error_covariate_spearman.csv`

Leitura:

- Não houve correlação significativa forte (`p < 0.05`) entre `|erro|` e uma variável isolada.
- Os menores p-valores foram para `macroporosity` e `macro_squared` (~0.061), sugerindo tendência fraca/borderline.

### 6.4 Erro por quadrante

Tabela:

- `outputs_explainability/diagnostics/error_by_quadrant.csv`

Resultados:

- SW: `R2_log=0.583`, `MAE_log=0.250`
- NW: `R2_log=0.499`, `MAE_log=0.272`
- SE: `R2_log=0.549`, `MAE_log=0.297`
- NE: `R2_log=0.667`, `MAE_log=0.276`

Leitura:

- O desempenho varia por região, indicando heterogeneidade espacial da dificuldade de predição.


## 7. Consistência física do modelo

Arquivos:

- `outputs_explainability/diagnostics/physical_consistency_checks.csv`
- `outputs_explainability/diagnostics/physical_consistency_curves.png`

Resultado dos testes de monotonia (Spearman):

- `macroporosity`: consistente (esperado positivo, observado `+0.965`)
- `bulk_density`: consistente (esperado negativo, observado `-0.585`)
- `sand`: consistente (esperado positivo, observado `+0.993`)
- `clay`: **inconsistente** (esperado negativo, observado `+0.744`)

Interpretação:

- O sinal invertido de `clay` é um alerta importante.
- Possíveis causas:
  - colinearidade/interação com variáveis derivadas;
  - amostragem pequena com regimes locais específicos;
  - comportamento regional da BHSE diferente do padrão simplificado esperado.


## 8. Síntese técnica

### 8.1 O que foi confirmado

- O modelo aprende forte estrutura não linear baseada em macroporosidade e interações.
- O bloco de variáveis de macroestrutura domina explicabilidade global e local.
- Há coerência entre SHAP e permutation para as variáveis principais.

### 8.2 O que exige cautela

- Desempenho robusto (SLOO) cai em relação ao holdout.
- Resíduos apresentam estrutura espacial remanescente (`GDE ~61%`).
- Relação com `clay` saiu fisicamente inconsistente na checagem monotônica.

### 8.3 Implicação prática

Antes de consolidar conclusões científicas fortes, recomenda-se:

1. testar modelagem híbrida `ML + krigagem de resíduos`;
2. revisar conjunto de features para reduzir redundância/colinearidade;
3. repetir checagem de consistência física após simplificação de features;
4. manter validação espacial como métrica principal de generalização.


## 9. Evidências visuais principais (figuras-chave)

Global:

![SHAP Beeswarm](outputs_explainability/global/shap_summary_beeswarm.png)

![Permutation Importance](outputs_explainability/global/permutation_importance.png)

![PDP + ICE](outputs_explainability/global/pdp_ice.png)

Local:

![Waterfall - Pior Predição](outputs_explainability/local/waterfall_worst_prediction.png)

![LIME - Pior Predição](outputs_explainability/local/lime_worst_prediction.png)

Espacial:

![Mapas de Resíduos (SLOO)](outputs_explainability/spatial/sloo_residual_maps.png)

![Variograma dos Resíduos](outputs_explainability/spatial/residual_variogram_plot.png)

Diagnóstico físico:

![Consistência Física](outputs_explainability/diagnostics/physical_consistency_curves.png)


## 10. Reprodutibilidade

Comando executado:

```bash
.venv/bin/python explainability_ksat.py --model rf --feature-set engineered --sloo-max-points 169
```

Relatório consolidado em:

- `RelatorioExplicabilidadeKsat.md`
- `RelatorioExplicabilidadeKsat.pdf`

