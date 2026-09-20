# PINN para previsão da condutividade hidráulica saturada do solo (Ksat)

## 1. O problema e por que não é uma PINN "clássica"

O dataset tem 482 amostras com Ksat medido (0,6 a 9034 cm/dia) e propriedades físicas do solo
(areia, silte, argila, densidade, porosidade total, macroporosidade, carbono orgânico,
profundidade, uso do solo). **Não há dados de carga hidráulica (h) nem de fluxo (q) medidos em
diferentes pontos do espaço/tempo** — o que seria necessário para resolver a Lei de Darcy como uma
equação diferencial parcial (o formato "clássico" de PINN, como em Raissi et al. 2019).

Por isso, a abordagem usada aqui é uma **PINN aplicada a uma função de pedotransferência (PTF)**:
uma rede neural prevê Ksat a partir das propriedades do solo, e a Lei de Darcy entra como uma
**restrição física suave sobre as derivadas da rede**, calculada via autograd (exatamente a mesma
técnica usada para impor resíduos de PDE em PINNs clássicas — aqui aplicada a uma relação
constitutiva em vez de uma PDE espacial).

## 2. A física incorporada

Da Lei de Darcy para fluxo saturado (q = -Ksat·dh/dz) combinada com a equação de Kozeny-Carman
(que descreve Ksat em função da geometria dos poros):

```
Ksat ~ (ρg/μ) · φ³ / ((1-φ)² · S₀²)
```

extraem-se os **sinais** físicos esperados (não os valores exatos — Kozeny-Carman é idealizada
para meios granulares, e solos reais têm agregação/estrutura):

| Variável | Efeito esperado sobre Ksat | Justificativa física |
|---|---|---|
| Porosidade total | positivo | mais poros → mais área de condução (Darcy) |
| Macroporosidade | positivo | poros maiores → fluxo cresce fortemente (Darcy-Poiseuille) |
| Densidade do solo | negativo | mais compactação → menos poros (Kozeny-Carman) |
| Argila | negativo | partículas finas → maior tortuosidade/superfície específica |

Esses sinais são impostos durante o treino como uma penalidade (*hinge loss*) sobre
∂Ksat_previsto/∂x, calculada com `jax.grad` para cada amostra do lote.

**Loss total:** `L = L_dados (MSE) + λ_fisica · L_física (violação de sinal) + λ_L2 · ||W||²`

## 3. Implementação

- **Framework:** JAX + Optax (autograd nativo, leve, ideal para derivadas de segunda ordem
  necessárias no treino de uma PINN).
- **Alvo:** log10(Ksat) — a distribuição original é muito assimétrica (span de 4 ordens de
  grandeza); log10 estabiliza a otimização.
- **Features:** areia, silte, argila, densidade, porosidade, macroporosidade, carbono orgânico,
  profundidade (ponto médio das faixas informadas) + uso do solo (one-hot).
- **Divisão:** 70% treino / 15% validação (early stopping) / 15% teste (avaliação final, nunca
  usado no treino).
- **Arquitetura:** MLP 14 → 32 → 32 → 1, ativação tanh.
- **Comparação:** dois modelos com a *mesma* inicialização — um baseline (λ_física = 0) e a PINN
  (λ_física = 0,05) — para isolar o efeito da restrição física.

Arquivos: `data_prep.py` (limpeza dos dados), `pinn_model.py` (rede + loss física + treino),
`run_experiment.py` (experimento completo, métricas e gráficos).

## 4. Resultados

| Métrica (teste, n=72) | Baseline (sem física) | PINN (Darcy/Kozeny-Carman) |
|---|---|---|
| R² (log10 Ksat) | 0,624 | **0,626** |
| RMSE (log10 Ksat) | 0,484 | **0,483** |
| RMSE (cm/dia) | 884 | 886 |

A acurácia preditiva ficou **praticamente igual** entre os dois modelos — esperado, já que a
restrição física é um regularizador suave, não uma fonte de novos dados. O ganho real da PINN
aparece na **consistência física das derivadas** (ver `training_curves.png` e a saída do script):

| Restrição | Baseline consistente | PINN consistente |
|---|---|---|
| ∂Ksat/∂Porosidade ≥ 0 | 48,6% | 52,8% |
| ∂Ksat/∂Macroporosidade ≥ 0 | 77,8% | 77,8% |
| ∂Ksat/∂Densidade ≤ 0 | 86,1% | 86,1% |
| ∂Ksat/∂Argila ≤ 0 | 62,5% | **69,4%** |

O ganho é modesto porque o *early stopping* (necessário para evitar overfitting — ver
`training_curves.png`) interrompe o treino cedo, antes que o termo físico tenha efeito pleno.
Rodando por mais épocas sem early stopping, o resíduo físico da PINN cai de forma consistente
(ver log de treino), mas ao custo de overfitting nos dados — um trade-off real que vale discutir
com você: dá para relaxar o early stopping e aumentar λ_física simultaneamente, ou usar a PINN
sobretudo como um **regularizador para extrapolação** (fora da faixa observada de treino), que é
onde restrições físicas normalmente mais ajudam.

## 5. Limitação observada e transparência

Nos gráficos de dependência parcial (`partial_dependence.png`), a curva de Porosidade ainda
aparece **decrescente** em ambos os modelos — contrariando o sinal físico esperado. Isso acontece
porque a dependência parcial fixa as demais variáveis na mediana, e Porosidade é fortemente
correlacionada com Densidade do solo e Macroporosidade no dataset real (solos mais densos tendem a
ter menor porosidade); ao variar só Porosidade e travar as demais, o modelo extrapola para
combinações fisicamente incomuns não vistas no treino. A restrição via autograd atua sobre a
derivada *pontual* em cada amostra real (onde as correlações naturais entre variáveis se
preservam), não sobre a curva de dependência parcial isolada — por isso os dois diagnósticos podem
discordar. Vale deixar isso registrado em vez de esconder.

## 6. Próximos passos possíveis

1. Aumentar λ_física gradualmente (*curriculum*) enquanto usa dropout/mais L2 para permitir treino
   mais longo sem overfitting.
2. Incluir uma penalidade adicional de magnitude (não só sinal), ancorada em Kozeny-Carman, para
   estimativas de Ksat em texturas fora da faixa observada.
3. Se houver possibilidade de coletar perfis de carga hidráulica/fluxo em pelo menos alguns pontos,
   migrar para uma PINN espacial "clássica" (resolvendo Darcy-Richards em 2D/3D com X, Y,
   profundidade), tratando os Ksat pontuais como dados esparsos de calibração.
