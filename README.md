# Ksat_PINNs

Previsão da condutividade hidráulica saturada do solo (Ksat) usando redes
neurais informadas por física (PINN), com a Lei de Darcy / Kozeny-Carman
como restrição física suave.

## Estrutura do repositório

```
Ksat_PINNs/
├── data/
│   └── ksat_dataset.xlsx                          # dataset bruto (5 fontes, 482 amostras)
├── src/                                           # código reutilizável
│   ├── data_preprocessing.py                       # leitura e limpeza dos dados
│   └── pinn_model.py                               # arquitetura da rede + funções de loss física
├── experiments/                                   # scripts de experimento, executáveis diretamente
│   ├── 01_train_baseline_vs_pinn.py                # NN comum vs PINN (split único treino/val/teste)
│   ├── 02_evaluate_leave_one_group_out.py          # validação rigorosa: cada fonte de dados fora por vez
│   └── 03_evaluate_physics_constraint_strategies.py # 5 estratégias físicas (baseline/mono/ptf/mono+ptf/full)
├── results/
│   ├── figures/                                    # gráficos gerados (.png)
│   └── metrics/                                    # métricas em .json (resumo e valores brutos)
├── docs/
│   └── methodology_and_results.md                  # relatório com a metodologia e discussão dos resultados
└── reference/
    └── related_work_water_stress_project/          # material de um colega (projeto similar, usado como referência)
```

## Como rodar

```bash
pip install -r requirements.txt
cd experiments
python 01_train_baseline_vs_pinn.py
python 02_evaluate_leave_one_group_out.py
python 03_evaluate_physics_constraint_strategies.py
```

Cada script salva seus resultados em `results/figures/` e `results/metrics/`.

## Estado atual do projeto (resumo)

- **01 — Baseline vs PINN (split único):** R² ≈ 0,62-0,63 em ambos os modelos.
  A restrição física não muda muito a acurácia aqui, mas melhora a consistência
  física das derivadas da rede (ver `docs/methodology_and_results.md`).
- **02 — Leave-one-source-out:** quando se testa em uma fonte de dados
  **inteira** nunca vista no treino, o R² cai para valores **negativos**
  (~-1,1 em média). Isso revela que o bom resultado do split único era
  otimista — o modelo não generaliza bem entre estudos/regiões/métodos
  diferentes (domain shift). Achado importante para reportar no artigo.
- **03 — Estratégias físicas do colega:** nenhuma das 5 variações testadas
  (baseline/mono/ptf/mono+ptf/full) supera o baseline sob a validação
  leave-one-source-out. O termo "ptf" piora o resultado porque usa
  coeficientes de exemplo não calibrados para este dataset.

## Contexto físico usado como restrição

Lei de Darcy (fluxo saturado): `q = -Ksat · dh/dz`. Combinada com a equação
de Kozeny-Carman, dá origem às restrições de sinal impostas via autograd
durante o treino:

| Variável | Efeito esperado sobre Ksat |
|---|---|
| Porosidade total | positivo |
| Macroporosidade | positivo |
| Densidade do solo | negativo |
| Argila | negativo |

Detalhes completos em `docs/methodology_and_results.md`.
