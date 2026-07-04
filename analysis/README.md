# Análise — melhor N para Graph-RKD (regression)

Pipeline de análise dos experimentos, **independente** dos notebooks antigos.
Puxa os dados direto do W&B (conta onde os experimentos foram logados) e gera
figuras + tabelas + um resumo em texto pronto pra compor o relatório.

## Escopo

`mode = regression` (técnica de comparação entre embeddings) ×
`method ∈ {profile, mds}` (2 técnicas de embedding) ×
`dataset ∈ {cars196, cub200}` = **4 análises**.
Para cada uma: varredura de **N** (nº de nós do grafo) ∈ {2,4,8,16,17}, 3 seeds.
Baseline `off` (destilação sem perda de grafo) usado como referência.
Métrica primária: **test mAP@R**; apoio: Recall@1, R-Precision.

## Como rodar

```bash
# 1) logar no W&B (uma vez). Os experimentos estão na conta rodz-ralm-v-ai.
wandb login
# (opcional) apontar p/ outra entity:
# export WANDB_ENTITY=rodz-ralm-v-ai

# 2) extrair -> analysis/data/{runs_summary,history}.csv
python analysis/pull_wandb.py

# 3) analisar -> analysis/out/{figuras, tables.tex, FINDINGS.md, summary_Nstar.csv}
python analysis/analyze.py
```

> Se você não tem acesso à conta W&B, use os CSVs já commitados em `analysis/data/`
> e rode só o passo 3.

## Saídas (`analysis/out/`)

| arquivo | o que é |
|---|---|
| `fig_mapr_vs_N.png/pdf` | mAP@R vs N, 4 painéis, com barras de erro, N\* marcado e linha do baseline |
| `fig_recall1_vs_N`, `fig_rprec_vs_N` | idem para métricas de apoio |
| `fig_convergence.png/pdf` | curvas test mAP@R por época, uma linha por N |
| `tables.tex` | tabelas booktabs por cenário (média±sem, N\* em negrito) |
| `summary_Nstar.csv` | resumo dos 4 N\* + deltas + Spearman + meio-termo |
| `FINDINGS.md` | **texto pronto** respondendo as 4 perguntas — cole no Claude-web p/ o LaTeX |

## Perguntas respondidas (por cenário)

1. Qual o melhor **N\*** (por mAP@R).
2. **Quão melhor** o N\* é (vs pior N e vs baseline `off`).
3. **Padrão com N** (Spearman(N, mAP@R): monotônico? pico interno? ruído?).
4. **Convergência**: qual N converge mais rápido (época p/ 95% do próprio teto)
   × qual converge melhor (maior teto), e um **meio-termo**.
