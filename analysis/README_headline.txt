GRAPH-RKD — RESULTADOS HEADLINE (Fase 5) — PARCIAL
Gerado em 13/07/2026 ~05h. Campanha ainda rodando (~161/210 alunos).

ARQUIVOS
  headline_partial.csv       Tabela resumo: test mAP@R (%) por célula (dataset×teacher)
                             × método × seed (s0/s1/s2) + média.
  headline_raw_metrics.csv   Cru, todas as métricas de teste por job concluído:
                             mAP@R, R_precision, recall@1/2/4/8 (frações 0-1).
  extract_headline.py        Script que gera os 2 CSVs. Rode de novo p/ atualizar:
                             uv run --no-sync python analysis/extract_headline.py

MÉTODOS (5 alunos do headline)
  triplet-only  = baseline (sem teacher; mesmo valor p/ r18 e cvt do mesmo dataset)
  RKD-D / RKD-A / RKD-D+A = clássicos (distance / angle / combinado)
  Graph-RKD = 3 configs: mds-N4 (headline principal), mds-N3 e prof-N3 (H5, arity-matched)
  λg=0.01, normalização per_graph, 120 épocas, teachers ResNet-18 e ConvNeXt-Tiny.

STATUS DAS CÉLULAS (13/07 ~05h)
  cars-r18 : COMPLETA (3 seeds)
  cars-cvt : seed 0 (seeds 1,2 rodando/pendentes)
  cub-r18  : seed 0 parcial — Graph-RKD AINDA RODANDO (sai hoje à tarde)
  cub-cvt  : só baseline até agora

ACHADOS ATÉ AGORA
  * CARS (2 teachers): Graph-RKD (~3.4-3.7) BATE RKD-D/A/ambos (~1.8-2.1) por +65-95%,
    e supera o triplet-only. Resultado positivo, generaliza por teacher.
  * CUB é regime DIFERENTE: aqui o RKD clássico AJUDA (3.4-3.6 vs triplet 2.84),
    ao contrário do Cars. O número do Graph-RKD no CUB ainda não fechou -> VEREDITO
    NO CUB PENDENTE. Pode virar "vence nos 2 datasets" ou "vence em Cars, empata/perde
    em CUB". Só com os dados de hoje à tarde dá pra cravar.

PREVISÃO (ritmo ~1 aluno/hora, 2 frentes)
  ~14h 13/07 : tabela 4 células com 1 seed completa
  ~08h 14/07 : 4 células com 2 seeds
  ~03-06h 15/07 : campanha 100% (210/210, 3 seeds)
