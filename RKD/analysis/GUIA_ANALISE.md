# Guia de Análise — Campanha Graph-RKD (Seção 7 do paper)

Passo a passo para reproduzir **todas** as tabelas e figuras da Seção 7 a partir
das runs do W&B — inclusive **com dados parciais**, enquanto a campanha ainda roda.
Escrito para o Gabriel conseguir rodar as análises sozinho a qualquer momento.

> **TL;DR** (ambiente já configurado, na raiz do repo):
> ```bash
> cd /mnt/b/_ai/mo434/Gabriel/RKD/analysis
> export WANDB_API_KEY=$(awk '/machine api.wandb.ai/{m=1} m&&/password/{print $2;exit}' ~/.netrc)
> uv run --no-sync jupyter nbconvert --to notebook --execute --inplace \
>     00_aggregate_results.ipynb 01_quantitative_H1.ipynb \
>     02_order_normalization_H0_H2.ipynb 03_descriptor_objective_H3_H4.ipynb \
>     04_n3_vs_rkda_H5.ipynb 06_findings.ipynb
> ```
> Rode o `00` **primeiro** (gera `results.csv`); os demais leem esse CSV.

---

## 1. Como o pipeline está montado

```
   W&B (gabomfim-unicamp/graph-rkd)          ← fonte única da verdade
        │  api.runs(...)  [só state=='finished']
        ▼
   analysis_utils.py :: fetch_runs()          ← puxa + classifica cada run
        │
        ▼
   00_aggregate_results.ipynb  ──►  results.csv   ← BASE (1 linha por run)
        │
        ├─►  01_quantitative_H1.ipynb        (§7.1  H1)
        ├─►  02_order_normalization_H0_H2    (§7.2  H0, H2)
        ├─►  03_descriptor_objective_H3_H4   (§7.3  H3, H4)
        ├─►  04_n3_vs_rkda_H5.ipynb          (§7.4  H5)
        ├─►  05_qualitative_retrieval.ipynb  (§7.5  painéis de retrieval)
        └─►  06_findings.ipynb               (§8    vereditos H0–H5)
```

**Regra de ouro:** o `00` é o único que fala com o W&B. Ele salva `results.csv`.
Todos os outros leem `results.csv` — então, para atualizar as análises depois que
mais runs terminam, **basta re-rodar o `00`** e depois os notebooks que interessam.

Toda a lógica pesada (classificação de run, agregação por seed, vereditos, figuras)
mora em **`analysis_utils.py`** (`import analysis_utils as au`). Os notebooks são
finos: chamam `au.<função>` e exibem o resultado. Se algo precisar mudar na
metodologia, muda-se ali num lugar só.

---

## 2. Pré-requisitos

- Ambiente já pronto no repo: venv `.venv`, gerido por `uv`. Rode sempre com
  `uv run --no-sync python ...` (o `--no-sync` evita reinstalar deps).
- **Credencial W&B** (necessária só para o notebook `00`): exporte a chave antes
  de abrir o Jupyter/rodar o `00`:
  ```bash
  export WANDB_API_KEY=$(awk '/machine api.wandb.ai/{m=1} m&&/password/{print $2;exit}' ~/.netrc)
  ```
  (a chave já está gravada no `~/.netrc` da máquina do Rodrigo). Entidade/projeto
  default: `gabomfim-unicamp/graph-rkd` — definidos no topo do `analysis_utils.py`.
- Dependências de análise: `pandas`, `matplotlib`, `wandb` (já no venv). O `wandb`
  é importado sob demanda, só dentro do `fetch_runs`.

Para trabalhar interativamente, em vez do `nbconvert`:
```bash
cd /mnt/b/_ai/mo434/Gabriel/RKD/analysis
export WANDB_API_KEY=$(awk '/machine api.wandb.ai/{m=1} m&&/password/{print $2;exit}' ~/.netrc)
uv run --no-sync jupyter lab      # abre o Jupyter; rode os notebooks na ordem
```

---

## 3. Passo a passo (o que cada notebook faz e produz)

Rode **nesta ordem**. Tempo total < 1 min (fora o `00`, que depende da rede do W&B).

### `00_aggregate_results.ipynb` — a base
- Chama `au.fetch_runs()` → puxa **só as runs `finished`** do W&B, achata
  `config` + `summary`, e **classifica** cada run em:
  - **fase**: `phase2..phase5` (+ `teacher`, `dev`, etc.) — via tag do job ou prefixo do nome;
  - **aluno**: `triplet_only`, `rkd_dist`, `rkd_angle`, `rkd_both`, `graph-rkd`, `teacher`;
  - metadados: `dataset`, `teacher`, `seed`, `norm`, `method` (prof/mds), `objective`
    (regression/contrastive), `N`, `lambda_g`, e todas as métricas de teste/val.
- **Salva `results.csv`** — a base que todo o resto lê.
- Célula de sanity: `df.groupby(['phase','student']).size()` — confira aqui quantas
  runs concluídas há por fase/aluno antes de seguir.

### `01_quantitative_H1.ipynb` — §7.1, resultado headline
- **Tabela headline** por célula `(dataset, teacher)`: os 5 alunos
  (triplet-only, +RKD-D, +RKD-A, +RKD-D+A, +Graph-RKD) com `mediana / mean ± sem`.
- Métricas secundárias (R-Precision, Recall@1).
- **Figura** `figures/fig_h1_headline.pdf`: barras de mAP@R por aluno, uma faceta
  por célula. Esta é a figura principal da Seção 7.

### `02_order_normalization_H0_H2.ipynb` — §7.2
- **H0** (`fig_h0_lambda`): val mAP@R vs λg no *gate* de viabilidade (phase1) +
  piso triplet-only. Mostra a faixa de λg que não colapsa.
- **H2** (`fig_h2_norm`): ablação de normalização (phase2), melhor λg por esquema.
- Qualidade por ordem N (phase3): mAP@R vs N por descritor.

### `03_descriptor_objective_H3_H4.ipynb` — §7.3
- **H3** (sonda offline, `au.fig_probe` + `descriptor_probe.csv`): taxa de
  degenerescência do MDS e de empate do profile vs N — **não precisa de treino**,
  roda na hora (evidência de mecanismo).
- **H3 (accuracy)**: profile vs MDS por `(N, dataset, teacher)` — precisa das runs.
- **H4** (`fig_h4_overlay`): robustez do objetivo — overlay val mAP@R vs λg para
  regression vs contrastive (phase4), com a largura da banda a <1 sem do melhor.

### `04_n3_vs_rkda_H5.ipynb` — §7.4
- Tabela **N=3 (profile e mds) vs RKD-A** por célula (`au.n3_vs_rkda`) — o teste
  arity-matched (Graph-RKD com ordem 3 comparado ao par angular do RKD).
- Veredito por célula: compara mean±sem de cada método contra o RKD-A na mesma célula.

### `05_qualitative_retrieval.ipynb` — §7.5
- Painéis qualitativos de retrieval (query → vizinhos). **Precisa de GPU +
  checkpoints dos alunos** (`student_last.pth`) e do dataset. É o único notebook
  que carrega modelo; rode por último e só quando quiser as figuras qualitativas.

### `06_findings.ipynb` — §8
- Consolida os **vereditos H0–H5** que alimentam a Seção 8 (FINDINGS). Usa a sonda
  (disponível já) + as tabelas dos notebooks anteriores.
- **Exporta as tabelas do paper:** roda `au.export_tables(df)` → grava
  `tables/*.{csv,tex}` (booktabs, com `\caption`/`\label`) para H1 (headline por
  célula), H2, H3, H4 e H5 — prontas pra `\input{}`/colar no `paper.tex`. As figuras
  já são salvas pelos `fig_*` em `figures/*.pdf`. (`tables/` é gitignored — deriva do
  `results.csv`; re-rode o `06` quando fechar mais seeds.)

---

## 4. Mapa notebook ↔ hipótese ↔ seção do paper

| Notebook | Hipótese | Seção | Precisa de |
|---|---|---|---|
| `00_aggregate_results` | — | — | W&B (chave) |
| `01_quantitative_H1` | **H1** headline (5 alunos) | §7.1 | `results.csv` |
| `02_order_normalization_H0_H2` | **H0** (λg), **H2** (norm) | §7.2 | `results.csv` |
| `03_descriptor_objective_H3_H4` | **H3** (descritor), **H4** (objetivo) | §7.3 | `results.csv` + `descriptor_probe.csv` |
| `04_n3_vs_rkda_H5` | **H5** (N=3 vs RKD-A) | §7.4 | `results.csv` |
| `05_qualitative_retrieval` | — (qualitativo) | §7.5 | GPU + checkpoints + dataset |
| `06_findings` | **H0–H5** (vereditos) | §8 | `results.csv` + sonda |

---

## 5. Como LER os resultados (regras de metodologia — §8)

Estas regras estão codificadas no `analysis_utils.py`; leia antes de concluir qualquer coisa:

1. **Estatística primária = MEDIANA** por config sobre as seeds (`au.agg` retorna
   median/mean/sem/n). O projeto reporta **o modelo da run mediana**
   (`au.median_run`), não uma média de pesos — não estamos aumentando o nº de seeds.
   mean±sem ficam como referência.
2. **Noise floor:** o `sem` entre seeds da célula é o ruído. Diferença **< ~1 sem**
   entre dois métodos = **indistinguível** (não afirme vitória).
3. **"Vence" só se:** a mediana/média do Graph-RKD excede a do **melhor baseline**
   por **> 1 sem** *E* o sinal é **consistente entre os dois teachers** (r18 e cvt).
4. **Reportar sempre por célula** `(dataset × teacher)` — os resultados são
   **dependentes do dataset** (ver Seção 6 abaixo). Não agregue Cars + CUB numa
   média só; isso esconde o efeito.
5. `au.h1_verdict(tabela)` já devolve o veredito textual (VENCE/IGUALA/PERDE) por
   célula seguindo essas regras.

---

## 6. Estado atual dos dados (rodar PARCIAL)

A campanha ainda está rodando (≈165/210 alunos concluídos). A ordem de execução foi
**reordenada por seed** (a pedido do Gabriel): **seed 0 de TODAS as 4 células
primeiro**, depois seed 1, depois seed 2. Então, com dados parciais, você já
consegue a **tabela de 4 células com 1 seed** antes de a campanha fechar.

Estado das células do headline (phase5), no momento da escrita:

| Célula (dataset·teacher) | seed 0 | seeds 1–2 |
|---|---|---|
| cars · r18 | ✅ completa | parcial |
| cars · cvt | ✅ completa | pendente |
| cub · r18 | ✅ completa | pendente |
| cub · cvt | 🔄 clássicos fechando; Graph-RKD ainda rodando | pendente |

**Achado científico consolidado (headline, seed 0):**
- **Cars (2 teachers):** Graph-RKD **VENCE** o RKD clássico por **+65–95%** — e supera
  o triplet-only. Sinal positivo e consistente entre teachers.
- **CUB / r18:** regime **diferente** — aqui o RKD clássico **ajuda** (3.36–3.61 vs
  triplet 2.84), e o Graph-RKD (melhor: prof-N3 = 3.26) **fica abaixo** do RKD, embora
  ainda **acima** do triplet. Falta fechar CUB/cvt para confirmar o padrão no 2º teacher.
- **Narrativa honesta:** *Graph-RKD supera o RKD onde o RKD atrapalha (Cars), mas
  não o supera onde o RKD já ajuda (CUB)* — resultado **dependente do dataset**.

**Rodar parcial:** os notebooks já tratam ausência de dados (funções de figura
imprimem "sem runs de phaseX ainda" e seguem). Basta re-rodar o `00` (puxa o que já
terminou) e depois `01`/`04`/`06`. Repita a cada seed que fechar — a única coisa que
muda é o `results.csv`.

> **Cadência combinada:** re-rodar/atualizar os notebooks **a cada seed que fecha**,
> começando quando o **seed 0** fechar nas 4 células (cub·cvt é o que falta).

---

## 7. Caminho alternativo: CSV local, sem W&B

Se o W&B estiver fora do ar ou você quiser um número rápido só do headline **a
partir dos logs locais** (`experiments_local/`), há um extrator independente:

```bash
cd /mnt/b/_ai/mo434/Gabriel
uv run --no-sync python analysis/extract_headline.py
# gera analysis/headline_partial.csv (resumo test mAP@R por célula×método×seed)
#   e analysis/headline_raw_metrics.csv (mAP@R, R-prec, recall@1/2/4/8)
```
Esse caminho **não** depende de credencial e é o que está sendo copiado para o
Desktop conforme as células fecham. Serve como conferência cruzada dos notebooks.

---

## 8. Troubleshooting

- **`00` falha em autenticar / lista vazia:** faltou exportar `WANDB_API_KEY`
  (Seção 2). Confirme com `python -c "import wandb; print(len(list(wandb.Api().runs('gabomfim-unicamp/graph-rkd'))))"`.
- **Uma run com métrica absurda (≈0 ou NaN):** provavelmente uma run `crashed`.
  O `fetch_runs` já filtra `state=='finished'`; se ainda aparecer, cheque `df.state`.
- **Métrica esperada não existe** (`test_mAP@R` NaN): a run não logou
  `final_test_*` (ainda rodando ou morreu antes do teste). Ela é ignorada nas
  agregações via `dropna`.
- **Figura "sem runs de phaseX ainda":** normal em dados parciais — aquela fase
  ainda não fechou nenhuma run. Re-rode quando fechar.
- **`05` (qualitativo) sem GPU:** cai para CPU (lento) e precisa dos checkpoints
  `student_last.pth` + dataset baixado. Rode só quando for gerar as figuras finais.
- **λg "estranho" (10/100/1000) nos dados:** são runs `dev`/exploratórias antigas;
  os filtros de fase (`phase2..phase5`) já as excluem das análises do paper.

---

*Dúvidas de metodologia → docstrings do `analysis_utils.py` (cada função explica a
regra que aplica). Este guia + o `analysis_utils.py` são a referência canônica.*
