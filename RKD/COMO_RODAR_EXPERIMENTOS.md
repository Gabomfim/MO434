# Como rodar os experimentos (campanha de METRIC LEARNING)

A campanha é de **metric learning / retrieval** (split disjunto, **recall@K**),
orquestrada por `run_all_experiments_metric.py`. Roda a grade inteira em ordem de
dependência, loga tudo no W&B e é **resumível** (pula o concluído, retoma o
interrompido na mesma run do W&B).

## 1. Pré-requisitos
```bash
uv sync
cd RKD
wandb login
export WANDB_ENTITY="<sua-entidade>"
```
- GPU recomendada (ex.: AWS SageMaker `ml.g5.2xlarge`).
- Datasets (Cars-196, CUB-200) baixam sozinhos na 1ª execução (`--data ../data`).
  (Não usamos Stanford Online Products.)

## 2. Ver o plano (sem treinar)
```bash
WANDB_ENTITY=$WANDB_ENTITY bash examples/run_all_experiments_metric.sh --dry_run
```
Com defaults: 4 teachers + 2 baselines + 8 clássicos + 256 Graph-RKD = **270 experimentos** (R=3 seeds).

## 3. Rodar tudo
```bash
WANDB_ENTITY=$WANDB_ENTITY bash examples/run_all_experiments_metric.sh
```

## 4. Por fase (paralelizar/controlar) — fases 1 e 2 dependem dos teachers
```bash
python run_all_experiments_metric.py --phases teachers          --wandb_entity $WANDB_ENTITY --amp
python run_all_experiments_metric.py --phases baseline          --wandb_entity $WANDB_ENTITY --amp
python run_all_experiments_metric.py --phases classic graph     --wandb_entity $WANDB_ENTITY --amp
# fatiar mais:
python run_all_experiments_metric.py --phases graph --datasets cub200 --teachers resnet18 ...
```

## 5. Retomada (nada se perde)
- Experimento concluído → grava `result.json`; re-rodar **pula** (`SKIP`).
- Interrompido no meio → estado completo salvo a cada época de avaliação; reinicia
  e **retoma da última época** (`--resume` automático).
- W&B → `wandb_id` estável + `resume="allow"`: a run reiniciada **continua a mesma**.
Se a máquina cair, **rode o mesmo comando de novo**.

## 6. Estrutura dos experimentos
- **Teachers** (`finetune_metric`): resnet18 / convnext_tiny como rede de
  embedding, **triplet loss**, recall@K.
- **Baseline** (`train_metric_baseline`): ConvNextMicro embedding, **triplet puro**.
- **Clássicos** (`distill_metric`): triplet + **RKD-distance isolado** e triplet +
  **RKD-angle isolado**, com **warmup** balanceando o termo relacional contra o triplet.
- **Graph-RKD** (`distill_metric` via `run_metric_search`): triplet + **loss de
  grafo** (regression/contrastive × profile/mds), com busca de N (varredura log),
  selecionada pelo **recall@1 de validação**.

## 7. Hiperparâmetros principais (defaults)
| Flag | Default | O quê |
|---|---|---|
| `--seeds` | 3 | seeds por candidato N |
| `--select` | argmax | seleção de N (`argmax` ou `1se`) |
| `--edge_budget` | 1024 | teto de N (K=128 → N_max=17 → 2,4,8,16,17) |
| `--rel_warmup_frac` | 0.1 | warmup do termo relacional vs triplet |
| `--teacher_epochs` | 60 | épocas do teacher métrico |
| `--student_epochs` | 120 | épocas finais do aluno |
| `--search_epochs` | 30 | épocas das runs de busca de N |
| `--recall` | 1 2 4 8 | K do recall@K |

Sem Hinton KD (metric learning não tem logits). Distância euclidiana nas arestas.

## 8. W&B
- **Teachers** → projeto `metric-teacher-finetune` (artefato `metric-<arch>-<dataset>:best`).
- **Alunos** (baseline, clássicos, Graph-RKD) → projeto `convnextmicro-metric-distill`;
  grupos `baseline-<ds>`, `classic-<arch>-<ds>`, `<mode>-<method>-<ds>`.
- Métricas: **recall@K** em train/val/test, `val/best_recall@1`, perdas por
  componente, `train/rel_scale`, e artefatos de modelo (`best`/`last`, TTL 30d).
- Saídas locais + `result.json` em `experiments_metric/` (`--save_root`).

## 9. Paralelização (AWS SageMaker)
A busca de N é não-adaptativa (varredura log) → quase tudo é independente. Lance
recortes da grade como jobs separados (`--phases`/`--datasets`/`--teachers`),
todos apontando para o mesmo `--save_root` compartilhado e a mesma entidade W&B;
o ledger evita refazer trabalho. Use Managed Spot Training (a retomada cobre
interrupções).

> Os scripts de **classificação** (`run_all_experiments.py`,
> `finetune_classifier.py`, `distill_to_convnextmicro.py`, `train_convnextmicro.py`)
> permanecem no repo, mas a campanha agora é métrica — use os `*_metric`.
