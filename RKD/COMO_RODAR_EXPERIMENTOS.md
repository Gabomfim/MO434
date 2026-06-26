# Como rodar os experimentos (campanha completa)

Tudo é orquestrado por `run_all_experiments.py`, que roda a grade inteira em
ordem de dependência e loga tudo no W&B. É **resumível**: re-executar pula o que
já terminou e retoma o que foi interrompido (inclusive a mesma run do W&B).

## 1. Pré-requisitos
```bash
uv sync                 # dependências (na raiz do repo)
cd RKD
wandb login             # cola a API key de wandb.ai/authorize
export WANDB_ENTITY="<sua-entidade>"
```
- GPU recomendada (ex.: AWS SageMaker `ml.g5.2xlarge`). Sem GPU é inviável.
- Os datasets (Cars-196, CUB-200) baixam sozinhos na 1ª execução (`--data ../data`).

## 2. Ver o plano antes de gastar GPU
```bash
WANDB_ENTITY=$WANDB_ENTITY bash examples/run_all_experiments.sh --dry_run
```
Mostra a contagem por fase e o total (com defaults: 4 professores + 2 baselines
CE + 12 clássicos + 256 Graph-RKD = **274 experimentos**; R=3 seeds).

## 3. Rodar tudo
```bash
WANDB_ENTITY=$WANDB_ENTITY bash examples/run_all_experiments.sh
```

## 4. Rodar por fase (recomendado p/ paralelizar / controlar)
As fases `classic` e `graph` dependem dos professores (fase `teachers`).
```bash
python run_all_experiments.py --phases teachers      --wandb_entity $WANDB_ENTITY --amp
python run_all_experiments.py --phases ce_baseline   --wandb_entity $WANDB_ENTITY --amp
python run_all_experiments.py --phases classic graph --wandb_entity $WANDB_ENTITY --amp
```
Dá para fatiar ainda mais por dataset/teacher (útil p/ jobs paralelos):
```bash
python run_all_experiments.py --phases graph --datasets cub200 --teachers resnet18 ...
```

## 5. Retomada (checkpoints) — nada se perde
- **Experimento concluído**: gera `result.json` no seu `save_dir`; re-rodar a
  campanha **pula** (imprime `SKIP (concluído)`).
- **Experimento interrompido no meio**: cada trainer salva o estado completo
  (`model/optimizer/scheduler/epoch/best`) **a cada época de avaliação**; ao
  reiniciar, ele **retoma da última época** (via `--resume`, já passado
  automaticamente).
- **W&B**: cada experimento usa um `wandb_id` estável, então a run reiniciada
  **continua a mesma run** no W&B (`resume="allow"`) em vez de criar outra.

Ou seja: se a máquina cair, basta **rodar o mesmo comando de novo**.

## 6. Hiperparâmetros principais (com defaults)
| Flag | Default | O quê |
|---|---|---|
| `--seeds` | 3 | seeds por candidato N na busca (R) |
| `--select` | argmax | seleção de N (`argmax` ou `1se`) |
| `--edge_budget` | 1024 | teto de N (K=128 → N_max=17 → candidatos log 2,4,8,16,17) |
| `--graph_warmup_frac` | 0.1 | warmup do peso da loss de grafo (10% das épocas) |
| `--finetune_epochs` | 60 | épocas do professor |
| `--student_epochs` | 300 | épocas finais do aluno |
| `--search_epochs` | 30 | épocas das runs curtas de busca de N |
| `--datasets/--teachers/--embeddings/--objectives` | grade cheia | recortes da grade |

Loss da destilação Graph-RKD = **só CE + a loss de grafo** (KD/RKD/AT desligados);
baselines clássicos = CE + **uma** técnica (Hinton / RKD-dist / RKD-angle).

## 7. Onde tudo é logado no W&B
- **Professores** → projeto `classifier-finetune` (artefato `<arch>-<dataset>:best`).
- **Alunos** (baseline CE, clássicos, Graph-RKD) → projeto `convnextmicro-distill`,
  em grupos: `baseline-<ds>`, `classic-<arch>-<ds>`, `<mode>-<method>-<ds>`.
- Métricas: top-1/top-5 em **train/val/test**, `val/best_top1`, perdas por
  componente, `train/graph_ratio` e `train/graph_temperature`, e os artefatos
  de modelo (`best`/`last`, com TTL 30 dias).
- Checkpoints locais e `result.json` ficam em `experiments/` (`--save_root`).

## 8. Paralelização (AWS SageMaker)
A busca de N agora é **não-adaptativa** (varredura log), então quase tudo é
independente. Para rodar em paralelo, lance recortes da grade como **jobs
separados** do SageMaker (cada um com seu `--phases`/`--datasets`/`--teachers`),
todos apontando para o mesmo `--save_root` (em volume/EFS compartilhado) e a
mesma entidade W&B. O ledger (`result.json`) evita refazer trabalho entre jobs.
Use **Managed Spot Training** para cortar custo — a retomada automática cobre as
interrupções.
