#!/usr/bin/env bash
set -euo pipefail

# Campanha completa de METRIC LEARNING (retrieval, recall@K) — substitui a de
# classificação. Teachers (embedding+triplet) -> baselines -> Graph-RKD métrico.
# Comece com --dry_run para ver o plano.
#
# Uso:
#   WANDB_ENTITY=<voce> bash examples/run_all_experiments_metric.sh --dry_run
#   WANDB_ENTITY=<voce> bash examples/run_all_experiments_metric.sh
# Por fase (dependem dos teachers):
#   ... --phases teachers
#   ... --phases baseline classic graph

WANDB_ENTITY="${WANDB_ENTITY:-}"
DATA_DIR="${DATA_DIR:-../data}"

cmd=(
  python run_all_experiments_metric.py
  --data "$DATA_DIR"
  --datasets cars196 cub200
  --teachers resnet18 convnext_tiny
  --embeddings profile mds
  --objectives regression contrastive
  --teacher_epochs 60
  --student_epochs 120
  --search_epochs 30
  --edge_budget 1024
  --seeds 3
  --select argmax
  --rel_warmup_frac 0.1
  --recall 1 2 4 8
  --amp
  --wandb_mode online
)
if [[ -n "$WANDB_ENTITY" ]]; then
  cmd+=(--wandb_entity "$WANDB_ENTITY")
fi
"${cmd[@]}" "$@"
