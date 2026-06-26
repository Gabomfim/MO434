#!/usr/bin/env bash
set -euo pipefail

# Roda TODA a campanha de experimentos (professores -> baselines -> Graph-RKD),
# logando tudo no W&B. Comece com --dry_run para ver o plano e a contagem.
#
# Uso:
#   WANDB_ENTITY=<voce> bash examples/run_all_experiments.sh --dry_run
#   WANDB_ENTITY=<voce> bash examples/run_all_experiments.sh        # roda de verdade
#
# Dica: use GPU e considere rodar fases separadas, ex.:
#   ... --phases teachers
#   ... --phases ce_baseline classic
#   ... --phases graph
# (as fases classic/graph dependem dos professores da fase teachers)

WANDB_ENTITY="${WANDB_ENTITY:-}"
DATA_DIR="${DATA_DIR:-../data}"

cmd=(
  python run_all_experiments.py
  --data "$DATA_DIR"
  --datasets cars196 cub200
  --teachers resnet18 convnext_tiny
  --embeddings profile mds
  --objectives regression contrastive
  --finetune_epochs 60
  --student_epochs 300
  --search_epochs 30
  --edge_budget 1024
  --seeds 1
  --select argmax
  --amp
  --wandb_mode online
)
if [[ -n "$WANDB_ENTITY" ]]; then
  cmd+=(--wandb_entity "$WANDB_ENTITY")
fi

# repassa flags extras (ex.: --dry_run, --phases graph)
"${cmd[@]}" "$@"
