#!/usr/bin/env bash
set -euo pipefail

# Experimentos Graph-RKD: para cada (modo de loss × método de embedding),
# busca binária do melhor N (guiada por top-1 de validação) e roda a destilação
# = SÓ cross-entropy + loss de grafo de N nós (distância euclidiana). Sem KD,
# dist, angle, quad nem attention. A temperatura da destilação (InfoNCE) varia
# ao longo do treino (--temp_schedule).
#
# Uso:
#   TEACHER_ARCH=resnet18 DATASET=cub200 bash examples/graph_rkd_search.sh
#
# ATENÇÃO: dispara MUITAS destilações (≈ log2(N) runs de busca + 1 final por
# combinação; 2 modos × 2 métodos). Ajuste SEARCH_EPOCHS/FINAL_EPOCHS e use GPU.

TEACHER_ARCH="${TEACHER_ARCH:-resnet18}"
DATASET="${DATASET:-cub200}"
DATA_DIR="../data"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# professor (escolha um): artefato do W&B (recomendado) ou checkpoint local
TEACHER_ARTIFACT="${TEACHER_ARTIFACT:-}"   # ex.: "$WANDB_ENTITY/classifier-finetune/${TEACHER_ARCH}-${DATASET}:best"
TEACHER_LOAD="${TEACHER_LOAD:-finetune/${TEACHER_ARCH}-${DATASET}/best.pth}"

cmd=(
  python run_graph_rkd_search.py
  --teacher_arch "$TEACHER_ARCH"
  --dataset "$DATASET"
  --data "$DATA_DIR"
  --batch 128
  --edge_budget 1024
  --modes regression contrastive
  --methods profile mds
  --search_epochs 30
  --final_epochs 300
  --temp_schedule cosine
  --temp_start 0.1
  --temp_end 0.05
  --amp
  --wandb_project "graph-rkd-node-search"
  --wandb_mode online
)

if [[ -n "$TEACHER_ARTIFACT" ]]; then
  cmd+=(--teacher_artifact "$TEACHER_ARTIFACT")
else
  cmd+=(--teacher_load "$TEACHER_LOAD")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
  cmd+=(--wandb_entity "$WANDB_ENTITY")
fi

"${cmd[@]}"
