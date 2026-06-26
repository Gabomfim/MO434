#!/usr/bin/env bash
set -euo pipefail

# Baseline para comparação: treina a ConvNextMicro do zero (só CE) nos DOIS
# datasets (Cars-196 e CUB-200), com a mesma política de métricas/seleção e os
# mesmos hiperparâmetros do aluno destilado. Compare o top-1 de teste destes
# baselines com os alunos convnextmicro-distill-<arch>-<dataset>.
#
# Usage:
#   bash examples/train_convnextmicro_baseline.sh

DATA_DIR="../data"
WANDB_PROJECT="convnextmicro-distill"   # mesmo projeto dos alunos destilados
WANDB_MODE="online"

for DATASET in cars196 cub200; do
  echo "=================================================================="
  echo ">> baseline ConvNextMicro: $DATASET"
  echo "=================================================================="
  python train_convnextmicro.py \
    --dataset "$DATASET" \
    --data "$DATA_DIR" \
    --epochs 300 \
    --warmup_epochs 20 \
    --batch 128 \
    --amp \
    --save_dir "baseline_runs/$DATASET" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_mode "$WANDB_MODE" \
    --wandb_run_name "baseline-$DATASET"
done
