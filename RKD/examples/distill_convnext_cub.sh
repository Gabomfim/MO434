#!/usr/bin/env bash
set -euo pipefail

# Distill a fine-tuned ResNet-18 teacher into ConvNextMicro on CUB-200,
# combining Hinton KD + RKD distance + RKD angle + attention map (stage 2).
#
# Usage:
#   bash examples/distill_convnext_cub.sh
#
# Set ONE of TEACHER_ARTIFACT / TEACHER_LOAD to your fine-tuned ResNet-18
# (see finetune_resnet18.py --dataset cub200, which logs resnet18-cub200:best).

DATA_DIR="../data"
SAVE_DIR="distill_runs/cub200"

# teacher (pick one)
TEACHER_ARTIFACT=""                       # e.g. "me/resnet18-finetune/resnet18-cub200:best"
TEACHER_LOAD="finetune/cub200/best.pth"   # local finetune_resnet18.py checkpoint

# distillation loss weights
CE_RATIO=1.0
KD_RATIO=1.0
KD_T=4.0
DIST_RATIO=25.0
ANGLE_RATIO=50.0
AT_RATIO=1000.0

# optimization
LR=1e-3
EPOCHS=120
WARMUP_EPOCHS=5
BATCH=64

# W&B
WANDB_PROJECT="resnet18-to-convnext-distill"
WANDB_MODE="online"

cmd=(
  python distill_resnet18_convnext.py
  --dataset cub200
  --data "$DATA_DIR"
  --ce_ratio "$CE_RATIO"
  --kd_ratio "$KD_RATIO"
  --kd_T "$KD_T"
  --dist_ratio "$DIST_RATIO"
  --angle_ratio "$ANGLE_RATIO"
  --at_ratio "$AT_RATIO"
  --lr "$LR"
  --epochs "$EPOCHS"
  --warmup_epochs "$WARMUP_EPOCHS"
  --batch "$BATCH"
  --amp
  --save_dir "$SAVE_DIR"
  --wandb_project "$WANDB_PROJECT"
  --wandb_mode "$WANDB_MODE"
  --wandb_run_name "distill-cub200"
)

if [[ -n "$TEACHER_ARTIFACT" ]]; then
  cmd+=(--teacher_artifact "$TEACHER_ARTIFACT")
else
  cmd+=(--teacher_load "$TEACHER_LOAD")
fi

"${cmd[@]}"
