#!/usr/bin/env bash
set -euo pipefail

# Distill a fine-tuned teacher into ConvNextMicro on CUB-200, combining
# Hinton KD + RKD distance + RKD angle + attention map (stage 2).
#
# Usage:
#   TEACHER_ARCH=resnet18      bash examples/distill_convnext_cub.sh
#   TEACHER_ARCH=convnext_tiny bash examples/distill_convnext_cub.sh
#
# Set ONE of TEACHER_ARTIFACT / TEACHER_LOAD to your fine-tuned teacher
# (see finetune_classifier.py --arch <arch> --dataset cub200, which logs
#  <arch>-cub200:best).

TEACHER_ARCH="${TEACHER_ARCH:-resnet18}"     # resnet18 | convnext_tiny
DATA_DIR="../data"
SAVE_DIR="distill_runs/${TEACHER_ARCH}-cub200"

# teacher (pick one)
TEACHER_ARTIFACT=""                                  # e.g. "me/classifier-finetune/${TEACHER_ARCH}-cub200:best"
TEACHER_LOAD="finetune/${TEACHER_ARCH}-cub200/best.pth"  # local finetune_classifier.py checkpoint

# distillation loss weights (source-paper / RepDistiller conventions)
CE_RATIO=1.0
KD_RATIO=0.9
KD_T=4.0
DIST_RATIO=25.0
ANGLE_RATIO=50.0
AT_RATIO=1000.0

# optimization (student trained from scratch -> long schedule)
LR=1e-3
EPOCHS=300
WARMUP_EPOCHS=20
BATCH=128

# W&B
WANDB_PROJECT="convnextmicro-distill"
WANDB_MODE="online"

cmd=(
  python distill_to_convnextmicro.py
  --teacher_arch "$TEACHER_ARCH"
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
  --wandb_run_name "distill-${TEACHER_ARCH}-cub200"
)

if [[ -n "$TEACHER_ARTIFACT" ]]; then
  cmd+=(--teacher_artifact "$TEACHER_ARTIFACT")
else
  cmd+=(--teacher_load "$TEACHER_LOAD")
fi

"${cmd[@]}"
