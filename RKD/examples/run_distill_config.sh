#!/usr/bin/env bash
set -euo pipefail

# Config file for run_distill.py (student distillation)
# Usage:
#   bash examples/run_distill_config.sh

# Dataset / paths
DATASET="cub200"
DATA_DIR="data"
SAVE_DIR="student"
LOAD_PATH=""  # optional student checkpoint to resume from

# Weights & Biases (optional)
WANDB_PROJECT="rkd-metric-learning"
WANDB_ENTITY=""           # leave empty to use your default account/entity
WANDB_RUN_NAME=""         # leave empty to auto-generate
WANDB_MODE="online"       # online | offline | disabled
WANDB_GROUP="distillation-experiments"

# Student model
BASE="resnet18"
EMBEDDING_SIZE=64
L2NORMALIZE="false"

# Teacher model
TEACHER_BASE="resnet50"
TEACHER_EMBEDDING_SIZE=512
TEACHER_L2NORMALIZE="true"
TEACHER_LOAD="teacher/best.pth"  # required

# Distillation loss weights
TRIPLET_RATIO=0
DIST_RATIO=1
ANGLE_RATIO=2
QUAD_RATIO=0
DARK_RATIO=0
DARK_ALPHA=2
DARK_BETA=3
AT_RATIO=0

# Triplet configuration for student
TRIPLET_SAMPLE="distance"
TRIPLET_MARGIN=0.2

# Optimization / schedule
LR=1e-4
EPOCHS=80
BATCH=128
ITER_PER_EPOCH=100
LR_DECAY_EPOCHS=(40 60)
LR_DECAY_GAMMA=0.1
RECALL=(1 2 4 8)
LOG_CONFUSION_MATRIX="true"
MAX_CONFUSION_CLASSES=200
MAX_CONFUSION_SAMPLES=2000

cmd=(
  python run_distill.py
  --dataset "$DATASET"
  --base "$BASE"
  --teacher_base "$TEACHER_BASE"
  --triplet_ratio "$TRIPLET_RATIO"
  --dist_ratio "$DIST_RATIO"
  --angle_ratio "$ANGLE_RATIO"
  --quad_ratio "$QUAD_RATIO"
  --dark_ratio "$DARK_RATIO"
  --dark_alpha "$DARK_ALPHA"
  --dark_beta "$DARK_BETA"
  --at_ratio "$AT_RATIO"
  --triplet_sample "$TRIPLET_SAMPLE"
  --triplet_margin "$TRIPLET_MARGIN"
  --l2normalize "$L2NORMALIZE"
  --embedding_size "$EMBEDDING_SIZE"
  --teacher_load "$TEACHER_LOAD"
  --teacher_l2normalize "$TEACHER_L2NORMALIZE"
  --teacher_embedding_size "$TEACHER_EMBEDDING_SIZE"
  --lr "$LR"
  --data "$DATA_DIR"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --iter_per_epoch "$ITER_PER_EPOCH"
  --lr_decay_epochs "${LR_DECAY_EPOCHS[@]}"
  --lr_decay_gamma "$LR_DECAY_GAMMA"
  --recall "${RECALL[@]}"
  --log_confusion_matrix "$LOG_CONFUSION_MATRIX"
  --max_confusion_classes "$MAX_CONFUSION_CLASSES"
  --max_confusion_samples "$MAX_CONFUSION_SAMPLES"
  --save_dir "$SAVE_DIR"
  --wandb_project "$WANDB_PROJECT"
  --wandb_run_name "$WANDB_RUN_NAME"
  --wandb_group "$WANDB_GROUP"
  --wandb_mode "$WANDB_MODE"
)

if [[ -n "$LOAD_PATH" ]]; then
  cmd+=(--load "$LOAD_PATH")
fi

if [[ -n "$WANDB_ENTITY" ]]; then
  cmd+=(--wandb_entity "$WANDB_ENTITY")
fi

"${cmd[@]}"
