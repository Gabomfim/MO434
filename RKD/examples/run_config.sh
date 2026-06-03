#!/usr/bin/env bash
set -euo pipefail

# Config file for run.py (teacher training / evaluation)
# Usage:
#   bash examples/run_config.sh train
#   bash examples/run_config.sh eval

MODE="${1:-train}"  # train or eval

# Common parameters
DATASET="cub200"
BASE="resnet50"
EMBEDDING_SIZE=512
L2NORMALIZE="true"
DATA_DIR="../data"
SAVE_DIR="teacher"
SEED=42

# Weights & Biases (optional)
WANDB_PROJECT="rkd-metric-learning"
WANDB_ENTITY=""           # leave empty to use your default account/entity
WANDB_RUN_NAME=""         # leave empty to auto-generate
WANDB_MODE="online"       # online | offline | disabled
WANDB_GROUP="teacher-runs"
MAX_CONFUSION_CLASSES=300

# Training parameters
SAMPLE="distance"
LOSS="l2_triplet"
MARGIN=0.2
LR=1e-5
LR_DECAY_EPOCHS=(25 30 35)
LR_DECAY_GAMMA=0.5
BATCH=128
NUM_IMAGE_PER_CLASS=5
EPOCHS=40
ITER_PER_EPOCH=100
RECALL=(1)

# Eval/load parameters
LOAD_PATH="${SAVE_DIR}/best.pth"

if [[ "$MODE" == "train" ]]; then
  cmd=(
    python run.py
    --mode train
    --dataset "$DATASET"
    --base "$BASE"
    --sample "$SAMPLE"
    --loss "$LOSS"
    --margin "$MARGIN"
    --embedding_size "$EMBEDDING_SIZE"
    --l2normalize "$L2NORMALIZE"
    --lr "$LR"
    --lr_decay_epochs "${LR_DECAY_EPOCHS[@]}"
    --lr_decay_gamma "$LR_DECAY_GAMMA"
    --batch "$BATCH"
    --num_image_per_class "$NUM_IMAGE_PER_CLASS"
    --epochs "$EPOCHS"
    --iter_per_epoch "$ITER_PER_EPOCH"
    --recall "${RECALL[@]}"
    --seed "$SEED"
    --data "$DATA_DIR"
    --save_dir "$SAVE_DIR"
    --wandb_project "$WANDB_PROJECT"
    --wandb_run_name "$WANDB_RUN_NAME"
    --wandb_group "$WANDB_GROUP"
    --max_confusion_classes "$MAX_CONFUSION_CLASSES"
    --wandb_mode "$WANDB_MODE"
  )

  if [[ -n "$WANDB_ENTITY" ]]; then
    cmd+=(--wandb_entity "$WANDB_ENTITY")
  fi

  "${cmd[@]}"
elif [[ "$MODE" == "eval" ]]; then
  cmd=(
    python run.py
    --mode eval
    --dataset "$DATASET"
    --base "$BASE"
    --embedding_size "$EMBEDDING_SIZE"
    --l2normalize "$L2NORMALIZE"
    --batch "$BATCH"
    --recall "${RECALL[@]}"
    --load "$LOAD_PATH"
    --data "$DATA_DIR"
    --wandb_project "$WANDB_PROJECT"
    --wandb_run_name "$WANDB_RUN_NAME"
    --wandb_group "$WANDB_GROUP"
    --max_confusion_classes "$MAX_CONFUSION_CLASSES"
    --wandb_mode "$WANDB_MODE"
  )

  if [[ -n "$WANDB_ENTITY" ]]; then
    cmd+=(--wandb_entity "$WANDB_ENTITY")
  fi

  "${cmd[@]}"
else
  echo "Invalid MODE: $MODE (expected train or eval)" >&2
  exit 1
fi
