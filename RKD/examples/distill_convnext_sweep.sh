#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep for distill_convnext.py
# (teacher ConvNeXt  ->  student ConvNextMicro, feature/embedding distillation).
#
# Usage:
#   bash examples/distill_convnext_sweep.sh
#
# Edit the TEACHER below (one of TEACHER_ARTIFACT / TEACHER_LOAD) and the CONFIGS
# list. Each config sweeps the distillation loss ratios. Attention Transfer (AT)
# is always applied ONLY at the student's 2nd non-pointwise layer (stage 2);
# here we sweep its weight.

# ---- dataset / paths ----
DATASET="cars196"
DATA_DIR="../data"
SAVE_ROOT="distill_runs"

# ---- teacher (pick ONE) ----
TEACHER_ARTIFACT=""                 # e.g. "me/convnext-micro/student_best:best" or full ref
TEACHER_LOAD="teacher/last.pth"     # local checkpoint (train_convnext.py format or raw state_dict)
TEACHER_DIMS=(96 192 384 768)
TEACHER_DEPTHS=(1 1 3 1)
TEACHER_WEIGHTS="auto"             # auto | model | ema

# ---- student ----
DIMS=(24 48 96 192)
DEPTHS=(1 1 3 1)
DROP_PATH=0.1
L2NORMALIZE="true"

# ---- optimization ----
LR=1e-3
WEIGHT_DECAY=0.05
EPOCHS=120
WARMUP_EPOCHS=5
BATCH=128
ITER_PER_EPOCH=100
NUM_IMAGE_PER_CLASS=5
RECALL=(1 2 4 8)
EVAL_EVERY=5

# ---- W&B ----
WANDB_PROJECT="convnext-distill"
WANDB_ENTITY=""
WANDB_MODE="online"
WANDB_GROUP="distill-sweep"

# ---- sweep configs ----
# Format: "name dist angle dark triplet at"
# AT-scale note: AttentionTransfer returns a small normalized MSE, so its weight
# is typically large; we sweep a few decades to find the right scale.
CONFIGS=(
  "rkd-da        1 2 0 0 0"        # RKD distance+angle baseline (no AT)
  "rkd-da-at100  1 2 0 0 100"      # + attention transfer (stage 2)
  "rkd-da-at1k   1 2 0 0 1000"
  "rkd-da-at10k  1 2 0 0 10000"
  "at-only-1k    0 0 0 0 1000"     # attention transfer alone
  "dark          0 0 1 0 0"        # HardDarkRank alone
  "triplet-rkd   1 2 0 1 0"        # RKD + student triplet
  "full          1 2 1 1 1000"     # everything together
)

mkdir -p "$SAVE_ROOT"

for cfg in "${CONFIGS[@]}"; do
  read -r NAME DIST ANGLE DARK TRIPLET AT <<< "$cfg"
  echo "=================================================================="
  echo ">> config: $NAME  (dist=$DIST angle=$ANGLE dark=$DARK triplet=$TRIPLET at=$AT)"
  echo "=================================================================="

  cmd=(
    python distill_convnext.py
    --dataset "$DATASET"
    --data "$DATA_DIR"
    --dims "${DIMS[@]}"
    --depths "${DEPTHS[@]}"
    --drop_path "$DROP_PATH"
    --l2normalize "$L2NORMALIZE"
    --teacher_dims "${TEACHER_DIMS[@]}"
    --teacher_depths "${TEACHER_DEPTHS[@]}"
    --teacher_weights "$TEACHER_WEIGHTS"
    --dist_ratio "$DIST"
    --angle_ratio "$ANGLE"
    --dark_ratio "$DARK"
    --triplet_ratio "$TRIPLET"
    --at_ratio "$AT"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --epochs "$EPOCHS"
    --warmup_epochs "$WARMUP_EPOCHS"
    --batch "$BATCH"
    --iter_per_epoch "$ITER_PER_EPOCH"
    --num_image_per_class "$NUM_IMAGE_PER_CLASS"
    --recall "${RECALL[@]}"
    --eval_every "$EVAL_EVERY"
    --save_dir "$SAVE_ROOT/$NAME"
    --wandb_project "$WANDB_PROJECT"
    --wandb_group "$WANDB_GROUP"
    --wandb_run_name "$NAME"
    --wandb_mode "$WANDB_MODE"
    --wandb_tags "$NAME"
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
done
