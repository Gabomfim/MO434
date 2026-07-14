#!/usr/bin/env bash
# ============================================================================
# OPTION A — run the experiments on a LOCAL GPU (no AWS).
# Logs to W&B gabomfim-unicamp/graph-rkd. Resumable.
#
# Prerequisites (on the machine with a GPU):
#   1) uv sync            # installs torch/torchvision/wandb/kagglehub
#   2) data in data/:     data/Cars196/{car_ims/,cars_annos.mat}
#                         data/CUB_200_2011/{images/,images.txt,...}
#   3) export WANDB_API_KEY=...   (or: wandb login)
#
# Runs SEVERAL experiments in parallel per GPU (auto by free VRAM) to use the
# GPU to the maximum; distributes across multiple GPUs in round-robin.
#
# Usage:
#   ./sm/run_local.sh                         # gate: teachers phase0 phase1 (auto parallel)
#   DATA=/path/data ./sm/run_local.sh         # data somewhere else
#   PHASES="phase5" ./sm/run_local.sh         # other phase(s)
#   MAX_PARALLEL=6 ./sm/run_local.sh          # fix 6 simultaneous jobs
#   PER_JOB_GB=3 ./sm/run_local.sh            # adjust the auto's VRAM/job
#   ./sm/run_local.sh --gate-dataset cub200   # gate on CUB instead of Cars
# ============================================================================
set -euo pipefail
DATA="${DATA:-data}"
PHASES="${PHASES:-teachers phase0 phase1}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"      # 0 = auto by VRAM
PER_JOB_GB="${PER_JOB_GB:-4.0}"
: "${WANDB_API_KEY:?set WANDB_API_KEY (or run 'wandb login') first}"

cd "$(dirname "$0")/.."   # -> RKD/
# AMP is already on per job (plan defaults); do not pass --amp here.
exec python sm/run_local.py --phases $PHASES --data "$DATA" \
    --max-parallel "$MAX_PARALLEL" --per-job-gb "$PER_JOB_GB" \
    --wandb-entity gabomfim-unicamp --wandb-project graph-rkd "$@"
