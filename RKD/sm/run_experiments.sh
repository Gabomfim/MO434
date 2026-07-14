#!/usr/bin/env bash
# ============================================================================
# RUN THE Graph-RKD EXPERIMENTS on a GPU — ONE COMMAND. Resumable.
#
# LEAN campaign (--trimmed): teachers + phases 2,3,4,5. Logs to W&B
# gabomfim-unicamp/graph-rkd. The datasets download themselves from the public S3 (no
# AWS credentials) and are cached. Uses the GPU to the MAXIMUM (packs several jobs
# per GPU by free VRAM; round-robin across GPUs).
#
# Prerequisites:
#   uv sync
#   export WANDB_API_KEY=...      # or: uv run wandb login
#
# Usage:
#   ./sm/run_experiments.sh                      # the whole lean campaign (~2-3 days)
#   PHASES="teachers phase2 phase3 phase4" ./sm/run_experiments.sh   # only the search
#   PHASES="phase5" ./sm/run_experiments.sh      # only the headline
#   MAX_PARALLEL=3 ./sm/run_experiments.sh       # fix the number of simultaneous jobs
#   PER_JOB_GB=3 ./sm/run_experiments.sh         # adjust the auto's VRAM/job
# ============================================================================
set -euo pipefail

PHASES="${PHASES:-teachers phase2 phase3 phase4 phase5}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"     # 0 = auto by VRAM (100% of the GPU)
PER_JOB_GB="${PER_JOB_GB:-4.0}"
: "${WANDB_API_KEY:?set WANDB_API_KEY (or run 'uv run wandb login')}"

cd "$(dirname "$0")/.."   # -> RKD/

# GPU check (training images on CPU is unfeasible)
uv run --no-sync python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo 'ERROR: no CUDA GPU visible. Run on a machine with a GPU.'; exit 1; }

echo ">> LEAN campaign | phases: [$PHASES] | parallelism: $MAX_PARALLEL (0=auto)"
echo ">> W&B: https://wandb.ai/gabomfim-unicamp/graph-rkd  (resumable: re-run this command)"
exec uv run --no-sync python sm/run_local.py --trimmed --phases $PHASES \
    --max-parallel "$MAX_PARALLEL" --per-job-gb "$PER_JOB_GB" \
    --wandb-entity gabomfim-unicamp --wandb-project graph-rkd
