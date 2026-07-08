#!/usr/bin/env bash
# ============================================================================
# OPÇÃO A — rodar os experimentos numa GPU LOCAL (sem AWS).
# Loga no W&B gabomfim-unicamp/graph-rkd. Resumível.
#
# Pré-requisitos (na máquina com GPU):
#   1) uv sync            # instala torch/torchvision/wandb/kagglehub
#   2) dados em data/:    data/Cars196/{car_ims/,cars_annos.mat}
#                         data/CUB_200_2011/{images/,images.txt,...}
#   3) export WANDB_API_KEY=...   (ou: wandb login)
#
# Roda VÁRIOS experimentos em paralelo por GPU (auto pela VRAM livre) p/ usar a
# GPU ao máximo; distribui entre múltiplas GPUs em round-robin.
#
# Uso:
#   ./sm/run_local.sh                         # gate: teachers phase0 phase1 (paralelo auto)
#   DATA=/caminho/data ./sm/run_local.sh      # dados em outro lugar
#   PHASES="phase5" ./sm/run_local.sh         # outra(s) fase(s)
#   MAX_PARALLEL=6 ./sm/run_local.sh          # fixar 6 jobs simultâneos
#   PER_JOB_GB=3 ./sm/run_local.sh            # ajustar VRAM/job do auto
#   ./sm/run_local.sh --gate-dataset cub200   # gate em CUB em vez de Cars
# ============================================================================
set -euo pipefail
DATA="${DATA:-data}"
PHASES="${PHASES:-teachers phase0 phase1}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"      # 0 = auto pela VRAM
PER_JOB_GB="${PER_JOB_GB:-4.0}"
: "${WANDB_API_KEY:?defina WANDB_API_KEY (ou rode 'wandb login') antes}"

cd "$(dirname "$0")/.."   # -> RKD/
# AMP já vem ligado por job (plan defaults); não passar --amp aqui.
exec python sm/run_local.py --phases $PHASES --data "$DATA" \
    --max-parallel "$MAX_PARALLEL" --per-job-gb "$PER_JOB_GB" \
    --wandb-entity gabomfim-unicamp --wandb-project graph-rkd "$@"
