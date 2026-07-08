#!/usr/bin/env bash
# ============================================================================
# RODAR OS EXPERIMENTOS Graph-RKD numa GPU — UM COMANDO. Resumível.
#
# Campanha ENXUTA (--trimmed): teachers + fases 2,3,4,5. Loga no W&B
# gabomfim-unicamp/graph-rkd. Os datasets baixam sozinhos do S3 público (sem
# credenciais AWS) e ficam em cache. Usa a GPU ao MÁXIMO (empacota vários jobs
# por GPU pela VRAM livre; round-robin entre GPUs).
#
# Pré-requisitos (ver RUNBOOK.md):
#   uv sync
#   export WANDB_API_KEY=...      # ou: uv run wandb login
#
# Uso:
#   ./sm/run_experiments.sh                      # campanha enxuta inteira (~2-3 dias)
#   PHASES="teachers phase2 phase3 phase4" ./sm/run_experiments.sh   # só a busca
#   PHASES="phase5" ./sm/run_experiments.sh      # só o headline
#   MAX_PARALLEL=3 ./sm/run_experiments.sh       # fixar nº de jobs simultâneos
#   PER_JOB_GB=3 ./sm/run_experiments.sh         # ajustar VRAM/job do auto
# ============================================================================
set -euo pipefail

PHASES="${PHASES:-teachers phase2 phase3 phase4 phase5}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"     # 0 = auto pela VRAM (100% da GPU)
PER_JOB_GB="${PER_JOB_GB:-4.0}"
: "${WANDB_API_KEY:?defina WANDB_API_KEY (ou rode 'uv run wandb login') — ver RUNBOOK.md}"

cd "$(dirname "$0")/.."   # -> RKD/

# checagem de GPU (o treino de imagens em CPU é inviável)
uv run --no-sync python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo 'ERRO: nenhuma GPU CUDA visível. Rode numa máquina com GPU (ver RUNBOOK.md).'; exit 1; }

echo ">> Campanha ENXUTA | fases: [$PHASES] | paralelismo: $MAX_PARALLEL (0=auto)"
echo ">> W&B: https://wandb.ai/gabomfim-unicamp/graph-rkd  (resumível: re-rode este comando)"
exec uv run --no-sync python sm/run_local.py --trimmed --phases $PHASES \
    --max-parallel "$MAX_PARALLEL" --per-job-gb "$PER_JOB_GB" \
    --wandb-entity gabomfim-unicamp --wandb-project graph-rkd
