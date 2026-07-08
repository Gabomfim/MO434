#!/usr/bin/env bash
# ============================================================================
# OPÇÃO B — rodar os experimentos no AWS SageMaker (jobs em PARALELO).
# Loga no W&B gabomfim-unicamp/graph-rkd.
#
# Pré-requisitos:
#   1) cota g5.xlarge de TREINO aprovada (pendente — ver status da campanha)
#   2) role de execução do SageMaker  -> export ROLE_ARN=arn:aws:iam::...:role/...
#   3) dados no S3:  aws s3 sync data/ s3://graph-rkd-832271495954/graph-rkd/data/
#   4) pip install sagemaker boto3   +   export WANDB_API_KEY=...
#
# Por padrão faz DRY-RUN (não cria nada). Para lançar de verdade: LAUNCH=1
#
# Uso:
#   ROLE_ARN=... ./sm/run_sagemaker.sh                 # dry-run do gate
#   LAUNCH=1 ROLE_ARN=... ./sm/run_sagemaker.sh        # cria os jobs
#   PHASES="phase5" LAUNCH=1 ROLE_ARN=... ./sm/run_sagemaker.sh
# ============================================================================
set -euo pipefail
export AWS_PROFILE="${AWS_PROFILE:-gabomfim}"
REGION="${REGION:-us-east-1}"
BUCKET="${BUCKET:-graph-rkd-832271495954}"
DATA_S3="${DATA_S3:-s3://graph-rkd-832271495954/graph-rkd/data/}"
PHASES="${PHASES:-teachers phase0 phase1}"
: "${ROLE_ARN:?defina ROLE_ARN (role de execução do SageMaker)}"

LAUNCH_FLAG=""
if [ "${LAUNCH:-0}" = "1" ]; then
    LAUNCH_FLAG="--launch"
    : "${WANDB_API_KEY:?defina WANDB_API_KEY (necessário p/ --launch)}"
fi

cd "$(dirname "$0")/.."   # -> RKD/
exec python sm/launch.py --phases $PHASES $LAUNCH_FLAG \
    --region "$REGION" --role "$ROLE_ARN" --bucket "$BUCKET" \
    --data-s3 "$DATA_S3" --instance-type ml.g5.xlarge \
    --wandb-entity gabomfim-unicamp --wandb-project graph-rkd "$@"
