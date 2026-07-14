#!/usr/bin/env bash
# ============================================================================
# OPTION B — run the experiments on AWS SageMaker (jobs in PARALLEL).
# Logs to W&B gabomfim-unicamp/graph-rkd.
#
# Prerequisites:
#   1) approved g5.xlarge TRAINING quota (pending — see campaign status)
#   2) SageMaker execution role  -> export ROLE_ARN=arn:aws:iam::...:role/...
#   3) data on S3:  aws s3 sync data/ s3://graph-rkd-832271495954/graph-rkd/data/
#   4) pip install sagemaker boto3   +   export WANDB_API_KEY=...
#
# By default it does a DRY-RUN (creates nothing). To launch for real: LAUNCH=1
#
# Usage:
#   ROLE_ARN=... ./sm/run_sagemaker.sh                 # dry-run of the gate
#   LAUNCH=1 ROLE_ARN=... ./sm/run_sagemaker.sh        # creates the jobs
#   PHASES="phase5" LAUNCH=1 ROLE_ARN=... ./sm/run_sagemaker.sh
# ============================================================================
set -euo pipefail
export AWS_PROFILE="${AWS_PROFILE:-gabomfim}"
REGION="${REGION:-us-east-1}"
BUCKET="${BUCKET:-graph-rkd-832271495954}"
DATA_S3="${DATA_S3:-s3://graph-rkd-832271495954/graph-rkd/data/}"
PHASES="${PHASES:-teachers phase0 phase1}"
: "${ROLE_ARN:?set ROLE_ARN (SageMaker execution role)}"

LAUNCH_FLAG=""
if [ "${LAUNCH:-0}" = "1" ]; then
    LAUNCH_FLAG="--launch"
    : "${WANDB_API_KEY:?set WANDB_API_KEY (required for --launch)}"
fi

cd "$(dirname "$0")/.."   # -> RKD/
exec python sm/launch.py --phases $PHASES $LAUNCH_FLAG \
    --region "$REGION" --role "$ROLE_ARN" --bucket "$BUCKET" \
    --data-s3 "$DATA_S3" --instance-type ml.g5.xlarge \
    --wandb-entity gabomfim-unicamp --wandb-project graph-rkd "$@"
