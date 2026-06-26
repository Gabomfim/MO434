#!/usr/bin/env bash
# Passo 2 - professor Cars (ResNet-18, 60 epocas). Destacado da sessao do Code
# via setsid/nohup -> sobrevive ao fechamento do editor. Grava .exit ao terminar.
set -u
cd /mnt/b/_ai/mo434/Gabriel/RKD || exit 99
export WANDB_ENTITY="rodz-ralm-v-ai"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/cars_passo2_teacher.log
MARK=logs/cars_passo2_teacher.exit
rm -f "$MARK"
echo "===== [$(date '+%F %T')] INICIANDO Passo 2 (professor Cars) pid=$$ =====" >> "$LOG"
python finetune_resnet18.py \
  --dataset cars196 --data ../data \
  --epochs 60 --batch 64 --amp \
  --save_dir finetune/cars196 \
  --wandb_project resnet18-finetune \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_run_name resnet18-cars196 >> "$LOG" 2>&1
rc=$?
echo "===== [$(date '+%F %T')] FIM Passo 2 (exit=$rc) =====" >> "$LOG"
echo "$rc" > "$MARK"
