#!/usr/bin/env bash
# Orquestrador autonomo do pipeline Cars (destacado via setsid -> sobrevive a
# queda da sessao). Espera o professor (Passo 2) terminar, valida o artefato
# :best e dispara a destilacao (Passo 4) AUTOMATICAMENTE. So para em erro fatal,
# anotando no status. Nao faz push de nada.
set -u
cd /mnt/b/_ai/mo434/Gabriel/RKD || exit 99
export WANDB_ENTITY="rodz-ralm-v-ai"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ORCH=logs/cars_pipeline_orchestrator.log
STATUS=logs/cars_pipeline_status.txt   # RUNNING_TEACHER | RUNNING_DISTILL | DONE | FATAL:...
T_EXIT=logs/cars_passo2_teacher.exit
D_LOG=logs/cars_passo4_distill.log
D_EXIT=logs/cars_passo4_distill.exit

olog(){ echo "[orch $(date '+%F %T')] $*" >> "$ORCH"; }
setstatus(){ echo "$*" > "$STATUS"; olog "STATUS -> $*"; }

olog "=== orquestrador Cars iniciado (pid=$$) ==="
setstatus "RUNNING_TEACHER (Passo 2 professor, 60ep)"

# 1) esperar o professor terminar (marcador .exit). teto ~2h.
for i in $(seq 1 7200); do
  [ -f "$T_EXIT" ] && break
  sleep 1
done
if [ ! -f "$T_EXIT" ]; then
  setstatus "FATAL: professor nao terminou em ~2h (sem .exit) -- verificar logs/cars_passo2_teacher.log"
  exit 1
fi
TRC=$(cat "$T_EXIT")
olog "professor terminou com exit=$TRC"
if [ "$TRC" != "0" ]; then
  setstatus "FATAL: professor Cars (Passo 2) falhou exit=$TRC -- ver logs/cars_passo2_teacher.log"
  exit 1
fi

# 2) validar que o artefato :best do professor resolve no W&B (retries p/ propagacao)
olog "validando artefato resnet18-cars196:best no W&B..."
OK=0
for try in $(seq 1 12); do
  if python - <<'PY'
import sys, wandb
try:
    a = wandb.Api().artifact("rodz-ralm-v-ai/resnet18-finetune/resnet18-cars196:best")
    print("best ok:", a.name); sys.exit(0)
except Exception as e:
    print("ainda nao:", e); sys.exit(1)
PY
  then OK=1; break; fi
  sleep 15
done
if [ "$OK" != "1" ]; then
  setstatus "FATAL: professor exit 0 mas resnet18-cars196:best nao resolveu no W&B apos retries"
  exit 1
fi
olog "artefato :best confirmado."

# 3) disparar a destilacao (Passo 4) -- automatico, sem gate
setstatus "RUNNING_DISTILL (Passo 4 aluno ConvNextMicro, 300ep, ~3h)"
olog "=== iniciando Passo 4 (destilacao Cars) ==="
rm -f "$D_EXIT"
echo "===== [$(date '+%F %T')] INICIANDO Passo 4 (destilacao Cars) =====" >> "$D_LOG"
python distill_resnet18_convnext.py \
  --dataset cars196 --data ../data \
  --teacher_artifact "$WANDB_ENTITY"/resnet18-finetune/resnet18-cars196:best \
  --amp \
  --save_dir distill_runs/cars196 \
  --wandb_project resnet18-to-convnext-distill \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_run_name distill-cars196 >> "$D_LOG" 2>&1
DRC=$?
echo "===== [$(date '+%F %T')] FIM Passo 4 (exit=$DRC) =====" >> "$D_LOG"
echo "$DRC" > "$D_EXIT"
olog "destilacao terminou com exit=$DRC"
if [ "$DRC" != "0" ]; then
  setstatus "FATAL: destilacao Cars (Passo 4) falhou exit=$DRC -- ver logs/cars_passo4_distill.log"
  exit 1
fi
setstatus "DONE: pipeline Cars completo (professor + aluno registrados no W&B)"
olog "=== pipeline Cars COMPLETO ==="
