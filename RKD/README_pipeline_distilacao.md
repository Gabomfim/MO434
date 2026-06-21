# Pipeline de Fine-tuning e Destilação (ResNet-18 → ConvNextMicro)

Este guia descreve o fluxo completo, registrando **todos** os modelos no
Weights & Biases (W&B):

1. Fine-tune da **ResNet-18** (pré-treinada na ImageNet-1k) no **CUB-200** → registra no W&B
2. Fine-tune da **ResNet-18** no **Cars-196** → registra no W&B
3. Destilação da ResNet-18 (professor, CUB) → **ConvNextMicro** (aluno) para classificação CUB → registra no W&B
4. Destilação da ResNet-18 (professor, Cars) → **ConvNextMicro** (aluno) para classificação Cars → registra no W&B

A destilação combina quatro técnicas: **Hinton KD** (logits), **RKD distance**,
**RKD angle** (embeddings) e **mapa de atenção** (stage-2 do aluno ↔ layer2 da
ResNet-18).

---

## Pré-requisitos

```bash
# 1. dependências (na raiz do repositório)
uv sync

# 2. entrar na pasta dos scripts
cd RKD

# 3. autenticar no W&B (uma única vez por máquina)
wandb login
```

- Os datasets (CUB-200 e Cars-196) são **baixados automaticamente** na primeira
  execução para `../data` (configurável via `--data`).
- Use `--amp` se tiver GPU (treino em precisão mista, mais rápido).
- **`--save_dir` é obrigatório para registrar o modelo no W&B**: o artefato só é
  enviado quando há um diretório de checkpoint. Sem ele, o modelo não é salvo
  nem registrado.

Defina sua entidade do W&B (usuário ou time) em uma variável para reaproveitar:

```bash
export WANDB_ENTITY="<sua-entidade>"   # ex.: seu usuário do W&B
```

---

## Passo 1 — Fine-tune da ResNet-18 no CUB-200 (e registro no W&B)

```bash
python finetune_resnet18.py \
  --dataset cub200 \
  --data ../data \
  --epochs 60 --batch 64 --amp \
  --save_dir finetune/cub200 \
  --wandb_project resnet18-finetune \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_run_name resnet18-cub200
```

- **Projeto W&B:** `resnet18-finetune`
- **Artefato registrado:** `resnet18-cub200` com os aliases `best` (melhor top-1)
  e `last` / `epoch-N`.
- **Referência para usar depois:** `"$WANDB_ENTITY"/resnet18-finetune/resnet18-cub200:best`

---

## Passo 2 — Fine-tune da ResNet-18 no Cars-196 (e registro no W&B)

```bash
python finetune_resnet18.py \
  --dataset cars196 \
  --data ../data \
  --epochs 60 --batch 64 --amp \
  --save_dir finetune/cars196 \
  --wandb_project resnet18-finetune \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_run_name resnet18-cars196
```

- **Projeto W&B:** `resnet18-finetune`
- **Artefato registrado:** `resnet18-cars196` (aliases `best`, `last`, `epoch-N`).
- **Referência para usar depois:** `"$WANDB_ENTITY"/resnet18-finetune/resnet18-cars196:best`

---

## Passo 3 — Destilar a ResNet-18 (CUB) para a ConvNextMicro (classificação CUB)

Usa o professor registrado no Passo 1 **direto do W&B** (não precisa de arquivo
local).

```bash
python distill_resnet18_convnext.py \
  --dataset cub200 \
  --data ../data \
  --teacher_artifact "$WANDB_ENTITY"/resnet18-finetune/resnet18-cub200:best \
  --amp \
  --save_dir distill_runs/cub200 \
  --wandb_project resnet18-to-convnext-distill \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_run_name distill-cub200
```

- **Projeto W&B:** `resnet18-to-convnext-distill`
- **Aluno registrado:** `convnextmicro-distill-cub200` (aliases `best`, `last`, `epoch-N`).
- O professor é puxado do W&B e fica registrado na *lineage* da run de destilação.

> ⚠️ **Importante:** a referência do professor precisa ser **completa**
> (`entidade/projeto/nome:alias`), porque o fine-tune foi registrado no projeto
> `resnet18-finetune`, enquanto a destilação roda no projeto
> `resnet18-to-convnext-distill`. A forma curta `resnet18-cub200:best` só
> funcionaria se ambos estivessem no mesmo projeto.

---

## Passo 4 — Destilar a ResNet-18 (Cars) para a ConvNextMicro (classificação Cars)

```bash
python distill_resnet18_convnext.py \
  --dataset cars196 \
  --data ../data \
  --teacher_artifact "$WANDB_ENTITY"/resnet18-finetune/resnet18-cars196:best \
  --amp \
  --save_dir distill_runs/cars196 \
  --wandb_project resnet18-to-convnext-distill \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_run_name distill-cars196
```

- **Projeto W&B:** `resnet18-to-convnext-distill`
- **Aluno registrado:** `convnextmicro-distill-cars196` (aliases `best`, `last`, `epoch-N`).

---

## Atalho: scripts de exemplo

Em vez dos comandos acima, dá para usar os launchers (edite o `TEACHER_ARTIFACT`
no topo de cada um com a referência completa do professor):

```bash
bash examples/distill_convnext_cub.sh
bash examples/distill_convnext_cars.sh
```

---

## Resumo dos artefatos gerados no W&B

| Passo | Projeto W&B | Artefato | Alias | Como referenciar |
|------|--------------|----------|-------|------------------|
| 1 | `resnet18-finetune` | `resnet18-cub200` | `best` | `<entidade>/resnet18-finetune/resnet18-cub200:best` |
| 2 | `resnet18-finetune` | `resnet18-cars196` | `best` | `<entidade>/resnet18-finetune/resnet18-cars196:best` |
| 3 | `resnet18-to-convnext-distill` | `convnextmicro-distill-cub200` | `best` | `<entidade>/resnet18-to-convnext-distill/convnextmicro-distill-cub200:best` |
| 4 | `resnet18-to-convnext-distill` | `convnextmicro-distill-cars196` | `best` | `<entidade>/resnet18-to-convnext-distill/convnextmicro-distill-cars196:best` |

---

## Hiperparâmetros da destilação (padrões já definidos)

Os scripts já vêm com valores baseados nos papers de origem — não precisa
passar nada além do acima, mas todos são ajustáveis por flag:

| Sinal | Flag | Padrão | Origem |
|-------|------|--------|--------|
| Hinton KD (temperatura) | `--kd_T` | 4.0 | Hinton et al. |
| Cross-entropy / KD | `--ce_ratio` / `--kd_ratio` | 1.0 / 0.9 | RepDistiller (γ, α) |
| RKD distance | `--dist_ratio` | 25 | Park et al. 2019 |
| RKD angle | `--angle_ratio` | 50 | Park et al. 2019 |
| Mapa de atenção | `--at_ratio` | 1000 | Zagoruyko & Komodakis |
| Otimização | — | AdamW, lr 1e-3, wd 0.05, 300 épocas, warmup 20 | aluno treinado do zero |

O mapa de atenção é aplicado **apenas na 2ª camada não-pointwise do aluno**
(stage-2, 28×28), pareada com a `layer2` da ResNet-18 (também 28×28).

---

## Observações

- Mantenha `--wandb_mode online` (padrão) nos quatro passos para que os modelos
  sejam de fato enviados ao W&B. Use `offline`/`disabled` apenas para testes
  locais (nesse caso nada é registrado).
- Para retomar um fine-tune interrompido: `--resume finetune/<dataset>/last.pth`.
- O aluno ConvNextMicro tem ~0,67M de parâmetros (vs. ~11M da ResNet-18) e é
  treinado **do zero** — por isso o agendamento longo (300 épocas).
