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

Há **dois caminhos** para rodar este pipeline:

- **A) Com o agente Claude** (automatizado, recomendado) — veja a seção logo abaixo.
- **B) Manualmente** — rode os comandos dos Passos 1 a 4 você mesmo.

---

## Executar com o agente Claude (slash command)

Existe um comando de projeto que faz um agente Claude executar estes passos por
você: [`/distill-pipeline`](../.claude/commands/distill-pipeline.md). Ele usa
**este README como fonte da verdade**, então o que estiver documentado aqui é o
que o agente segue.

### Pré-requisitos para usar o agente

1. Abrir **este repositório** no Claude Code (CLI, app ou extensão do VS Code) —
   o comando só aparece quando o projeto está aberto, pois vive em
   `.claude/commands/`.
2. Estar logado no W&B na máquina onde o agente roda (`wandb login`) e,
   idealmente, ter GPU (o agente verifica e avisa se for CPU).

### Como acionar

Digite, na conversa do Claude Code:

```text
/distill-pipeline <entidade-wandb> [passo]
```

- `<entidade-wandb>`: seu usuário/time do W&B. Se omitir, o agente tenta
  `$WANDB_ENTITY`, depois `wandb status` e, em último caso, pergunta.
- `[passo]` (opcional, default `all`):

| Você digita | O agente executa |
|-------------|------------------|
| `/distill-pipeline meu-usuario` | Pipeline inteiro (Passos 1→4) |
| `/distill-pipeline meu-usuario finetune` | Só os fine-tunes (1 e 2) |
| `/distill-pipeline meu-usuario distill` | Só as destilações (3 e 4) |
| `/distill-pipeline meu-usuario cub` | Caminho CUB (1 e 3) |
| `/distill-pipeline meu-usuario cars` | Caminho Cars (2 e 4) |
| `/distill-pipeline meu-usuario 3` | Apenas o Passo 3 |

### O que o agente faz

- Verifica login no W&B e disponibilidade de GPU antes de começar.
- Roda os comandos dos Passos 1 a 4 **na ordem certa**, respeitando dependências
  (a destilação CUB só começa depois que o fine-tune CUB registrou `:best`).
- Garante `--save_dir` e `--wandb_mode online` (sem os dois o modelo não é
  registrado) e usa a **referência completa** do professor entre projetos.
- Ao final, reporta uma tabela com os artefatos registrados e as métricas.

### Alternativa em linguagem natural

Sem o slash command, você também pode pedir diretamente, por exemplo:

> "Siga o `RKD/README_pipeline_distilacao.md` e rode o pipeline completo na
> entidade `meu-usuario`."

O agente segue o mesmo README. O `/distill-pipeline` é só o atalho
padronizado (já embute as travas de `--save_dir`, modo online e a referência
cruzada de projeto).

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

## Configuração do W&B (login e projeto)

### 1. Pegar a API key e logar

1. Crie/acesse sua conta em <https://wandb.ai> e copie sua chave em
   <https://wandb.ai/authorize>.
2. Faça login (uma vez por máquina). A chave fica salva em `~/.netrc`:

   ```bash
   wandb login
   # cole a API key quando pedir
   ```

   Alternativas equivalentes:

   ```bash
   wandb login <SUA_API_KEY>          # passando a chave direto
   export WANDB_API_KEY="<SUA_API_KEY>"   # via variável de ambiente (útil em servidores/CI)
   ```

3. Verifique se está logado:

   ```bash
   wandb status     # mostra a conta e a entidade default
   ```

### 2. Descobrir sua entidade (`entity`)

A **entidade** é seu usuário do W&B ou o nome de um time. Você a vê no canto
superior esquerdo em <https://wandb.ai> ou na URL dos seus projetos
(`https://wandb.ai/<entidade>/<projeto>`). Use-a em `--wandb_entity` (ou deixe
em branco para usar a entidade default da sua conta).

### 3. Projetos usados neste pipeline

Os scripts já definem projetos padrão — **não precisa criar nada à mão**, o W&B
cria o projeto na primeira run:

| Etapa | Projeto (`--wandb_project`) |
|-------|------------------------------|
| Fine-tune (Passos 1 e 2) | `resnet18-finetune` |
| Destilação (Passos 3 e 4) | `resnet18-to-convnext-distill` |

### 4. Formas de configurar entidade/projeto

- **Por flag** (recomendado, explícito): `--wandb_entity` e `--wandb_project`
  em cada comando (é o que os comandos abaixo fazem).
- **Por variável de ambiente** (aplica a todas as runs do shell):

  ```bash
  export WANDB_ENTITY="<sua-entidade>"
  export WANDB_PROJECT="resnet18-finetune"   # opcional; a flag tem prioridade
  ```

  > As flags dos scripts têm prioridade sobre as variáveis de ambiente.

### 5. Modos de execução (`--wandb_mode`)

| Modo | Quando usar |
|------|-------------|
| `online` (padrão) | Envia métricas **e registra os modelos** no W&B. Use nos 4 passos. |
| `offline` | Salva localmente; sincronize depois com `wandb sync <dir>`. |
| `disabled` | Desliga o W&B (testes locais). **Nada é registrado** — nem o modelo. |

> Para que os modelos sejam registrados como artefatos, você precisa de
> `--wandb_mode online` **e** `--save_dir <dir>` (ambos já estão nos comandos).

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
