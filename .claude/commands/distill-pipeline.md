---
description: Executa o pipeline de fine-tune + destilação (professor -> ConvNextMicro) registrando tudo no W&B
argument-hint: "[entidade-wandb] [all|finetune|distill|cub|cars|1|2|3|4] [resnet18|convnext_tiny]"
---

Você é um agente que executa o pipeline documentado neste README (fonte da verdade — siga-o exatamente):

@RKD/README_pipeline_distilacao.md

## Argumentos recebidos

- `$1` = entidade do W&B (usuário/time). Se vazio: tente `echo "$WANDB_ENTITY"`; se ainda vazio, rode `wandb status` para descobrir a entidade default; se mesmo assim não souber, **pergunte ao usuário** antes de continuar.
- `$2` = quais passos rodar. Default `all` quando vazio. Mapeamento:
  - `all` → passos 1, 2, 3, 4
  - `finetune` → passos 1 e 2
  - `distill` → passos 3 e 4 (exige os professores já registrados no W&B)
  - `cub` → passos 1 e 3 | `cars` → passos 2 e 4
  - `1` | `2` | `3` | `4` → passo individual
- `$3` = arquitetura do professor: `resnet18` (default) ou `convnext_tiny`. Passe a mesma em `--arch` (fine-tune) e `--teacher_arch` (destilação).

## Antes de começar (verificações)

1. Trabalhe sempre a partir de `RKD/` (todos os comandos do README rodam de lá).
2. Confirme login no W&B: `wandb status`. Se não estiver logado, oriente `wandb login` e pare.
3. Verifique GPU: rode `python -c "import torch; print(torch.cuda.is_available())"`. Se `False`, avise que o treino será MUITO lento na CPU e confirme com o usuário antes de seguir; quando rodar com GPU use `--amp`.
4. Os datasets baixam sozinhos na primeira execução (`--data ../data`).

## Como executar cada passo

Use **exatamente** os comandos do README (`finetune_classifier.py` e `distill_to_convnextmicro.py`), substituindo a entidade por `$1` e a arquitetura por `$3` (default `resnet18`). Pontos não-negociáveis:

- **Sempre** passe `--save_dir` e `--wandb_mode online` — sem os dois o modelo NÃO é registrado no W&B.
- Passe `--arch $3` no fine-tune e `--teacher_arch $3` na destilação.
- Nos passos de destilação, o professor vem do W&B com **referência completa** (o fine-tune fica no projeto `classifier-finetune` e a destilação em `convnextmicro-distill`):
  - CUB: `--teacher_artifact $1/classifier-finetune/$3-cub200:best`
  - Cars: `--teacher_artifact $1/classifier-finetune/$3-cars196:best`
- Treinos são longos. Rode cada passo em background e acompanhe a saída até concluir antes de iniciar o próximo passo dependente (um passo de destilação só pode começar depois que o fine-tune correspondente registrou o artefato `:best`).

## Ordem e dependências

- Passo 3 depende do Passo 1 (professor CUB). Passo 4 depende do Passo 2 (professor Cars).
- Se o usuário pedir só `distill`, verifique antes que os artefatos `$3-cub200:best` / `$3-cars196:best` existem no W&B (ex.: tentar resolvê-los); se não existirem, avise que é preciso rodar o fine-tune primeiro.

## Ao final

Confirme, para cada passo executado, que o artefato foi registrado e reporte uma tabela com:
- passo, projeto W&B, nome do artefato, melhor métrica (top-1 de validação + test) e a referência completa `entidade/projeto/nome:alias`.

Use os padrões de hiperparâmetro já definidos nos scripts (não sobrescreva, a menos que o usuário peça). Reporte falhas com a saída real do comando — não declare sucesso sem o artefato confirmado no W&B.
