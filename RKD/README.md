# Relational Knowledge Distillation

Implementação oficial de [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068?context=cs.LG), CVPR 2019\
Este repositório contém o código-fonte dos experimentos de *metric learning*.


## Início Rápido

```bash
python run.py --help    
python run_distill.py --help

# Use os scripts no estilo de configuração (recomendado)
bash examples/run_config.sh train
bash examples/run_config.sh eval
bash examples/run_distill_config.sh

# Logging no W&B (opcional)
# Use --wandb_mode disabled para rodar sem logging.
python run.py --mode train \
              --dataset cub200 \
              --base resnet50 \
              --save_dir teacher \
              --wandb_project rkd-metric-learning \
              --wandb_run_name teacher-resnet50 \
              --wandb_mode online

python run_distill.py --dataset cub200 \
                      --base resnet18 \
                      --teacher_base resnet50 \
                      --teacher_load teacher/best.pth \
                      --quad_ratio 1 \
                      --save_dir student \
                      --wandb_project rkd-metric-learning \
                      --wandb_run_name distill-resnet18 \
                      --wandb_mode online

# Flags de W&B disponíveis em ambos os scripts:
#   --wandb_project, --wandb_entity, --wandb_run_name, --wandb_mode

# Treina uma rede de embedding professora (resnet50, d=512)
# usando triplet loss (margin=0.2) com distance weighted sampling.
python run.py --mode train \ 
               --dataset cub200 \
               --base resnet50 \
               --sample distance \ 
               --margin 0.2 \ 
               --embedding_size 512 \
               --save_dir teacher

# Avalia a rede de embedding professora
python run.py --mode eval \ 
               --dataset cub200 \
               --base resnet50 \
               --embedding_size 512 \
               --load teacher/best.pth 

# Destila a professora para a rede de embedding aluna
python run_distill.py --dataset cub200 \
                      --base resnet18 \
                      --embedding_size 64 \
                      --l2normalize false \
                      --teacher_base resnet50 \
                      --teacher_embedding_size 512 \
                      --teacher_load teacher/best.pth \
                      --dist_ratio 1  \
                      --angle_ratio 2 \
                      --save_dir student
                      
# Avalia o modelo aluno treinado
python run.py --mode eval \ 
               --dataset cub200 \
               --base resnet18 \
               --l2normalize false \
               --embedding_size 64 \
               --load student/best.pth 
            
```

## Extensões MO434 — ConvNextMicro, Fine-tuning e Destilação de Classificação

Além do código original de *metric learning* acima, este repositório adiciona um
classificador compacto **ConvNextMicro** (<1M de parâmetros) e um pipeline de
destilação de **classificação** que transforma uma **ResNet-18** ajustada
(*fine-tuned*) em uma ConvNextMicro.

Novos scripts / módulos:

* `model/convnext_block.py`: `ConvNextBlock` + `Downsample`.
* `model/convnext_micro.py`: classificador `ConvNextMicro` (`dims`/`depths` configuráveis, ~0,67M de parâmetros).
* `train_convnext.py`: treina a ConvNextMicro do zero (receita ConvNeXt: AdamW, cosine+warmup, mixup/cutmix, RandAugment, EMA).
* `finetune_resnet18.py`: faz fine-tune de uma ResNet-18 da ImageNet em `cars196`/`cub200` (split oficial de classificação, top-1/top-5).
* `distill_resnet18_convnext.py`: destila ResNet-18 → ConvNextMicro (Hinton KD + RKD distance/angle + mapa de atenção).
* `distill_convnext.py`: destilação de embedding/métrica para a ConvNextMicro (split disjunto de *metric learning*).
* `wandb_artifacts.py`: busca um professor no W&B (`resolve_teacher_checkpoint`) e registra modelos treinados como artefatos do W&B (`log_model_artifact`).

### Destilação de Classificação (ResNet-18 → ConvNextMicro)

Combina quatro técnicas: **Hinton KD** (logits), **RKD distance**, **RKD angle**
(embeddings agrupados) e um **mapa de atenção** na 2ª camada não-pointwise do
aluno (stage-2, 28×28) pareada com a `layer2` da ResNet-18. Tanto o professor
quanto o aluno são classificadores no split *oficial*, então o KD de logits é
válido.

```bash
export WANDB_ENTITY="<sua-entidade>"

# 1) Fine-tune do professor e registro no W&B (--save_dir + online obrigatórios)
python finetune_resnet18.py --dataset cub200 --data ../data --amp \
  --save_dir finetune/cub200 --wandb_project resnet18-finetune \
  --wandb_entity "$WANDB_ENTITY" --wandb_run_name resnet18-cub200
#   -> artefato: $WANDB_ENTITY/resnet18-finetune/resnet18-cub200:best

# 2) Destila o professor registrado para a ConvNextMicro (professor puxado do W&B)
python distill_resnet18_convnext.py --dataset cub200 --data ../data --amp \
  --teacher_artifact "$WANDB_ENTITY"/resnet18-finetune/resnet18-cub200:best \
  --save_dir distill_runs/cub200 --wandb_project resnet18-to-convnext-distill \
  --wandb_entity "$WANDB_ENTITY" --wandb_run_name distill-cub200
#   -> artefato: $WANDB_ENTITY/resnet18-to-convnext-distill/convnextmicro-distill-cub200:best
```

Troque `cub200` → `cars196` para o pipeline do Cars. Launchers por dataset:
`examples/distill_convnext_cub.sh` e `examples/distill_convnext_cars.sh`.

Padrões ajustados (baseados na literatura): KD `T=4`, `ce=1.0`/`kd=0.9`; RKD
`dist=25`, `angle=50`; atenção `at=1000`; AdamW `lr=1e-3`, `wd=0.05`, 300
épocas, 20 épocas de warmup. A referência do professor precisa ser
**totalmente qualificada** (`entidade/projeto/nome:alias`), porque o fine-tune e
a destilação usam projetos diferentes no W&B.

> **Guia passo a passo completo (PT-BR)** — incluindo login/configuração do W&B
> e como rodar o pipeline inteiro com um **agente Claude** (`/distill-pipeline`)
> — está em [`README_pipeline_distilacao.md`](README_pipeline_distilacao.md).

## Arquivos do Repositório

* `run.py`: Script principal do professor para *metric learning*. Suporta treino e avaliação com `--mode train|eval`, salva checkpoints (`best.pth`, `last.pth`) e registra métricas/artefatos no W&B.
* `run_distill.py`: Script de destilação do aluno. Treina um aluno a partir de um checkpoint do professor usando perdas RKD (distance/angle e perdas auxiliares opcionais), salva checkpoints e registra métricas/artefatos no W&B.
* `examples/run_config.sh`: Wrapper no estilo de configuração para `run.py`, com variáveis centralizadas de dataset/modelo/treino/W&B.
* `examples/run_distill_config.sh`: Wrapper no estilo de configuração para `run_distill.py`, com variáveis centralizadas de professor/aluno/destilação/W&B.
* `examples/`: Scripts de exemplo com presets de hiperparâmetros reprodutíveis.
* `data/` (criado em tempo de execução): Diretório de download/cache dos datasets usado por `--data`.
* `teacher/` e `student/` (criados quando `--save_dir` é usado): Diretórios de saída com checkpoints e `result.txt`.

## O Que Rodar Primeiro

1. Treine ou avalie o modelo professor com `run.py` (ou `examples/run_config.sh`).
2. Destile o aluno com `run_distill.py` (ou `examples/run_distill_config.sh`) usando o checkpoint do professor (por exemplo `teacher/best.pth`).
3. Avalie o aluno com `run.py --mode eval` usando `student/best.pth`.

Ordem rápida de comandos:

```bash
bash examples/run_config.sh train
bash examples/run_config.sh eval
bash examples/run_distill_config.sh
python run.py --mode eval --dataset cub200 --base resnet18 --embedding_size 64 --l2normalize false --load student/best.pth
```

### Logging do W&B nos Scripts

Tanto `run.py` quanto `run_distill.py` suportam:

* `--wandb_project`: Nome do projeto.
* `--wandb_entity`: Namespace de time/usuário (opcional).
* `--wandb_run_name`: Nome explícito da run (opcional).
* `--wandb_mode`: `online`, `offline` ou `disabled`.

Peso de perda extra específico de destilação:

* `--quad_ratio`: peso da perda de correspondência de soma de distâncias relacionais em conjuntos de 4 amostras.

Quando habilitado, os scripts registram:

* Hiperparâmetros/valores de configuração.
* Métricas por época (loss, recall, learning rate e componentes da perda de destilação).
* Artefato de metadados do dataset.
* Artefatos de checkpoint do modelo (`best`, `last`) quando `--save_dir` é definido.

## Template de Relatório W&B (Comparação de Destilação)

Use este template para comparar várias runs de destilação em um único relatório.

### 1) Organize as Runs para Comparação

Use um projeto e grupo comuns para runs relacionadas.

```bash
python run_distill.py ... \
  --wandb_project rkd-metric-learning \
  --wandb_group distillation-experiments \
  --wandb_run_name distill-r18-a2-d1-q0
```

Padrão de nomenclatura recomendado:

* `distill-<student>-a<angle_ratio>-d<dist_ratio>-q<quad_ratio>-dark<dark_ratio>-seed<seed>`

### 2) Crie Painéis em um Relatório W&B

Adicione um filtro de conjunto de runs:

* `group = distillation-experiments`

Adicione estes gráficos de linha (eixo x = `epoch`):

* `eval/test_recall1`, `eval/test_recall2`, `eval/test_recall4`, `eval/test_recall8`
* `eval/train_recall1`, `eval/train_recall2`, `eval/train_recall4`, `eval/train_recall8`
* `train/loss`
* `train/dist_loss`, `train/angle_loss`, `train/quad_loss`, `train/dark_loss`, `train/triplet_loss`, `train/at_loss`
* `lr`

Adicione painéis de resumo:

* `best_test_recall_primary`
* `final_test_recall_primary`
* `best_test_recall1`, `best_test_recall2`, `best_test_recall4`, `best_test_recall8`

Adicione tabelas de comparação:

* Tabela de runs com colunas: `name`, `group`, `config.dist_ratio`, `config.angle_ratio`, `config.quad_ratio`, `config.dark_ratio`, `summary.best_test_recall_primary`, `summary.final_test_recall_primary`.
* Tabela de artefatos para artefatos de `metrics` (`distill-metrics-<run_id>`) para baixar e comparar os históricos em CSV.

### 3) Use os Artefatos Registrados

Cada run de destilação registra:

* `metrics/epoch_table` (tabela na run com métricas por época).
* `distill_epoch_metrics.csv` como artefato `metrics` do W&B.
* Artefatos de modelo `best` e `last`.
* Gráficos de matriz de confusão (`eval/test/confusion_matrix`, `best/test/confusion_matrix`) quando os limites de classe/amostra permitem.

### 4) Fluxo de Comparação Sugerido

1. Rode uma baseline (`quad_ratio=0`) e ao menos uma variante (`quad_ratio>0`).
2. Mantenha todos os outros hiperparâmetros fixos, exceto o que está sendo testado.
3. Compare `best_test_recall_primary` primeiro, depois inspecione as curvas dos componentes de perda.
4. Use matrizes de confusão para diagnosticar mudanças de comportamento por classe.
5. Exporte os artefatos `distill_epoch_metrics.csv` para análise externa, se necessário.

## Template de Relatório W&B (Comparação de Professores)

Use este template para comparar runs apenas de professor (escolhas de backbone, loss, sampling e margin).

### 1) Organize as Runs

Use um grupo dedicado para experimentos de professor:

```bash
python run.py --mode train ... \
  --wandb_project rkd-metric-learning \
  --wandb_group teacher-runs \
  --wandb_run_name teacher-r50-distance-l2triplet
```

Padrão de nomenclatura recomendado:

* `teacher-<backbone>-<sample>-<loss>-m<margin>-seed<seed>`

### 2) Crie Painéis no W&B

Adicione um filtro de conjunto de runs:

* `group = teacher-runs`

Adicione estes gráficos de linha (eixo x = `epoch`):

* `train/loss`
* `eval/train_recall@1`, `eval/train_recall@2`, `eval/train_recall@4`, `eval/train_recall@8`
* `eval/test_recall@1`, `eval/test_recall@2`, `eval/test_recall@4`, `eval/test_recall@8`
* `best_recall@1`
* `lr`

Adicione painéis de resumo:

* `best_recall@1`
* `final_recall@1`
* `final_recall@2`, `final_recall@4`, `final_recall@8` (se registrados via `--recall`)

Adicione tabelas de comparação:

* Tabela de runs com colunas: `name`, `config.base`, `config.sample`, `config.loss`, `config.margin`, `summary.best_recall@1`, `summary.final_recall@1`.
* Tabela de artefatos para artefatos de métrica do professor (`teacher-history-<base>-<seed>`).

### 3) Use os Artefatos do Professor

Cada run de professor registra:

* `artifacts/epoch_history_table`.
* `teacher_epoch_history.csv` como artefato `metrics` do W&B.
* Artefatos de modelo `best` e `last`.
* Matriz de confusão final/de avaliação quando a contagem de classes está dentro dos limites.

### 4) Fluxo de Professor Sugerido

1. Escolha um backbone baseline e uma configuração de loss/sampler.
2. Mude um fator por vez (por exemplo, sampler ou margin).
3. Compare `best_recall@1` primeiro, depois inspecione as curvas de recall e a loss de treino.
4. Use diferenças de matriz de confusão para inspecionar o comportamento de retrieval por classe.
5. Mantenha o melhor checkpoint de professor como o professor fixo para os sweeps de destilação.


##  Dependências

* Python 3.6
* Pytorch 1.0
* tqdm (pip install tqdm)
* h5py (pip install h5py)
* scipy (pip install scipy)
* wandb (pip install wandb)

### Nota
* Os hiperparâmetros usados nos experimentos do paper estão especificados nos scripts em ```examples/```.
* A rede professora pesada (ResNet50 com 512 dimensões) requer mais de 12GB de memória de GPU se o batch size for 128.  
  Portanto, talvez seja necessário reduzir o batch size. (Os experimentos do paper foram conduzidos em uma P40 com 24GB de memória de GPU.
)

## Citação
Caso utilize este código-fonte em sua pesquisa, por favor cite o paper.

```
@inproceedings{park2019relational,
  title={Relational Knowledge Distillation},
  author={Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3967--3976},
  year={2019}
}
```
