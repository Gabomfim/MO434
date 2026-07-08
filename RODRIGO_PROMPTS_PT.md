# Playbook de prompts (PT) — rodar os experimentos Graph-RKD na GPU

Guia para o **Rodrigo** rodar a campanha na máquina com GPU usando o **Claude Code
dele**. Cole **um prompt por vez** na sessão do Claude, siga as **bifurcações
(➤ SE … ENTÃO …)** conforme o que aparecer. O Rodrigo **só roda** os experimentos e
**manda relatórios pro Gabriel**; a **análise/gráficos/paper o Gabriel faz depois**,
na máquina dele.

- Repositório já tem tudo: script único `RKD/sm/run_experiments.sh`, dados baixam
  sozinhos do S3 público (sem credencial AWS) e ficam em cache, tudo é **resumível**.
- Logs de treino vão pro W&B **`gabomfim-unicamp/graph-rkd`**.
- Onde reportar pro Gabriel: (combinar canal — WhatsApp/e-mail/Slack). Mande o
  **RELATÓRIO** que o Claude gerar nos Prompts 3 e 6.

---

## Prompt 1 — Setup e checagem de recursos

```
Você é um agente rodando no repositório Graph-RKD (pasta MO434) numa máquina com
GPU. Objetivo desta etapa: preparar o ambiente e checar recursos, SEM treinar ainda.

Faça:
1. `git pull` (ou clone) e `uv sync`.
2. Confirme a GPU: rode `nvidia-smi` e reporte modelo, VRAM total/livre, driver e
   versão de CUDA. Rode `uv run python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
3. Confirme que o WANDB_API_KEY está no ambiente (`echo ${WANDB_API_KEY:+ok}`); se
   não, peça pro usuário exportar (ou `uv run wandb login`).
4. Cheque espaço em disco (`df -h .`) — os datasets extraídos ocupam ~6 GB + cache.
5. NÃO treine. Ao final, gere um bloco "RELATÓRIO SETUP" com: GPU/VRAM, torch+CUDA
   ok?, WANDB ok?, disco livre, e qualquer risco (driver antigo p/ Blackwell, pouco
   disco). Leia RUNBOOK.md se tiver dúvida.
```

➤ **SE** `torch.cuda.is_available()` for **False** ou o driver não suportar a GPU
(ex.: RTX 5070/Blackwell precisa de driver recente + torch cu128) **ENTÃO** envie o
**Prompt 1b**. **SENÃO** siga pro **Prompt 2**.

### Prompt 1b — Corrigir ambiente de GPU
```
A GPU não está visível pro PyTorch. Diagnostique e corrija: verifique driver NVIDIA
(nvidia-smi funciona?), e garanta o torch com CUDA certo. Este projeto fixa torch
2.7.0/torchvision 0.22.0 do índice cu128 (pyproject.toml) — compatível com Blackwell
(RTX 5070). Se preciso, atualize o driver NVIDIA e rode `uv sync` de novo. NÃO troque
as versões do torch no pyproject sem avisar. Ao final, reporte se a GPU passou a
aparecer. Se não resolver, gere um "RELATÓRIO DE BLOQUEIO" curto pro Gabriel com o
erro exato e pare.
```

---

## Prompt 2 — Iniciar a campanha (enxuta) + monitoramento em background

```
Agora rode a campanha ENXUTA inteira (teachers + fases 2,3,4,5), resumível, logando
no W&B gabomfim-unicamp/graph-rkd. Use o script pronto:

    ./RKD/sm/run_experiments.sh

Rode-o EM BACKGROUND (não bloqueie a sessão) e capture o log num arquivo, ex.:
    nohup ./RKD/sm/run_experiments.sh > run.log 2>&1 &

Regras:
- Usa a GPU ao máximo automaticamente (empaca vários jobs por VRAM). Se a VRAM
  estourar depois, dá pra reduzir com PER_JOB_GB (ver fork abaixo).
- É resumível: se cair/reiniciar, é só rodar o mesmo comando de novo.
- Treina os teachers primeiro, depois os alunos. Tempo esperado ~2–3 dias.

Depois de iniciar, confirme que começou (primeiras linhas do run.log, um job
aparecendo, GPU ocupando em `nvidia-smi`) e gere um "RELATÓRIO INÍCIO" pro Gabriel:
hora de início, comando usado, nº de jobs no plano, link do W&B, GPU em uso.
```

➤ **SE** aparecer **erro de "no GPU CUDA visible"** → volte ao **Prompt 1b**.
➤ **SE** aparecer **OOM / CUDA out of memory** nos primeiros minutos → **Prompt 2b**.
➤ **SENÃO** (rodando ok) → use o **Prompt 3** periodicamente.

### Prompt 2b — Reduzir uso de VRAM
```
Deu out-of-memory. Pare o run (mate o processo do run_experiments.sh), e reinicie com
menos jobs simultâneos por GPU:
    PER_JOB_GB=6 nohup ./RKD/sm/run_experiments.sh > run.log 2>&1 &
(ou PER_JOB_GB=8 se persistir; ou MAX_PARALLEL=1 para um job por vez). É resumível —
os jobs já concluídos são pulados. Confirme que voltou a rodar sem OOM e reporte o
novo PER_JOB_GB usado.
```

---

## Prompt 3 — Monitorar recursos/erros e gerar RELATÓRIO (repetir a cada ~6–12 h)

```
Faça um check de saúde do treino e gere um RELATÓRIO pro Gabriel. Não interrompa o
treino. Colete:
1. GPU: `nvidia-smi` — utilização %, VRAM usada/total, temperatura, processos.
2. Progresso: quantos jobs terminaram vs total. Use o W&B:
   `uv run --no-sync python -c "import wandb;api=wandb.Api();rs=list(api.runs('gabomfim-unicamp/graph-rkd'));from collections import Counter;print('total',len(rs));print(Counter(r.state for r in rs))"`
   e as últimas linhas de run.log (qual fase/epoch está rodando).
3. Erros: procure no run.log por 'Error', 'Traceback', 'CUDA', 'NaN', 'FAIL', 'rc='.
4. Disco: `df -h .`.
5. Sanidade dos números (só um olhar, NÃO é análise): algum run com mAP@R de teste
   praticamente zero/estagnado? (sinal de colapso).

Gere o bloco "RELATÓRIO <data/hora>" com: fases concluídas/total, jobs ok/rodando/
falhos, GPU %/VRAM/temp, disco, erros encontrados (ou "nenhum"), ETA aproximado, link
W&B. Deixe pronto para o Gabriel copiar.
```

➤ **SE** houver **jobs FALHOS ou Traceback repetido** → **Prompt 4**.
➤ **SE** algum run mostrar **loss NaN/inf ou mAP@R colado em ~0** (colapso) →
**Prompt 5** (avisar o Gabriel; não tente "consertar" o método sozinho).
➤ **SE** **disco < ~5 GB livre** → limpe caches antigos/checkpoints não usados e
reporte; se persistir, avise o Gabriel.
➤ **SENÃO** (tudo saudável) → continue treinando e repita o **Prompt 3** mais tarde.
Quando o W&B mostrar **todos os jobs `finished`** → **Prompt 6**.

---

## Prompt 4 — Diagnosticar e resumir jobs que falharam (não é análise)

```
Alguns jobs falharam. Diagnostique SEM refazer o método:
1. Ache a causa no run.log e no diretório do job (experiments_local/<nome>/run.log):
   OOM? erro de dados? erro transitório de rede? bug de código?
2. Se for TRANSITÓRIO (rede/pré-empção) ou OOM → é resumível: reduza PER_JOB_GB se
   OOM e re-rode `./RKD/sm/run_experiments.sh` (pula os concluídos, refaz os que
   faltam).
3. Se for BUG DE CÓDIGO claro e pequeno (ex.: caminho, flag), você PODE corrigir e,
   se corrigir, **atualize a documentação/commit** correspondente com mensagem clara
   (e cite no relatório). Não faça mudanças grandes de método.
4. Se não souber a causa ou for algo do método/experimento → NÃO mexa; gere um
   "RELATÓRIO DE FALHA" com o traceback exato, o job afetado e o que tentou, e mande
   pro Gabriel decidir.
Reporte o que fez e o estado após a correção.
```

---

## Prompt 5 — Suspeita de colapso do treino (decisão do Gabriel)

```
Vi possível colapso (loss NaN/inf ou mAP@R de teste ~0 em vários runs). NÃO altere o
método nem os hiperparâmetros por conta própria. Faça:
1. Confirme: liste os runs afetados (nome, fase, config, mAP@R val/test) do W&B.
2. Verifique se é geral (todos) ou de uma config específica (ex.: uma normalização
   ou λg). Colapso só em λg alto/uma norm pode ser esperado.
3. Deixe os demais jobs saudáveis continuarem.
4. Gere um "RELATÓRIO DE COLAPSO" pro Gabriel: quais configs colapsaram, quais estão
   bem, e sua hipótese — e PARE de tomar ação corretiva de método até o Gabriel
   responder.
```

---

## Prompt 6 — Fim: relatório final e entrega (NÃO rodar análise)

```
O W&B mostra todos os jobs `finished`. Faça o encerramento (SEM rodar análise —
o Gabriel fará isso na máquina dele):
1. Confirme no W&B a contagem por fase (teachers, phase2, phase3, phase4, phase5) e
   que não há runs `crashed`/`failed` pendentes; se houver, re-rode para completá-los.
2. Confirme que a GPU está livre (`nvidia-smi`) e nada está mais treinando.
3. Se você fez qualquer correção de código/config durante o processo, garanta que
   está commitada com mensagem clara e liste esses commits.
4. Gere o "RELATÓRIO FINAL" pro Gabriel: total de runs por fase, tempo total, custo/
   recursos, quaisquer falhas e como foram resolvidas, commits feitos, e o link do
   W&B. Deixe explícito: "experimentos concluídos — pronto para análise local do
   Gabriel".
NÃO gere tabelas/figuras nem rode os notebooks de análise — isso é responsabilidade
do Gabriel (ver ANALYSIS_AND_REPORTING_EN.md).
```

---

## Formato do RELATÓRIO (para o Gabriel)

Curto, copiável:
```
RELATÓRIO Graph-RKD — <data/hora> — <nome da máquina/GPU>
Fase atual: <ex. phase3>  | Jobs: <ok>/<total> finished, <n> running, <n> failed
GPU: <modelo> | util <..%> | VRAM <usada>/<total> GB | temp <..°C>
Disco livre: <.. GB> | ETA: <~.. h/dias>
Erros: <nenhum | descrição curta>
Mudanças de doc/código: <nenhuma | commits X, Y>
W&B: https://wandb.ai/gabomfim-unicamp/graph-rkd
Observações: <ex. colapso em λg=1 na norm none (esperado); resto ok>
```

## Regras gerais (todos os prompts)
- **Só rodar experimentos.** Análise, gráficos, tabelas e paper = Gabriel, depois.
- **Resumível sempre.** Em dúvida após uma queda, re-rode `./RKD/sm/run_experiments.sh`.
- **Não presuma resultado** nem "conserte" o método para melhorar números.
- **Documentação:** se corrigir algo, atualize o doc/commit e cite no relatório.
- **Em bloqueio ou dúvida de método:** gere um relatório claro e passe pro Gabriel.
