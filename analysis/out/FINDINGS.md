# Resultados — melhor N para Graph-RKD (regression)

**Métrica primária:** test mAP@R (Recall@1 e R-Precision de apoio), média ± sem sobre 3 seeds. `N` = nº de nós do grafo relacional. Baseline `off` = destilação sem perda de grafo.

> **Escopo:** o Gabriel definiu 4 análises (2 métodos × 2 datasets). Os dados têm **2 teachers** (`resnet18`, `convnext_tiny`); geramos os 4 cenários para cada um — **escolham qual teacher manter** no trabalho.

> **Caveats (importantes):** (1) o sweep de N usa runs de *busca* de **~30 épocas** (subtreinados: mAP@R ~0.006–0.02). Ele compara N **entre si** no mesmo budget — não é o número final de qualidade. (2) O baseline `off` e os runs `final` rodam **120 épocas**, então **NÃO são comparáveis** ao sweep @30ép (o off aparece maior só por isso — não é o grafo 'perdendo'). (3) O único N com budget cheio (120 ép) é **N=2**, porque a seleção automática (val Recall@1) elegeu N=2 em todas as células. Ainda assim, as curvas de treino já permitem concluir (ver **Veredito** no fim): os N>2 ficam planos, não apenas lentos.

## Resumo dos cenários

| teacher | dataset | method | Nstar | mAPR_sweep30ep | delta_vs_worstN_pct | off_ref_120ep | spearman_N | meio_termo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| convnext_tiny | cars196 | profile | 2 | 0.0061 | 266.4 | 0.0206 | -0.1 | 2 |
| convnext_tiny | cars196 | mds | 2 | 0.0065 | 343.9 | 0.0206 | -0.9 | 2 |
| convnext_tiny | cub200 | profile | 2 | 0.0205 | 854.4 | 0.0308 | 0.0 | 2 |
| convnext_tiny | cub200 | mds | 2 | 0.0208 | 915.1 | 0.0308 | -0.3 | 2 |
| resnet18 | cars196 | profile | 2 | 0.0061 | 268.2 | 0.0185 | -0.1 | 2 |
| resnet18 | cars196 | mds | 2 | 0.0065 | 316.7 | 0.0185 | -0.9 | 2 |
| resnet18 | cub200 | profile | 2 | 0.0204 | 880.8 | 0.034 | 0.0 | 2 |
| resnet18 | cub200 | mds | 2 | 0.0207 | 911.7 | 0.034 | -0.7 | 2 |


# Teacher = convnext_tiny

## cars196 · profile

- **N\* = 2** (test mAP@R = 0.0061, sweep @30ép).
- **Quão melhor (dentro do sweep):** +266.4% vs o pior N (0.0017).
- _Ref.: baseline off @120ép = 0.0206 (não comparável ao sweep @30ép — só contexto)._
- **Padrão com N:** Spearman(N, mAP@R) = -0.10 → pico interno / não-monotônico.
- **Convergência:** só **N=2** aprende de fato (maior teto 0.0061; atinge 95% dele na época 25). Os demais N ficam **estagnados** (teto ≤ 0.0034) em 30 ép — não é só 'convergir mais devagar', eles quase não saem do lugar.
- **Meio-termo:** não há trade-off — N=2 domina em velocidade **e** qualidade, então é a escolha única (não se ganha nada indo pra N maior neste budget).

## cars196 · mds

- **N\* = 2** (test mAP@R = 0.0065, sweep @30ép).
- **Quão melhor (dentro do sweep):** +343.9% vs o pior N (0.0015).
- _Ref.: baseline off @120ép = 0.0206 (não comparável ao sweep @30ép — só contexto)._
- **Padrão com N:** Spearman(N, mAP@R) = -0.90 → ↓ cai com N.
- **Convergência:** só **N=2** aprende de fato (maior teto 0.0065; atinge 95% dele na época 25). Os demais N ficam **estagnados** (teto ≤ 0.0018) em 30 ép — não é só 'convergir mais devagar', eles quase não saem do lugar.
- **Meio-termo:** não há trade-off — N=2 domina em velocidade **e** qualidade, então é a escolha única (não se ganha nada indo pra N maior neste budget).

## cub200 · profile

- **N\* = 2** (test mAP@R = 0.0205, sweep @30ép).
- **Quão melhor (dentro do sweep):** +854.4% vs o pior N (0.0021).
- _Ref.: baseline off @120ép = 0.0308 (não comparável ao sweep @30ép — só contexto)._
- **Padrão com N:** Spearman(N, mAP@R) = 0.00 → pico interno / não-monotônico.
- **Convergência:** só **N=2** aprende de fato (maior teto 0.0205; atinge 95% dele na época 25). Os demais N ficam **estagnados** (teto ≤ 0.0024) em 30 ép — não é só 'convergir mais devagar', eles quase não saem do lugar.
- **Meio-termo:** não há trade-off — N=2 domina em velocidade **e** qualidade, então é a escolha única (não se ganha nada indo pra N maior neste budget).

## cub200 · mds

- **N\* = 2** (test mAP@R = 0.0208, sweep @30ép).
- **Quão melhor (dentro do sweep):** +915.1% vs o pior N (0.0020).
- _Ref.: baseline off @120ép = 0.0308 (não comparável ao sweep @30ép — só contexto)._
- **Padrão com N:** Spearman(N, mAP@R) = -0.30 → tendência fraca.
- **Convergência:** só **N=2** aprende de fato (maior teto 0.0208; atinge 95% dele na época 25). Os demais N ficam **estagnados** (teto ≤ 0.0021) em 30 ép — não é só 'convergir mais devagar', eles quase não saem do lugar.
- **Meio-termo:** não há trade-off — N=2 domina em velocidade **e** qualidade, então é a escolha única (não se ganha nada indo pra N maior neste budget).


# Teacher = resnet18

## cars196 · profile

- **N\* = 2** (test mAP@R = 0.0061, sweep @30ép).
- **Quão melhor (dentro do sweep):** +268.2% vs o pior N (0.0016).
- _Ref.: baseline off @120ép = 0.0185 (não comparável ao sweep @30ép — só contexto)._
- **Padrão com N:** Spearman(N, mAP@R) = -0.10 → pico interno / não-monotônico.
- **Convergência:** só **N=2** aprende de fato (maior teto 0.0062; atinge 95% dele na época 25). Os demais N ficam **estagnados** (teto ≤ 0.0030) em 30 ép — não é só 'convergir mais devagar', eles quase não saem do lugar.
- **Meio-termo:** não há trade-off — N=2 domina em velocidade **e** qualidade, então é a escolha única (não se ganha nada indo pra N maior neste budget).

## cars196 · mds

- **N\* = 2** (test mAP@R = 0.0065, sweep @30ép).
- **Quão melhor (dentro do sweep):** +316.7% vs o pior N (0.0015).
- _Ref.: baseline off @120ép = 0.0185 (não comparável ao sweep @30ép — só contexto)._
- **Padrão com N:** Spearman(N, mAP@R) = -0.90 → ↓ cai com N.
- **Convergência:** só **N=2** aprende de fato (maior teto 0.0065; atinge 95% dele na época 25). Os demais N ficam **estagnados** (teto ≤ 0.0018) em 30 ép — não é só 'convergir mais devagar', eles quase não saem do lugar.
- **Meio-termo:** não há trade-off — N=2 domina em velocidade **e** qualidade, então é a escolha única (não se ganha nada indo pra N maior neste budget).

## cub200 · profile

- **N\* = 2** (test mAP@R = 0.0204, sweep @30ép).
- **Quão melhor (dentro do sweep):** +880.8% vs o pior N (0.0021).
- _Ref.: baseline off @120ép = 0.0340 (não comparável ao sweep @30ép — só contexto)._
- **Padrão com N:** Spearman(N, mAP@R) = 0.00 → pico interno / não-monotônico.
- **Convergência:** só **N=2** aprende de fato (maior teto 0.0204; atinge 95% dele na época 25). Os demais N ficam **estagnados** (teto ≤ 0.0023) em 30 ép — não é só 'convergir mais devagar', eles quase não saem do lugar.
- **Meio-termo:** não há trade-off — N=2 domina em velocidade **e** qualidade, então é a escolha única (não se ganha nada indo pra N maior neste budget).

## cub200 · mds

- **N\* = 2** (test mAP@R = 0.0207, sweep @30ép).
- **Quão melhor (dentro do sweep):** +911.7% vs o pior N (0.0020).
- _Ref.: baseline off @120ép = 0.0340 (não comparável ao sweep @30ép — só contexto)._
- **Padrão com N:** Spearman(N, mAP@R) = -0.70 → ↓ cai com N.
- **Convergência:** só **N=2** aprende de fato (maior teto 0.0207; atinge 95% dele na época 25). Os demais N ficam **estagnados** (teto ≤ 0.0021) em 30 ép — não é só 'convergir mais devagar', eles quase não saem do lugar.
- **Meio-termo:** não há trade-off — N=2 domina em velocidade **e** qualidade, então é a escolha única (não se ganha nada indo pra N maior neste budget).


# Veredito (com os dados existentes no W&B)

- **Melhor N = 2**, de forma consistente nas **8 células** (2 teachers × 2 datasets × 2 métodos). Vale para ambos os teachers (`resnet18` e `convnext_tiny`) — a escolha de teacher **não muda** o N ótimo.
- **Não é só 'N=2 converge mais rápido':** nas curvas de treino, os N>2 ficam **planos** (~0.002 de mAP@R ao longo das 30 épocas), enquanto N=2 **sobe de forma consistente**. Uma curva plana não é 'lenta' — é sinal de que o sinal relacional do grafo com muitos nós praticamente não é aproveitado neste setup. Não há indício de cruzamento: os N maiores não estão subindo em direção ao N=2.
- **Tendência com N:** aumentar N não ajuda (Spearman(N, mAP@R) ≤ 0 na maioria; cai claramente no `mds`). A mensagem do trabalho pode ser direta: **para a comparação por regressão, o grafo relacional mínimo (N=2) é o que entrega — adicionar nós não traz ganho e chega a atrapalhar.**
- **Corroboração em budget cheio:** o único N que rodou 120 épocas (runs `final`) é o N=2, que sobe até mAP@R ~0.03–0.04 — coerente com N=2 sendo o operacional.

## Limite honesto desta conclusão
- O sweep de N existe apenas em **~30 épocas**; não há N>2 em 120 épocas. Logo, não é possível **provar** que nenhum N maior superaria N=2 num treino longo. Mas, com os dados disponíveis, a evidência (curvas planas dos N>2, ausência de tendência de cruzamento) aponta **de forma consistente** para N=2 — é a conclusão defensável.
- O baseline `off` (120 ép) **não** é comparável ao sweep (30 ép); entra só como contexto, não como veredito sobre 'o grafo ajuda vs não-grafo'.
