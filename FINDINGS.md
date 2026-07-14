# FINDINGS — Graph-RKD (campaign of 2026-07, recompute pass)

> Produced by `RKD/analysis/07_todo_analyses.ipynb` (recompute-only, no retraining) per
> `ANALYSES_TODO.md`. Primary metric everywhere: **median test mAP@R in %** (selection on
> validation mAP@R). No min–max ranges (author decision). CUB is provisional (n=2).

## 0. Campaign metadata
- Seeds per cell (Graph-RKD headline mds/N4): Cars/R18=**3**, Cars/CvT=**2**, CUB/R18=**2**, CUB/CvT=**2**
- Budget: search = **60** epochs, headline = **120** epochs
- Current headline config: descriptor=**MDS**, N=**4**, norm=**per-graph**, λg=**0.01**, objective=**regression**
- Figure convention after §1 fix: **median test mAP@R in %, aggregated across the 4 cells** (same as the headline table). Exception: the H0 gate stays **val** mAP@R % (it is a viability gate; selection is on val), with the axis relabelled accordingly.
- **Cause of the ~4× text↔figure mismatch:** the old figures plotted **`val_mAP@R` as a raw fraction** and aggregated by **best-λg max over (N, cells)**, whereas the text/tables use **`test_mAP@R` in %** with a **cross-cell median**. Two independent offsets (val vs test; fraction vs %; max vs median) compounded. All mAP figures were regenerated in the unified test-% / cross-cell-median convention; H0 is the only val figure and is labelled "val mAP@R (%)".

## 1. Headline table (H1) — 5 students × 4 cells (median test mAP@R %)
| student | Cars/R18 | Cars/CvT | CUB/R18 | CUB/CvT |
|---|---|---|---|---|
| triplet-only (floor) | 3.07 | 3.07 | 3.05 | 3.05 |
| +RKD-D | 1.79 | 2.05 | 3.29 | 2.98 |
| +RKD-A | 1.93 | 2.29 | 3.50 | 3.07 |
| +RKD-D+RKD-A | 1.76 | 1.93 | 3.48 | 3.08 |
| +Graph-RKD (MDS, N=4) | 3.20 | 3.35 | 3.12 | 3.08 |

n per cell: Cars/R18=3, Cars/CvT=2, CUB/R18=2, CUB/CvT=2 (all students; Graph-RKD = strict mds/N4).

Classification vs floor (noise band = 13% relative):
| cell | Graph-RKD vs floor | best classic-RKD vs floor |
|---|---|---|
| Cars/R18 | within-noise (3.20 vs 3.07) | below-floor (best RKD-A 1.93) |
| Cars/CvT | within-noise (3.35 vs 3.07) | below-floor (best RKD-A 2.29) |
| CUB/R18 | within-noise (3.12 vs 3.05) | **above-beyond-noise** (RKD-A 3.50, RKD-D+A 3.48) |
| CUB/CvT | within-noise (3.08 vs 3.05) | within-noise (RKD-D+A 3.08, RKD-A 3.07) |

**Veredito H1: dependente-de-regime — com recalibração importante.** Nenhum student, **incluindo o Graph-RKD**, fica acima do piso além do ruído, **exceto** o RKD clássico (RKD-A e RKD-D+A) na célula **CUB/R18**. As afirmações defensáveis são: (a) no **Cars**, o RKD clássico **cai abaixo do piso** (prejudica) enquanto o Graph-RKD fica **dentro do ruído** do piso (não prejudica); (b) no **CUB**, o RKD clássico **ajuda** (RKD-A além-do-ruído no R18) enquanto o Graph-RKD fica **dentro do ruído** (nem ajuda nem prejudica).
Frase-resumo: *O valor do Graph-RKD aqui é "não prejudicar onde o RKD clássico prejudica (Cars)"; ele não é "o melhor" em nenhuma célula além do ruído.*

## 2. H0 — gate de viabilidade (λg) [Cars/R18, phase1, val mAP@R %]
- Faixa viável de λg: **só λg=0.01**
- val mAP@R no melhor λg vs piso: **2.80 vs 2.32** (acima do piso)
- λg≥0.1: cai abaixo do piso, **monotônico** (1.70 → 1.12 → 0.86 → 0.94 → 0.81 para 0.1/1/10/100/1000)
- Sobreviveu ao treino longo? **sim** (o item §8 mostra os finais de 120 ép platôando acima do piso de val)
- **Veredito H0: suportada (marginal)** — janela estreita, só em λg pequeno.

## 3. H2 — normalização (phase2, best-λg median test mAP@R %)
| norm | N=3 | N=4 | N=8 |
|---|---|---|---|
| per-graph | 2.34 | 2.36 | 2.29 |
| minibatch | 2.32 | 2.33 | 2.29 |
| none | 2.19 | 2.00 | 1.03 |
| hybrid | NÃO NO GRID (só sondagem — ver §12) | — | — |
- Diagnóstico de invariância da híbrida: **não testado** (a híbrida não entrou no grid; ver §12).
- **Veredito H2:** per-graph ≈ minibatch (indistinguíveis); **none piora e o déficit cresce com N** (2.19→1.03 de N=3 a N=8). A escolha entre os dois viáveis é imaterial na acurácia; mantém-se per-graph pela correspondência direta com a normalização do RKD clássico.

## 4. H3 — descritor (profile vs MDS), acurácia em ORÇAMENTO CHEIO (phase5, N=3)
| cell | profile | MDS | Δ(prof−mds) |
|---|---|---|---|
| Cars/R18 | 3.57 | 3.28 | +0.30 |
| Cars/CvT | 3.31 | 3.34 | −0.02 |
| CUB/R18 | 3.31 | 3.03 | +0.28 |
| CUB/CvT | 3.13 | 2.98 | +0.16 |
- Mecanismo (sonda offline): MDS quase-degeneração **>90% em N=16/17** (0.1% em N=3); profile tie-rate cresce só levemente. Ambos seguros em N≤8.
- **Recomendação de manchete: TROCAR para profile.** No orçamento cheio (N=3) o profile é ≥ MDS em 3 de 4 células (empate na 4ª, −0.02), é o descritor **mais estável** e **mais interpretável**. A escolha do MDS veio do agregado de 60 épocas; em 120 épocas o profile é igual ou melhor. (A conclusão de design-space não muda — Δ dentro do ruído — mas o descritor de vitrine defensável é o profile.)
- **Veredito H3: equivalente em acurácia; separado por mecanismo** (fragilidade do MDS em N grande).

## 5. H4 — objetivo (regression vs contrastive) [phase4, test mAP@R %]
| objetivo | λg=0.01 | λg=0.1 | λg=1 | queda 0.01→0.1 |
|---|---|---|---|---|
| regression | 1.50 | 1.21 | 0.26 | −19% |
| contrastive | 1.63 | 0.93 | 0.20 | −43% |
- n de seeds deste sweep: **n=1** (single-seed)
- **Veredito H4: não suportada / inconclusiva a n=1.** Contra a expectativa (InfoNCE mais robusto a λg), a **regression degrada MENOS** de 0.01→0.1 (−19% vs −43%). Nenhuma banda robusta claramente mais larga; precisa de múltiplas seeds antes de qualquer claim. Mantém-se regression como objetivo de manchete.

## 6. H5 — N=3 vs RKD-A (arity casada, phase5, test mAP@R %)
| cell | Graph-RKD (profile) | Graph-RKD (MDS) | RKD-A |
|---|---|---|---|
| Cars/R18 | 3.57 | 3.28 | 1.93 |
| Cars/CvT | 3.31 | 3.34 | 2.29 |
| CUB/R18 | 3.31 | 3.03 | 3.50 |
| CUB/CvT | 3.13 | 2.98 | 3.07 |
**Veredito H5: dependente-de-regime.** No **Cars** o descritor de distância (profile/MDS 3.3–3.6) **bate o RKD-A decisivamente** (1.9–2.3). No **CUB**, RKD-A é competitivo: à frente no R18 (3.50 vs profile 3.31 / MDS 3.03) e comparável no CvT (3.07; profile 3.13, MDS 2.98). O ângulo do RKD-A carrega informação que o descritor só-distância não captura no CUB → **motiva descritor angular/híbrido** (trabalho futuro).

## 7. Per-order (N=3/4/8) — median test mAP@R % (phase3, agregado entre células)
| N | MDS mAP@R | profile mAP@R | custo | estabilidade |
|---|---|---|---|---|
| 3 | 2.18 | 2.03 | menor (menos grafos; dim N ou N(N−1)) | segura (degeneração MDS ~0%) |
| 4 | 2.01 | 2.01 | intermediário | segura |
| 8 | 1.75 | 1.57 | maior | segura em N≤8 (none colapsa; MDS ainda ok) |
Melhor-por-trade-off: **N=3–4** (acurácia decresce monotonicamente com N; N=4 na manchete atual, N=3 igualmente defensável e mais barato). N≥16 excluído (MDS >90% degenerado). N=2 excluído (Lema: degenera ou vira RKD-D).

## 8. Convergência (item §3 do TODO) — val mAP@R vs época, headline mds/N4/seed0
| cell | época melhor ckpt | platôou? | val pico → final (%) | veredito |
|---|---|---|---|---|
| Cars/R18 | 100 | sim | 16.76 → 16.68 | convergiu |
| Cars/CvT | 120 | sim | 16.07 → 16.07 | convergiu (pico na última ép., mas platô nas últimas 20) |
| CUB/R18 | 35 | sim | 14.58 → 12.67 | convergiu; **leve overfit de val após ~ép35** (ckpt no pico) |
| CUB/CvT | 55 | sim | 14.18 → 12.21 | convergiu; leve overfit de val após ~ép55 |
- Gap teacher→student (test mAP@R %): Cars/R18 19.1→3.2; Cars/CvT 36.4→3.35; CUB/R18 18.1→3.12; CUB/CvT 29.3→3.08.
- **Gap val↔test grande e sistemático:** val mAP@R chega a **~14–17%** enquanto o test fica em **~3%**. Isso é a **generalização para classes NÃO vistas** (split disjunto: val = classes de treino retidas; test = classes oficiais de teste), não subtreino.
- **Veredito global: students convergiram (SEM subtreino).** Nos 4 casos as curvas platôaram; o CUB inclusive faz leve overfit de val depois do pico (o oposto de subtreino). **Os números baixos de test são o gap de conjunto-aberto + student minúsculo (0.67M), não falta de treino.** Descarta a hipótese de subtreino do §3.

## 9. Métricas secundárias (apêndice) — mediana test %, por dataset
### Cars-196
| student | R@1 | R@2 | R@4 | R@8 | R-Prec |
|---|---|---|---|---|---|
| triplet-only | 20.96 | 30.08 | 41.34 | 54.05 | 8.75 |
| +RKD-D | 14.24 | 21.08 | 30.28 | 41.67 | 6.50 |
| +RKD-A | 14.28 | 21.40 | 30.86 | 42.84 | 6.89 |
| +RKD-D+RKD-A | 13.59 | 20.38 | 29.66 | 40.92 | 6.49 |
| +Graph-RKD | 23.67 | 33.07 | 44.05 | 56.36 | 9.11 |
### CUB-200
| student | R@1 | R@2 | R@4 | R@8 | R-Prec |
|---|---|---|---|---|---|
| triplet-only | 14.30 | 21.79 | 31.76 | 44.02 | 8.34 |
| +RKD-D | 15.27 | 22.48 | 32.22 | 44.84 | 8.45 |
| +RKD-A | 15.92 | 23.70 | 33.55 | 46.03 | 8.77 |
| +RKD-D+RKD-A | 15.77 | 23.50 | 34.19 | 46.72 | 8.87 |
| +Graph-RKD | 14.58 | 21.75 | 31.16 | 43.76 | 8.23 |
(Recall@K collapsed across both teachers per dataset; Graph-RKD = strict mds/N4. Secondary metrics track mAP@R: Graph-RKD best on Cars, classic RKD slightly ahead on CUB.)
- **Recall@1 bate com o citado na análise qualitativa? PARCIALMENTE — e é explicável.** A análise qualitativa cita R@1 do **checkpoint seed-0** (25.3% / 24.6% Cars; 14.3% / 15.0% CUB). O R@1 **mediano por célula** (mds/N4) é 22.0 / 24.1 (Cars) e 14.6 / 14.1 (CUB). A diferença é **agregação de seeds** (qualitativo = seed 0; tabela = mediana sobre seeds), **não um bug**. Recomendação: no texto, ou citar "R@1 do checkpoint seed-0 mostrado no painel", ou trocar pelos valores medianos para consistência com as tabelas.

## 10. Q1 — comparação de teachers
- Qualidade própria dos teachers (test mAP@R %): ConvNeXt-Tiny = **36.4 / 29.3** (Cars/CUB); ResNet-18 = **19.1 / 18.1**.
- Isso se traduz em students melhores? **NÃO de forma proporcional.** Graph-RKD: Cars R18 3.20 vs CvT 3.35 (CvT levemente melhor); CUB R18 3.12 vs CvT 3.08 (R18 levemente melhor). Diferenças ~0.1–0.2 ponto, dentro do ruído, apesar do teacher CvT ser ~2× melhor.
- **Veredito Q1:** nesta compressão (student 0.67M), a **capacidade do teacher não é o gargalo**; o sinal de transferência e o regime do dataset dominam.

## 11. Limitações (texto pronto para o autor)
- **Peso do RKD não re-tunado (FORA DE ESCOPO).** A comparação mantém RKD-D=25 / RKD-A=50, os pesos do setting de **classificação** de Park et al. O próprio H0 mostra que um peso relacional grande demais empurra o student **abaixo do piso**; portanto o colapso do RKD clássico no Cars **pode ser um artefato de escala de peso**, não uma propriedade do método. O controle correto seria re-tunar o peso do RKD na validação (mesmo protocolo do λg). **Isto não foi rodado** (restrição de tempo/quota). Toda afirmação "RKD hurts / cai abaixo do piso" deve ser lida como **"aos pesos prescritos"**, não como claim incondicional.
- **n baixo de seeds:** Cars n=2–3, CUB n=2. Sem estatística inferencial; diferenças ≤ ~13% relativo são indistinguíveis; **CUB provisório**.
- **Convergência OK, números absolutos baixos:** os students convergiram (§8) — os ~3% de test refletem o **gap para classes não vistas** (val ~14–17% vs test ~3%) e o student minúsculo, não subtreino. Todas as conclusões são **relativas** entre variantes de destilação.
- **Híbrida não testada no grid:** existe só uma **sondagem** de 4 runs (dev, Cars/R18, N=4, n=1; ver §12). A propriedade de invariância da híbrida fica **em aberto**, não refutada.
- **Recalibração da manchete:** no Cars, Graph-RKD está **dentro do ruído** do piso (não "é o melhor"); o achado defensável é "não prejudica enquanto o RKD clássico cai abaixo do piso".

## 12. Notas livres para o assistente (mudanças que o texto precisa refletir)
- **§4 Híbrida — sondagem existe (n=1, dev, Cars/R18, N=4, λg=0.01):**
  | run | val mAP@R % | test mAP@R % |
  |---|---|---|
  | profile+regression+hybrid | 2.97 | 0.71 |
  | profile+contrastive+hybrid | 1.72 | 0.58 |
  | mds+regression+hybrid | 1.13 | 0.19 |
  | mds+contrastive+hybrid | 1.08 | 0.17 |
  São runs de **dev** (schedule curto → test muito baixo, não comparável ao grid de 120 ép). Melhor híbrida = profile+regression (val 2.97). **Não** entrou no grid completo e o diagnóstico de invariância (μ-teacher vs μ-student) **não** foi computado → reportar como "sondada preliminarmente, invariância não testada".
- **Figuras (todas regeneradas na convenção test %):** `fig_h1_headline.pdf` (bars, test %), `fig_h0_lambda.pdf` (val %, gate), `fig_h2_norm.pdf` (agora **grouped bars norm×N em test %** — não mais fração/val), `fig_h4_overlay.pdf` (test %), `fig_h3_probe.pdf` (inalterada, taxas %). **NOVA:** `fig_convergence.pdf` (val mAP@R vs época, 4 células — evidência do item §8). Rótulos val/test agora coerentes com as legendas.
- **Descritor de manchete:** recomendação de trocar MDS→**profile** (§4/H3). Se aceito, atualizar a config da manchete no texto ("Graph-RKD = MDS, N=4" → "profile, N=4") e recomputar figuras/tabelas da manchete com profile. **Decisão do autor** — os números das duas versões estão em §4.
- **R@1 qualitativo vs tabela:** ver §9 — citar como seed-0 ou trocar pelos medianos.
- **Tabelas exportadas** atualizadas em `RKD/analysis/tables/*.{csv,tex}` (8 tabelas).
