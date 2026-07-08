# FINDINGS — H0–H5 verdicts (SCAFFOLD)

Maps each hypothesis to its verdict + evidence, per `EXPERIMENTS_EN.md` §7–§8.
H0 is decided (Modal gate/dev/conv); **H1–H5 are filled from the local trimmed
run** via `RKD/analysis/` notebooks `00`→`06` (median test mAP@R, per the §10.4
amendment). Replace each _TBD_ once `results.csv` has the phase-2–5 runs.

> Global rule (§8, amended §10.4): report the **median** test mAP@R and the
> median-run checkpoint; "beats" = higher median, reported **per (dataset ×
> teacher) cell**, never averaged across a sign flip.

## H0 — λg viability gate — **ACCEPT (with caveat)** ✅ decided
- Gate (30 ep): viable band only at λg≈0.01; larger λg fell below floor
  (`PHASE_1_REPORT.md`).
- Dev probe: 11/14 configs beat floor at λg=0.01; **`hybrid` worst**; regression
  more consistent.
- **conv (80 ep): Graph-RKD beats the triplet-only floor by ~14–18% test mAP@R**
  (`mds/reg/per_graph` 0.0252, `prof/reg/mb` 0.0243 vs floor 0.0214). The edge
  survives convergence → method worth the full run.

## H1 — headline (Graph-RKD vs RKD-D / RKD-A / combined) — _TBD_
Source: notebook `01`, phase 5. Per cell, median test mAP@R of the 5 students.
_Verdict per (dataset, teacher): TBD._

## H2 — normalization (per_graph / minibatch / none [hybrid dropped]) — _TBD_
Source: notebook `02`, phase 2. Best scheme by median, crossed with stability.
_Verdict: TBD (dev suggested per_graph/minibatch/none over hybrid)._

## H3 — descriptor (profile vs MDS) — _partly decided_
- Mechanism (offline probe, `descriptor_probe.csv`): **MDS near-degenerate
  >90% at N=16/17**, ~0.7% at N≤4; profile tie rate 3→32% with N. → MDS fragile
  at large N (why the trimmed grid uses N∈{3,4,8}).
- Accuracy by (N, dataset, teacher): _TBD_ (notebook `03`, phase 3).

## H4 — objective robustness (regression vs contrastive) — _TBD_
Source: notebook `03`, phase 4. λg-band width within 10% of each objective's best.
_Verdict: TBD (dev: regression more consistent at λg=0.01)._

## H5 — N=3 vs RKD-A — _TBD_
Source: notebook `04`, phase 5. Median test mAP@R of Graph-RKD N=3 (profile & mds)
vs RKD-A, per cell. _Verdict: TBD._

## Qualitative — _TBD_
Notebook `05`: top-k retrieval panels, Graph-RKD vs strongest classic baseline.
