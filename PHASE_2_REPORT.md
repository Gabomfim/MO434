# PHASE 2 REPORT — Normalization ablation (H2) — SCAFFOLD

Source: `RKD/analysis/02_order_normalization_H0_H2.ipynb`, phase 2 runs
(norm ∈ {per_graph, minibatch, none} — hybrid dropped, §10.5; N∈{3,4,8},
λg∈{0.01,0.1,1}, search_epochs=60). Metric: **median** val/test mAP@R.

## Table (best λg per norm) — _TBD from results.csv_
| norm | N | best λg | val mAP@R | test mAP@R |
|---|---|---|---|---|
| per_graph | | | _TBD_ | _TBD_ |
| minibatch | | | _TBD_ | _TBD_ |
| none | | | _TBD_ | _TBD_ |

**Decision rule (H2):** minibatch/none ≥ per_graph where the median gap is
clear. **Verdict:** _TBD._
