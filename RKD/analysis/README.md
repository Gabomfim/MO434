# Analysis — fills Section 7 of the paper (`Graph-RKD-MO434.pdf`)

These notebooks turn the W&B runs into the paper's tables/figures for hypotheses
H0–H5. All real logic lives in `analysis_utils.py`; the notebooks are thin
callers, so the analysis is reproducible and testable. Regenerate the notebooks
with `python _make_notebooks.py`.

## Prerequisites
```bash
uv sync
export WANDB_API_KEY=...        # to read runs from wandb.ai/gabomfim-unicamp/graph-rkd
cd RKD/analysis
```

## Notebooks (run in order)
| Notebook | Paper § | Produces |
|---|---|---|
| `00_aggregate_results.ipynb` | — | pulls all W&B runs → `results.csv` (used by the rest) |
| `01_quantitative_H1.ipynb` | §7.1 | 5-student tables (mAP@R / R-Prec / Recall@K, mean±sem) + grouped bar; **H1** verdict per cell |
| `02_order_normalization_H0_H2.ipynb` | §7.2 | λg-viability gate (**H0**), normalization ablation (**H2**), per-order quality |
| `03_descriptor_objective_H3_H4.ipynb` | §7.3 | descriptor probe (**H3** mechanism), profile-vs-MDS accuracy, λg-robustness overlay (**H4**) |
| `04_n3_vs_rkda_H5.ipynb` | §7.4 | N=3 vs RKD-A matched-arity table (**H5**) |
| `05_qualitative_retrieval.ipynb` | §7.5 | top-k retrieval panels (needs `STUDENT_CKPT` env → a trained checkpoint) |
| `06_findings.ipynb` | §8 | consolidated H0–H5 verdicts text |

Decision rules follow paper §8: seed **noise floor** first; "beats" only if mean
> best baseline + 1 sem **and** sign-consistent across teachers; reported per
(dataset × teacher) cell, never averaged over a sign flip.

## Offline descriptor probe (no training, no W&B)
`descriptor_probe.py` computes profile-vs-MDS **collision rate**, **MDS eigengap
near-degeneracy**, and **profile tie rate** across N∈{3,4,8,16,17} and all norm
schemes, on random graphs. It is the mechanism evidence for **H3** and already
has real results in `descriptor_probe.csv` (+ `figures/fig_h3_probe.pdf`):
```bash
python descriptor_probe.py --n-graphs 2000      # regenerate the CSV
```
Key finding: MDS near-degeneracy rises 0.1% (N=3) → 12.7% (N=8) → 94.6% (N=17);
profile tie rate 3.4% → 32.1%; collisions ≈ 0 for both.

## Notes
- Figures are written to `figures/*.pdf` (paper-ready).
- Notebooks degrade gracefully before runs exist (print "sem runs ainda"); the
  probe (H3 mechanism) works with zero training.
