# Graph-RKD — Analysis & Reporting Guide (for Claude Code)

> **Scope.** This document is the *analysis and reporting* companion to
> `EXPERIMENTS_EN.md` (which specifies **what to run**) and the scaffolded
> `paper.tex` (which specifies **where results go**). This file specifies:
> (1) the hypotheses and the exact analysis that validates or falsifies each,
> (2) which metrics to use and how to compute uncertainty, (3) the tables and
> charts to produce, and (4) how to write the results, interpretation, and
> conclusion honestly. Read `EXPERIMENTS_EN.md` first; then use this to turn logged
> runs into validated conclusions.

> **Prime directive.** Do not presuppose a positive or negative outcome. The job is
> to *test* the hypotheses, report what the data show per cell, and write a
> conclusion that follows from the evidence — including "no significant difference"
> or "worse than baseline" if that is what the numbers say. A rigorous negative
> result is a valid, publishable outcome for this project.

---

## 0. Definitions used throughout

- **Cell.** One combination of (dataset, teacher). There are 4: {Cars-196, CUB-200}
  × {ResNet-18, ConvNeXt-Tiny}. Most analysis is done *per cell*, then examined for
  consistency across cells.
- **Configuration.** One point in the Graph-RKD design grid: (normalization,
  descriptor, objective, N). Baselines (RKD-D, RKD-A, combined, triplet-only) are
  also "configurations" for comparison purposes.
- **Seed spread.** For a fixed configuration+cell, run multiple seeds and report the
  **median** as the central value and the **min–max range** as the spread — e.g.
  `0.41 (0.39–0.43, n=3)`. With only 3 seeds this is descriptive, not inferential:
  do **not** report standard error of the mean (sem), confidence intervals, or
  p-values, because from 3 points the sem is itself wildly uncertain and implies a
  precision you do not have. Median (not mean) is the central value because a single
  collapsed/diverged seed — a known failure mode here — would drag a 3-point mean but
  not the median.
- **"Beyond noise" decision rule (n=3).** Two configurations are **distinguishable**
  only if their seed **ranges do not overlap** (stricter mini-check: all seeds of A
  beat all seeds of B). If the ranges overlap, call it **within noise / a tie** —
  never declare a winner on overlapping ranges. This range-based rule is the
  small-n-honest replacement for any "±1 sem" test. A ~13% relative run-to-run gap
  was observed historically; treat that as a sanity anchor for how wide the spread
  can be.
- **Primary metric = mAP@R.** All model selection and all headline claims use test
  mAP@R (selected on validation mAP@R — see §2). Recall@K and R-Precision are
  secondary/reported-only.

---

## 1. Metrics — which to use, and how

### 1.1 The metrics
- **mAP@R (primary).** Mean Average Precision at R, where R is the number of true
  same-class items for each query. Order-sensitive over *all* relevant items.
  Use for: every selection decision, every headline comparison, every plot axis
  unless stated otherwise.
- **R-Precision (secondary).** Fraction of the top-R that is correct. Report
  alongside mAP@R; do not select on it.
- **Recall@K, K ∈ {1,2,4,8} (secondary, comparability).** Report for comparison with
  the metric-learning literature only. Recall@1 is known to saturate and mislead
  (Musgrave et al.), so it is never a selection or headline metric here.

### 1.2 Seed variability and how to report it (mandatory)
- **Report median and min–max range over seeds**, with n stated: e.g.
  `0.41 (0.39–0.43, n=3)`. Do **not** report `mean ± sem` at small n — see below.
- **Why not sem at n=3.** The sem is `std/√n`, and with n=3 the `std` is estimated
  from three points, so the sem has huge sampling uncertainty; a "±0.003" would
  falsely imply a calibrated interval. Whether a "±1 sem" comparison looks
  significant can flip on which 3 seeds you drew. So we avoid inferential summaries
  at small n and report the raw spread instead.
- **Central value = median**, because a single collapsed seed distorts a 3-point mean
  but not the median. If you prefer, additionally list all seed values in an appendix
  table so nothing is hidden.
- **Comparisons** use the range-overlap rule of §0: distinguishable iff seed ranges do
  not overlap (or, stricter, all seeds of one beat all seeds of the other). Report the
  **effect size** as the difference of medians, Δ mAP@R, alongside both ranges. No
  p-values, no CIs at n=3.
- **Seed budget (amended §10.4 — seeds NOT increased for now).** The executed run
  uses **single-seed search phases** (2/3/4) and **≥3 seeds for the headline** (phase 5);
  we do not bump seeds beyond that. Report the **median test mAP@R** and, as the
  representative model, the **run at that median** (`analysis_utils.median_run`). At
  n=1 there is no range — report the single value and label it single-seed; only the
  headline (n≥3) gets a min–max range and the range-overlap comparison. State n
  everywhere. (If a headline call is close and compute frees up, more seeds help, but
  that is optional, not the default.)

### 1.3 Selection discipline (do not leak the test set)
- Choose checkpoints, N, λg (and τ, M for contrastive) on **validation mAP@R only**.
- Read the test set **once**, with the validation-selected configuration, and report
  all metrics on it. Never tune anything to improve a test number.

---

## 2. Hypotheses and the analysis that validates each

For every hypothesis: state the **test**, the **artifact** (table/chart) that
supports it, and the **decision rule** (accept / reject / inconclusive). Write the
verdict + evidence into the corresponding paper section (§ noted).

### H0 — Active-order viability gate  → paper §7.2 (order/normalization)
- **Claim.** There exists a λg band in which active-order Graph-RKD (N≥3) trains
  without collapsing to the triplet-only floor.
- **Test.** Phase-1 slice (Cars-196, ResNet-18, profile, regression, minibatch norm,
  N=4). Sweep λg ∈ {1e-2, 1e-1, 1, 10, 1e2, 1e3}, short schedule, 1–2 seeds.
- **Artifact.** *Chart:* validation mAP@R vs λg (log x-axis), with a horizontal line
  at the triplet-only floor.
- **Decision.** ACCEPT if some λg yields val mAP@R at or above the triplet-only floor
  (ranges overlapping the floor or better); record the viable band. REJECT and **stop
  the program**, reporting this as the finding, if all λg collapse to ≈ the historical
  floor (~0.002). Inconclusive → widen the λg grid or lengthen the schedule before
  proceeding.
- **Executed (decided).** ACCEPT (marginal): the 30-ep gate had a viable band only at
  **λg≈0.01** (larger λg fell below the floor), and the `conv` 80-ep test confirmed
  Graph-RKD **beats the triplet-only floor by ~14–18% test mAP@R** — the edge survives
  convergence. The full run trims λg to **{0.01, 0.1, 1}**. See `PHASE_1_REPORT.md`.

### H1 — Headline: Graph-RKD vs classic RKD  → paper §7.1 (quantitative)
- **Claim.** Graph-RKD at its best (normalization, descriptor, objective, N) matches
  or exceeds RKD-D, RKD-A, and combined RKD-D+RKD-A in test mAP@R, per cell.
- **Test.** The five students at full budget (120 ep), **≥5 seeds** (this is a
  headline comparison — use 5 if compute allows, 3 only as a fallback), all 4 cells,
  under the fair protocol (per-config λg for Graph-RKD; prescribed weights + full RKD
  recipe for baselines).
- **Artifact.** *Table* per dataset (rows = students, cols = mAP@R, R-Prec,
  R@1/2/4/8, each as **median (min–max), n**); *chart* = grouped bar of test mAP@R
  with the bar at the median and whiskers spanning the seed min–max, grouped by cell.
- **Decision (per cell), using the range-overlap rule (§0).**
  - "Graph-RKD **matches**": its seed range overlaps the best baseline's range.
  - "Graph-RKD **beats**": its range lies **entirely above** the best baseline's
    (ideally every Graph-RKD seed exceeds every baseline seed) AND the direction is
    consistent across both teachers of that dataset.
  - "Graph-RKD **loses**": a baseline's range lies entirely above Graph-RKD's.
  - Report the median difference Δ mAP@R with both ranges. Report each cell
    separately. **If the direction flips across datasets, report the flip** — do not
    average it away; a dataset-dependent result is itself a finding.
  - Report each cell separately. **If the sign flips across datasets, report the
    flip** — do not average it away; a dataset-dependent result is itself a finding.

### H2 — Normalization  → paper §7.2
- **Amended §10.5.** The executed run **drops `hybrid`** (consistently weakest in the
  dev probe) and compares **per-graph / minibatch / none**. The hybrid/`hybrid-specratio`
  material below is retained for reference / a possible follow-up, not the main run.
- **Claim.** Minibatch or hybrid normalization outperforms per-graph and none at
  N≥3; hybrid additionally preserves teacher/student scale-invariance.
- **Canonical hybrid recipe.** μ_batch denominator **+** per-dimension z-scoring of the
  descriptor across the sampled graphs (eval statistics frozen from training; ε-guard
  on the per-dimension std). This one recipe applies to BOTH profile and MDS, so the
  normalization axis stays separable from the descriptor axis. A sample-independent
  **MDS-only** alternative — spectrum ratios (spectrum ÷ largest eigenvalue) — is run
  as a labeled ablation `hybrid-specratio`, never as the default hybrid (it has no
  profile analogue).
- **Test.** Vary normalization ∈ {per-graph, minibatch, none, hybrid}, λg re-tuned
  within each, N ∈ {3,4,8}, both datasets, ResNet-18 (extend to ConvNeXt-Tiny if
  budget allows). Additionally run `hybrid-specratio` on the MDS descriptor only.
- **Artifact.** *Table:* mAP@R by (normalization × N), each as median (min–max), with a
  separate `hybrid-specratio` row under the MDS block. *Chart:* grouped bars or a
  small-multiples line plot (mAP@R vs N, one line per scheme).
- **Decision.** ACCEPT "minibatch/hybrid > per-graph/none" where the seed ranges are
  disjoint. For **hybrid specifically**, additionally verify the invariance property:
  show that teacher-μ and student-μ track (or that descriptor distributions across
  teacher/student align) — report this diagnostic, not just accuracy. A hybrid that
  wins on accuracy but shows no invariance advantage should be reported as "wins on
  accuracy, invariance claim unsupported." Compare canonical `hybrid` vs
  `hybrid-specratio` on MDS: within noise → the sampled-set dependence of z-scoring
  costs nothing; specratio better → prefer it for MDS only and say so.

### H3 — Descriptor: profile vs MDS  → paper §7.3 (descriptor/objective)
- **Amended §10.5.** Executed at **N ∈ {3,4,8}** (16/17 dropped): the offline probe
  (`RKD/analysis/descriptor_probe.csv`, already produced) shows MDS spectra are
  **>90% near-degenerate at N=16/17** (0.1% at N=3), so MDS there is numerically
  fragile and not worth its cost. The stability/fidelity instrumentation below is thus
  available **offline now** (no training needed); cross it with the phase-3 accuracy.
- **Claim.** Profile and MDS differ in accuracy, stability, and fidelity in
  characterizable ways; one is preferable in identifiable regimes.
- **Test.** Both descriptors across N ∈ {3,4,8}, all cells, best normalization,
  regression objective, λg re-tuned per (descriptor, N).
- **Artifacts.**
  - *Table:* profile vs MDS mAP@R by (N, cell), each as median (min–max).
  - *Instrumentation table/plot:* MDS eigenvalue-gap distribution and count of
    near-degenerate spectra (gap < ε); MDS gradient-norm spike frequency; profile
    edge-tie / sort-churn frequency — all as functions of N.
  - *Fidelity:* collision-probe rate (fraction of structurally distinct sampled
    graphs whose descriptors fall within ε), separately for profile and MDS.
- **Decision.** ACCEPT a scenario recommendation ("use profile when …, MDS when …")
  only where (a) the accuracy gap clears noise AND (b) it is explained by a measured
  mechanism (e.g. MDS underperforms exactly where near-degenerate spectra and
  gradient spikes are frequent; profile's edge grows costly/unstable at large N).
  A raw accuracy gap with no mechanistic support is reported as "observed, mechanism
  unconfirmed."

### H4 — Objective: regression vs contrastive  → paper §7.3
- **Claim.** The contrastive objective is more robust to λg mis-scaling than
  regression, because InfoNCE is scale-normalized.
- **Test.** At a fixed active-order cell, sweep the relational weight for BOTH
  objectives over a wide log range; also track raw (unweighted) loss magnitude vs
  N/descriptor and contrastive negative-quality diagnostics.
- **Artifacts.**
  - *Chart (key):* val mAP@R vs weight (log x), one line per objective, on the same
    axes. Mark each objective's best-median and the weight range whose seed spread overlaps that best (the "robust band").
  - *Table:* width of the weight range whose seed range still overlaps that objective's
    best (the "robust band width") for each objective.
  - *Table/plot:* raw loss magnitude vs N and descriptor (explains regression's scale
    sensitivity). *Diagnostics:* fraction of negatives sharing nodes with the anchor
    (overlap contamination); anchor-positive vs anchor-negative similarity gap.
- **Decision.** ACCEPT if regression shows a sharp peak (collapses outside a narrow
  band) while contrastive stays within its own best's seed range across a **substantially
  wider** band (quantify the ratio of band widths). Rule out the alternative
  explanation that contrastive's robustness is merely an easy-negatives artifact by
  citing the negative-quality diagnostics. If contrastive is robust but *worse* in
  peak mAP@R, report both facts — robustness and peak quality are separate claims.

### H5 — N=3 vs RKD-A (matched arity)  → paper §7.4
- **Claim.** At order 3, a distance-based graph descriptor transfers
  comparably/better/worse than RKD-A's angle.
- **Test.** Graph-RKD N=3 with BOTH descriptors vs RKD-A, matched conditions, per
  cell.
- **Artifact.** *Table:* {Graph-RKD-N3-profile, Graph-RKD-N3-MDS, RKD-A} × cells,
  mAP@R as median (min–max), n stated.
- **Decision.** If RKD-A > both Graph-RKD-N3 variants beyond noise → conclude the
  angular signal carries information the distance descriptor misses → this is the
  evidence base for the angular-descriptor future work. If comparable/better → the
  distance descriptor already captures the triadic structure. Report per cell.

### Per-order characterization (supports H1/H2/H3)  → paper §7.2
- For each N ∈ {3,4,8,16,17} report: **convergence** (epochs to reach X% — e.g. 95% —
  of that run's final mAP@R, or area under the val-mAP@R curve), **stability** (seed
  seed range and count of collapsed seeds, instrumentation spikes), **peak** (best median mAP@R), and
  **cost** (measured per-step time, G(N) sampled graphs).
- *Artifacts:* a per-N table with those four columns; a chart of mAP@R vs N with min–max seed
  bands; optionally convergence curves (val mAP@R vs epoch) overlaid by N.
- **Recommendation rule:** report the **best-by-trade-off** N explicitly. It need not
  be argmax mAP@R — e.g. if N=4 reaches 98% of N=8's mAP@R at half the cost and lower
  seed variance, recommend N=4 and say why. State the trade-off you optimized.

---

## 3. Tables to produce (canonical list)

**Implemented:** `analysis_utils.export_tables(df)` (run automatically by notebook
`06_findings`) writes the H1/H2/H3/H4/H5 tables to `RKD/analysis/tables/*.{csv,tex}`
(booktabs, with `\caption`/`\label`), ready to `\input{}`/paste into `paper.tex`.
Figures are saved by the `fig_*` functions to `figures/*.pdf`. `tables/` is gitignored
(derived from `results.csv`) — re-run `06` as more seeds finish.

Emit each as (a) a CSV under `results/tables/` and (b) a LaTeX `table` (booktabs)
ready to paste into `paper.tex`. Always report median and min–max range (state n). Bold the best in each
column *only when its range is entirely above all others* (§0 rule); otherwise bold nothing and note the tie.

1. **T1 — Headline, Cars-196** (H1): rows = 5 students; cols = mAP@R, R-Prec,
   R@1/2/4/8. One block per teacher.
2. **T2 — Headline, CUB-200** (H1): same shape.
3. **T3 — Normalization ablation** (H2): rows = {per-graph, minibatch, none,
   hybrid (canonical z-scoring), and `hybrid-specratio` (MDS-only)}; cols = mAP@R at
   N∈{3,4,8}; one block per dataset. Plus a hybrid invariance-diagnostic column.
4. **T4 — Descriptor characterization** (H3): rows = N∈{3,4,8,16,17}; cols =
   profile mAP@R, MDS mAP@R, and their Δ; plus instrumentation columns (MDS
   near-degeneracy rate, profile tie rate, collision-probe rate).
5. **T5 — Objective robustness** (H4): rows = {regression, contrastive}; cols =
   peak mAP@R, robust-band width, raw-loss magnitude, negative-overlap rate.
6. **T6 — N=3 vs RKD-A** (H5): rows = {Graph-RKD-N3-profile, Graph-RKD-N3-MDS,
   RKD-A}; cols = mAP@R per cell.
7. **T7 — Per-order profile**: rows = N∈{3,4,8,16,17}; cols = peak mAP@R,
   epochs-to-95%, seed range (min–max), per-step time, G(N).

---

## 4. Charts to produce (canonical list)

Emit each as a vector PDF under `results/figures/` (matplotlib, no seaborn styling
assumptions), sized ~\linewidth, with median lines and min–max whiskers/shaded seed range (not sem). Label
axes with units; state K=128 where relevant.

1. **F_headline_bars** (H1): grouped bar chart, test mAP@R, groups = cells, bars =
   5 students, whiskers spanning the seed min–max.
2. **F_lambda_viability** (H0): val mAP@R vs λg (log x), horizontal floor line.
3. **F_norm_lines** (H2): mAP@R vs N, one line per normalization scheme, shaded min–max seed band.
4. **F_descriptor_by_N** (H3): mAP@R vs N, profile vs MDS lines; a companion panel
   with the MDS near-degeneracy rate and profile tie rate vs N.
5. **F_objective_robustness** (H4): val mAP@R vs relational weight (log x),
   regression vs contrastive on one axes, with each objective's robust band (seed-range overlap) marked.
6. **F_order_tradeoff** (per-order): dual-axis or small-multiples — mAP@R vs N with
   min–max seed band, plus cost (per-step time) vs N.
7. **F_convergence** (optional): val mAP@R vs epoch, one line per N (or per objective).
8. **F_qualitative** (§7 qualitative): top-k retrieval panels, Graph-RKD vs strongest
   baseline, same-class hits boxed; caption must say these illustrate error modes,
   not ranking.

**Chart hygiene.** Use a log x-axis wherever a quantity spans orders of magnitude
(λg, weights, the tuples/graphs counts). Never plot a single-seed line without showing the
seed range. Do not use dual y-axes to imply a correlation that is not tested.

---

## 5. How to write the Results section (§7)

Order: quantitative headline → order/normalization → descriptor/objective → N=3 →
qualitative. For each:

1. **State what was run** in one sentence (config, seeds, budget) — refer to the
   setup section, do not restate hyperparameters.
2. **Present the artifact** (table/chart) and read off the numbers with uncertainty:
   "Graph-RKD (minibatch, profile, regression, N=4) reaches 0.xxx ± 0.00y mAP@R vs
   RKD-D's 0.xxx ± 0.00y."
3. **Apply the decision rule** explicitly and per cell. Use the exact language
   "within noise (ranges overlap)" / "beyond noise (ranges disjoint)".
4. **No interpretation yet** — Results states *what happened*; Discussion says *what
   it means*. Keep them separate.

Writing rules:
- Report per cell (dataset × teacher). Never collapse a sign flip into an average.
- Every comparative claim carries its uncertainty and its decision verdict.
- If a result is inconclusive, say so; do not round a tie into a win.
- Cite the primary metric (mAP@R) for claims; mention Recall@K only for
  literature comparability.

---

## 6. How to write the Discussion (§8)

The durable framing (subsumption: Graph-RKD contains N=1 individual and N=2 RKD-D as
special cases; and the per-term-logging lesson) is already in `paper.tex` — keep it.
Then add **one short paragraph per hypothesis**, each with this structure:

1. **Verdict** in the first sentence ("H2 is supported: minibatch normalization
   beats per-graph at every N≥3 tested, beyond noise, on both datasets.").
2. **Evidence** — the specific numbers/artifact, with uncertainty.
3. **Mechanism / interpretation** — *why*, tied to the analysis (e.g. "consistent
   with the λg-scale coupling: per-graph's per-graph rescaling inflates raw-loss
   variance, which the tuned λg cannot fully absorb").
4. **Scope / caveat** — where it holds and where it does not (which cells, which N),
   and any confound not ruled out.

Special cases:
- **H1 dataset-dependence.** If Graph-RKD wins on one dataset and not another,
  discuss *why* the datasets differ (e.g. CUB's fine-grained plumage vs Cars'
  rigid geometry) as a hypothesis, clearly labeled as a hypothesis, not a proven
  cause.
- **H4 robustness ≠ quality.** Keep "contrastive is more λg-robust" and "contrastive
  reaches higher/lower peak mAP@R" as two separate claims with separate verdicts.
- **Q1 (best teacher).** State which teacher transferred better on the new numbers,
  per dataset, with uncertainty.

---

## 7. How to write the Conclusion (§10)

1. **One-paragraph synthesis** of the verdicts: what Graph-RKD does relative to
   classic RKD at N≥3, stated at the evidence's confidence level. If the headline is
   negative or mixed, say so plainly — "Graph-RKD did not outperform properly-tuned
   RKD-D/RKD-A under the fair protocol on either dataset; however, …". If positive,
   state exactly where (which cells, which config) and by how much.
2. **What is genuinely established** vs **what remains open** — separate the two.
   The design-space characterizations (H2/H3/H4) can hold as contributions even if
   the headline (H1) is negative: "the study's contribution is a characterization of
   the normalization/descriptor/objective axes, plus the analytical N=2 reduction,
   independent of whether Graph-RKD beats RKD at these scales."
3. **Future work** — carry the pre-registered items: the N*(K) rule (needs a
   batch-size sweep — a study of its own) and angular/hybrid descriptors (motivated
   by the H5 result). Add any new open question the results surfaced.

**Honesty gates for the conclusion:**
- Do not claim a positive headline unless H1 accepted (beyond noise) in at least
  some clearly named cell, and state where it fails.
- Do not bury a negative H1 — a rigorous negative result is the honest outcome and
  is scientifically valuable here.
- Every superlative ("best", "most robust") must trace to a table row that clears
  the range-overlap rule (§0).

---

## 8. Interpretation pitfalls to avoid (checklist)

- **Averaging across a sign flip.** Report per cell; a mean that hides
  dataset-dependence is misleading.
- **Selecting on test.** All selection is on validation; test is read once.
- **Single-seed claims.** No comparative claim from one seed; report median and min–max over ≥3 seeds (≥5 for headline).
- **Recall@1 as headline.** It saturates; keep it comparability-only.
- **Confusing robustness with quality (H4).** Separate claims, separate verdicts.
- **Mechanism-free scenario rules (H3).** An accuracy gap needs a measured mechanism
  before it becomes a recommendation.
- **λg leakage.** Graph-RKD's λg is tuned on validation and its sensitivity reported;
  the classic baselines use prescribed weights — state this asymmetry so the
  comparison is transparent.
- **Presupposing the outcome.** The conclusion must follow the data, not the
  method's motivation.

---

## 9. Artifacts checklist (what "done" looks like)

- `results/results.csv` — one row per (student/config, normalization, descriptor,
  objective, N, dataset, teacher, seed) with all metric and instrumentation columns.
- `results/tables/T1..T7.{csv,tex}` — the canonical tables.
- `results/figures/F_*.pdf` — the canonical charts.
- `FINDINGS.md` — each hypothesis H0–H5 (+ Q1) mapped to: verdict
  (accept/reject/inconclusive), the supporting artifact, the effect size (median Δ) with seed ranges,
  and one-line interpretation.
- Filled §7, §8, §10 in `paper.tex`, plus the Abstract/Summary headline sentences
  (the `% TODO` markers) — written only from `FINDINGS.md`, never ahead of it.

---

## 10. Execution amendments & implementation (2026-07-08)

Mirrors `EXPERIMENTS_EN.md` §10. Where these conflict with §§1–9, these win.

- **Median reporting (not mean±sem).** Already the policy here (§1.2); seeds are
  **not** increased — single-seed search phases, ≥3 seeds headline. Report the
  **median** test mAP@R and the **median-run checkpoint** (`analysis_utils.median_run`).
- **Trimmed grid (executed).** normalization ∈ {per-graph, minibatch, none} (drop
  `hybrid`); λg ∈ {0.01, 0.1, 1}; N ∈ {3,4,8}; headline = mds/regression/per-graph/N4/
  λg0.01. Search phases use **60 epochs** (was 30, near-floor). Tables T3/T4/T7 shrink
  accordingly; the hybrid row and N∈{16,17} rows become "not run (see amendment)".
- **H0/H3 partly decided offline.** H0 accepted (gate + conv); H3 mechanism quantified
  by `descriptor_probe.csv`. Fill the accuracy halves from the local run.
- **Implementation to use (no need to reimplement).** `RKD/analysis/analysis_utils.py`
  (`fetch_runs`→`results.csv`, `agg` median, `median_run`, `h1_verdict`, and the
  `fig_*` builders) + notebooks `00_aggregate` … `06_findings`. W&B project:
  **`gabomfim-unicamp/graph-rkd`** (personal entity, not the org). `descriptor_probe.py`
  produces the H3 fidelity/stability evidence with no training.
- **Division of labor.** The colleague only *runs* the experiments (logs to W&B).
  **Gabriel runs the analysis locally** from the finished W&B runs using this guide +
  the notebooks, then writes the paper §§7–8–10.
