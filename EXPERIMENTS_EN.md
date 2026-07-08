# Graph-RKD — Experimental Guide (for Claude Code)

> **Purpose.** This document specifies the experimental program for the Graph-RKD
> project so that an agent (Claude Code) can execute it end-to-end. It defines the
> hypotheses, the runs, the protocol invariants, the metrics, the instrumentation,
> and — crucially — **how to analyze the results to accept or reject each
> hypothesis**. Read the whole file before running anything.

---

## 0. How to use this document

You are an autonomous coding agent working in the Graph-RKD repository.

1. **First, map the spec onto the code.** Do NOT assume file names or CLI flags.
   Inspect the repo: locate the training entry point, the config system, the loss
   definitions (triplet, RKD-D, RKD-A, graph regression, graph contrastive), the
   graph-descriptor code (profile, MDS), the normalization step, the sampler, and
   the evaluation/metrics code. Produce a short `REPO_MAP.md` listing where each
   concept below lives. Only then begin implementing/running.
2. **Respect the invariants in §2.** They are the reason the study is fair; violating
   them silently invalidates every result.
3. **Stage the work as in §3.** Do not launch the full grid blindly — Phase 1 gates
   everything else.
4. **Log everything specified in §6.** Analysis depends on per-term and per-config logs.
5. After each phase, write a short `PHASE_k_REPORT.md` with the tables/plots and an
   explicit accept/reject verdict on the phase's hypotheses (§8 gives the criteria).

---

## 1. Background & the core reframing

Graph-RKD generalizes Relational KD by modeling each minibatch as several complete,
undirected, weighted graphs (one per sampled node set of size `N`), summarized by a
permutation-invariant descriptor (profile or MDS) and matched between teacher and
student. The project's central claim is **not** "does the graph term help in
isolation" but:

> **Does Graph-RKD at relational order N ≥ 3 match or beat the classic RKD baselines
> (RKD-D, RKD-A, and their combination), all added to a triplet task loss, under a
> fair shared protocol?**

Two facts fixed by prior analysis:

- **N = 2 is not run.** A two-node graph is a single edge = the pairwise case. Under
  per-graph normalization it degenerates (constant descriptor, zero gradient); under
  minibatch normalization it provably reduces to RKD-D (Lemma). Either way it carries
  no information beyond the RKD-D baseline, which represents the pairwise case.
- **N = 1 is not run.** A one-node graph has no edges → no relation (this is
  individual distillation, out of scope).

So Graph-RKD is evaluated at **N ∈ {3, 4, 8, 16, 17}** (K = 128), where it is a
genuinely higher-order method. N = 3 is the smallest non-degenerate order and the
arity-matched comparison to RKD-A.

---

## 2. Protocol invariants (DO NOT VIOLATE)

These make the comparison fair. If any cannot be satisfied, STOP and report.

- **I1 — Per-configuration λg tuning.** Graph-RKD's relational weight `λg` is tuned
  **per configuration** on the **validation** split. Never a global constant; never
  tuned on test. (The original failure was a fixed λg=1000 calibrated for the tiny
  N=2 loss regime, which swamped the triplet term ~500:1 at active orders.)
- **I2 — Classic baselines at prescribed weights + full RKD recipe.** RKD-D weight
  = 25, RKD-A weight = 50 (the original 1:2 ratio); combined student uses the same
  1:2 ratio. Baselines use the original RKD recipe: μ-normalization of the distance
  potential (batch-mean), the **Huber** potential, and RKD's pair/triplet sampling.
  Do NOT re-tune the classic baselines — using authors' validated weights avoids
  handicapping them.
- **I3 — Shared everything else.** All students share the same teacher, dataset
  splits, training budget (epochs), task (triplet) loss, warm-up schedule, optimizer,
  and batch size K=128. Only the relational term differs.
- **I4 — Selection metric is validation mAP@R** everywhere: teacher checkpoint,
  student checkpoint, and order selection. Recall@K is reported but never used to
  select.
- **I5 — Multi-seed finals.** Every headline number is mean ± sem over ≥ 3 seeds.
  No single-seed point estimates in the final comparison.
- **I6 — Warm-up ramps to λg, not to 1.** The warm-up multiplier goes 0→1 over the
  first ~10% of epochs; the effective relational weight reaches λg. Confirm the code
  matches this.
- **I7 — Per-term logging always on.** Log `train/graph_loss` (and `train/triplet_loss`)
  separately, every run. A silently-zero term must be detectable.
- **I8 — No mixup/cutmix** in distillation runs (breaks the per-example teacher/student
  node correspondence the graph loss needs).

---

## 3. Staged execution plan

Do the phases in order. Each phase gates the next.

### Phase 0 — Repo map & smoke test
- Produce `REPO_MAP.md`.
- Confirm you can: fine-tune/load a teacher, train the student with triplet-only,
  and evaluate mAP@R/Recall@K/R-Precision on the test split.
- Confirm per-term logging (I7) works: run 1–2 epochs of a graph configuration and
  verify `train/graph_loss` is logged and non-zero at N=3.
- **Gate:** triplet-only student trains and evaluates end-to-end on both datasets.

### Phase 1 — Establish active-order viability (the critical gate)
This is the make-or-break phase: does Graph-RKD produce a *usable* signal at an active
order once λg is sane? Run on ONE slice to keep it cheap:
- Dataset = Cars-196, Teacher = ResNet-18, Descriptor = profile, Objective = regression,
  Normalization = minibatch, Order N = 4.
- **Sweep λg** over a wide log range (e.g. 1e-2, 1e-1, 1, 10, 1e2, 1e3) at a short
  schedule (e.g. 30 epochs), 1–2 seeds.
- Plot validation mAP@R vs λg. Identify whether there exists a λg band where the
  graph configuration is **at least competitive with triplet-only** (not collapsed to
  the floor).
- **Gate (H0 below):** if NO λg avoids collapse, STOP and report — the method as
  implemented cannot be fairly tested further without a fix; document this as the
  finding. If a viable band exists, record it and proceed.

### Phase 2 — Normalization ablation
- Fix a reasonable slice (Cars-196 + CUB-200, ResNet-18, profile, regression, N ∈ {3,4,8}).
- Vary **Normalization ∈ {per-graph, minibatch, none, hybrid}**, λg re-tuned within each.
- Establishes which normalization to carry forward (H2).

### Phase 3 — Descriptor characterization (profile vs MDS)
- Best normalization from Phase 2, both datasets, both teachers, N ∈ {3,4,8,16,17},
  regression objective, λg re-tuned per (descriptor, N).
- Adds the stability & fidelity instrumentation (§6). Answers H3.

### Phase 4 — Objective characterization (regression vs contrastive)
- Best normalization + descriptor, one dataset + teacher for the robustness sweep,
  then confirm on a second cell.
- Centerpiece: the **λg / weight-robustness sweep** for both objectives (H4).

### Phase 5 — Headline comparison (multi-seed finals)
- The 5 students (§4) at the selected (normalization, descriptor, objective, N),
  full budget (120 epochs), ≥ 3 seeds, both datasets, both teachers.
- Plus the **N=3 standalone vs RKD-A** comparison (H5).
- This produces Tables 2–3 and the headline verdict (H1).

### Phase 6 (optional / future) — N*(K)
- Only if Phase 3's per-order curves are non-flat and meaningful. Requires sweeping K.
  Treat as a separate study; likely out of scope for the current deliverable.

---

## 4. The five students (same architecture, differ only in relational term)

All add their relational term to the shared triplet loss under I2/I3.

1. **triplet-only** — ablation FLOOR (does any relational term beat plain triplet?).
2. **triplet + RKD-D** (weight 25, Huber, μ-norm).
3. **triplet + RKD-A** (weight 50, Huber).
4. **triplet + RKD-D + RKD-A** (combined, 1:2 ratio) — strongest classic reference.
5. **triplet + Graph-RKD** — the method, swept over the design axes below.

---

## 5. The Graph-RKD design grid

| Axis | Levels |
|---|---|
| Normalization | per-graph, minibatch, none, hybrid |
| Order N | 3, 4, 8, 16, 17 (K=128; **no N=2**, no N=1) |
| Descriptor | profile (dim N(N−1)), MDS (dim N) |
| Objective | regression (Minkowski p=2), contrastive (InfoNCE; hyperparams M negatives, τ temperature) |
| Dataset × Teacher | {Cars-196, CUB-200} × {ResNet-18, ConvNeXt-Tiny} |

**Do not run the full cross-product blind** (≈256 cells before seeds/λg). Use the
staged slices in §3. λg (and τ, M for contrastive) tuned per configuration on validation.

Hybrid normalization = μ_batch denominator (restores cross-graph scale) + scale-invariant
descriptor (MDS-spectrum ratios, or z-score descriptors across the sampled graphs).

---

## 6. Metrics & instrumentation

**Retrieval metrics** (test split, but SELECT on validation — I4):
- **mAP@R** — primary, order-sensitive, used for all selection.
- **R-Precision** — secondary, order-sensitive.
- **Recall@K**, K ∈ {1,2,4,8} — secondary, for literature comparability only.

**Per-run instrumentation** (log to whatever tracker the repo uses; also dump to CSV):
- `train/graph_loss`, `train/triplet_loss` per step (I7). Raw (unweighted) and weighted.
- Effective λg·(warm-up) over training.
- **MDS stability:** per-step min eigenvalue gap of the double-centered Gram, count of
  near-degenerate spectra (gap < ε), gradient-norm spikes / NaNs.
- **Profile stability:** frequency of edge-weight ties / near-ties (sort-order churn).
- **Descriptor fidelity:** on a sample of graphs, fraction of structurally distinct
  graphs whose descriptors are within ε (collision probe) — separately for profile
  (correspondence loss) and MDS (cospectral).
- **Per-order profile:** epochs-to-X%-of-final-mAP@R (convergence), seed variance
  (stability), best mAP@R (quality), measured per-step time and G(N) (cost).
- **Contrastive negative quality:** fraction of sampled negatives sharing nodes with
  the anchor (overlap contamination); anchor-positive vs anchor-negative similarity gap.

**Efficiency:** compute the K×K teacher (and student) distance matrix once per step and
slice submatrices for sampled graphs. Do NOT cache teacher embeddings across epochs
(augmentation changes them — deliberate).

---

## 7. Hypotheses

State each as falsifiable; §8 gives the accept/reject rule.

- **H0 (viability gate).** There exists a λg band in which active-order Graph-RKD
  (N≥3) trains without collapsing to the floor. *If false, the method as implemented
  is not fairly testable and that is the finding.*
- **H1 (headline).** Graph-RKD at its best (normalization, descriptor, objective, N)
  matches or exceeds the classic baselines (RKD-D, RKD-A, combined) in test mAP@R,
  per dataset/teacher. *Direction unknown a priori — do not presuppose.*
- **H2 (normalization).** Minibatch (or hybrid) normalization outperforms per-graph and
  none at N≥3; hybrid recovers both cross-graph scale and teacher/student invariance.
- **H3 (descriptor).** Profile and MDS differ in accuracy, stability, and fidelity in
  characterizable ways; one is preferable in identifiable regimes (order, dataset).
- **H4 (objective).** The contrastive objective is more robust to λg mis-scaling than
  regression (flat mAP@R across a wide λg band vs a sharp regression peak), because
  InfoNCE is scale-normalized.
- **H5 (N=3 vs RKD-A).** At matched arity (order 3), a distance-based graph descriptor
  transfers comparably/better/worse than RKD-A's angle. An RKD-A win is evidence the
  missing ingredient is angular (motivating angular descriptors as future work).

---

## 8. How to analyze results to accept/reject each hypothesis

For every phase, produce a `PHASE_k_REPORT.md` with the relevant table/plot and an
explicit verdict. Use these decision rules (all comparisons use test mAP@R unless
noted; all "beats/differs" claims must clear the noise floor):

- **Noise floor first.** From the multi-seed runs, compute the seed sem for each cell.
  Treat two numbers as indistinguishable if they are within ~1 sem (and note the ~13%
  single-seed run-to-run gap observed historically as a sanity anchor). Never call a
  winner inside the noise band.

- **H0.** Plot val mAP@R vs λg (Phase 1). ACCEPT if some λg gives mAP@R ≥ triplet-only
  floor within noise; the "viable band" is every λg clearing that bar. REJECT (and
  stop) if all λg collapse to ≈ the floor (~0.002 historically) — report as the finding
  and do not proceed to headline claims.

- **H1.** For each (dataset, teacher), rank the 5 students by mean mAP@R with sem.
  ACCEPT "Graph-RKD matches" if its interval overlaps the best baseline's; ACCEPT
  "beats" only if its mean exceeds the best baseline by > 1 sem on that cell AND the
  sign is consistent across teachers. Report per cell — the sign may be
  dataset-dependent (Cars vs CUB), which is itself a finding, not a failure. Do NOT
  average away a sign flip.

- **H2.** For fixed (descriptor, N, dataset, teacher), compare normalization schemes.
  ACCEPT "minibatch/hybrid > per-graph/none" where the gap clears noise. For hybrid
  specifically, check it is ≥ minibatch on accuracy AND shows the teacher/student
  scale-invariance property (e.g. teacher-μ and student-μ track, or descriptor
  distributions align) — report both, not just accuracy.

- **H3.** Build a per-scenario table: profile-vs-MDS mAP@R by (N, dataset, teacher).
  Cross with the instrumentation: does MDS underperform where near-degenerate spectra
  or gradient spikes are frequent? Does profile's advantage shrink at large N (where
  its dimension grows)? ACCEPT a scenario recommendation only where the accuracy gap
  clears noise AND is explained by a measured mechanism (stability/fidelity), not just
  a raw number. Report the collision-probe rates as the fidelity evidence.

- **H4.** Overlay val mAP@R vs λg for regression and contrastive at a fixed active-order
  cell. ACCEPT if regression shows a sharp peak (collapses outside a narrow band) while
  contrastive stays within noise of its own best across a much wider λg range. Quantify:
  width of the λg range keeping mAP@R within 1 sem of that objective's best. Also report
  raw-loss magnitude vs N/descriptor (explains regression's scale sensitivity) and the
  contrastive negative-quality diagnostics (rule out that robustness is just an easy-
  negatives artifact).

- **H5.** At N=3, matched conditions, compare Graph-RKD (profile AND MDS) vs RKD-A.
  Report per (dataset, teacher). If RKD-A > Graph-RKD-N3 beyond noise, conclude the
  angular signal carries information the distance-based descriptor misses → recommend
  angular/ hybrid descriptors as future work. If comparable/worse, the distance
  descriptor already captures the triadic structure.

- **Per-order (feeds H1/H3).** For each N, report convergence (epochs-to-target),
  stability (seed sem, collapse count), peak mAP@R, and cost (per-step time, G(N)).
  Recommend the best-**by-trade-off** N explicitly; it need not be the argmax.

**Global honesty rules for the write-up:**
- Do not claim a positive or negative headline until Phase 5 multi-seed finals exist.
- Every "beats"/"helps"/"hurts" must clear the seed noise floor and be reported per
  cell (dataset × teacher), not averaged across a sign flip.
- Tuning of λg is on validation only; report the λg-sensitivity so the chosen value is
  transparent (the classic baselines use prescribed weights — state this asymmetry).

---

## 9. Deliverables

- `REPO_MAP.md`, `PHASE_1_REPORT.md` … `PHASE_5_REPORT.md`.
- Regenerated tables (headline mAP@R/R-Prec/Recall@K, multi-seed) and figures
  (per-order curves; λg-robustness overlay; normalization ablation bars;
  profile-vs-MDS by scenario).
- A `results.csv` with one row per (student, normalization, descriptor, objective, N,
  dataset, teacher, seed) and all metrics + instrumentation columns, so the paper's
  tables/plots can be regenerated deterministically.
- A short `FINDINGS.md` mapping each hypothesis H0–H5 to its verdict and the evidence.

---

## 10. Execution amendments & findings (2026-07-08)

These record how the plan was **adapted during execution**, and why. They amend
§§5–8 above; where they conflict, these win.

### 10.1 Run infrastructure (backends)
The campaign is enumerated once as independent jobs (`RKD/sm/plan.py`, phases
`0–5` plus cheap iteration phases `dev`/`conv`) and run through any of three
backends that share that plan: a **parallel local runner** (`RKD/sm/run_local.py`
— packs jobs per GPU by free VRAM, round-robins GPUs, 100% utilization), **Modal**
(`RKD/sm/run_modal.py` — rented GPUs, no quota, server-side driver so runs survive
local disconnects), and **AWS SageMaker** (`RKD/sm/launch.py`). *Reason:* AWS
SageMaker GPU quota was denied for a new account (lack of history), so Modal is
used for cheap iteration and a local GPU for the full run.

### 10.2 Gate finding (H0) — reason for the amendments below
On the gate slice (Cars-196 / R18 / profile / regression / minibatch / N=4,
30 ep, 1 seed) Graph-RKD only stayed **at/above the triplet-only floor at very
small λg (≈0.01)**; every λg ≥ 0.1 fell **below** the floor and hurt monotonically
(see `PHASE_1_REPORT.md`). A cheap `dev` design-space probe (descriptor × objective
× normalization at λg=0.01) then showed **`hybrid` normalization is consistently
worst** and **regression is the more consistent objective**. This motivated the
following.

### 10.3 `search_epochs` bumped 30 → 60
The 30-epoch search runs sat near the floor (test mAP@R ≈ 0.007 vs teacher 0.188),
so λg/normalization/descriptor **selection was unreliable**. Search phases (2–4)
now use **60 epochs** for a more discriminative signal. *(Cost trade-off accepted;
the headline phase 5 stays at 120.)*

### 10.4 Seeds unchanged; **report the median-mAP@R model** (amends I5 / §8)
We are **not increasing seeds** for now. Instead of mean ± sem over ≥3 seeds, the
reported number is the **median test mAP@R**, and the reported checkpoint is the
**run at that median** (`analysis_utils.median_run`). *Reason:* median is robust to
the run-to-run noise seen here without the cost of more seeds; §8's noise-floor/
sem rules are superseded by a median comparison (`analysis_utils.h1_verdict`).

### 10.5 Trimmed grid for the full run (amends §5), enabled by `--trimmed`
Derived from the gate + dev findings:
- **normalization** ∈ {per_graph, minibatch, none} — **drop `hybrid`** (worst).
- **λg** ∈ {0.01, 0.1, 1} (3 pts, not 6) — the signal lives at small λg.
- **order N** ∈ {3, 4, 8} — drop 16/17: the offline descriptor probe
  (`RKD/analysis/descriptor_probe.py`) shows MDS spectra are **>90% near-degenerate
  at N=16/17**, so MDS there is numerically fragile and not worth its cost.
- **headline config** = `mds / regression / per_graph / N=4 / λg=0.01` (best in the
  dev probe; to be reconfirmed by the `conv` convergence test).
This cuts the full campaign from **477 → 213 jobs**. With `search_epochs=60`
(§10.3): full ≈ 371 GPU-h vs **trimmed ≈ 196 GPU-h** (~$215 on Modal A10G;
~20 h at Modal's 10-GPU cap; **~2.7 days on a single RTX 5070**, 2–3 jobs packed
in 12 GB). Phase 5 (multi-seed headline, ~104 GPU-h) is unchanged and dominates —
it is the essential result; the trim mostly shrinks the search phases 2–4.

### 10.6 Cheap iteration phases (Modal, within free credits)
- `dev`: descriptor × objective × normalization at small λg, short, reuses the
  teacher (via `--only`) — design-space probe.
- `conv`: floor + top-2 configs at a longer schedule — tests whether Graph-RKD's
  edge **survives convergence** before committing the full local run.
