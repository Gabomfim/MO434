# REPO_MAP — where each concept lives

Concept → code map for the Graph-RKD metric-learning campaign (the Phase-0
deliverable of `EXPERIMENTS_EN.md` §0). Everything is under `RKD/`.

## Training entry points
| Concept | File | Notes |
|---|---|---|
| Teacher fine-tune (embedding + triplet) | `finetune_metric.py` | `run_with_params`; logs `metric-<arch>-<dataset>:best` W&B artifact |
| Student baseline (triplet-only, floor) | `train_metric_baseline.py` | student #1 |
| Student distillation (triplet + relational term) | `distill_metric.py` | students #2–#5; RKD-D/RKD-A/Graph-RKD via ratios/flags |
| Shared loaders + metrics | `metric_common.py` | disjoint-class splits, `retrieval_metrics` (mAP@R, R-Prec, Recall@K), `score_of` |
| Teacher backbones | `teacher_models.py` | ResNet-18 / ConvNeXt-Tiny (torchvision), `load_teacher` |
| Student architecture | `model/convnext_micro.py` | ConvNextMicro (~0.67M params) |
| Resume ledger / W&B artifacts | `experiment_ledger.py`, `wandb_artifacts.py` | |

## The 5 students (EXPERIMENTS_EN §4)
Defined by loss weights/flags in `distill_metric.py`, enumerated in `sm/plan.py`
(`CLASSIC` + `_graph_spec`): triplet-only · +RKD-D (25) · +RKD-A (50) ·
+RKD-D+RKD-A (1:2) · +Graph-RKD.

## Graph-RKD design axes (§5)
| Axis | Flag (`distill_metric.py`) | Implementation |
|---|---|---|
| Objective | `--graph_rkd_mode {regression,contrastive}` | `graph_rkd/loss.py` (Minkowski), `graph_rkd/contrastive.py` (InfoNCE) |
| Descriptor | `--graph_rkd_method {profile,mds}` | `graph_rkd/embeddings.py` |
| Normalization | `--graph_rkd_norm {per_graph,minibatch,none,hybrid}` | `graph_rkd/embeddings.py` (`_apply_norm`, `batch_distance_mean`, `zscore_descriptor`), mapped by `loss.norm_flags` |
| Order N | `--graph_rkd_nodes` | sampling in `graph_rkd/loss.py` (`sample_graphs`), sizing in `graph_rkd/node_search.py` |
| λg | `--graph_rkd_ratio` | applied with warm-up in `distill_metric.py` (`rel_scale_at`, I6) |

## Classic RKD recipe (I2)
`metric/loss.py`: `RkdDistance` (μ-norm + Huber), `RKdAngle` (Huber);
triplet in `metric/loss.py` (`L2Triplet`) with samplers in `metric/pairsampler.py`,
class-balanced batches in `metric/batchsampler.py`.

## Instrumentation (§6)
- Per-term raw+weighted losses + `train/step_time_ms`: `distill_metric.py` (I7).
- Descriptor fidelity/stability (offline, no training): `analysis/descriptor_probe.py`
  → `analysis/descriptor_probe.csv` (collision, MDS eigengap, profile ties).

## Datasets (§5) & data flow
- Loaders: `dataset/cars196.py`, `dataset/cub200.py` (canonical `car_ims/+cars_annos.mat`
  and `CUB_200_2011/images.txt`; CUB URL = Caltech Data mirror; Cars URL env-overridable).
- S3 archives: `s3://graph-rkd-832271495954/graph-rkd/data/{Cars196.tar,CUB_200_2011.tgz}`.
- Stage to S3: `sm/stage_data.py`. Pull S3→local in jobs: `sm/data_prep.py`.

## Running the campaign (backends, all share `sm/plan.py`)
| Backend | File | Quota? |
|---|---|---|
| Local GPU (parallel) | `sm/run_local.py` / `run_local.sh` | none |
| Modal (rented GPUs) | `sm/run_modal.py` | none |
| AWS SageMaker | `sm/launch.py` / `run_sagemaker.sh` (+`entry.py`) | needs g5 quota |
| SageMaker local mode | `sm/launch.py --local` | none (Docker) |

Phases 0–5 (`sm/plan.py`): 0 smoke · 1 λg gate (H0) · 2 normalization (H2) ·
3 descriptor (H3) · 4 objective (H4) · 5 headline multi-seed (H1, H5).

## Analysis → paper §7 (`RKD/analysis/`)
Notebooks `00`–`06` (see `analysis/README.md`) consume W&B runs via
`analysis_utils.py` and produce the tables/figures for H0–H5.
