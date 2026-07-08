# Graph-RKD — Relational Knowledge Distillation for Metric Learning (MO434)

Graph-RKD generalizes Relational KD: instead of matching pairwise distances
(RKD-D) or triplet angles (RKD-A), it treats each minibatch as several complete
weighted graphs (one per sampled node set of size `N`), summarizes each with a
permutation-invariant descriptor (**profile** or **MDS**), and matches those
descriptors between teacher and student — as a relational term added to a triplet
task loss. The central question, and the full experimental program, is defined in
[`EXPERIMENTS_EN.md`](EXPERIMENTS_EN.md).

**Scope: metric learning only** (image retrieval on Cars-196 / CUB-200,
mAP@R primary). This repo has been trimmed to exactly that campaign.

## What gets trained

Teacher (ResNet-18 / ConvNeXt-Tiny embedding, triplet) → **ConvNextMicro** student.
The 5 students compared (all add their relational term to the same triplet loss):

1. triplet-only (floor) · 2. +RKD-D (25) · 3. +RKD-A (50) ·
4. +RKD-D+RKD-A (1:2) · 5. **+Graph-RKD** (swept over normalization ∈
{per_graph, minibatch, none, hybrid}, descriptor ∈ {profile, mds}, objective ∈
{regression, contrastive}, order N ∈ {3,4,8,16,17}, and λg).

Selection is on **validation mAP@R**; metrics are mAP@R, R-Precision, Recall@K.

## Repo layout

```
RKD/
  distill_metric.py        # student distillation (triplet + relational term)
  finetune_metric.py       # teacher (embedding + triplet)
  train_metric_baseline.py # triplet-only student (floor)
  metric_common.py         # loaders, retrieval metrics (mAP@R, R-Prec, Recall@K)
  metric/                  # triplet loss, RKD-D/RKD-A, samplers
  graph_rkd/               # graph descriptors (profile/mds), regression & InfoNCE losses
  model/                   # ConvNextMicro student
  dataset/                 # Cars-196, CUB-200 (metric splits)
  teacher_models.py        # ResNet-18 / ConvNeXt-Tiny teachers (torchvision)
  sm/                      # run the campaign: local GPU or AWS SageMaker
EXPERIMENTS_EN.md          # the experimental spec (hypotheses, phases, analysis)
```

## Data

Place under `data/` (a `../data → RKD/data` symlink exists):

```
data/Cars196/{car_ims/, cars_annos.mat}
data/CUB_200_2011/{images/, images.txt, ...}
```

CUB downloads automatically (Caltech Data mirror). Stanford's Cars source is
offline — obtain it separately; the loader URL is env-overridable
(`CARS196_IMG_URL` / `CARS196_ANNO_URL`).

## Running the experiments

Everything logs to **W&B `gabomfim-unicamp/graph-rkd`**. Full details in
[`RKD/sm/README.md`](RKD/sm/README.md). The campaign is **staged** (per
`EXPERIMENTS_EN.md` §3): `teachers phase0 phase1` is the gate — run it and check
H0 before scaling to phases 2–5.

Datasets live in S3 as archives (`s3://graph-rkd-832271495954/graph-rkd/data/
{Cars196.tar,CUB_200_2011.tgz}`); every backend pulls the per-dataset archive and
extracts it (`sm/data_prep.py`), so no manual data placement is needed.

### Option A — local GPU (recommended now)
Packs many experiments per GPU in parallel (auto-sized to free VRAM),
round-robins across GPUs; pulls datasets from S3 on first run.
```bash
uv sync
export WANDB_API_KEY=...
./RKD/sm/run_local.sh                    # gate; auto parallelism; pulls data from S3
MAX_PARALLEL=6 PHASES="phase5" ./RKD/sm/run_local.sh
# offline (data already extracted locally): add --data-s3 ''
```

### Option A′ — Modal (rented GPUs, no AWS quota) ⭐
Runs jobs in parallel on Modal GPUs, per-second billing, no quota.
```bash
uv sync                                   # includes modal
modal setup
modal secret create wandb WANDB_API_KEY=...
modal secret create aws AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=us-east-1
modal run RKD/sm/run_modal.py --phases "teachers phase0 phase1"
```

### Option B — AWS SageMaker (parallel cloud jobs)
Needs g5 quota approved + an execution role. Dry-run unless `LAUNCH=1`.
```bash
pip install sagemaker boto3
aws s3 sync data/ s3://graph-rkd-832271495954/graph-rkd/data/
export WANDB_API_KEY=... ROLE_ARN=arn:aws:iam::832271495954:role/<role>
LAUNCH=1 ROLE_ARN=$ROLE_ARN ./RKD/sm/run_sagemaker.sh
```

### Option B′ — SageMaker Local Mode (no quota)
Runs the SageMaker container on a local GPU via Docker:
```bash
export WANDB_API_KEY=...
python RKD/sm/launch.py --launch --local --phases teachers phase0 phase1 \
    --data-local data --wandb-entity gabomfim-unicamp
```

## Setup

See [`SETUP.md`](SETUP.md). Python ≥3.10, managed with `uv` (`uv sync`).
