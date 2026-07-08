# Graph-RKD on AWS SageMaker (parallel campaign)

Runs the `EXPERIMENTS_EN.md` program as **independent SageMaker training jobs,
one per experiment, in parallel**, each logging to **your** W&B project.

- `plan.py` — pure enumeration: config → list of `JobSpec` (one per experiment),
  organized by the phases of `EXPERIMENTS_EN.md` §3. No AWS, no torch.
- `entry.py` — runs *inside* each SageMaker job: resolves data + checkpoint dirs
  from the SageMaker env and dispatches to the right trainer
  (`finetune_metric` / `train_metric_baseline` / `distill_metric`).
- `launch.py` — turns the plan into parallel SageMaker jobs. **Dry-run by
  default**; only `--launch` touches AWS (and costs money).

The container is the AWS-managed **PyTorch DLC** (no Docker build); extra deps
come from `RKD/requirements.txt`.

---

## Two ways to run

Both run the **same plan** (`plan.py`) and log to **`gabomfim-unicamp-org/graph-rkd`**.

### Option A — local GPU (no AWS) — `sm/run_local.sh`
For whoever has a GPU + the datasets. Runs jobs sequentially, resumable.
```bash
uv sync                                    # torch/torchvision/wandb/kagglehub
export WANDB_API_KEY=...                    # or: wandb login
# data/ must have Cars196/{car_ims/,cars_annos.mat} and CUB_200_2011/{images/,images.txt}
./sm/run_local.sh                          # runs the gate: teachers phase0 phase1
# variants:
DATA=/path/to/data ./sm/run_local.sh
PHASES="phase5" ./sm/run_local.sh
./sm/run_local.sh --gate-dataset cub200    # gate on CUB instead of Cars
```

### Option B — AWS SageMaker (parallel jobs) — `sm/run_sagemaker.sh`
Needs g5 quota approved + a SageMaker execution role. **Dry-run unless `LAUNCH=1`.**
```bash
pip install sagemaker boto3
aws s3 sync data/ s3://graph-rkd-832271495954/graph-rkd/data/   # once
export WANDB_API_KEY=... ROLE_ARN=arn:aws:iam::832271495954:role/<role>
ROLE_ARN=$ROLE_ARN ./sm/run_sagemaker.sh                # dry-run (prints plan)
LAUNCH=1 ROLE_ARN=$ROLE_ARN ./sm/run_sagemaker.sh       # actually launch
```

Region defaults to `us-east-1`, bucket to `graph-rkd-832271495954`, profile to
`gabomfim` — override via `REGION=` / `BUCKET=` / `AWS_PROFILE=`.

The rest of this file documents Option B (SageMaker) in detail.

---

## 1. Prerequisites (one-time)

You must supply four things (flags or env vars):

| What | Flag | Env var | Notes |
|---|---|---|---|
| AWS region | `--region` | `AWS_REGION` | e.g. `us-east-1` |
| SageMaker execution role ARN | `--role` | `SAGEMAKER_ROLE_ARN` | needs S3 + SageMaker + (optional) ECR read |
| S3 bucket | `--bucket` | `SAGEMAKER_BUCKET` | code, checkpoints, outputs land here |
| W&B API key | — | `WANDB_API_KEY` | **required** to log/pull artifacts |
| W&B entity | `--wandb-entity` | `WANDB_ENTITY` | your username/team (optional but recommended) |

Authenticate first (this repo uses the AWS `wehandle` profile — run the
`aws-auth` skill or `aws sso login --profile wehandle`), then confirm access:

```bash
aws sts get-caller-identity
python sm/discover_aws.py         # prints candidate role / bucket / region
```

## 2. Dry-run (no AWS, safe)

```bash
python sm/launch.py --phases teachers phase0 phase1 \
    --wandb-entity YOUR_ENTITY --wandb-project graph-rkd
```

Prints the full job list + counts and writes `plan.json`. Inspect it before
launching. Try `--phases phase5` to see the full multi-seed headline grid size.

## 3. Data staging (recommended)

The datasets (Stanford Cars, CUB-200) auto-download via torchvision, but the
upstream URLs are flaky and re-downloading in every parallel job is wasteful.
Stage **once** to S3 and pass it as the `data` channel:

```
s3://<bucket>/graph-rkd/data/
    Cars196/cars_annos.mat + car_ims/...
    CUB_200_2011/images.txt + images/...
```

Then add `--data-s3 s3://<bucket>/graph-rkd/data/` (or set `GRAPH_RKD_DATA_S3`).
Without it, each job downloads on first use.

## 4. Launch

```bash
export WANDB_API_KEY=...        # required
python sm/launch.py --launch \
    --phases teachers phase0 \
    --region us-east-1 \
    --role   arn:aws:iam::<acct>:role/<SageMakerRole> \
    --bucket <your-bucket> \
    --data-s3 s3://<your-bucket>/graph-rkd/data/ \
    --wandb-entity YOUR_ENTITY --wandb-project graph-rkd \
    --instance-type ml.g5.xlarge
```

Waves: **teachers + baselines** launch together; the launcher waits only for the
**teachers** to finish, then launches all **distillation** jobs in parallel.
Distill jobs pull their teacher via the W&B artifact `metric-<arch>-<dataset>:best`.

## 5. Staged execution (matches `EXPERIMENTS_EN.md` §3)

Run phases in order — each gates the next:

1. `teachers phase0` — smoke test (2 epochs): pipeline trains, evaluates, logs
   per-term losses, pulls the teacher.
2. `phase1` — the **λg viability gate (H0)**. Sweeps λg on one cheap slice
   (Cars-196 / ResNet-18 / profile / regression / minibatch / N=4) + the
   triplet-only floor. In W&B, plot `val mAP@R` vs λg. **Stop if all λg collapse
   to the floor.**
3. `phase2` — normalization ablation (H2). `phase3` — descriptor (H3).
   `phase4` — objective robustness (H4).
4. Set the chosen headline config in `plan.py` `DEFAULTS`
   (`headline_norm/method/objective/nodes/lambda`) from the sweep results, then
   run `phase5` — the 5 students, full budget, ≥3 seeds, both datasets/teachers.

## 6. Resuming / re-launching

Job names get a unique suffix (SageMaker forbids name reuse), but each logical
experiment has a **stable** `wandb_id` (resumes the same W&B run) and a stable
`checkpoint_s3_uri` (resumes training). Re-running a phase continues where it
left off. Use `--skip-teachers` to go straight to distillation when the teacher
artifacts already exist.

## 7. Quotas

Launching all phases can create hundreds of jobs. Check your account's
*"training jobs"* and *per-instance-type* quotas; SageMaker errors on
`ResourceLimitExceeded` rather than queueing. Launch phase-by-phase to stay
within limits.

## Not yet automated (do after the gate passes)

- **λg selection**: phases 1–4 emit the λg grid as separate jobs; picking the
  winner (val mAP@R) and writing it into the headline config is still manual /
  a follow-up aggregation script.
- **`results.csv` + `PHASE_k_REPORT.md`** aggregation from W&B, and the §6
  stability/fidelity instrumentation (MDS eigen-gap, collision probe, contrastive
  negative-quality) — tracked as next steps.
