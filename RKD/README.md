# Relational Knowledge Distillation

Official implementation of [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068?context=cs.LG), CVPR 2019\
This repository contains source code of experiments for metric learning.


## Quick Start

```bash
python run.py --help    
python run_distill.py --help

# Use config-style scripts (recommended)
bash examples/run_config.sh train
bash examples/run_config.sh eval
bash examples/run_distill_config.sh

# W&B logging (optional)
# Use --wandb_mode disabled to run without logging.
python run.py --mode train \
              --dataset cub200 \
              --base resnet50 \
              --save_dir teacher \
              --wandb_project rkd-metric-learning \
              --wandb_run_name teacher-resnet50 \
              --wandb_mode online

python run_distill.py --dataset cub200 \
                      --base resnet18 \
                      --teacher_base resnet50 \
                      --teacher_load teacher/best.pth \
                      --quad_ratio 1 \
                      --save_dir student \
                      --wandb_project rkd-metric-learning \
                      --wandb_run_name distill-resnet18 \
                      --wandb_mode online

# W&B flags available in both scripts:
#   --wandb_project, --wandb_entity, --wandb_run_name, --wandb_mode

# Train a teacher embedding network of resnet50 (d=512)
# using triplet loss (margin=0.2) with distance weighted sampling.
python run.py --mode train \ 
               --dataset cub200 \
               --base resnet50 \
               --sample distance \ 
               --margin 0.2 \ 
               --embedding_size 512 \
               --save_dir teacher

# Evaluate the teacher embedding network
python run.py --mode eval \ 
               --dataset cub200 \
               --base resnet50 \
               --embedding_size 512 \
               --load teacher/best.pth 

# Distill the teacher to student embedding network
python run_distill.py --dataset cub200 \
                      --base resnet18 \
                      --embedding_size 64 \
                      --l2normalize false \
                      --teacher_base resnet50 \
                      --teacher_embedding_size 512 \
                      --teacher_load teacher/best.pth \
                      --dist_ratio 1  \
                      --angle_ratio 2 \
                      --save_dir student
                      
# Distill the trained model to student network
python run.py --mode eval \ 
               --dataset cub200 \
               --base resnet18 \
               --l2normalize false \
               --embedding_size 64 \
               --load student/best.pth 
            
```

## Repository Files

* `run.py`: Main teacher script for metric learning. Supports training and evaluation with `--mode train|eval`, saves checkpoints (`best.pth`, `last.pth`), and logs metrics/artifacts to W&B.
* `run_distill.py`: Student distillation script. Trains a student from a teacher checkpoint using RKD losses (distance/angle and optional auxiliary losses), saves checkpoints, and logs metrics/artifacts to W&B.
* `examples/run_config.sh`: Config-style wrapper for `run.py` with centralized variables for dataset/model/training/W&B.
* `examples/run_distill_config.sh`: Config-style wrapper for `run_distill.py` with centralized variables for teacher/student/distillation/W&B.
* `examples/`: Example launcher scripts with reproducible hyperparameter presets.
* `data/` (created at runtime): Dataset download/cache directory used by `--data`.
* `teacher/` and `student/` (created when `--save_dir` is used): Output directories containing checkpoints and `result.txt`.

## What To Run First

1. Train or evaluate the teacher model with `run.py` (or `examples/run_config.sh`).
2. Distill the student with `run_distill.py` (or `examples/run_distill_config.sh`) using the teacher checkpoint (for example `teacher/best.pth`).
3. Evaluate the student with `run.py --mode eval` using `student/best.pth`.

Quick command order:

```bash
bash examples/run_config.sh train
bash examples/run_config.sh eval
bash examples/run_distill_config.sh
python run.py --mode eval --dataset cub200 --base resnet18 --embedding_size 64 --l2normalize false --load student/best.pth
```

### W&B Logging in Scripts

Both `run.py` and `run_distill.py` support:

* `--wandb_project`: Project name.
* `--wandb_entity`: Team/user namespace (optional).
* `--wandb_run_name`: Explicit run name (optional).
* `--wandb_mode`: `online`, `offline`, or `disabled`.

Distillation-specific extra loss weight:

* `--quad_ratio`: weight for 4-sample-set relational distance-sum matching loss.

When enabled, scripts log:

* Hyperparameters/config values.
* Epoch metrics (loss, recall, learning rate, and distillation loss components).
* Dataset metadata artifact.
* Model checkpoint artifacts (`best`, `last`) when `--save_dir` is set.

## W&B Report Template (Distillation Comparison)

Use this template to compare multiple distillation runs in a single report.

### 1) Organize Runs for Comparison

Use a common project and group for related runs.

```bash
python run_distill.py ... \
  --wandb_project rkd-metric-learning \
  --wandb_group distillation-experiments \
  --wandb_run_name distill-r18-a2-d1-q0
```

Recommended naming pattern:

* `distill-<student>-a<angle_ratio>-d<dist_ratio>-q<quad_ratio>-dark<dark_ratio>-seed<seed>`

### 2) Create Panels in a W&B Report

Add a run set filter:

* `group = distillation-experiments`

Add these line charts (x-axis = `epoch`):

* `eval/test_recall1`, `eval/test_recall2`, `eval/test_recall4`, `eval/test_recall8`
* `eval/train_recall1`, `eval/train_recall2`, `eval/train_recall4`, `eval/train_recall8`
* `train/loss`
* `train/dist_loss`, `train/angle_loss`, `train/quad_loss`, `train/dark_loss`, `train/triplet_loss`, `train/at_loss`
* `lr`

Add summary panels:

* `best_test_recall_primary`
* `final_test_recall_primary`
* `best_test_recall1`, `best_test_recall2`, `best_test_recall4`, `best_test_recall8`

Add comparison tables:

* Run table with columns: `name`, `group`, `config.dist_ratio`, `config.angle_ratio`, `config.quad_ratio`, `config.dark_ratio`, `summary.best_test_recall_primary`, `summary.final_test_recall_primary`.
* Artifact table for `metrics` artifacts (`distill-metrics-<run_id>`) to download and compare CSV histories.

### 3) Use Logged Artifacts

Each distillation run logs:

* `metrics/epoch_table` (in-run table with per-epoch metrics).
* `distill_epoch_metrics.csv` as a W&B `metrics` artifact.
* `best` and `last` model artifacts.
* Confusion matrix plots (`eval/test/confusion_matrix`, `best/test/confusion_matrix`) when class/sample limits allow.

### 4) Suggested Comparison Workflow

1. Run a baseline (`quad_ratio=0`) and at least one variant (`quad_ratio>0`).
2. Keep all other hyperparameters fixed except the one being tested.
3. Compare `best_test_recall_primary` first, then inspect loss-component curves.
4. Use confusion matrices to diagnose class-level behavior changes.
5. Export `distill_epoch_metrics.csv` artifacts for external analysis if needed.

## W&B Report Template (Teacher Comparison)

Use this template to compare teacher-only runs (backbone, loss, sampling, and margin choices).

### 1) Organize Runs

Use a dedicated group for teacher experiments:

```bash
python run.py --mode train ... \
  --wandb_project rkd-metric-learning \
  --wandb_group teacher-runs \
  --wandb_run_name teacher-r50-distance-l2triplet
```

Recommended naming pattern:

* `teacher-<backbone>-<sample>-<loss>-m<margin>-seed<seed>`

### 2) Create Panels in W&B

Add a run set filter:

* `group = teacher-runs`

Add these line charts (x-axis = `epoch`):

* `train/loss`
* `eval/train_recall@1`, `eval/train_recall@2`, `eval/train_recall@4`, `eval/train_recall@8`
* `eval/test_recall@1`, `eval/test_recall@2`, `eval/test_recall@4`, `eval/test_recall@8`
* `best_recall@1`
* `lr`

Add summary panels:

* `best_recall@1`
* `final_recall@1`
* `final_recall@2`, `final_recall@4`, `final_recall@8` (if logged via `--recall`)

Add comparison tables:

* Run table with columns: `name`, `config.base`, `config.sample`, `config.loss`, `config.margin`, `summary.best_recall@1`, `summary.final_recall@1`.
* Artifact table for teacher metric artifacts (`teacher-history-<base>-<seed>`).

### 3) Use Teacher Artifacts

Each teacher run logs:

* `artifacts/epoch_history_table`.
* `teacher_epoch_history.csv` as a W&B `metrics` artifact.
* `best` and `last` model artifacts.
* Final/eval confusion matrix when class count is within limits.

### 4) Suggested Teacher Workflow

1. Choose one baseline backbone and loss/sampler setup.
2. Change one factor at a time (for example, sampler or margin).
3. Compare `best_recall@1` first, then inspect recall curves and training loss.
4. Use confusion matrix differences to inspect class-level retrieval behavior.
5. Keep top teacher checkpoint as the fixed teacher for distillation sweeps.


##  Dependency

* Python 3.6
* Pytorch 1.0
* tqdm (pip install tqdm)
* h5py (pip install h5py)
* scipy (pip install scipy)
* wandb (pip install wandb)

### Note
* Hyper-parameters that used for experiments in the paper are specified at scripts in ```examples/```.
* Heavy teacher network (ResNet50 w/ 512 dimension) requires more than 12GB of GPU memory if batch size is 128.  
  Thus, you might have to reduce the batch size. (The experiments in the paper were conducted on P40 with 24GB of gpu memory. 
)

## Citation
In case of using this source code for your research, please cite our paper.

```
@inproceedings{park2019relational,
  title={Relational Knowledge Distillation},
  author={Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3967--3976},
  year={2019}
}
```
