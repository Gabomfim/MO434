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

When enabled, scripts log:

* Hyperparameters/config values.
* Epoch metrics (loss, recall, learning rate, and distillation loss components).
* Dataset metadata artifact.
* Model checkpoint artifacts (`best`, `last`) when `--save_dir` is set.


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
