#!/usr/bin/env bash

WANDB_PROJECT="rkd-metric-learning"
WANDB_MODE="online"
WANDB_GROUP_TEACHER="stanford-teacher"
WANDB_GROUP_DISTILL="stanford-distill"

# Teacher
python run.py --dataset stanford --epochs 40 --lr_decay_epochs 25 30 35 --iter_per_epoch 1000  --lr_decay_gamma 0.5 --batch 128\
              --base resnet50 --embedding_size 512 --sample distance --margin 0.2 --save_dir stanford_resnet50_512 \
              --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" --wandb_group "$WANDB_GROUP_TEACHER" --wandb_run_name stanford_teacher_resnet50_512

# Student with small embedding
python run_distill.py --dataset stanford --epochs 120 --iter_per_epoch 500 --lr_decay_epochs 40 80 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 64 --l2normalize false --dist_ratio 1 --angle_ratio 2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load stanford_resnet50_512/best.pth \
                      --save_dir stanford_student_resnet18_64 --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
                      --wandb_group "$WANDB_GROUP_DISTILL" --wandb_run_name stanford_student_resnet18_64

# Student with small embedding (Quadruplet-only)
python run_distill.py --dataset stanford --epochs 120 --iter_per_epoch 500 --lr_decay_epochs 40 80 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 64 --l2normalize false --quad_ratio 3 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load stanford_resnet50_512/best.pth \
                      --save_dir stanford_student_resnet18_64_quad --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
                      --wandb_group "$WANDB_GROUP_DISTILL" --wandb_run_name stanford_student_resnet18_64_quad

# Self-Distillation
python run_distill.py --dataset stanford --epochs 120 --iter_per_epoch 500 --lr_decay_epochs 40 80 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load stanford_resnet50_512/best.pth \
                      --save_dir stanford_student_resnet50_512 --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
                      --wandb_group "$WANDB_GROUP_DISTILL" --wandb_run_name stanford_student_resnet50_512

# Self-Distillation (Quadruplet-only)
python run_distill.py --dataset stanford --epochs 120 --iter_per_epoch 500 --lr_decay_epochs 40 80 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet50 --embedding_size 512 --l2normalize false --quad_ratio 3 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load stanford_resnet50_512/best.pth \
                      --save_dir stanford_student_resnet50_512_quad --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
                      --wandb_group "$WANDB_GROUP_DISTILL" --wandb_run_name stanford_student_resnet50_512_quad
