#!/usr/bin/env bash

WANDB_PROJECT="rkd-metric-learning"
WANDB_MODE="online"
WANDB_GROUP_TEACHER="cub200-teacher"
WANDB_GROUP_DISTILL="cub200-distill"

# Teacher Network
python run.py --dataset cub200 --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 128\
              --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --save_dir cub200_resnet50_512 \
              --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" --wandb_group "$WANDB_GROUP_TEACHER" --wandb_run_name cub200_teacher_resnet50_512

# Student with small embedding
python run_distill.py --dataset cub200 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1  --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize false --dist_ratio 1 --angle_ratio 2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cub200_resnet50_512/best.pth \
                      --save_dir cub200_student_resnet18_128 --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
                      --wandb_group "$WANDB_GROUP_DISTILL" --wandb_run_name cub200_student_resnet18_128

# Student with small embedding (Quadruplet-only)
python run_distill.py --dataset cub200 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1  --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize false --quad_ratio 3 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cub200_resnet50_512/best.pth \
                      --save_dir cub200_student_resnet18_128_quad --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
                      --wandb_group "$WANDB_GROUP_DISTILL" --wandb_run_name cub200_student_resnet18_128_quad

# Self-Distillation (batch 64: ResNet50 student + ResNet50 teacher nao cabem em 128 na RTX 5070 12GB -> OOM)
python run_distill.py --dataset cub200 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1  --batch 64\
                      --base resnet50 --embedding_size 512 --l2normalize false --dist_ratio 1 --angle_ratio 2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cub200_resnet50_512/best.pth \
                      --save_dir cub200_student_resnet50_512 --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
                      --wandb_group "$WANDB_GROUP_DISTILL" --wandb_run_name cub200_student_resnet50_512

# Self-Distillation (Quadruplet-only) (batch 64: idem, evita OOM com teacher+student ResNet50)
python run_distill.py --dataset cub200 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1  --batch 64\
                      --base resnet50 --embedding_size 512 --l2normalize false --quad_ratio 3 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cub200_resnet50_512/best.pth \
                      --save_dir cub200_student_resnet50_512_quad --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" \
                      --wandb_group "$WANDB_GROUP_DISTILL" --wandb_run_name cub200_student_resnet50_512_quad
