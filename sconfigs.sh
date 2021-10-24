#!/bin/bash

JOB_INFO="Train unlearnable simclr with differentiable data augmentation. Larger step 3.2"

MYCOMMEND="python3 ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 10 3 32 32 --epsilon 16 --num_steps 1 --step_size 4 --attack_type min-min --perturb_type classwise --universal_train_target 'classwise' --train_step 10 --epochs 151"