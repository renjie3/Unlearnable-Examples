#!/bin/bash
cd `dirname $0`
MY_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $MY_JOB_ROOT_PATH

JOB_INFO="New small dataset"

MYCOMMEND="python3 ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 4 3 32 32 --epsilon 8 --num_steps 1 --step_size 1.6 --attack_type min-min --perturb_type classwise --universal_train_target 'classwise' --train_step 1 --epochs 1000 --min_min_attack_fn neg --strong_aug"

# JOB_INFO="Retrain SimCLR to test the transferability."

# MYCOMMEND="python simclr_transfer.py --batch_size 512 --epochs 1000 --arch resnet18"

cat ./slurm_files/sconfigs1.sb > submit.sb
echo "JOB_INFO=\"${JOB_INFO}\"" >> submit.sb
echo "MYCOMMEND=\"${MYCOMMEND}\"" >> submit.sb
cat ./slurm_files/sconfigs2.sb >> submit.sb
MY_RETURN=`sbatch submit.sb`

echo $MY_RETURN

MY_SLURM_JOB_ID=`echo $MY_RETURN | awk '{print $4}'`

#print the information of a job into one file
date >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MY_SLURM_JOB_ID >>${MY_JOB_ROOT_PATH}/history_job.log
echo $JOB_INFO >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MYCOMMEND >>${MY_JOB_ROOT_PATH}/history_job.log
echo "---------------------------------------------------------------" >>${MY_JOB_ROOT_PATH}/history_job.log
