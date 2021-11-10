#!/bin/bash
cd `dirname $0`
MY_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $MY_JOB_ROOT_PATH

MYTIME="2:00:00"

# JOB_INFO="New small dataset"
# MYCOMMEND="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 4 3 32 32 --epsilon 32 --num_steps 10 --step_size 3.2 --attack_type min-min --perturb_type classwise --universal_train_target 'classwise' --train_step 10 --epochs 1200 --min_min_attack_fn non_eot --strong_aug --model_group 3"

JOB_INFO="Retrain SimCLR to test the transferability."
MYCOMMEND="python -u simclr_transfer.py --batch_size 512 --epochs 1000 --arch resnet18 --perturbation_budget 1 --class_4 --pre_load_name random_noise32_perturbation"

# 32_10_0.8 32_10_1.6 32_1_0.8 8_10_0.8 ./my_experiments/random_noise/perturbation
# JOB_INFO="Retrain SimCLR with perturbation from supervised"
# MYCOMMEND="python simclr_transfer.py --batch_size 512 --epochs 1000 --arch resnet18 --perturbation_budget 32 --class_4 --pre_load_name randomassign_supervised_10class_classwise_8_10_0.8_checkpoint_perturbation"
# MYCOMMEND="python simclr_transfer.py --batch_size 512 --epochs 1000 --arch resnet18 --perturbation_budget 32 --orglabel --class_4 --pre_load_name supervised_orglabel_4class_perturbation"

# JOB_INFO="Retrain SimCLR with perturbation from feature space"
# MYCOMMEND="python3 ssl_perturbation_feature_space.py --pre_load_name differentiable_20211102231654_0.5_200_512 --noise_shape 1024 3 32 32 --epsilon 8 --num_steps 30 --step_size 0.4 --batch_size=512 --model_group 10"
# MYCOMMEND="python simclr_transfer.py --batch_size 512 --epochs 1000 --arch resnet18 --class_4 --perturbation_budget 32 --pre_load_name differentiable_20211102231654_0.5_200_512_32.0_30_0.8_perturb_on_random_initial_model --samplewise"

# JOB_INFO="Retrain SimCLR with perturbation from supervised using larger steps and larger parameters"
# MYCOMMEND="python3 perturbation.py --config_path configs/cifar10 --exp_name my_experiments/class_wise_cifar10_random_assign_32_1_0.8 --version resnet18 --train_data_type CIFAR10 --noise_shape 10 3 32 32 --epsilon 32 --num_steps 1 --step_size 0.8 --attack_type min-min --perturb_type classwise --universal_train_target train_dataset --universal_stop_error 0.1"

# JOB_INFO="random_baseline"
# MYCOMMEND="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 4 3 32 32 --epsilon 32 --num_steps 5 --step_size 3.2 --attack_type min-min --perturb_type classwise --universal_train_target 'classwise' --train_step 10 --epochs 1000 --min_min_attack_fn neg --strong_aug"

# python3 perturbation.py --config_path configs/cifar10 --exp_name my_experiments/class_wise_cifar10_random_assign_8_10_0.8 --version resnet18 --train_data_type CIFAR10 --noise_shape 10 3 32 32 --epsilon 8 --num_steps 10 --step_size 0.8 --attack_type min-min --perturb_type classwise --universal_train_target 'train_dataset' --universal_stop_error 0.1 --use_subset

cat ./slurm_files/sconfigs1_cmse.sb > submit.sb
echo "#SBATCH --time=${MYTIME}             # limit of wall clock time - how long the job will run (same as -t)" >> submit.sb
# echo "#SBATCH --nodelist=nvl-001" >> submit.sb
echo "#SBATCH -o ${MY_JOB_ROOT_PATH}/logfile/%j.log" >> submit.sb
echo "#SBATCH -e ${MY_JOB_ROOT_PATH}/logfile/%j.err" >> submit.sb
cat ./slurm_files/sconfigs2.sb >> submit.sb
echo "JOB_INFO=\"${JOB_INFO}\"" >> submit.sb
echo "MYCOMMEND=\"${MYCOMMEND} --job_id \${SLURM_JOB_ID}\"" >> submit.sb
cat ./slurm_files/sconfigs3.sb >> submit.sb
MY_RETURN=`sbatch submit.sb`

echo $MY_RETURN

MY_SLURM_JOB_ID=`echo $MY_RETURN | awk '{print $4}'`

#print the information of a job into one file
date >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MY_SLURM_JOB_ID >>${MY_JOB_ROOT_PATH}/history_job.log
echo $JOB_INFO >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MYCOMMEND >>${MY_JOB_ROOT_PATH}/history_job.log
echo "---------------------------------------------------------------" >>${MY_JOB_ROOT_PATH}/history_job.log
