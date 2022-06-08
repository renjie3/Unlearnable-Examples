#!/bin/bash
cd `dirname $0`
MY_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $MY_JOB_ROOT_PATH

MYTIME="3:59:00"
MYCPU="3"
MYGRES="gpu:v100:1"

# MobileNetV2 DenseNet121 resnet18 resnet50 VGG19

JOB_INFO="samplewise perturbation"
MYCOMMEND="python simclr_transfer.py --batch_size 512 --epochs 1000 --arch resnet18 --perturbation_budget 1 --pre_load_name unlearnable_samplewise_55592549_1_20220606190119_0.5_512_2_checkpoint_perturbation_epoch_20 --train_data_type CIFAR10 --samplewise --pytorch_aug --no_save"
# MYCOMMEND="python supervised_cifar10.py --train_data_type cifar100 --arch resnet18 --pre_load_name unlearnable_samplewise_54777661_1_20220525162248_0.5_512_2_checkpoint_perturbation --samplewise"
# MYCOMMEND="python3 -u ssl_perturbation_v2.py --epochs 200 --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 5 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step 100 --min_min_attack_fn eot_v1 --strong_aug --eot_size 1 --shuffle_train_perturb_data --pytorch_aug --simclr_weight 1 --linear_noise_dbindex_weight 1 --seed 1"

MYCOMMEND2="python simclr_transfer.py --batch_size 512 --epochs 1000 --arch resnet18 --perturbation_budget 1 --pre_load_name unlearnable_samplewise_52260377_1_20220502113459_0.5_512_300_checkpoint_perturbation --train_data_type CIFAR10 --samplewise --pytorch_aug"

MYCOMMEND3="python3 -u ssl_perturbation_v2.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise_dbindex --train_step 10 --epochs 1000 --min_min_attack_fn eot_v1 --class_4 --strong_aug --not_shuffle_train_data --eot_size 10 --dbindex_weight 0.1 --kmeans_index 2"

MYCOMMEND2="No_commend2"
MYCOMMEND3="No_commend3"


# cat ./slurm_files/sconfigs1_cmse.sb > submit.sb
# cat ./slurm_files/sconfigs1_scavenger.sb > submit.sb
cat ./slurm_files/sconfigs1.sb > submit.sb
echo "#SBATCH --time=${MYTIME}             # limit of wall clock time - how long the job will run (same as -t)" >> submit.sb
echo "#SBATCH --cpus-per-task=${MYCPU}           # number of CPUs (or cores) per task (same as -c)" >> submit.sb
echo "#SBATCH --gres=${MYGRES}" >> submit.sb
# echo "#SBATCH --nodelist=nvl-001" >> submit.sb
echo "#SBATCH -o ${MY_JOB_ROOT_PATH}/logfile/%j.log" >> submit.sb
echo "#SBATCH -e ${MY_JOB_ROOT_PATH}/logfile/%j.err" >> submit.sb
cat ./slurm_files/sconfigs2.sb >> submit.sb
echo "JOB_INFO=\"${JOB_INFO}\"" >> submit.sb
echo "MYCOMMEND=\"${MYCOMMEND} --job_id \${SLURM_JOB_ID}_1\"" >> submit.sb
echo "MYCOMMEND2=\"${MYCOMMEND2} --job_id \${SLURM_JOB_ID}_2\"" >> submit.sb
echo "MYCOMMEND3=\"${MYCOMMEND3} --job_id \${SLURM_JOB_ID}_3\"" >> submit.sb
cat ./slurm_files/sconfigs3.sb >> submit.sb
MY_RETURN=`sbatch submit.sb`

echo $MY_RETURN

MY_SLURM_JOB_ID=`echo $MY_RETURN | awk '{print $4}'`

#print the information of a job into one file
date >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MY_SLURM_JOB_ID >>${MY_JOB_ROOT_PATH}/history_job.log
echo $JOB_INFO >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MYCOMMEND >>${MY_JOB_ROOT_PATH}/history_job.log
if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
then
    echo $MYCOMMEND2 >>${MY_JOB_ROOT_PATH}/history_job.log
fi
if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
then
    echo $MYCOMMEND3 >>${MY_JOB_ROOT_PATH}/history_job.log
fi
echo "---------------------------------------------------------------" >>${MY_JOB_ROOT_PATH}/history_job.log
