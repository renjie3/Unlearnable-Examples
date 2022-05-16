#!/bin/bash
cd `dirname $0`
MY_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $MY_JOB_ROOT_PATH

MYTIME="30:50:00"
MYCPU="5"
MYGRES="gpu:v100:1"

# JOB_INFO="noise_ave_value"
# MYCOMMEND="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 4 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type classwise --universal_train_target 'classwise' --train_step 10 --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4"

# 1.0 0.5 0.8 0.2

# unlearnable_samplewise_51030219_1_20220409114042_0.5_512_1000_checkpoint_perturbation
# unlearnable_samplewise_51073519_1_20220410221405_0.5_512_1000perturbation

# model_zoo = {'VGG19': VGG,
# 'resnet18': ResNet18,
# 'PreActResNet18': PreActResNet18,
# 'GoogLeNet': GoogLeNet,
# 'DenseNet121': DenseNet121,
# 'ResNeXt29_2x64d': ResNeXt29_2x64d,
# 'MobileNet': MobileNet,
# 'MobileNetV2': MobileNetV2,
# 'DPN92': DPN92,
# 'SENet18': SENet18,
# 'EfficientNetB0': EfficientNetB0,
# 'RegNetX_200MF': RegNetX_200MF,
# 'simpledla': SimpleDLA}

# 52865694 unlearnable_samplewise_52865694_1_20220507200052_0.5_512_2_checkpoint_perturbation
# 52866132 unlearnable_samplewise_52866132_1_20220507200522_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866270 unlearnable_samplewise_52866270_1_20220507200522_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866443 unlearnable_samplewise_52866443_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866444 unlearnable_samplewise_52866444_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866445 unlearnable_samplewise_52866445_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866446 unlearnable_samplewise_52866446_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866447 unlearnable_samplewise_52866447_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866448 unlearnable_samplewise_52866448_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866449 unlearnable_samplewise_52866449_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866450 unlearnable_samplewise_52866450_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866563 unlearnable_samplewise_52866563_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866589 unlearnable_samplewise_52866589_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866590 unlearnable_samplewise_52866590_1_20220507200823_0.5_512_2_checkpoint_perturbation_epoch_10

# 52802192
# 52802195
# 52802196
# 52802236
# 52802261

JOB_INFO="samplewise perturbation"
# MYCOMMEND="python3 -u ssl_perturbation_v2_byol.py --epochs 100 --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step 20 --min_min_attack_fn eot_v1 --strong_aug --eot_size 1 --shuffle_train_perturb_data --pytorch_aug --simclr_weight 1 --cl_algorithm byol"
# MYCOMMEND="python supervised_cifar10.py --train_data_type cifar10 --arch resnet18 --pre_load_name unlearnable_samplewise_53657706_1_20220513184955_0.5_512_2_checkpoint_perturbation --samplewise"
MYCOMMEND="python moco_transfer.py --batch_size 512 --epochs 1000 --pre_load_name unlearnable_samplewise_52260407_1_20220502114610_0.5_512_300_checkpoint_perturbation_epoch_40 --samplewise"

MYCOMMEND2="python simclr_transfer.py --batch_size 512 --epochs 1000 --arch resnet11 --perturbation_budget 1 --pre_load_name unlearnable_samplewise_52260377_1_20220502113459_0.5_512_300_checkpoint_perturbation --train_data_type CIFAR10 --samplewise --pytorch_aug"

MYCOMMEND3="python3 -u ssl_perturbation_v2.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise_dbindex --train_step 10 --epochs 1000 --min_min_attack_fn eot_v1 --class_4 --strong_aug --not_shuffle_train_data --eot_size 10 --dbindex_weight 0.1 --kmeans_index 2"

MYCOMMEND2="No_commend2"
MYCOMMEND3="No_commend3"


cat ./slurm_files/sconfigs1_cmse.sb > submit.sb
# cat ./slurm_files/sconfigs1_scavenger.sb > submit.sb
# cat ./slurm_files/sconfigs1.sb > submit.sb
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
