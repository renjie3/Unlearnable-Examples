#!/bin/bash
cd `dirname $0`
cd ..
PIERMARO_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $PIERMARO_JOB_ROOT_PATH
DATE_NAME=${1}
echo $$

WHOLE_EPOCH=30
SINGLE_EPOCH=3
REJOB_TIMES=`expr $WHOLE_EPOCH / $SINGLE_EPOCH`
MYGRES="gpu:v100s:4"

JOB_INFO="cifar10 baseline"
# MYCOMMEND="python main.py --batch_size 512 --epochs 300 --arch resnet18 --data_name cifar10_20000_4class --train_data_drop_last --train_mode inst_suppress --not_shuffle_train_data"

# MYCOMMEND2="python main.py --batch_size 512 --epochs 300 --arch resnet18 --data_name cifar10_20000_4class --train_mode inst_suppress --not_shuffle_train_data"

# whole_cifar10 DBindex_cluster_momentum_kmeans_wholeset DBindex_cluster_momentum_kmeans_repeat_v2 normal_48210871_1_20220316164538_0.5_200_512_1000_model
# normal_49260763_1_20220322155227_0.5_200_512_200_piermaro_model       200_epoch_whole_cifar10_base
# normal_48899799_1_20220319160643_0.5_200_512_1000_model       880_epoch_base
# normal_49449742_1_20220323194707_0.5_200_512_200_piermaro_model  200_epch_base_pytroch_transform
# 10 30 100
# 200 500
# 1000 1500
# random_initial_model1
# train_dbindex_loss

# python3 -u -m torch.distributed.launch --nproc_per_node=4 ssl_perturbation_v2_ddp.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 12 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step 10 --min_min_attack_fn eot_v1 --strong_aug --not_shuffle_train_data --eot_size 3 --batch_size 1024 --num_workers 8 --gpu_num 4

PIERMARO_MYCOMMEND="python3 -u -m torch.distributed.launch --nproc_per_node=4 ssl_perturbation_v2_ddp.py --piermaro_whole_epoch ${WHOLE_EPOCH} --epochs $SINGLE_EPOCH --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 12 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step 10 --min_min_attack_fn eot_v1 --strong_aug --not_shuffle_train_data --eot_size 3 --batch_size 1024 --num_workers 8 --gpu_num 4 --dbindex_weight 0.3 --simclr_weight 0 --train_simclr_dbindex"

PIERMARO_MYCOMMEND2=""

PIERMARO_MYCOMMEND3=""

PIERMARO_MYCOMMEND2="No_commend2"
PIERMARO_MYCOMMEND3="No_commend3"

echo "MYCOMMEND=\"${PIERMARO_MYCOMMEND}\"" > re_job_cmd/${DATE_NAME}.sh
echo "MYCOMMEND2=\"${PIERMARO_MYCOMMEND2}\"" >> re_job_cmd/${DATE_NAME}.sh
echo "MYCOMMEND3=\"${PIERMARO_MYCOMMEND3}\"" >> re_job_cmd/${DATE_NAME}.sh
echo "MYGRES=\"${MYGRES}\"" >> re_job_cmd/${DATE_NAME}.sh

PIERMARO_RETURN=`sh ./re_job.sh ${DATE_NAME}`
echo $PIERMARO_RETURN
PIERMARO_SLURM_JOB_ID=`echo $PIERMARO_RETURN | awk '{print $4}'`
SUBJOB_ID=$PIERMARO_SLURM_JOB_ID

# subjob_id_list
subjob_id_str=${SUBJOB_ID}

date >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log
echo $PIERMARO_SLURM_JOB_ID >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log
echo $JOB_INFO >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log
echo $PIERMARO_MYCOMMEND >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log
if [[ "$PIERMARO_MYCOMMEND2" != *"No_commend2"* ]]
then
    echo $PIERMARO_MYCOMMEND2 >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log
fi
if [[ "$PIERMARO_MYCOMMEND3" != *"No_commend3"* ]]
then
    echo $PIERMARO_MYCOMMEND3 >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log
fi
echo -e "---------------------------------------------------------------" >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log

for((i=1;i<${REJOB_TIMES};i++));
do
    test -e ./FLAG_ROOM/RUNNING_FLAG_${SUBJOB_ID}
    while [ $? -eq 0 ]
    do
        sleep 30s
        echo $i
        test -e ./FLAG_ROOM/RUNNING_FLAG_${SUBJOB_ID}
    done

    PIERMARO_RESTART_EPOCH=`expr $i \* $SINGLE_EPOCH`

    PIERMARO_MODEL_PATH=`ls ./results | grep ${PIERMARO_SLURM_JOB_ID}_1 | grep _piermaro_model | awk 'BEGIN{FS=".pth"} {print $1}'`
    MYCOMMEND="${PIERMARO_MYCOMMEND} --load_piermaro_model --load_piermaro_model_path ${PIERMARO_MODEL_PATH} --piermaro_restart_epoch ${PIERMARO_RESTART_EPOCH}"

    if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
    then
        PIERMARO_MODEL_PATH=`ls ./results | grep ${PIERMARO_SLURM_JOB_ID}_2 | grep _piermaro_model | awk 'BEGIN{FS=".pth"} {print $1}'`
        MYCOMMEND2="${PIERMARO_MYCOMMEND2} --load_piermaro_model --load_piermaro_model_path ${PIERMARO_MODEL_PATH} --piermaro_restart_epoch ${PIERMARO_RESTART_EPOCH}"
    fi

    if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
    then
        PIERMARO_MODEL_PATH=`ls ./results | grep ${PIERMARO_SLURM_JOB_ID}_3 | grep _piermaro_model | awk 'BEGIN{FS=".pth"} {print $1}'`
        MYCOMMEND3="${PIERMARO_MYCOMMEND3} --load_piermaro_model --load_piermaro_model_path ${PIERMARO_MODEL_PATH} --piermaro_restart_epoch ${PIERMARO_RESTART_EPOCH}"
    fi

    echo "MYCOMMEND=\"${MYCOMMEND}\"" > re_job_cmd/${DATE_NAME}.sh
    echo "MYCOMMEND2=\"${MYCOMMEND2}\"" >> re_job_cmd/${DATE_NAME}.sh
    echo "MYCOMMEND3=\"${MYCOMMEND3}\"" >> re_job_cmd/${DATE_NAME}.sh
    echo "MYGRES=\"${MYGRES}\"" >> re_job_cmd/${DATE_NAME}.sh

    SUBJOB_RETURN=`sh ./re_job.sh ${DATE_NAME}`
    echo $SUBJOB_RETURN
    SUBJOB_ID=`echo $SUBJOB_RETURN | awk '{print $4}'`
    subjob_id_str="${subjob_id_str} ${SUBJOB_ID}"

done

test -e ./FLAG_ROOM/RUNNING_FLAG_${SUBJOB_ID}
while [ $? -eq 0 ]
do
    sleep 30s
    echo $i
    test -e ./FLAG_ROOM/RUNNING_FLAG_${SUBJOB_ID}
done

date >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
echo $PIERMARO_SLURM_JOB_ID >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
echo $JOB_INFO >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
echo $MYCOMMEND >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
then
    echo $MYCOMMEND2 >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
fi
if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
then
    echo $MYCOMMEND3 >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
fi
echo ${subjob_id_str} >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
echo -e "---------------------------------------------------------------" >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
