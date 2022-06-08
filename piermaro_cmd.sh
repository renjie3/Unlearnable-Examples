#!/bin/bash
cd `dirname $0`
cd ..
PIERMARO_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $PIERMARO_JOB_ROOT_PATH
DATE_NAME=${1}
echo $$

WHOLE_EPOCH=42
SINGLE_EPOCH=2
REJOB_TIMES=`expr $WHOLE_EPOCH / $SINGLE_EPOCH`
MYGRES="gpu:v100s:1"
MYCPU="2"

JOB_INFO="cifar10 based on samplewise"
# python main_train_transfer.py --method simsiam --arch resnet18 --dataset cifar10 --batch_size 512 --eval_batch_size 512 --num_workers 8 --knn_eval_freq 10 --lr 0.06 --wd 5e-4 --gpu 0 --trial 0 --pre_load_name unlearnable_samplewise_53967350_1_20220516215150_0.5_512_2_checkpoint_perturbation_epoch_40
# unlearnable_samplewise_52866270_1_20220507200522_0.5_512_2_checkpoint_perturbation
# unlearnable_samplewise_54120121_1_20220519102435_0.5_512_2_checkpoint_model_epoch_40
# unlearnable_samplewise_54120137_1_20220519103718_0.5_512_2_checkpoint_perturbation_epoch_40

# PIERMARO_MYCOMMEND="python simclr_transfer.py --piermaro_whole_epoch ${WHOLE_EPOCH} --epochs ${SINGLE_EPOCH} --arch resnet18 --perturbation_budget 1 --pre_load_name unlearnable_samplewise_53599430_1_20220512202310_0.5_512_2_checkpoint_perturbation_epoch_30_2 --train_data_type CIFAR10 --samplewise --pytorch_aug"
# not_use_normalized

PIERMARO_MYCOMMEND="python3 -u ssl_perturbation_v2.py --piermaro_whole_epoch ${WHOLE_EPOCH} --epochs ${SINGLE_EPOCH} --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step 20 --min_min_attack_fn eot_v1 --strong_aug --eot_size 1 --shuffle_train_perturb_data --pytorch_aug --simclr_weight 1 --linear_noise_dbindex_weight 0 --seed 1"

PIERMARO_MYCOMMEND2=""

PIERMARO_MYCOMMEND3=""

PIERMARO_MYCOMMEND2="No_commend2"
PIERMARO_MYCOMMEND3="No_commend3"

echo "MYCOMMEND=\"${PIERMARO_MYCOMMEND}\"" > re_job_cmd/${DATE_NAME}.sh
echo "MYCOMMEND2=\"${PIERMARO_MYCOMMEND2}\"" >> re_job_cmd/${DATE_NAME}.sh
echo "MYCOMMEND3=\"${PIERMARO_MYCOMMEND3}\"" >> re_job_cmd/${DATE_NAME}.sh
echo "MYGRES=\"${MYGRES}\"" >> re_job_cmd/${DATE_NAME}.sh
echo "MYCPU=\"${MYCPU}\"" >> re_job_cmd/${DATE_NAME}.sh

PIERMARO_RETURN=`sh ./re_job_cpu.sh ${DATE_NAME}`
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

    PIERMARO_MODEL_PATH=`ls ./results | grep ${PIERMARO_SLURM_JOB_ID}_1 | grep _piermaro_model | grep -v retrain | awk 'BEGIN{FS=".pth"} {print $1}'`
    MYCOMMEND="${PIERMARO_MYCOMMEND} --load_piermaro_model --load_piermaro_model_path ${PIERMARO_MODEL_PATH} --piermaro_restart_epoch ${PIERMARO_RESTART_EPOCH}"

    if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
    then
        PIERMARO_MODEL_PATH=`ls ./results | grep ${PIERMARO_SLURM_JOB_ID}_2 | grep _piermaro_model | grep -v retrain | awk 'BEGIN{FS=".pth"} {print $1}'`
        MYCOMMEND2="${PIERMARO_MYCOMMEND2} --load_piermaro_model --load_piermaro_model_path ${PIERMARO_MODEL_PATH} --piermaro_restart_epoch ${PIERMARO_RESTART_EPOCH}"
    fi

    if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
    then
        PIERMARO_MODEL_PATH=`ls ./results | grep ${PIERMARO_SLURM_JOB_ID}_3 | grep _piermaro_model | grep -v retrain | awk 'BEGIN{FS=".pth"} {print $1}'`
        MYCOMMEND3="${PIERMARO_MYCOMMEND3} --load_piermaro_model --load_piermaro_model_path ${PIERMARO_MODEL_PATH} --piermaro_restart_epoch ${PIERMARO_RESTART_EPOCH}"
    fi

    echo "MYCOMMEND=\"${MYCOMMEND}\"" > re_job_cmd/${DATE_NAME}.sh
    echo "MYCOMMEND2=\"${MYCOMMEND2}\"" >> re_job_cmd/${DATE_NAME}.sh
    echo "MYCOMMEND3=\"${MYCOMMEND3}\"" >> re_job_cmd/${DATE_NAME}.sh
    echo "MYGRES=\"${MYGRES}\"" >> re_job_cmd/${DATE_NAME}.sh
    echo "MYCPU=\"${MYCPU}\"" >> re_job_cmd/${DATE_NAME}.sh

    SUBJOB_RETURN=`sh ./re_job_cpu.sh ${DATE_NAME}`
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
