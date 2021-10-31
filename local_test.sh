#!/bin/bash
cd `dirname $0`

array=(3 10 20 30 50 75 100 300 500 1000 5000 10000)

for element in ${array[@]}
do
echo $element

python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape ${element} 3 32 32 --epsilon 8 --num_steps 5 --step_size 6 --attack_type min-min --perturb_type classwise --universal_train_target 'classwise' --train_step 10 --epochs 151 --min_min_attack_fn eot_v2 --strong_aug --local_dev --no_save >./local_log/8_5_6_eotv2_noise_${element}class.log 2>&1 &

MYPID=$!
echo $MYPID
sleep 300
kill -9 $MYPID
done


for element in ${array[@]}
do
echo $element
python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape ${element} 3 32 32 --epsilon 16 --num_steps 5 --step_size 3.2 --attack_type min-min --perturb_type classwise --universal_train_target 'classwise' --train_step 10 --epochs 151 --min_min_attack_fn eot_v2 --strong_aug --local_dev --no_save >./local_log/16_5_3.2_eotv2_noise_${element}class.log 2>&1 &

MYPID=$!
echo $MYPID
sleep 300
kill -9 $MYPID
done


for element in ${array[@]}
do
echo $element
python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape ${element} 3 32 32 --epsilon 16 --num_steps 5 --step_size 3.2 --attack_type min-min --perturb_type classwise --universal_train_target 'classwise' --train_step 10 --epochs 151 --min_min_attack_fn non_eot --strong_aug --local_dev --no_save >./local_log/16_5_3.2_noneot_noise_${element}class.log 2>&1 &

MYPID=$!
echo $MYPID
sleep 300
kill -9 $MYPID
done

