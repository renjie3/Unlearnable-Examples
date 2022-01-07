MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 2048 3 32 32 --perturb_type clean_train_softmax --epochs 1000 --min_min_attack_fn non_eot --batch_size 2048 --strong_aug --class_4 --not_shuffle_train_data --augmentation simclr --class_4_train_size 2048 --local_dev 2 --no_save"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
$MY_CMD

# for i in {4..28..4}
# do
    
#     MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type clean_train --train_step 10 --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --shuffle_train_perturb_data --gray_train gray --gray_test freq_cifar10_1024_4class_gray_low_${i} --augmentation ReCrop_Hflip_Bri --load_model --load_model_path unlearnable_cleantrain_42483880_2_20211221153535_0.5_512_1000_final_model --just_test --local_dev 2 --no_save"

#     # echo $MY_CMD
#     # echo ${MY_CMD}>>local_history.log
#     $MY_CMD
# done

# unlearnable_cleantrain_42483880_1_20211221153535_0.5_512_1000_final_model
# unlearnable_cleantrain_41406147_1_20211203004142_0.5_512_1000_statistics pos
# unlearnable_cleantrain_41459622_2_20211203162721_0.5_512_1000_final_model normal
# unlearnable_cleantrain_41406143_1_20211203004141_0.5_512_1000_statistics pos/neg
# unlearnable_cleantrain_42873264_1_20211228104252_0.5_512_1000_final_model --augmentation_prob 0.0 0.5 0.8 0.2
# unlearnable_cleantrain_42873264_2_20211228104252_0.5_512_1000_final_model --augmentation_prob 1.0 0.0 0.8 0.2
# unlearnable_cleantrain_42873265_1_20211228104344_0.5_512_1000_final_model --augmentation_prob 1.0 0.5 0.0 0.2
# unlearnable_cleantrain_42873265_2_20211228104344_0.5_512_1000_final_model --augmentation_prob 1.0 0.5 0.8 0.0