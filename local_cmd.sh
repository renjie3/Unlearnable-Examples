# models=(unlearnable_theory_43990132_1_20220116172552_0.5_512_1000_final_model unlearnable_theory_43990132_2_20220116172552_0.5_512_1000_final_model unlearnable_theory_43990132_3_20220116172552_0.5_512_1000_final_model unlearnable_theory_43990133_1_20220116172556_0.5_512_1000_final_model unlearnable_theory_43990133_2_20220116172556_0.5_512_1000_final_model unlearnable_theory_43990133_3_20220116172556_0.5_512_1000_final_model unlearnable_theory_43990134_1_20220116172605_0.5_512_1000_final_model unlearnable_theory_43990134_2_20220116172605_0.5_512_1000_final_model unlearnable_theory_43990134_3_20220116172605_0.5_512_1000_final_model unlearnable_theory_43990135_1_20220116172633_0.5_512_1000_final_model)
# test_datas=(hierarchical_test_knn256 hierarchical_test_knn64 hierarchical_test_knn16 hierarchical_test_knn4)

# for((i=0;i<4;i++));
# do
#     for((j=0;j<10;j++));
#     do
#         MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical_knn4 --theory_test_data ${test_datas[${i}]} --random_drop_feature_num 0 --theory_normalize --just_test --load_model --load_model_path ${models[${j}]} --local_dev 2 --no_save"

#         echo $MY_CMD
#         # echo ${MY_CMD}>>local_history.log
#         $MY_CMD
#     done
# done


MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical_period_dim20_knn4 --theory_test_data hierarchical_period_dim20_test_knn16 --random_drop_feature_num 4 4 4 4 4 --theory_normalize --local 2 --no_save"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
$MY_CMD


# MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type just_test --plot_be_mode single_augmentation --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --augmentation ReCrop_Hflip_Bri --augmentation_prob 1.0 0.5 0.0 0.2 --load_model --load_model_path unlearnable_cleantrain_42873265_1_20211228104344_0.5_512_1000_final_model --not_shuffle_train_data --mix no --local_dev 2 --no_save"

# echo $MY_CMD
# echo ${MY_CMD}>>local_history.log
# $MY_CMD

# unlearnable_theory_43990132_1_20220116172552_0.5_512_1000_statistics
# unlearnable_theory_43990132_2_20220116172552_0.5_512_1000_statistics
# unlearnable_theory_43990132_3_20220116172552_0.5_512_1000_statistics
# unlearnable_theory_43990133_1_20220116172556_0.5_512_1000_statistics
# unlearnable_theory_43990133_2_20220116172556_0.5_512_1000_statistics
# unlearnable_theory_43990133_3_20220116172556_0.5_512_1000_statistics
# unlearnable_theory_43990134_1_20220116172605_0.5_512_1000_statistics
# unlearnable_theory_43990134_2_20220116172605_0.5_512_1000_statistics
# unlearnable_theory_43990134_3_20220116172605_0.5_512_1000_statistics
# unlearnable_theory_43990135_1_20220116172633_0.5_512_1000_statistics

# random_samplewise_mnist
# random_samplewise_center_all_18_budget128
# unlearnable_cleantrain_43810886_1_20220115005203_0
# unlearnable_cleantrain_43810887_1_20220115005214_0
# unlearnable_cleantrain_43810887_2_20220115005214_0
# unlearnable_cleantrain_43810886_2_20220115005203_0
# unlearnable_cleantrain_43810884_1_20220115005150_0

# samplewise_mnist
# samplewise_all_center_18_128
# unlearnable_cleantrain_43809527_1_20220115003409_0
# unlearnable_cleantrain_43809528_1_20220115003402_0
# unlearnable_cleantrain_43809528_2_20220115003402_0
# unlearnable_cleantrain_43809527_2_20220115003409_0
# unlearnable_cleantrain_41798562_2_20211210195041_0

# for i in {4..28..4}
# do
    
#     MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type clean_train --train_step 10 --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --shuffle_train_perturb_data --gray_train gray --gray_test freq_cifar10_1024_4class_gray_low_${i} --augmentation ReCrop_Hflip_Bri --load_model --load_model_path unlearnable_cleantrain_42483880_2_20211221153535_0.5_512_1000_final_model --just_test --local_dev 2 --no_save"

#     # echo $MY_CMD
#     # echo ${MY_CMD}>>local_history.log
#     $MY_CMD
# done

# python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type test_find_positive_pair --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --not_shuffle_train_data --augmentation ReCrop_Hflip --batch_size 256 --gray_train mnist_train_4gray --gray_test mnist_train_4gray --load_model --load_model_path unlearnable_cleantrain_43557131_1_20220111124039_0.5_256_1000_final_model --local_dev 2 --no_save

# unlearnable_cleantrain_42483880_1_20211221153535_0.5_512_1000_final_model
# unlearnable_cleantrain_41406147_1_20211203004142_0.5_512_1000_statistics pos
# unlearnable_cleantrain_41459622_2_20211203162721_0.5_512_1000_final_model normal
# unlearnable_cleantrain_41406143_1_20211203004141_0.5_512_1000_statistics pos/neg
# unlearnable_cleantrain_42873264_1_20211228104252_0.5_512_1000_final_model --augmentation_prob 0.0 0.5 0.8 0.2
# unlearnable_cleantrain_42873264_2_20211228104252_0.5_512_1000_final_model --augmentation_prob 1.0 0.0 0.8 0.2
# unlearnable_cleantrain_42873265_1_20211228104344_0.5_512_1000_final_model --augmentation_prob 1.0 0.5 0.0 0.2
# unlearnable_cleantrain_42873265_2_20211228104344_0.5_512_1000_final_model --augmentation_prob 1.0 0.5 0.8 0.0

# grayshiftlarge_font_randomdigit_mnist grayshift_mnist grayshift_font_mnist mnist_train_4gray mnist_4position
# unlearnable_cleantrain_43556970_1_20220111122328_0.5_256_1000_statistics mnist_train_4gray
# unlearnable_cleantrain_43556970_2_20220111122328_0.5_256_1000_statistics CIFAR10 gray
# unlearnable_cleantrain_43557131_1_20220111124039_0.5_256_1000_statistics not_shuffle_train

# unlearnable_cleantrain_42873264_1_20211228104252_0.5_512_1000_statistics
# unlearnable_cleantrain_42873264_2_20211228104252_0.5_512_1000_statistics
# unlearnable_cleantrain_42873265_1_20211228104344_0.5_512_1000_statistics
# unlearnable_cleantrain_42873265_2_20211228104344_0.5_512_1000_statistics

# random_samplewise_mnist
# unlearnable_cleantrain_43810886_1_20220115005203_0
# unlearnable_cleantrain_43810887_1_20220115005214_0
# unlearnable_cleantrain_43810887_2_20220115005214_0
# unlearnable_cleantrain_43810886_2_20220115005203_0
# unlearnable_cleantrain_43810884_1_20220115005150_0

# samplewise_mnist
# unlearnable_cleantrain_43809527_1_20220115003409_0
# unlearnable_cleantrain_43809528_1_20220115003402_0
# unlearnable_cleantrain_43809528_2_20220115003402_0
# unlearnable_cleantrain_43809527_2_20220115003409_0
# unlearnable_cleantrain_41798562_2_20211210195041_0